"""Write a retargeted motion as the layers of a Rerun store, and read a robot layer back.

A layer is a stage of the pipeline, a directory, and one file per motion:

    base/        <motion>.rrd   the source clip as published, at its own rate
    <robot>/     <motion>.rrd   everything the robot needs, at the control rate
    assets/      <robot>.rrd    the geometry of that robot, once for every motion
    blueprints/  <robot>.rbl    the viewer layout for that robot

``base/`` is provenance. Nothing downstream reads it, and a motion is complete without
it. ``<robot>/`` is what training reads: its joints reset the robot, and its body poses
are the goal the robot tracks.

A robot layer stores the **goal**, not the pose the solve reached, and says so in the
path: ``/<robot>/goal/`` and not ``/<robot>/body/``. The reached pose is the forward
kinematics of the joints in the same file, so storing it would repeat what is already
there. Nothing else in the file gives the goal. The joint chain under
``/<robot>/tf`` already draws the robot where the solve put it, so the gap between the
robot and its goal triads is the error of the solve, in one file.

Every layer of one motion shares an ``application_id`` and a ``recording_id``. The viewer
pools data by those two ids, so ``rerun <root>/*/<motion>.rrd`` opens the layers as one
recording. Each file also embeds the layout, so any one of them opens sensibly alone.
Blueprints do not merge, and the last one that the viewer loads wins, so put the ``.rbl``
last on the command line.

Both layers store bodies and joints the same way, so one writer serves both:

    /base/body/poses/<body>           Transform3D, an entity per body, a row per frame
    /base/body/frames/<body>          an axis triad bound to that body's frame, static
    /base/body/linear_velocities      Arrows3D, a row per frame, an arrow per body
    /base/body/angular_velocities     Arrows3D
    /<robot>/goal/...                 the same four, for where the bodies are meant to be
    /<layer>/joint/positions          Scalars, a row per frame, a scalar per joint
    /<layer>/joint/velocities         Scalars
    /<robot>/tf                       a Transform3D per joint, and the root placement
    /<robot>/tf_static                the URDF's fixed transforms
    /<robot>/visual_geometries/...    the URDF's geometry, named by ``rr.urdf``

A pose names a transform frame and does not use the entity hierarchy. Rerun recommends
this for robot data, and ``rr.urdf`` writes it. The layer prefixes the frame name, so a
source body and a robot link of the same name stay apart, and so do
``<robot>/goal/<link>`` and the ``<robot>/<link>`` that the joint chain already defines.
A frame has one parent.

Every entity drawn in 3D must name a frame. One that names none is an orphan: the viewer
invents a frame for it, finds no route to the root of the view, and draws an error in its
place. Scalars are exempt, because the viewer plots them against time.

The geometry of a robot is 9 MB, more than most motions, so ``assets/<robot>.rrd`` holds
it once, as static data under a recording id of its own. ``rerun --asset`` (Rerun 0.38.1
and later) attaches it to every motion the viewer opens:

    rerun <root>/*/<motion>.rrd --asset <root>/assets/<robot>.rrd <root>/blueprints/<robot>.rbl

Only the native viewer loads assets. A motion that the web viewer must draw on its own,
such as one the Hub card links to, carries its own copy of the geometry instead.

Rerun stores a rotation as xyzw. MuJoCo, and every array this module takes, uses wxyz.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from rerun.chunk import RrdReader

from .kinematics import quat_mul
from .robots import model_hash

#: The rate of every written motion, which is the control rate of the training
#: environment. A motion at another rate would force the loader to interpolate.
FPS: int = 50

#: Every motion uses this id, so a directory of motions registers as one dataset.
APPLICATION_ID: str = "lite-motion-tracking"

#: Every temporal column has both timelines. A loader indexes by ``frame``. ``time`` plays
#: the motion at true speed, and lines up two layers at different rates, whose frame
#: counts do not match.
TIMELINE_FRAME: str = "frame"
TIMELINE_TIME: str = "time"

#: The layer that holds the source clip as published.
BASE_LAYER: str = "base"
WORLD_FRAME: str = "world"

#: The length of each axis of a triad, in metres. It is long enough to clear the limbs of
#: the robot, which would otherwise hide it.
AXIS_LENGTH: float = 0.12


def write_base_layer(
    root: Path,
    name: str,
    fps: int,
    joint_names: tuple[str, ...],
    joint_pos: np.ndarray,
    body_names: tuple[str, ...],
    body_pos: np.ndarray,
    body_quat: np.ndarray,
) -> Path:
    """Write ``base/<name>.rrd``: the source clip as published, at its own rate.

    This layer keeps the source inside the store, so you can derive a motion again without
    the upstream dataset. It holds the joints of a source robot, and the world poses of the
    bodies of the source, robot or human, in the frame convention of the source. Thus
    nothing here is directly comparable to a robot layer. A human source has no joint
    angles to keep, and ``joint_names`` is empty.

    ``body_pos`` is ``(T, NB, 3)`` and ``body_quat`` ``(T, NB, 4)`` wxyz, both world.
    """
    path = root / BASE_LAYER / f"{name}.rrd"
    path.parent.mkdir(parents=True, exist_ok=True)
    with rr.RecordingStream(APPLICATION_ID, recording_id=name) as stream:
        stream.save(str(path), _base_blueprint())
        stream.send_recording_name(name)
        stream.send_property("motion", rr.AnyValues(
            fps=fps, joint_names=list(joint_names), body_names=list(body_names)
        ))
        if joint_names:
            _send_joints(stream, BASE_LAYER, fps, joint_names, joint_pos)
        _send_bodies(stream, f"{BASE_LAYER}/body", fps, body_names, body_pos, body_quat)
    return path


def write_robot_layer(
    root: Path,
    name: str,
    urdf_path: Path,
    fps: int,
    joint_names: tuple[str, ...],
    joint_pos: np.ndarray,
    body_names: tuple[str, ...],
    source_body_names: tuple[str, ...],
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    geometry: bool,
    metrics: dict[str, float],
    active_joints: tuple[str, ...],
) -> Path:
    """Write ``<robot>/<name>.rrd``: everything one robot needs for one motion.

    The layer holds the joints that the solve reached, and the world pose that each of
    ``body_names`` is meant to reach. With ``geometry``, it also holds the geometry of the
    robot and the static transforms of the URDF, so the file animates the robot on its own.
    Without ``geometry``, the viewer takes them from the asset that :func:`write_asset`
    writes. Everything is under ``/<robot>/``, so a second robot can open beside it.

    ``body_pos`` and ``body_quat`` are the **goal**, in the frame convention of this robot,
    which is what a tracking reward compares against. They go under ``/<robot>/goal/`` so
    that no reader mistakes them for where the robot is. ``source_body_names`` names the
    body of the source rig that each goal came from, in the order of ``body_names``,
    because two rigs need not share a vocabulary.

    ``body_names`` must contain the root link of the URDF, whose goal places the robot.
    Its goal is the base trajectory of the retarget, so the goal and the reached pose are
    the same pose for that one body.

    ``metrics`` are the quality numbers of ``retargeting.motion_metrics``. The layer keeps
    them as the ``metrics`` property, so a motion records how well the retarget went.

    ``active_joints`` are the joints that the source drives, kept as
    ``active_joint_names``. Every other joint holds its default, 0, so a motion says which
    part of the robot it can train.

    The ``model`` property is the ``robots.model_hash`` of the MJCF beside the URDF, so that a
    reader can tell which version of the robot a motion is for.
    """
    robot = _robot_name(urdf_path)
    tree = _urdf_tree(urdf_path, robot)
    indexes = _indexes(joint_pos.shape[0], fps)

    path = root / robot / f"{name}.rrd"
    path.parent.mkdir(parents=True, exist_ok=True)
    with rr.RecordingStream(APPLICATION_ID, recording_id=name) as stream:
        stream.save(str(path), _robot_blueprint(robot))
        stream.send_recording_name(name)
        stream.send_property(robot, rr.AnyValues(
            robot=robot,
            model=model_hash(urdf_path.parent.parent / "mjcf" / f"{urdf_path.stem}.xml"),
            fps=fps,
            joint_names=list(joint_names),
            body_names=list(body_names),
            source_body_names=list(source_body_names),
            active_joint_names=list(active_joints),
        ))
        stream.send_property("metrics", rr.AnyValues(**metrics))
        if geometry:
            tree.log_urdf_to_recording(recording=stream)
        _send_joints(stream, robot, fps, joint_names, joint_pos)
        _send_bodies(stream, f"{robot}/goal", fps, body_names, body_pos, body_quat)

        # The frame graph puts the geometry where the joints say it is. Each joint is a chunk
        # of its own, so a row carries one parent-child relation.
        for joint in tree.joints():
            if joint.name not in joint_names:
                continue
            angles = joint_pos[:, joint_names.index(joint.name)].astype(np.float64)
            stream.send_columns(
                f"/{robot}/tf",
                indexes=indexes,
                columns=joint.compute_transform_columns(angles, clamp=False),
            )

        root_link = tree.root_link().name
        placement = body_names.index(root_link)
        stream.send_columns(
            f"/{robot}/tf",
            indexes=indexes,
            columns=_pose_columns(
                body_pos[:, placement], body_quat[:, placement], f"{robot}/{root_link}"
            ),
        )
    return path


def read_robot_layer(path: Path, robot: str) -> dict:
    """Read the ``robot`` layer at ``path``: its joints, its goals, and the joints that its source drives.

    ``model`` is the hash of the robot model that the layer was written for, and ``active``
    the joints that its source drives. Each is None for a layer older than its property.
    """
    columns: dict[str, list[pa.RecordBatch]] = {}
    properties = None
    for chunk in RrdReader(path).stream():
        entity = chunk.entity_path
        if entity == f"/__properties/{robot}":
            properties = chunk.to_record_batch().to_pydict()
        elif entity == f"/{robot}/joint/positions" or entity.startswith(f"/{robot}/goal/poses/"):
            columns.setdefault(entity, []).append(chunk.to_record_batch())
    if properties is None:
        raise SystemExit(f"{path} has no /__properties/{robot}, so it is not a robot layer.")

    def values(entity: str, component: str, width: int) -> np.ndarray:
        table = pa.Table.from_batches(columns[entity]).sort_by("frame")
        flat = table.column(component).combine_chunks().flatten()
        if pa.types.is_fixed_size_list(flat.type):
            flat = flat.flatten()
        return flat.to_numpy().astype(np.float32).reshape(-1, width)

    joint_names = properties["joint_names"][0]
    bodies = properties["body_names"][0]
    poses = [f"/{robot}/goal/poses/{body}" for body in bodies]
    goal_quat = np.stack([values(entity, "Transform3D:quaternion", 4) for entity in poses], axis=1)
    return {
        "fps": properties["fps"][0][0],
        "model": properties["model"][0][0] if "model" in properties else None,
        "joint_names": joint_names,
        "active": properties["active_joint_names"][0] if "active_joint_names" in properties else None,
        "bodies": bodies,
        "sources": properties["source_body_names"][0],
        "joint_pos": values(f"/{robot}/joint/positions", "Scalars:scalars", len(joint_names)),
        "goal_pos": np.stack([values(entity, "Transform3D:translation", 3) for entity in poses], axis=1),
        "goal_quat": goal_quat[..., [3, 0, 1, 2]],  # Rerun stores xyzw.
    }


def write_asset(root: Path, urdf_path: Path) -> Path:
    """Write ``assets/<robot>.rrd``: the URDF's geometry and static transforms.

    An asset can hold only static data. Its recording id is its own, because the viewer
    attaches an asset to every motion of the dataset, whatever their ids.
    """
    robot = _robot_name(urdf_path)
    tree = _urdf_tree(urdf_path, robot)
    path = root / "assets" / f"{robot}.rrd"
    path.parent.mkdir(parents=True, exist_ok=True)
    with rr.RecordingStream(APPLICATION_ID, recording_id=f"{robot}_geometry") as stream:
        stream.save(str(path))
        tree.log_urdf_to_recording(recording=stream)
    return path


def write_blueprint(root: Path, urdf_path: Path) -> Path:
    """Write ``blueprints/<robot>.rbl``, the layout ``rerun`` takes after the layers."""
    robot = _robot_name(urdf_path)
    path = root / "blueprints" / f"{robot}.rbl"
    path.parent.mkdir(parents=True, exist_ok=True)
    _robot_blueprint(robot).save(APPLICATION_ID, path)
    return path


def _robot_name(urdf_path: Path) -> str:
    """Return the robot name in the URDF, which names its layer, asset, and blueprint."""
    return rr.urdf.UrdfTree.from_file_path(urdf_path).name


def _urdf_tree(urdf_path: Path, robot: str) -> rr.urdf.UrdfTree:
    """Load the URDF with its frames and static transforms under ``<robot>/``.

    The robot layer and the asset both load it this way, so the geometry of the asset names
    the frames that the joints of the layer move.
    """
    return rr.urdf.UrdfTree.from_file_path(
        urdf_path,
        frame_prefix=f"{robot}/",
        static_transform_entity_path=f"{robot}/tf_static",
    )


def _send_joints(
    stream: rr.RecordingStream,
    layer: str,
    fps: int,
    joint_names: tuple[str, ...],
    joint_pos: np.ndarray,
) -> None:
    """Write the joint positions of ``layer``, and the velocities differenced from them."""
    frames = joint_pos.shape[0]
    indexes = _indexes(frames, fps)
    joint_vel = np.zeros_like(joint_pos)
    joint_vel[1:] = np.diff(joint_pos, axis=0) * fps
    rows = [len(joint_names)] * frames
    for entity, values in (("positions", joint_pos), ("velocities", joint_vel)):
        stream.send_columns(
            f"/{layer}/joint/{entity}",
            indexes=indexes,
            columns=rr.Scalars.columns(scalars=values.reshape(-1)).partition(rows),
        )


def _indexes(frames: int, fps: int) -> list[rr.TimeColumn]:
    """Return the two timelines of a temporal column, for ``frames`` samples at ``fps``."""
    steps = np.arange(frames)
    return [
        rr.TimeColumn(TIMELINE_FRAME, sequence=steps),
        rr.TimeColumn(TIMELINE_TIME, duration=steps / float(fps)),
    ]


def _pose_columns(pos: np.ndarray, quat: np.ndarray, child_frame: str) -> rr.ComponentColumnList:
    """Return the columns that place ``child_frame`` in the world, a row per frame.

    ``pos`` is ``(T, 3)`` and ``quat`` ``(T, 4)`` wxyz.
    """
    frames = len(pos)
    return rr.Transform3D.columns(
        translation=pos,
        quaternion=np.ascontiguousarray(quat[..., [1, 2, 3, 0]], dtype=np.float32),
        child_frame=[child_frame] * frames,
        parent_frame=[WORLD_FRAME] * frames,
    )


def _axis_triad() -> rr.Arrows3D:
    """Return a red, green, and blue triad along +X, +Y, and +Z, as geometry that names a frame."""
    return rr.Arrows3D(
        vectors=np.eye(3) * AXIS_LENGTH,
        colors=[[255, 70, 70], [70, 220, 70], [90, 140, 255]],
    )


def _velocities(
    pos: np.ndarray, quat: np.ndarray, fps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the linear and angular world velocity of a ``(T, NB, ...)`` pose trajectory.

    Both are backward differences, so frame 0, which has no earlier sample, keeps zero. The
    angular velocity comes from the rotation between two frames, because an Euler rate is
    not an angular velocity. That rotation must take the short way: a quaternion can flip
    sign between frames, and the flipped quaternion reads as an almost full turn.
    """
    linear = np.zeros_like(pos)
    linear[1:] = np.diff(pos, axis=0) * fps

    step = quat_mul(quat[1:], quat[:-1] * (1.0, -1.0, -1.0, -1.0))
    step = np.where(step[..., :1] < 0.0, -step, step)
    axis = step[..., 1:]
    sin_half = np.linalg.norm(axis, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, step[..., 0])
    # As the step goes to zero, angle / sin_half goes to 2, which is the value to use there.
    scale = np.divide(
        angle, sin_half, out=np.full_like(angle, 2.0), where=sin_half > 1e-12
    )
    angular = np.zeros_like(pos)
    angular[1:] = axis * (scale * fps)[..., None]
    return linear, angular


def _send_bodies(
    stream: rr.RecordingStream,
    group: str,
    fps: int,
    body_names: tuple[str, ...],
    body_pos: np.ndarray,
    body_quat: np.ndarray,
) -> None:
    """Write a group of body poses, triads, and velocities under ``/<group>/``.

    ``group`` is ``<layer>/body`` when the poses are where the bodies are, and
    ``<layer>/goal`` when the poses are where the bodies are meant to be.
    """
    frames = body_pos.shape[0]
    indexes = _indexes(frames, fps)
    for i, body in enumerate(body_names):
        frame = f"{group}/{body}"
        stream.send_columns(
            f"/{group}/poses/{body}",
            indexes=indexes,
            columns=_pose_columns(body_pos[:, i], body_quat[:, i], frame),
        )
        stream.log(
            f"/{group}/frames/{body}",
            _axis_triad(),
            rr.CoordinateFrame(frame=frame),
            static=True,
        )

    linear, angular = _velocities(body_pos, body_quat, fps)
    rows = [len(body_names)] * frames
    for entity, vectors in (("linear", linear), ("angular", angular)):
        # The velocities are siblings of the poses, not children. An Arrows3D below a posed
        # entity would inherit that pose and turn its vectors.
        stream.log(
            f"/{group}/{entity}_velocities",
            rr.CoordinateFrame(frame=WORLD_FRAME),
            static=True,
        )
        stream.send_columns(
            f"/{group}/{entity}_velocities",
            indexes=indexes,
            columns=rr.Arrows3D.columns(
                origins=body_pos.reshape(-1, 3), vectors=vectors.reshape(-1, 3)
            ).partition(rows),
        )


def _hidden_velocities(group: str) -> dict[str, rrb.EntityBehavior]:
    """Return the overrides that hide the velocity arrows of ``group`` at the start.

    At walking speed, the arrows are metres long and bury the robot. A hidden entity stays
    in the entity tree, one click away. An excluded entity would need an edit of the view
    filter.
    """
    return {
        f"/{group}/{quantity}_velocities": rrb.EntityBehavior(visible=False)
        for quantity in ("linear", "angular")
    }


def _base_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/", name="source", overrides=_hidden_velocities(f"{BASE_LAYER}/body")
        ),
        rrb.TimePanel(state="expanded"),
    )


def _robot_blueprint(robot: str) -> rrb.Blueprint:
    """Return the layout: the robot in 3D beside its joint plots, with its goal triads over it.

    The view shows both, because the gap between a goal triad and the link under it is the
    error of the solve, frame by frame. If the source layer is open too, it draws its own
    triads in its own frame convention, so they are off by a fixed angle. Its velocities
    also start hidden.
    """
    hidden = (
        _hidden_velocities(f"{BASE_LAYER}/body") | _hidden_velocities(f"{robot}/goal")
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", name=robot, overrides=hidden),
            rrb.TimeSeriesView(origin=f"/{robot}/joint/positions", name="joint angles"),
            column_shares=[3, 1],
        ),
        rrb.TimePanel(state="expanded"),
    )
