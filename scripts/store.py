"""Write a retargeted motion as the layers of a Rerun store.

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
there, and the goal is recoverable from nothing else. The joint chain under
``/<robot>/tf`` already draws the robot where the solve put it, so the gap between the
robot and its goal triads is the error of the solve, in one file.

Every layer of one motion shares an ``application_id`` and a ``recording_id``, and the
viewer pools data by those two ids, so ``rerun <root>/*/<motion>.rrd`` opens them as one
recording. Each file also embeds the layout, so any one of them opens sensibly alone.
Blueprints do not merge and the last one loaded wins, so pass the ``.rbl`` last.

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

A pose names a transform frame instead of riding the entity hierarchy, which is what
Rerun recommends for robot data and what ``rr.urdf`` writes. The layer prefixes the frame
name, so a source body and a robot link of the same name stay apart, and so do
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
import rerun as rr
import rerun.blueprint as rrb
from common import quat_mul

#: Shared by every motion, so a directory of them registers as one dataset.
APPLICATION_ID: str = "lite-motion-tracking"

#: Every temporal column carries both. A loader indexes by ``frame``. ``time`` is what
#: plays the motion at true speed, and what lines up two layers written at different
#: rates, whose frame counts do not match.
TIMELINE_FRAME: str = "frame"
TIMELINE_TIME: str = "time"

BASE_LAYER: str = "base"
WORLD_FRAME: str = "world"

#: Metres. Long enough to clear the limbs of the robot, which would otherwise hide it.
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

    This layer keeps the source inside the store, so a motion can be derived again
    without the upstream dataset. It holds the joints of a source robot, and the world
    poses of the bodies of the source, robot or human, in its own frame convention, so
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
) -> Path:
    """Write ``<robot>/<name>.rrd``: everything one robot needs for one motion.

    The layer holds the joints the solve reached and the world pose ``body_names`` is
    meant to reach. With ``geometry``, it also holds the robot's geometry and the URDF's
    static transforms, so the file animates the robot on its own. Without it, the
    viewer takes them from the asset that :func:`write_asset` writes. Everything sits
    under ``/<robot>/``, so a second robot can open beside it.

    ``body_pos`` and ``body_quat`` are the **goal**, in this robot's frame convention,
    which is what a tracking reward compares against, and they go under ``/<robot>/goal/``
    so no reader mistakes them for where the robot is. ``source_body_names`` names the
    body of the source rig each goal came from, in the order of ``body_names``, because
    two rigs need not share a vocabulary.

    ``body_names`` must contain the root link of the URDF, whose goal places the robot.
    Its goal is the base trajectory of the retarget, so the goal and the reached pose are
    the same pose for that one body.

    ``metrics`` are the quality numbers of ``retarget.motion_metrics``, kept as the
    ``metrics`` property so a motion carries how well it was made.
    """
    robot = _robot_name(urdf_path)
    tree = rr.urdf.UrdfTree.from_file_path(
        urdf_path,
        frame_prefix=f"{robot}/",
        static_transform_entity_path=f"{robot}/tf_static",
    )
    frames = joint_pos.shape[0]
    indexes = _indexes(frames, fps)

    path = root / robot / f"{name}.rrd"
    path.parent.mkdir(parents=True, exist_ok=True)
    with rr.RecordingStream(APPLICATION_ID, recording_id=name) as stream:
        stream.save(str(path), _robot_blueprint(robot))
        stream.send_recording_name(name)
        stream.send_property(robot, rr.AnyValues(
            robot=robot,
            fps=fps,
            joint_names=list(joint_names),
            body_names=list(body_names),
            source_body_names=list(source_body_names),
        ))
        stream.send_property("metrics", rr.AnyValues(**metrics))
        if geometry:
            tree.log_urdf_to_recording(recording=stream)
        _send_joints(stream, robot, fps, joint_names, joint_pos)
        _send_bodies(stream, f"{robot}/goal", fps, body_names, body_pos, body_quat)

        # The frame graph that puts the geometry where the joints say it is. Each joint
        # is its own chunk, so a row still carries one parent-child relation.
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
            columns=rr.Transform3D.columns(
                translation=body_pos[:, placement],
                quaternion=_wxyz_to_xyzw(body_quat[:, placement]),
                child_frame=[f"{robot}/{root_link}"] * frames,
                parent_frame=[WORLD_FRAME] * frames,
            ),
        )
    return path


def write_asset(root: Path, urdf_path: Path) -> Path:
    """Write ``assets/<robot>.rrd``: the URDF's geometry and static transforms.

    An asset may hold static data only, and this one holds nothing else. Its recording
    id is its own, because the viewer attaches an asset to every motion of the dataset,
    whatever their ids.
    """
    robot = _robot_name(urdf_path)
    tree = rr.urdf.UrdfTree.from_file_path(
        urdf_path,
        frame_prefix=f"{robot}/",
        static_transform_entity_path=f"{robot}/tf_static",
    )
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
    """The name the URDF gives the robot, which names its layer and its blueprint."""
    return rr.urdf.UrdfTree.from_file_path(urdf_path).name


def _send_joints(
    stream: rr.RecordingStream,
    layer: str,
    fps: int,
    joint_names: tuple[str, ...],
    joint_pos: np.ndarray,
) -> None:
    """Write one layer's joint positions, and the velocities differenced from them."""
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
    """The two time axes of a temporal column, over ``frames`` samples at ``fps``."""
    steps = np.arange(frames)
    return [
        rr.TimeColumn(TIMELINE_FRAME, sequence=steps),
        rr.TimeColumn(TIMELINE_TIME, duration=steps / float(fps)),
    ]


def _wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(quat[..., [1, 2, 3, 0]], dtype=np.float32)


def _axis_triad() -> rr.Arrows3D:
    """A red/green/blue triad along +X/+Y/+Z, as geometry that can name a frame."""
    return rr.Arrows3D(
        vectors=np.eye(3) * AXIS_LENGTH,
        colors=[[255, 70, 70], [70, 220, 70], [90, 140, 255]],
    )


def _velocities(
    pos: np.ndarray, quat: np.ndarray, fps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Linear and angular velocity of a ``(T, NB, ...)`` pose trajectory, world frame.

    Backward difference, so frame 0 has no earlier sample and keeps its zero. The angular
    part has to come from the rotation between two frames, because an Euler rate is not
    an angular velocity, and that rotation has to go the short way round: a quaternion
    can flip sign between frames, and the flipped one reads as a near-full turn.
    """
    linear = np.zeros_like(pos)
    linear[1:] = np.diff(pos, axis=0) * fps

    step = quat_mul(quat[1:], quat[:-1] * (1.0, -1.0, -1.0, -1.0))
    step = np.where(step[..., :1] < 0.0, -step, step)
    axis = step[..., 1:]
    sin_half = np.linalg.norm(axis, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, step[..., 0])
    # angle / sin_half tends to 2 as the step vanishes, which is the value to use there
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

    ``group`` is ``<layer>/body`` where the poses are where the bodies are, and
    ``<layer>/goal`` where they are where the bodies are meant to be.
    """
    frames = body_pos.shape[0]
    indexes = _indexes(frames, fps)
    for i, body in enumerate(body_names):
        frame = f"{group}/{body}"
        stream.send_columns(
            f"/{group}/poses/{body}",
            indexes=indexes,
            columns=rr.Transform3D.columns(
                translation=body_pos[:, i],
                quaternion=_wxyz_to_xyzw(body_quat[:, i]),
                child_frame=[frame] * frames,
                parent_frame=[WORLD_FRAME] * frames,
            ),
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
        # The velocities are siblings of the poses, not children: an Arrows3D below a
        # posed entity would inherit that pose and turn its vectors.
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
    """Start one group's velocity arrows switched off.

    At walking speed the arrows are metres long and bury the robot. Hiding keeps them in
    the entity tree, one click away; excluding them would take a view filter edit.
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
    """The robot in 3D beside its joint plots, with its goal triads over it.

    Seeing both is the point: the gap between a goal triad and the link under it is the
    error of the solve, frame by frame. The source layer, if it is open too, draws its
    own triads in its own frame convention, so it is off by a fixed angle by
    construction, and its velocities start switched off with everything else.
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
