"""Write a retargeted motion as the layers of a Rerun store.

A layer is a stage of the pipeline, a directory, and one file per motion:

    reference/   <motion>.rrd   the bodies a solve aims at, robot-agnostic
    <robot>/     <motion>.rrd   the joints it reached, and the geometry to draw them
    blueprints/  <robot>.rbl    the viewer layout for that robot

Every layer of one motion shares an ``application_id`` and a ``recording_id``, and the
viewer pools data by those two ids, so ``rerun <root>/*/<motion>.rrd`` opens them as one
recording. Each file also embeds the layout, so any one of them opens sensibly alone.
Blueprints do not merge and the last one loaded wins, so pass the ``.rbl`` last.

Both layers store bodies the same way, so one writer serves both:

    /<layer>/body/poses/<body>        Transform3D, an entity per body, a row per frame
    /<layer>/body/frames/<body>       an axis triad bound to that body's frame, static
    /<layer>/body/linear_velocities   Arrows3D, a row per frame, an arrow per body
    /<layer>/body/angular_velocities  Arrows3D
    /<robot>/joint/positions          Scalars, a row per frame, a scalar per joint
    /<robot>/joint/velocities         Scalars
    /<robot>/tf                       a Transform3D per joint, and the root placement
    /<robot>/tf_static                the URDF's fixed transforms
    /<robot>/visual_geometries/...    the URDF's geometry, named by ``rr.urdf``

A pose names a transform frame instead of riding the entity hierarchy, which is what
Rerun recommends for robot data and what ``rr.urdf`` writes. The layer prefixes the frame
name, so a reference body and a robot link of the same name stay apart, and so do
``<robot>/body/<link>`` and the ``<robot>/<link>`` that the joint chain already defines.
A frame has one parent.

Every entity drawn in 3D must name a frame. One that names none is an orphan: the viewer
invents a frame for it, finds no route to the root of the view, and draws an error in its
place. Scalars are exempt, because the viewer plots them against time.

Each motion carries its own copy of the robot's geometry. A shared file would need a
different recording id, which the viewer would not compose against the motion.

Rerun stores a rotation as xyzw. MuJoCo, and every array this module takes, uses wxyz.
"""

from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

#: Shared by every motion, so a directory of them registers as one dataset.
APPLICATION_ID: str = "lite-motion-tracking"

TIMELINE: str = "frame"
REFERENCE_LAYER: str = "reference"
WORLD_FRAME: str = "world"

#: Metres. Long enough to clear the limbs of the robot, which would otherwise hide it.
AXIS_LENGTH: float = 0.12


def write_reference_layer(
    root: Path,
    name: str,
    fps: int,
    body_names: tuple[str, ...],
    body_pos: np.ndarray,
    body_quat: np.ndarray,
) -> Path:
    """Write ``reference/<name>.rrd``: the bodies a solve aims at, and no robot.

    Joints never go here. A joint angle only means something for one robot, so it belongs
    to that robot's layer. ``body_pos`` is ``(T, NB, 3)`` and ``body_quat`` ``(T, NB, 4)``
    wxyz, both world frame.
    """
    path = root / REFERENCE_LAYER / f"{name}.rrd"
    path.parent.mkdir(parents=True, exist_ok=True)
    with rr.RecordingStream(APPLICATION_ID, recording_id=name) as stream:
        stream.save(str(path), _reference_blueprint())
        stream.send_recording_name(name)
        stream.send_property("motion", rr.AnyValues(
            fps=fps, body_names=list(body_names)
        ))
        _send_bodies(stream, REFERENCE_LAYER, fps, body_names, body_pos, body_quat)
    return path


def write_robot_layer(
    root: Path,
    name: str,
    urdf_path: Path,
    fps: int,
    joint_names: tuple[str, ...],
    joint_pos: np.ndarray,
    body_names: tuple[str, ...],
    reference_body_names: tuple[str, ...],
    body_pos: np.ndarray,
    body_quat: np.ndarray,
) -> Path:
    """Write ``<robot>/<name>.rrd``: what a solve reached, and the robot that reached it.

    The layer holds the joints, the world poses the solve put ``body_names`` at, and the
    robot's geometry, so the file animates the robot on its own. Everything sits under
    ``/<robot>/``, the URDF's frames and static transforms included, so a second robot
    can open beside it.

    ``body_names`` must contain the root link of the URDF, whose pose places the robot.
    ``reference_body_names`` names the reference body each of ``body_names`` was solved
    from, in the same order, because two rigs need not share a vocabulary.
    """
    robot = _robot_name(urdf_path)
    tree = rr.urdf.UrdfTree.from_file_path(
        urdf_path,
        frame_prefix=f"{robot}/",
        static_transform_entity_path=f"{robot}/tf_static",
    )
    frames = joint_pos.shape[0]
    index = rr.TimeColumn(TIMELINE, sequence=np.arange(frames))

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
            reference_body_names=list(reference_body_names),
        ))
        tree.log_urdf_to_recording(recording=stream)
        _send_bodies(stream, robot, fps, body_names, body_pos, body_quat)

        joint_vel = np.zeros_like(joint_pos)
        joint_vel[1:] = np.diff(joint_pos, axis=0) * fps
        rows = [len(joint_names)] * frames
        for entity, values in (("positions", joint_pos), ("velocities", joint_vel)):
            stream.send_columns(
                f"/{robot}/joint/{entity}",
                indexes=[index],
                columns=rr.Scalars.columns(scalars=values.reshape(-1)).partition(rows),
            )

        # The frame graph that puts the geometry where the joints say it is. Each joint
        # is its own chunk, so a row still carries one parent-child relation.
        for joint in tree.joints():
            if joint.name not in joint_names:
                continue
            angles = joint_pos[:, joint_names.index(joint.name)].astype(np.float64)
            stream.send_columns(
                f"/{robot}/tf",
                indexes=[index],
                columns=joint.compute_transform_columns(angles, clamp=False),
            )

        root_link = tree.root_link().name
        placement = body_names.index(root_link)
        stream.send_columns(f"/{robot}/tf", indexes=[index], columns=rr.Transform3D.columns(
            translation=body_pos[:, placement],
            quaternion=_wxyz_to_xyzw(body_quat[:, placement]),
            child_frame=[f"{robot}/{root_link}"] * frames,
            parent_frame=[WORLD_FRAME] * frames,
        ))
    return path


def write_blueprint(root: Path, urdf_path: Path) -> Path:
    """Write ``blueprints/<robot>.rbl``, the layout to pass to ``rerun`` after the layers."""
    robot = _robot_name(urdf_path)
    path = root / "blueprints" / f"{robot}.rbl"
    path.parent.mkdir(parents=True, exist_ok=True)
    _robot_blueprint(robot).save(APPLICATION_ID, path)
    return path


def _robot_name(urdf_path: Path) -> str:
    """The name the URDF gives the robot, which names its layer and its blueprint."""
    return rr.urdf.UrdfTree.from_file_path(urdf_path).name


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

    step = _quat_mul(quat[1:], quat[:-1] * (1.0, -1.0, -1.0, -1.0))
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


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternion arrays, broadcast over leading axes."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def _send_bodies(
    stream: rr.RecordingStream,
    layer: str,
    fps: int,
    body_names: tuple[str, ...],
    body_pos: np.ndarray,
    body_quat: np.ndarray,
) -> None:
    """Write one layer's body poses, triads, and velocities under ``/<layer>/body/``."""
    frames = body_pos.shape[0]
    index = rr.TimeColumn(TIMELINE, sequence=np.arange(frames))
    for i, body in enumerate(body_names):
        frame = f"{layer}/body/{body}"
        stream.send_columns(
            f"/{layer}/body/poses/{body}",
            indexes=[index],
            columns=rr.Transform3D.columns(
                translation=body_pos[:, i],
                quaternion=_wxyz_to_xyzw(body_quat[:, i]),
                child_frame=[frame] * frames,
                parent_frame=[WORLD_FRAME] * frames,
            ),
        )
        stream.log(
            f"/{layer}/body/frames/{body}",
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
            f"/{layer}/body/{entity}_velocities",
            rr.CoordinateFrame(frame=WORLD_FRAME),
            static=True,
        )
        stream.send_columns(
            f"/{layer}/body/{entity}_velocities",
            indexes=[index],
            columns=rr.Arrows3D.columns(
                origins=body_pos.reshape(-1, 3), vectors=vectors.reshape(-1, 3)
            ).partition(rows),
        )


def _hidden_velocities(layer: str) -> dict[str, rrb.EntityBehavior]:
    """Start one layer's velocity arrows switched off.

    At walking speed the arrows are metres long and bury the robot. Hiding keeps them in
    the entity tree, one click away; excluding them would take a view filter edit.
    """
    return {
        f"/{layer}/body/{quantity}_velocities": rrb.EntityBehavior(visible=False)
        for quantity in ("linear", "angular")
    }


def _reference_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/", name="motion", overrides=_hidden_velocities(REFERENCE_LAYER)
        ),
        rrb.TimePanel(state="expanded"),
    )


def _robot_blueprint(robot: str) -> rrb.Blueprint:
    """The robot in 3D beside its joint plots, with the reference triads over it.

    Seeing both is the point: the gap between a reference triad and the link under it is
    the error of the solve, frame by frame. The robot's own triads start switched off,
    because they sit inside its geometry.
    """
    hidden = (
        {f"/{robot}/body/frames": rrb.EntityBehavior(visible=False)}
        | _hidden_velocities(REFERENCE_LAYER)
        | _hidden_velocities(robot)
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", name=robot, overrides=hidden),
            rrb.TimeSeriesView(origin=f"/{robot}/joint/positions", name="joint angles"),
            column_shares=[3, 1],
        ),
        rrb.TimePanel(state="expanded"),
    )
