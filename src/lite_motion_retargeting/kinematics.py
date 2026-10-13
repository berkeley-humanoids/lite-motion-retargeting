"""Resampling, forward kinematics, the soles of the feet, and foot contacts."""

from itertools import product
from typing import NamedTuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .robots import body_id

#: A point of a source foot can plant the foot only below this speed, in m/s. See
#: :func:`planted`.
CONTACT_SPEED: float = 0.2
#: A point of a source foot can plant the foot only below this height above the floor, in
#: metres.
CONTACT_HEIGHT: float = 0.05


# ── Resampling ────────────────────────────────────────────────────────────────


def resample_motion(
    motion: dict[str, np.ndarray], src_fps: int, dst_fps: int
) -> dict[str, np.ndarray]:
    """Resample the output of a G1 source loader from ``src_fps`` to ``dst_fps``.

    The pipeline resamples the source, not the retarget, so that the IK solves every frame
    that goes into the file, and the body poses stay the exact kinematics of the joints
    beside them.

    Positions and joint angles interpolate linearly, and the base rotation by slerp. The
    output grid is uniform, because a motion declares one rate and a loader indexes it by
    frame. Thus the last output sample is within one output period of the last input
    sample. The clip keeps it from going past the last input sample, which ``Slerp``
    rejects.
    """
    frames = motion["base_pos"].shape[0]
    if dst_fps == src_fps or frames < 2:
        return motion
    t_src, t_dst = _resample_times(frames, src_fps, dst_fps)
    return {
        "base_pos": _interp(t_dst, t_src, motion["base_pos"]),
        "base_quat_wxyz": _slerp(t_dst, t_src, motion["base_quat_wxyz"]),
        "g1_joint_pos": _interp(t_dst, t_src, motion["g1_joint_pos"]),
    }


def resample_poses(
    pos: np.ndarray, quat: np.ndarray, src_fps: int, dst_fps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resample body poses ``(T, B, 3)`` and wxyz ``(T, B, 4)`` as :func:`resample_motion` does."""
    frames = pos.shape[0]
    if dst_fps == src_fps or frames < 2:
        return pos, quat
    t_src, t_dst = _resample_times(frames, src_fps, dst_fps)
    pos_out = _interp(t_dst, t_src, pos.reshape(frames, -1)).reshape(len(t_dst), *pos.shape[1:])
    quat_out = np.stack([_slerp(t_dst, t_src, quat[:, b]) for b in range(quat.shape[1])], axis=1)
    return pos_out, quat_out


def _resample_times(frames: int, src_fps: int, dst_fps: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the sample times of the source, and the uniform output grid that spans them."""
    t_src = np.arange(frames) / src_fps
    t_dst = np.clip(np.arange(round(t_src[-1] * dst_fps) + 1) / dst_fps, 0.0, t_src[-1])
    return t_src, t_dst


def _interp(t_dst: np.ndarray, t_src: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate each column of a ``(T, N)`` array onto ``t_dst``."""
    return np.stack(
        [np.interp(t_dst, t_src, values[:, k]) for k in range(values.shape[1])], axis=1
    ).astype(np.float32)


def _slerp(t_dst: np.ndarray, t_src: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Slerp a ``(T, 4)`` wxyz rotation track onto ``t_dst``."""
    rotations = Slerp(t_src, Rotation.from_quat(quat[:, [1, 2, 3, 0]]))
    return rotations(t_dst).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Hamilton product of two wxyz quaternion arrays, broadcast over leading axes."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


# ── Forward kinematics ────────────────────────────────────────────────────────


def world_poses(
    model: mujoco.MjModel,
    joint_addrs: np.ndarray,
    joint_pos: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    body_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the world pose of :data:`robots.BASE_BODY`, then of each of ``body_names``, per frame.

    The base comes first because it is the frame that the kinematics runs in, not a body of
    the result. The Lite MJCF welds its pelvis to the origin, and MuJoCo fuses the root link
    of the G1 URDF, which has no joint, into ``world``. Thus MuJoCo gives every body pose
    relative to the pelvis, and the base trajectory ``(base_pos, base_quat)``, wxyz,
    composes directly onto it.

    ``joint_addrs`` gives the ``qpos`` address of each column of ``joint_pos``. A joint of
    the model that no column addresses keeps its model default.

    Returns ``(T, 1 + NB, 3)`` positions and ``(T, 1 + NB, 4)`` wxyz rotations.
    """
    data = mujoco.MjData(model)
    body_ids = [body_id(model, name) for name in body_names]
    base_pos = np.asarray(base_pos, dtype=np.float64)
    base_quat = np.asarray(base_quat, dtype=np.float64)

    frames = joint_pos.shape[0]
    pos = np.empty((frames, 1 + len(body_ids), 3), dtype=np.float32)
    quat = np.empty((frames, 1 + len(body_ids), 4), dtype=np.float32)
    pos[:, 0], quat[:, 0] = base_pos, base_quat
    world_pos, world_quat = np.empty(3), np.empty(4)

    for t in range(frames):
        data.qpos[joint_addrs] = joint_pos[t]
        mujoco.mj_kinematics(model, data)
        for k, bid in enumerate(body_ids, start=1):
            mujoco.mju_mulPose(
                world_pos, world_quat,
                base_pos[t], base_quat[t], data.xpos[bid], data.xquat[bid],
            )
            pos[t, k], quat[t, k] = world_pos, world_quat
    return pos, quat


class Sole(NamedTuple):
    """The collision geoms of each foot, in the frame of its body, one entry per foot.

    Each geom is a set of points and a radius: the lowest point of the geom is the lowest of
    its points, moved down by its radius. A box is its 8 corners with no radius, a capsule
    the 2 ends of its segment with its radius, and a sphere its center with its radius. Thus
    the lowest point is exact at any tilt. The visual mesh does not count, because it does
    not touch the floor in simulation.
    """

    points: list[np.ndarray]
    radii: list[np.ndarray]


def sole(model: mujoco.MjModel, foot_names: tuple[str, ...]) -> Sole:
    """Return the collision geoms of ``foot_names`` on ``model``."""
    points, radii = [], []
    for name in foot_names:
        foot = body_id(model, name)
        foot_points, foot_radii = [], []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != foot or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            size, kind = model.geom_size[g], model.geom_type[g]
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                local, radius = np.array(list(product(*((-s, s) for s in size)))), 0.0
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                local, radius = np.array([[0.0, 0.0, -size[1]], [0.0, 0.0, size[1]]]), size[0]
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                local, radius = np.zeros((1, 3)), size[0]
            else:
                raise ValueError(
                    f"{name} has a {mujoco.mjtGeom(kind).name} collision geom. "
                    "A sole holds only boxes, capsules and spheres."
                )
            rotation = Rotation.from_quat(model.geom_quat[g][[1, 2, 3, 0]]).as_matrix()
            foot_points.append(model.geom_pos[g] + local @ rotation.T)
            foot_radii.append(np.full(len(local), radius))
        points.append(np.concatenate(foot_points))
        radii.append(np.concatenate(foot_radii))
    return Sole(points, radii)


def lowest_points(foot_sole: Sole, foot_pos: np.ndarray, foot_rot: np.ndarray) -> np.ndarray:
    """Return the world position of the lowest point of each foot, ``(T, F, 3)``.

    ``foot_pos`` ``(T, F, 3)`` and the rotation matrices ``foot_rot`` ``(T, F, 3, 3)`` are
    the world pose of each foot body.
    """
    points = np.empty(foot_pos.shape)
    frames = np.arange(foot_pos.shape[0])
    for f in range(foot_pos.shape[1]):
        world = foot_pos[:, f, None] + np.einsum("tij,kj->tki", foot_rot[:, f], foot_sole.points[f])
        lowest = np.argmin(world[..., 2] - foot_sole.radii[f], axis=1)
        points[:, f] = world[frames, lowest]
        points[:, f, 2] -= foot_sole.radii[f][lowest]
    return points


def sole_points(
    model: mujoco.MjModel,
    joint_addrs: np.ndarray,
    joint_pos: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    foot_names: tuple[str, ...],
) -> np.ndarray:
    """Return the world position of the lowest point of each of ``foot_names``, ``(T, F, 3)``.

    The arguments are those of :func:`world_poses`.
    """
    pos, quat = world_poses(model, joint_addrs, joint_pos, base_pos, base_quat, foot_names)
    rot = Rotation.from_quat(quat[:, 1:].reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix()
    return lowest_points(sole(model, foot_names), pos[:, 1:], rot.reshape(*pos[:, 1:].shape, 3))


def planted(points: np.ndarray, fps: int) -> np.ndarray:
    """Return which feet of a source stand on the floor, per frame, ``(T, F)`` bool.

    ``points`` ``(T, F, P, 3)`` are P world points of each of F feet, such as the ankle and
    the ball of the foot. A foot is planted while one of its points moves slower than
    :data:`CONTACT_SPEED` and is within :data:`CONTACT_HEIGHT` of the floor. The floor of a
    point is the lowest it gets in the clip, taken as its 5th percentile, so that one bad
    frame does not move it.
    """
    speed = np.linalg.norm(np.gradient(points, axis=0), axis=-1) * fps
    near = points[..., 2] - np.percentile(points[..., 2], 5, axis=0) < CONTACT_HEIGHT
    return np.any((speed < CONTACT_SPEED) & near, axis=-1)
