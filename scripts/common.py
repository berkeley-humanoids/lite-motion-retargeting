"""Shared helpers for the retargeting pipeline.

Conventions:
- Quaternions are scalar-first wxyz, unit-norm, with ``w >= 0``. A source file can store
  xyzw, and its loader in ``sources.py`` converts it exactly once.
- MuJoCo sets the column order of the Lite ``joint_pos`` from the MJCF
  (:func:`lite_joint_names`). No code hardcodes it.
"""

import re
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np
from lite_description import get_mjcf_path, get_urdf_path
from scipy.spatial.transform import Rotation, Slerp

# ── Constants ─────────────────────────────────────────────────────────────────

#: The rate of every written motion, which is the control rate of the training
#: environment. A motion at another rate would force the loader to interpolate.
FPS: int = 50

#: The Hub dataset repo that holds the published motions.
DATASET_REPO_ID: str = "Berkeley-Humanoids/Lite-LAFAN1-Dataset"

#: The one source that the Hub repo holds. The terms of the other sources forbid it: AMASS
#: bars any redistribution, and ChingMu bars hosting MotionDecode as a competing dataset.
#: Their motions stay in the local store, for training.
PUBLIC_SOURCE: str = "lafan1"

#: The static Space that hosts the motion browser. The browser reads the motions from the
#: dataset repo.
SPACE_REPO_ID: str = "Berkeley-Humanoids/Lite-Motion-Dataset"

#: The local copy of each source and of the G1 model, one directory each.
CACHE_ROOT: Path = Path(__file__).resolve().parent.parent / ".cache"

#: A point of a source foot can plant the foot only below this speed, in m/s. See
#: :func:`planted`.
CONTACT_SPEED: float = 0.2
#: A point of a source foot can plant the foot only below this height above the floor, in
#: metres.
CONTACT_HEIGHT: float = 0.05

#: The upstream authors retargeted every G1 source to this G1 model. The LAFAN1 repo
#: bundles it.
G1_REPO_ID: str = "lvhaidong/LAFAN1_Retargeting_Dataset"
G1_URDF_PATH: Path = CACHE_ROOT / "g1" / "robot_description" / "g1" / "g1_29dof_rev_1_0.urdf"

#: The body that both robots are rooted at. Neither model gives it a joint, so it is the
#: frame that forward kinematics runs in, not a body of the result. The base trajectory of
#: a motion is its world pose.
BASE_BODY: str = "pelvis"

#: Lite link -> the G1 link whose pose the IK aims it at. The Lite names come from the
#: MJCF. The G1 names come from its URDF after the MuJoCo import, which fuses the
#: fixed-joint ``*_rubber_hand`` link into ``*_wrist_yaw_link``, the last named body of
#: each arm chain.
IK_BODY_PAIRS: dict[str, str] = {
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
    "left_hand": "left_wrist_yaw_link",
    "right_hand": "right_wrist_yaw_link",
}

#: The bodies that each layer of a motion from a G1 source stores, the root body first.
#: The robot layer names them after the Lite links, and the base layer after the G1 links.
ROBOT_BODIES: tuple[str, ...] = (BASE_BODY, *IK_BODY_PAIRS)
SOURCE_BODIES: tuple[str, ...] = (BASE_BODY, *IK_BODY_PAIRS.values())

#: The joints of the 29-DoF G1, in the order that every G1 source stores them after the
#: base position and rotation.
G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

#: G1 joint -> (Lite joint, sign, offset in rad). Step 1 of the G1 retarget sets
#: ``lite_q = sign * g1_q + offset`` for each joint:
#:
#: - Legs: the chain order and the axes match, so the sign is +1 and the offset is 0.
#: - Waist: the chain order differs (G1 yaw, roll, pitch, and Lite yaw, pitch, roll), but
#:   the axes match by name. The table maps by name, which is a small approximation when
#:   more than one waist joint is not zero.
#: - Shoulders and elbows: the chain order matches. The offsets (±π/2 on shoulder roll and
#:   elbow) move the G1 rest pose, arms down, to the Lite rest pose, a T-pose.
#: - Wrists: the chain order differs (G1 roll, pitch, yaw, and Lite yaw, roll, pitch), but
#:   the physical 3-DoF wrist is the same, so the table maps by position in the chain: G1
#:   wrist roll to Lite wrist yaw, and so on. The first wrist DoF has sign -1, because the
#:   two URDFs use opposite positive directions.
G1_TO_LITE: dict[str, tuple[str, int, float]] = {
    # Legs (12)
    "left_hip_pitch_joint":      ("left_hip_pitch",      +1, 0.0),
    "left_hip_roll_joint":       ("left_hip_roll",       +1, 0.0),
    "left_hip_yaw_joint":        ("left_hip_yaw",        +1, 0.0),
    "left_knee_joint":           ("left_knee_pitch",     +1, 0.0),
    "left_ankle_pitch_joint":    ("left_ankle_pitch",    +1, 0.0),
    "left_ankle_roll_joint":     ("left_ankle_roll",     +1, 0.0),
    "right_hip_pitch_joint":     ("right_hip_pitch",     +1, 0.0),
    "right_hip_roll_joint":      ("right_hip_roll",      +1, 0.0),
    "right_hip_yaw_joint":       ("right_hip_yaw",       +1, 0.0),
    "right_knee_joint":          ("right_knee_pitch",    +1, 0.0),
    "right_ankle_pitch_joint":   ("right_ankle_pitch",   +1, 0.0),
    "right_ankle_roll_joint":    ("right_ankle_roll",    +1, 0.0),
    # Waist (3)
    "waist_yaw_joint":           ("waist_yaw",           +1, 0.0),
    "waist_roll_joint":          ("waist_roll",          +1, 0.0),
    "waist_pitch_joint":         ("waist_pitch",         +1, 0.0),
    # Arms (14) — left
    "left_shoulder_pitch_joint": ("left_shoulder_pitch", +1, 0.0),
    "left_shoulder_roll_joint":  ("left_shoulder_roll",  +1, -1.5708),
    "left_shoulder_yaw_joint":   ("left_shoulder_yaw",   +1, 0.0),
    "left_elbow_joint":          ("left_elbow_pitch",    +1, -1.5708),
    "left_wrist_roll_joint":     ("left_wrist_yaw",      -1, 0.0),
    "left_wrist_pitch_joint":    ("left_wrist_roll",     +1, 0.0),
    "left_wrist_yaw_joint":      ("left_wrist_pitch",    +1, 0.0),
    # Arms (14) — right
    "right_shoulder_pitch_joint":("right_shoulder_pitch",+1, 0.0),
    "right_shoulder_roll_joint": ("right_shoulder_roll", +1, +1.5708),
    "right_shoulder_yaw_joint":  ("right_shoulder_yaw",  +1, 0.0),
    "right_elbow_joint":         ("right_elbow_pitch",   +1, -1.5708),
    "right_wrist_roll_joint":    ("right_wrist_yaw",     -1, 0.0),
    "right_wrist_pitch_joint":   ("right_wrist_roll",    +1, 0.0),
    "right_wrist_yaw_joint":     ("right_wrist_pitch",   +1, 0.0),
}

#: The Lite joints that a G1 source drives. The neck, the ankle yaw, and the fingers have no
#: G1 joint, and hold their default.
G1_DRIVEN: tuple[str, ...] = tuple(lite for lite, _, _ in G1_TO_LITE.values())


# ── Model loaders ─────────────────────────────────────────────────────────────


def load_lite_model() -> mujoco.MjModel:
    """Load the Lite Pro MJCF, the same asset that the training repo loads."""
    return mujoco.MjModel.from_xml_path(str(get_mjcf_path("lite_pro")))


def lite_urdf_path() -> Path:
    """Return the path of the Lite Pro URDF, whose geometry goes into the robot layer.

    The MJCF drives the retarget and this URDF only draws it, so the two must describe
    one robot. They come from one asset package, and their joint and link names match.
    """
    return get_urdf_path("lite_pro")


def load_g1_model() -> mujoco.MjModel:
    """Load the Unitree G1 URDF that ``scripts/download.py`` puts in the cache.

    The URDF declares ``<compiler meshdir="meshes"/>`` and also gives each mesh as
    ``meshes/foo.STL``, so MuJoCo would look for ``meshes/meshes/foo.STL``. The loader sets
    meshdir to ``"."`` and gives MuJoCo the mesh bytes through ``assets``, so MuJoCo does
    not read the mesh files a second time.
    """
    urdf = G1_URDF_PATH
    if not urdf.exists():
        raise FileNotFoundError(
            f"Cannot find the G1 URDF at {urdf}. Run scripts/download.py first."
        )
    text = re.sub(r'meshdir="[^"]*"', 'meshdir="."', urdf.read_text())
    assets = {f"meshes/{p.name}": p.read_bytes() for p in (urdf.parent / "meshes").glob("*.STL")}
    return mujoco.MjModel.from_xml_string(text, assets=assets)


def joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
    """Return the ``qpos`` address of the 1-DoF joint ``joint_name``."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise KeyError(f"The model has no joint {joint_name!r}.")
    return int(model.jnt_qposadr[jid])


def body_id(model: mujoco.MjModel, body_name: str) -> int:
    """Return the id of the body ``body_name``."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise KeyError(f"The model has no body {body_name!r}.")
    return bid


def lite_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """Return the names of the Lite joints, except a free joint, in the order of MuJoCo.

    MuJoCo orders the joints depth first. The names give the column order of ``joint_pos``,
    and the motion file stores them, so a loader maps a column to a robot joint by name.
    """
    names: list[str] = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None:
            raise RuntimeError(f"Lite joint {jid} has no name.")
        names.append(name)
    return tuple(names)


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
    """Return the world pose of :data:`BASE_BODY`, then of each of ``body_names``, per frame.

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
    """The collision boxes of each foot, in the frame of its body, one entry per foot.

    Each collision geom counts as the box that MuJoCo bounds it with (``geom_aabb``), which
    is exact for a box and for a level capsule. The visual mesh does not count, because it
    does not touch the floor in simulation.
    """

    centers: list[np.ndarray]
    rotations: list[np.ndarray]
    halves: list[np.ndarray]


def sole(model: mujoco.MjModel, foot_names: tuple[str, ...]) -> Sole:
    """Measure the collision boxes of ``foot_names`` on ``model``."""
    centers, rotations, halves = [], [], []
    for name in foot_names:
        foot = body_id(model, name)
        geoms = [
            g for g in range(model.ngeom)
            if model.geom_bodyid[g] == foot and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]
        rotation = Rotation.from_quat(model.geom_quat[geoms][:, [1, 2, 3, 0]]).as_matrix()
        centers.append(
            model.geom_pos[geoms] + np.einsum("gij,gj->gi", rotation, model.geom_aabb[geoms, :3])
        )
        rotations.append(rotation)
        halves.append(model.geom_aabb[geoms, 3:])
    return Sole(centers, rotations, halves)


def lowest_points(foot_sole: Sole, foot_pos: np.ndarray, foot_rot: np.ndarray) -> np.ndarray:
    """Return the world position of the lowest point of each foot, ``(T, F, 3)``.

    ``foot_pos`` ``(T, F, 3)`` and the rotation matrices ``foot_rot`` ``(T, F, 3, 3)`` are
    the world pose of each foot body.
    """
    points = np.empty(foot_pos.shape, dtype=np.float32)
    for f in range(foot_pos.shape[1]):
        rotations = np.einsum("tij,gjk->tgik", foot_rot[:, f], foot_sole.rotations[f])
        centers = foot_pos[:, f, None] + np.einsum(
            "tij,gj->tgi", foot_rot[:, f], foot_sole.centers[f]
        )
        # A box reaches lowest at the corner whose every axis points down.
        down = -np.sign(rotations[:, :, 2, :]) * foot_sole.halves[f]
        lowest = centers + np.einsum("tgij,tgj->tgi", rotations, down)
        points[:, f] = lowest[np.arange(len(lowest)), np.argmin(lowest[..., 2], axis=1)]
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
