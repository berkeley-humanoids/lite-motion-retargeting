"""Shared helpers for the LAFAN1 → Lite retargeting pipeline.

Conventions:
- Quaternions are scalar-first WXYZ, unit-norm, ``w >= 0``. The only non-WXYZ
  appearance is the LAFAN1 CSV row, converted exactly once in
  :func:`load_lafan_csv`.
- The Lite ``joint_pos`` column order is whatever MuJoCo emits from the MJCF
  (:func:`lite_joint_names`); we never hardcode it.
"""

import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# ── Constants ─────────────────────────────────────────────────────────────────

#: The rate of the LAFAN1 CSVs.
SOURCE_FPS: int = 30

#: The rate a motion is written at, which is the control rate of the training
#: environment. A motion at another rate would force the loader to interpolate.
FPS: int = 50

LAFAN_REPO_ID: str = "lvhaidong/LAFAN1_Retargeting_Dataset"
DATASET_REPO_ID: str = "Berkeley-Humanoids/Lite-LAFAN1-Dataset"

#: The body both robots are rooted at. Neither model gives it a joint, so it is the
#: frame forward kinematics runs in rather than a body of the result, and the base
#: trajectory of the motion is its world pose.
BASE_BODY: str = "pelvis"

#: Lite link -> the G1 link whose pose the IK aims it at. The Lite names come from the
#: MJCF. The G1 names come from its URDF after MuJoCo import, where the fixed-jointed
#: ``*_rubber_hand`` link fuses into ``*_wrist_yaw_link``, the last named body of each
#: arm chain.
IK_BODY_PAIRS: dict[str, str] = {
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
    "left_hand": "left_wrist_yaw_link",
    "right_hand": "right_wrist_yaw_link",
}

#: The bodies each layer of a motion stores, the root body first. The robot layer names
#: them after the links of Lite, the base layer after the links of the G1.
ROBOT_BODIES: tuple[str, ...] = (BASE_BODY, *IK_BODY_PAIRS)
SOURCE_BODIES: tuple[str, ...] = (BASE_BODY, *IK_BODY_PAIRS.values())

LAFAN_G1_URDF_RELPATH: str = "robot_description/g1/g1_29dof_rev_1_0.urdf"

# Authoritative LAFAN1 G1 CSV joint order. Every CSV row is
# ``[base_x, base_y, base_z, base_qx, base_qy, base_qz, base_qw,
#   *G1_LAFAN_JOINT_NAMES]`` at :data:`SOURCE_FPS`.
G1_LAFAN_JOINT_NAMES: tuple[str, ...] = (
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

# G1 joint → (Lite joint, sign, offset_rad). Step-1 retargeting applies
# ``lite_q[t] = sign * g1_q[t] + offset`` per joint.
#
# Decisions baked into this table:
#   - Legs: chain order and axes match; sign=+1, offset=0.
#   - Waist: chain order DIFFERS (G1 yaw→roll→pitch vs Lite yaw→pitch→roll),
#     but axes match by *name*, so we map by name and accept the small
#     rotation-order approximation when multiple waist joints are non-zero.
#   - Shoulders + elbows: chain order matches; offsets bring G1's arms-down
#     rest pose to Lite's T-pose rest (±π/2 on shoulder_roll and elbow).
#   - Wrists: chain order DIFFERS (G1 roll→pitch→yaw vs Lite yaw→roll→pitch)
#     but the physical 3-DoF wrist is the same, so we map by *position in
#     the chain* (G1 wrist_roll → Lite wrist_yaw, etc.). Sign of the first
#     wrist DoF is -1 because the URDFs picked opposite positive directions.
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


# ── Model loaders ─────────────────────────────────────────────────────────────


def load_lite_model() -> mujoco.MjModel:
    """Load the Lite Pro MJCF, the same asset that the training repo loads."""
    from robot_assets import load

    return mujoco.MjModel.from_xml_path(str(load("robots/lite_pro/mjcf/lite_pro.xml")))


def lite_urdf_path() -> Path:
    """Path to the Lite Pro URDF, whose geometry goes into the robot layer.

    The MJCF drives the retarget and this URDF only draws it, so the two must describe
    one robot. They come from one asset package, and their joint and link names match.
    """
    from robot_assets import load

    return Path(load("robots/lite_pro/urdf/lite_pro.urdf"))


def load_g1_model(lafan_root: Path) -> mujoco.MjModel:
    """Load the Unitree G1 URDF bundled in the downloaded LAFAN1 dataset.

    The URDF declares ``<compiler meshdir="meshes"/>`` and also references
    meshes as ``meshes/foo.STL`` — MuJoCo would concatenate to
    ``meshes/meshes/foo.STL``. We rewrite meshdir to ``"."`` and supply the
    mesh bytes through the ``assets`` dict so nothing touches the filesystem
    a second time.
    """
    urdf = Path(lafan_root) / LAFAN_G1_URDF_RELPATH
    if not urdf.exists():
        raise FileNotFoundError(
            f"G1 URDF not found at {urdf}. Run scripts/download_lafan.py first."
        )
    text = re.sub(r'meshdir="[^"]*"', 'meshdir="."', urdf.read_text())
    assets = {f"meshes/{p.name}": p.read_bytes() for p in (urdf.parent / "meshes").glob("*.STL")}
    return mujoco.MjModel.from_xml_string(text, assets=assets)


def joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
    """Index into ``data.qpos`` for a 1-DoF joint by name."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise KeyError(f"Joint {joint_name!r} not found in model")
    return int(model.jnt_qposadr[jid])


def body_id(model: mujoco.MjModel, body_name: str) -> int:
    """Body id for ``body_name``, raising if absent."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise KeyError(f"Body {body_name!r} not found in model")
    return int(bid)


def lite_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """All hinge joint names of the Lite model, in the depth-first order of MuJoCo.

    Skips the free joint at index 0 when the model has one. The result gives the
    order of the ``joint_pos`` columns, and the motion file stores it, so a
    loader maps a column to a robot joint by name.
    """
    names: list[str] = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None:
            raise RuntimeError(f"Lite joint id {jid} has no name")
        names.append(name)
    return tuple(names)


# ── LAFAN1 CSV I/O ────────────────────────────────────────────────────────────


def load_lafan_csv(path: Path) -> dict[str, np.ndarray]:
    """Read one LAFAN1 G1 CSV into base pose + joint positions.

    On-disk format: headerless, :data:`SOURCE_FPS`, ``[base_xyz(3), base_quat_xyzw(4),
    joint_pos(29)]`` per row. Converts XYZW → WXYZ and flips quaternion sign
    so ``w >= 0``.

    Returns:
        Dict with ``base_pos`` ``(T, 3)``, ``base_quat_wxyz`` ``(T, 4)``,
        ``g1_joint_pos`` ``(T, 29)`` — all float32. Joint columns follow
        :data:`G1_LAFAN_JOINT_NAMES`.
    """
    raw = np.loadtxt(path, delimiter=",", dtype=np.float32)
    expected_cols = 7 + len(G1_LAFAN_JOINT_NAMES)
    if raw.ndim != 2 or raw.shape[1] != expected_cols:
        raise ValueError(
            f"{path}: expected {expected_cols} columns, got shape {raw.shape}"
        )
    base_quat = np.empty((raw.shape[0], 4), dtype=np.float32)
    base_quat[:, 0] = raw[:, 6]
    base_quat[:, 1:4] = raw[:, 3:6]
    base_quat /= np.linalg.norm(base_quat, axis=1, keepdims=True)
    base_quat[base_quat[:, 0] < 0] *= -1.0
    return {
        "base_pos": raw[:, 0:3],
        "base_quat_wxyz": base_quat,
        "g1_joint_pos": raw[:, 7:],
    }


def resample_motion(
    motion: dict[str, np.ndarray], src_fps: int, dst_fps: int
) -> dict[str, np.ndarray]:
    """Resample the output of :func:`load_lafan_csv` from ``src_fps`` onto ``dst_fps``.

    The source is resampled, and not the retarget, so that the IK solves every frame
    that reaches the file and the body poses stay the exact kinematics of the joints
    beside them.

    Positions and joint angles interpolate linearly, and the base rotation by slerp.

    The output grid is uniform, because a motion declares one rate and a loader indexes
    it by frame. The last output sample therefore lands within one output period of the
    last input sample, and the clip keeps it from landing past it, which ``Slerp``
    rejects.
    """
    frames = motion["base_pos"].shape[0]
    if dst_fps == src_fps or frames < 2:
        return motion

    t_src = np.arange(frames) / src_fps
    t_dst = np.clip(
        np.arange(round(t_src[-1] * dst_fps) + 1) / dst_fps, 0.0, t_src[-1]
    )
    rotations = Slerp(
        t_src, Rotation.from_quat(motion["base_quat_wxyz"][:, [1, 2, 3, 0]])
    )
    return {
        "base_pos": _interp(t_dst, t_src, motion["base_pos"]),
        "base_quat_wxyz": rotations(t_dst).as_quat().astype(np.float32)[:, [3, 0, 1, 2]],
        "g1_joint_pos": _interp(t_dst, t_src, motion["g1_joint_pos"]),
    }


def _interp(t_dst: np.ndarray, t_src: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate each column of a ``(T, N)`` array onto ``t_dst``."""
    return np.stack(
        [np.interp(t_dst, t_src, values[:, k]) for k in range(values.shape[1])], axis=1
    ).astype(np.float32)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternion arrays, broadcast over leading axes."""
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
    """World pose of :data:`BASE_BODY`, then of each of ``body_names``, per frame.

    The base leads the arrays because it is the frame the kinematics runs in and not a
    body of the result: the Lite MJCF welds its pelvis to the origin, and MuJoCo fuses
    the jointless root link of the G1 URDF into ``world``. Every pose a pass returns is
    therefore pelvis-local, and the base trajectory ``(base_pos, base_quat)``, wxyz,
    composes straight onto it.

    ``joint_addrs`` gives the ``qpos`` address of each column of ``joint_pos``. A joint
    of the model that no column addresses keeps its model default.

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
