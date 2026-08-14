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

# ── Constants ─────────────────────────────────────────────────────────────────

FPS: int = 30
LAFAN_REPO_ID: str = "lvhaidong/LAFAN1_Retargeting_Dataset"
LITE_DATASET_REPO_ID: str = "Berkeley-Humanoids/Lite-Motion-Tracking-Dataset"
LITE_TASK_NAME: str = "track_lafan1"

# End-effector body names — Lite (matches the MJCF) and the G1 URDF after MuJoCo
# import (the ``*_rubber_hand`` link is fixed-jointed and fuses into
# ``*_wrist_yaw_link``, which is the last named body in each arm chain).
LITE_FOOT_BODIES: tuple[str, str] = ("left_foot", "right_foot")
LITE_HAND_BODIES: tuple[str, str] = ("left_hand", "right_hand")
G1_FOOT_BODIES: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
G1_HAND_BODIES: tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link")

LAFAN_G1_URDF_RELPATH: str = "robot_description/g1/g1_29dof_rev_1_0.urdf"

# Authoritative LAFAN1 G1 CSV joint order. Every CSV row is
# ``[base_x, base_y, base_z, base_qx, base_qy, base_qz, base_qw,
#   *G1_LAFAN_JOINT_NAMES]`` at 30 FPS.
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
    """Load the Berkeley Lite humanoid MJCF via the Berkeley-Humanoids package."""
    from robot_descriptions import load_asset

    return mujoco.MjModel.from_xml_path(str(load_asset("robots/lite/mjcf/lite.xml")))


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
    """All hinge joint names in the Lite model in MuJoCo's depth-first order.

    Skips the free joint at index 0 if present. The result is the single
    source of truth for the LeRobotDataset ``joint_pos`` column order.
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

    On-disk format: headerless, 30 FPS, ``[base_xyz(3), base_quat_xyzw(4),
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


# ── Derivatives ───────────────────────────────────────────────────────────────


def finite_diff(x: np.ndarray, fps: int) -> np.ndarray:
    """Central finite difference along axis 0."""
    return np.gradient(x, 1.0 / fps, axis=0).astype(x.dtype, copy=False)


def angular_velocity_from_quat(q_wxyz: np.ndarray, fps: int) -> np.ndarray:
    """World-frame angular velocity (rad/s) from a sequence of WXYZ quaternions.

    Uses the central stencil ``omega = axis_angle(q[t+1] * q[t-1]^-1) / 2dt``;
    endpoints are repeated to keep length T.
    """
    dt = 1.0 / fps
    rel = _quat_mul(q_wxyz[2:], _quat_conj(q_wxyz[:-2]))
    omega_mid = _axis_angle_from_quat(rel) / (2.0 * dt)
    omega = np.empty((q_wxyz.shape[0], 3), dtype=q_wxyz.dtype)
    omega[1:-1] = omega_mid
    omega[0] = omega_mid[0]
    omega[-1] = omega_mid[-1]
    return omega


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty_like(a)
    out[..., 0] = aw * bw - ax * bx - ay * by - az * bz
    out[..., 1] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 2] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 3] = aw * bz + ax * by - ay * bx + az * bw
    return out


def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _axis_angle_from_quat(q: np.ndarray) -> np.ndarray:
    """WXYZ quaternion → rotation vector ``axis * angle``, shape ``(..., 3)``."""
    w = np.clip(q[..., 0], -1.0, 1.0)
    xyz = q[..., 1:]
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    safe = sin_half > 1e-8
    axis = np.zeros_like(xyz)
    axis[safe] = xyz[safe] / sin_half[safe, None]
    return axis * angle[..., None]


# ── LeRobotDataset feature schema ─────────────────────────────────────────────


def dataset_features(joint_count: int) -> dict:
    """LeRobotDataset feature schema for the retargeted Lite motion dataset.

    Six flat fields, scalar-first WXYZ for the base orientation, no ``action``
    (this is a kinematic reference).
    """
    return {
        "base_pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "base_quat": {"dtype": "float32", "shape": (4,), "names": ["w", "x", "y", "z"]},
        "base_lin_vel": {"dtype": "float32", "shape": (3,), "names": ["vx", "vy", "vz"]},
        "base_ang_vel": {"dtype": "float32", "shape": (3,), "names": ["wx", "wy", "wz"]},
        "joint_pos": {"dtype": "float32", "shape": (joint_count,), "names": None},
        "joint_vel": {"dtype": "float32", "shape": (joint_count,), "names": None},
    }
