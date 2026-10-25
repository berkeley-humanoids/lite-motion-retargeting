"""The G1 path: copy each G1 joint to its Lite joint, then refine the copy with IK.

The authors of a G1 source already retargeted it to the 29-DoF Unitree G1. Step 1 copies
each G1 joint by :data:`G1_TO_LITE`. Step 2 moves the feet and the hands of Lite to those
of the G1, relative to the pelvis, and changes step 1 only a little.
"""

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from ...robots import BASE_BODY, G1_JOINT_NAMES, body_id, joint_qpos_addr

if TYPE_CHECKING:
    from .. import Models

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

#: Newton steps per frame of :func:`step2_ik_refine`.
_NEWTON_STEPS: int = 15


def _rest_frame_conversions(g1: mujoco.MjModel, lite: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Return a constant ``R_conv`` per IK link, such that ``R_target = R_g1_actual @ R_conv``.

    The world-frame motion of a Lite link relative to its matched rest must equal the
    world-frame motion of its G1 link relative to the G1 rest. Solve for the target rotation:
    ``R_target = R_g1_actual @ (R_g1_rest^-1 @ R_lite_matched_rest)``. The matched rest of
    Lite is the pose that step 1 makes from the G1 rest.
    """
    g1_data = mujoco.MjData(g1)
    mujoco.mj_kinematics(g1, g1_data)
    lite_data = mujoco.MjData(lite)
    for lite_name, _, offset in G1_TO_LITE.values():
        lite_data.qpos[joint_qpos_addr(lite, lite_name)] = offset
    mujoco.mj_kinematics(lite, lite_data)

    out: dict[str, np.ndarray] = {}
    for lite_name, g1_name in IK_BODY_PAIRS.items():
        R_g1 = g1_data.xmat[body_id(g1, g1_name)].reshape(3, 3)
        R_lite = lite_data.xmat[body_id(lite, lite_name)].reshape(3, 3)
        out[lite_name] = R_g1.T @ R_lite
    return out


def goal_conversions(g1: mujoco.MjModel, lite: mujoco.MjModel) -> np.ndarray:
    """Return the wxyz rotation per body from the G1 frame convention to Lite, root body first.

    A G1 link and the Lite link that it drives disagree about which way their axes point at
    rest, by up to 120 degrees. Thus a goal in the raw G1 convention is not comparable to
    the link that it is the goal for, and a tracking reward on orientation would never reach
    zero. The IK aims through the same conversion, so the goal that a motion stores is the
    goal that the IK got.

    The root body is not an IK target, and its goal is the base trajectory, so it converts
    by the identity.
    """
    conversions = _rest_frame_conversions(g1, lite)
    quats = np.zeros((len(ROBOT_BODIES), 4))
    quats[0] = (1.0, 0.0, 0.0, 0.0)
    for index, lite_name in enumerate(IK_BODY_PAIRS, start=1):
        mujoco.mju_mat2Quat(quats[index], conversions[lite_name].reshape(9))
    return quats


def step1_direct_remap(g1_joint_pos: np.ndarray, joint_names: tuple[str, ...]) -> np.ndarray:
    """Copy each G1 joint to its Lite joint as ``lite_q = sign * g1_q + offset``.

    Return ``(T, J)`` joint positions in the order of ``joint_names``. A Lite joint with no
    G1 joint (the neck, the fingers, and the ankle yaw) holds 0.
    """
    lite_names, signs, offsets = zip(*(G1_TO_LITE[name] for name in G1_JOINT_NAMES), strict=True)
    joint_pos = np.zeros((g1_joint_pos.shape[0], len(joint_names)), dtype=np.float32)
    joint_pos[:, [joint_names.index(name) for name in lite_names]] = (
        np.asarray(signs, dtype=np.float32) * g1_joint_pos + np.asarray(offsets, dtype=np.float32)
    )
    return joint_pos


def _posture_cost(lite: mujoco.MjModel) -> np.ndarray:
    """Return the posture cost of each DoF, in three tiers, so that step 2 only adjusts step 1.

    The posture task pulls each joint toward step 1 with this cost. The frame task of each
    foot and hand pulls the joints away with cost 1.0 for position and 1.0 for orientation.

    * ``1e3`` locks a joint. The neck, the ankle yaw and the fingers have no G1 joint, so
      they stay at the 0 of step 1. The motion holds them at 0, so the IK must not use them.
    * ``10.0`` makes a joint stiff but adjustable. The waist turns the torso and the head,
      so a high cost keeps the IK from twisting the whole upper body to reach a hand goal.
      The IK moves the arm joints first.
    * ``1.0`` matches the frame tasks. The arms and the legs get a balanced trade-off, and
      the correction in each frame is small.
    """
    cost = np.full(lite.nv, 1e3, dtype=np.float64)
    for jid in range(lite.njnt):
        name = mujoco.mj_id2name(lite, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name not in G1_DRIVEN:
            continue
        dof = int(lite.jnt_dofadr[jid])
        if any(limb in name for limb in ("hip", "knee", "ankle", "shoulder", "elbow", "wrist")):
            cost[dof] = 1.0
        elif "waist" in name:
            cost[dof] = 10.0
    return cost


def step2_ik_refine(models: "Models", g1_joint_pos: np.ndarray, step1_joint_pos: np.ndarray) -> np.ndarray:
    """Refine the step-1 joints to match the feet and hands of the G1 relative to the pelvis.

    The IK solves each frame on its own, from the step-1 seed.
    """
    import mink

    g1_data = mujoco.MjData(models.g1)
    conversions = _rest_frame_conversions(models.g1, models.lite)
    configuration = mink.Configuration(models.lite)
    frame_tasks = {
        name: mink.FrameTask(name, "body", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
        for name in IK_BODY_PAIRS
    }
    posture_task = mink.PostureTask(models.lite, cost=_posture_cost(models.lite))
    tasks = [*frame_tasks.values(), posture_task]
    limits = [mink.ConfigurationLimit(models.lite)]

    out = step1_joint_pos.copy()
    seed = np.zeros(models.lite.nq, dtype=np.float64)
    for t in range(step1_joint_pos.shape[0]):
        g1_data.qpos[models.g1_addrs] = g1_joint_pos[t]
        mujoco.mj_kinematics(models.g1, g1_data)
        for lite_name, g1_name in IK_BODY_PAIRS.items():
            bid = body_id(models.g1, g1_name)
            target = np.eye(4)
            target[:3, :3] = g1_data.xmat[bid].reshape(3, 3) @ conversions[lite_name]
            target[:3, 3] = g1_data.xpos[bid]
            frame_tasks[lite_name].set_target(mink.SE3.from_matrix(target))

        seed[models.joint_addrs] = step1_joint_pos[t]
        configuration.q[:] = seed
        posture_task.set_target(seed.copy())
        for _ in range(_NEWTON_STEPS):
            vel = mink.solve_ik(
                configuration, tasks, 1.0, solver="daqp", damping=1e-1, limits=limits
            )
            configuration.integrate_inplace(vel, 1.0)
        out[t] = configuration.q[models.joint_addrs]
    return out
