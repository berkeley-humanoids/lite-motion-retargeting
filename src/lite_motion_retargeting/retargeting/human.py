"""Map human motion onto Lite: the goal pose of each Lite link, and the IK that reaches it.

Every retargeting backend shares this mapping. It turns the human bodies of
``sources/human.py`` into a goal pose for each Lite link in :data:`LITE_PAIRS`:

* **Calibration.** Both skeletons stand in the T-pose, facing +x. The scale is the ratio of
  their pelvis heights, and the rest pose of each Lite link comes from the model.
* **Goal.** Each human body turns about a joint, and :data:`ANCHORS` names the Lite link at
  the same joint. The pelvis goes to the human pelvis, scaled. Every other joint hangs off
  the joint above it by the segment between them in the Lite T-pose. The rest-normalized
  rotation of the human segment that joins the two joints turns this segment. A Lite link
  then turns with its human body about that joint. Thus the goals form a skeleton with the
  proportions of Lite and the joint rotations of the human, and Lite can reach this
  skeleton wherever its joints allow.
* **Planting.** A skeleton with other proportions does not step where the human steps, so
  a foot that the human plants would slide. :func:`plant` moves the whole goal skeleton,
  frame by frame, so that the sole point where a planted foot touches the floor stays in
  place and on the floor. A shift of all the goals together keeps them reachable. If a foot
  still dips below the floor, only the goal of that foot rises.

The IK backend (:func:`solve_ik`) solves for the joints that reach these goals, frame by
frame. An RL backend such as ReActor tracks the same goals in simulation.
"""

from typing import NamedTuple

import mujoco
import numpy as np
from lite_description import get_mjcf_path
from scipy.ndimage import uniform_filter1d

from ..kinematics import Sole, lowest_points, quat_mul, sole_points
from ..robots import FEET, body_id, joint_qpos_addr
from ..sources.human import HUMAN_BODIES, HUMAN_PARENTS
from ..store import FPS

#: Human body -> the Lite link at the joint that the body turns about. A toe has none.
ANCHORS: dict[str, str] = {
    "pelvis": "pelvis",
    "left_hip": "left_hip_pitch",
    "left_knee": "left_knee_pitch",
    "left_foot": "left_foot",
    "right_hip": "right_hip_pitch",
    "right_knee": "right_knee_pitch",
    "right_foot": "right_foot",
    "waist": "waist_yaw",
    "chest": "chest",
    "neck": "neck_yaw",
    "head": "head",
    "left_shoulder": "left_shoulder_pitch",
    "left_elbow": "left_elbow_pitch",
    "left_hand": "left_hand",
    "right_shoulder": "right_shoulder_pitch",
    "right_elbow": "right_elbow_pitch",
    "right_hand": "right_hand",
}


class Pair(NamedTuple):
    """The human body that a Lite link turns with, and the IK cost of each half of its pose."""

    human: str
    position_cost: float
    orientation_cost: float


#: Lite link -> its :class:`Pair`, for every link that the IK aims at, the root first. A
#: cost weighs metres against radians: a position cost of 10 on 1 cm weighs as much as an
#: orientation cost of 1 on 0.1 rad. The thigh, shin, upper arm, and forearm have an
#: orientation goal, so that no limb can twist about itself while its end stays on its goal.
LITE_PAIRS: dict[str, Pair] = {
    "pelvis": Pair("pelvis", 2.0, 1.0),
    "left_hip_yaw": Pair("left_hip", 0.0, 1.0),
    "left_knee_pitch": Pair("left_knee", 2.0, 1.0),
    "left_foot": Pair("left_foot", 10.0, 2.0),
    "right_hip_yaw": Pair("right_hip", 0.0, 1.0),
    "right_knee_pitch": Pair("right_knee", 2.0, 1.0),
    "right_foot": Pair("right_foot", 10.0, 2.0),
    "chest": Pair("chest", 0.0, 1.0),
    "head": Pair("head", 0.0, 0.3),
    "left_shoulder_yaw": Pair("left_shoulder", 0.0, 1.0),
    "left_elbow_pitch": Pair("left_elbow", 2.0, 1.0),
    "left_hand": Pair("left_hand", 5.0, 0.5),
    "right_shoulder_yaw": Pair("right_shoulder", 0.0, 1.0),
    "right_elbow_pitch": Pair("right_elbow", 2.0, 1.0),
    "right_hand": Pair("right_hand", 5.0, 0.5),
}

#: Posture cost of each DoF. It only settles what the goals leave open.
_POSTURE_COST: float = 1e-2
#: Posture cost of a finger. No human body drives a finger, and the posture holds it straight.
_FINGER_COST: float = 1e3
#: Posture cost of an ankle yaw. It keeps the ankle from turning the foot in place of the hip.
_ANKLE_YAW_COST: float = 1.0

#: rad/s. The IK turns no joint faster than this. A human motion rarely asks for more. The
#: exception is an arm that passes the T-pose: there the shoulder pitch and yaw of Lite share
#: an axis, and without the limit the solution would swap one for the other in one frame.
_JOINT_SPEED: float = 20.0

#: Frames. :func:`plant` averages its height shift over this window, so that the body does
#: not jump when it changes the foot it stands on.
_HEIGHT_WINDOW: int = 5

#: Newton steps on the first frame, which starts from the T-pose and not from the frame before.
_FIRST_FRAME_ITERS: int = 50


class Calibration(NamedTuple):
    """The constants that take one human skeleton onto Lite."""

    scale: float
    #: Each link of :data:`LITE_PAIRS` in the Lite T-pose, on the floor at the origin and
    #: facing +x: its position, and its world rotation in wxyz.
    rest_pos: np.ndarray
    rest_quat: np.ndarray
    #: Human body -> the position of its :data:`ANCHORS` link in that T-pose.
    joints: dict[str, np.ndarray]


def load_floating_model() -> mujoco.MjModel:
    """Load the Lite Pro MJCF with a free joint on the pelvis, so that the IK moves the base."""
    spec = mujoco.MjSpec.from_file(str(get_mjcf_path("lite_pro")))
    spec.body("pelvis").add_freejoint()
    return spec.compile()


def calibrate(lite: mujoco.MjModel, human_rest: np.ndarray) -> Calibration:
    """Measure the Lite T-pose, which is its zero pose, against ``human_rest`` (B, 3).

    ``lite`` is the model with the pelvis welded to the origin.
    """
    data = mujoco.MjData(lite)
    mujoco.mj_kinematics(lite, data)
    # One frame of the zero pose: no joint columns, and the base at the origin.
    soles = sole_points(lite, np.zeros(0, dtype=np.int32), np.zeros((1, 0)), np.zeros((1, 3)),
                        np.array([[1.0, 0.0, 0.0, 0.0]]), FEET)
    # The lift puts the soles on the floor. The pelvis is at the origin, so the lift is also
    # the height of the pelvis.
    lift = np.array([0.0, 0.0, -soles[0, :, 2].min()])
    ids = [body_id(lite, link) for link in LITE_PAIRS]
    return Calibration(
        scale=float(lift[2] / human_rest[HUMAN_BODIES.index("pelvis"), 2]),
        rest_pos=data.xpos[ids] + lift,
        rest_quat=data.xquat[ids],
        joints={body: data.xpos[body_id(lite, link)] + lift for body, link in ANCHORS.items()},
    )


def driven_joints(lite: mujoco.MjModel) -> tuple[str, ...]:
    """Return the joints that the goals drive, sorted by name.

    These are the joints between the root and a link of :data:`LITE_PAIRS`. No goal lies
    beyond a hand, so the fingers are not among them.
    """
    driven = set()
    for link in LITE_PAIRS:
        body = body_id(lite, link)
        while body > 0:
            for joint in range(lite.body_jntadr[body], lite.body_jntadr[body] + lite.body_jntnum[body]):
                driven.add(mujoco.mj_id2name(lite, mujoco.mjtObj.mjOBJ_JOINT, joint))
            body = lite.body_parentid[body]
    return tuple(sorted(driven))


def goals(pos: np.ndarray, quat: np.ndarray, calibration: Calibration) -> tuple[np.ndarray, np.ndarray]:
    """Return the world goal pose of each link of :data:`LITE_PAIRS`, per frame.

    ``pos`` ``(T, B, 3)`` and ``quat`` ``(T, B, 4)`` are human bodies as
    ``sources/human.py`` defines them. Returns ``(T, P, 3)`` positions and ``(T, P, 4)`` wxyz rotations.
    """
    rotation = _matrices(quat)
    joints = {"pelvis": calibration.scale * pos[:, HUMAN_BODIES.index("pelvis")]}
    for body, parent in HUMAN_PARENTS.items():
        if body in ANCHORS:
            segment = calibration.joints[body] - calibration.joints[parent]
            joints[body] = joints[parent] + rotation[:, HUMAN_BODIES.index(parent)] @ segment

    bodies = [HUMAN_BODIES.index(pair.human) for pair in LITE_PAIRS.values()]
    goal_pos = np.stack([
        joints[pair.human] + rotation[:, body] @ (rest - calibration.joints[pair.human])
        for pair, body, rest in zip(LITE_PAIRS.values(), bodies, calibration.rest_pos, strict=True)
    ], axis=1)
    goal_quat = quat_mul(quat[:, bodies], calibration.rest_quat)
    return goal_pos.astype(np.float32), goal_quat.astype(np.float32)


def plant(goal_pos: np.ndarray, goal_quat: np.ndarray, feet_planted: np.ndarray, foot_sole: Sole) -> np.ndarray:
    """Shift the goals of :func:`goals` so that a planted foot does not slide or leave the floor.

    ``feet_planted`` ``(T, 2)`` tells which of :data:`robots.FEET` the source stands on. Between two
    frames where a foot is planted, the lowest point of its sole moves as a point of the
    foot. The goals shift back by that slide, averaged over the planted feet. In a frame
    with a planted foot, the goals shift down by the height of the lowest sole point,
    averaged over the planted feet. Across frames with no planted foot, such as the flight
    of a jump, the height shift changes linearly from the frame before to the frame after.
    If the sole of a foot is still below the floor, the goal of that foot rises out of it.
    Returns the shifted ``goal_pos``.
    """
    feet = [list(LITE_PAIRS).index(foot) for foot in FEET]
    foot_pos = goal_pos[:, feet].astype(np.float64)
    foot_rot = _matrices(goal_quat[:, feet])
    lowest = lowest_points(foot_sole, foot_pos, foot_rot)

    # The horizontal shift cancels the slide of the planted feet.
    local = np.einsum("tfji,tfj->tfi", foot_rot[:-1], lowest[:-1] - foot_pos[:-1])
    moved = foot_pos[1:] + np.einsum("tfij,tfj->tfi", foot_rot[1:], local)
    both = feet_planted[1:] & feet_planted[:-1]
    slide = np.sum((moved[..., :2] - lowest[:-1, :, :2]) * both[..., None], axis=1)
    step = -slide / np.maximum(both.sum(axis=1), 1)[:, None]
    shift_xy = np.concatenate([np.zeros((1, 2)), np.cumsum(step, axis=0)])

    # The vertical shift puts the planted feet on the floor.
    frames = np.arange(goal_pos.shape[0])
    standing = feet_planted.any(axis=1)
    height = np.sum(lowest[..., 2] * feet_planted, axis=1) / np.maximum(feet_planted.sum(axis=1), 1)
    shift_z = np.zeros(len(frames))
    if standing.any():
        shift_z = -np.interp(frames, frames[standing], height[standing])
        shift_z = uniform_filter1d(shift_z, _HEIGHT_WINDOW, mode="nearest")

    out = goal_pos.copy()
    out[..., :2] += shift_xy[:, None].astype(np.float32)
    out[..., 2] += shift_z[:, None].astype(np.float32)
    # A foot that lands while it still moves can dip below the floor.
    depth = -lowest[..., 2] - shift_z[:, None]
    out[:, feet, 2] += np.maximum(depth, 0.0).astype(np.float32)
    return out


def solve_ik(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
    iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the IK of the floating-base ``model`` for the goals of :func:`goals`, per frame.

    Each frame starts from the solution of the frame before, so that the solution moves
    continuously. ``iters`` is the number of Newton steps on each frame after the first. No
    joint turns faster than :data:`_JOINT_SPEED` between two frames. Returns the base
    position ``(T, 3)``, the base rotation ``(T, 4)`` in wxyz, and the ``(T, J)`` joint
    positions in the order of ``joint_names``.
    """
    import mink

    configuration = mink.Configuration(model)
    tasks = [
        mink.FrameTask(link, "body", position_cost=pair.position_cost,
                       orientation_cost=pair.orientation_cost, lm_damping=1.0)
        for link, pair in LITE_PAIRS.items()
    ]
    posture_cost = np.full(model.nv, _POSTURE_COST)
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        dof = model.jnt_dofadr[joint]
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            posture_cost[dof:dof + 6] = 0.0
        elif "_finger" in name:
            posture_cost[dof] = _FINGER_COST
        elif "ankle_yaw" in name:
            posture_cost[dof] = _ANKLE_YAW_COST
    posture = mink.PostureTask(model, cost=posture_cost)
    position_limit = mink.ConfigurationLimit(model)
    speed_limit = mink.VelocityLimit(model, dict.fromkeys(joint_names, _JOINT_SPEED))
    # The Newton steps of one frame add up to one frame, so that the speed limit holds
    # between frames.
    step = 1.0 / (FPS * iters)
    joint_addrs = np.asarray([joint_qpos_addr(model, name) for name in joint_names])

    # The joints at zero are the T-pose. The posture task has no cost on the base, so its
    # target base has no effect.
    start = np.zeros(model.nq)
    start[:3], start[3:7] = goal_pos[0, 0], goal_quat[0, 0]
    posture.set_target(start)
    configuration.update(start)

    frames = goal_pos.shape[0]
    base_pos = np.empty((frames, 3), dtype=np.float32)
    base_quat = np.empty((frames, 4), dtype=np.float32)
    joint_pos = np.empty((frames, len(joint_names)), dtype=np.float32)
    for t in range(frames):
        for index, task in enumerate(tasks):
            task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3(goal_quat[t, index].astype(np.float64)), goal_pos[t, index].astype(np.float64)
            ))
        if t == 0:
            # The first frame moves from the T-pose in no time, so no speed limit applies.
            dt, limits, newton_steps = 1.0, [position_limit], _FIRST_FRAME_ITERS
        else:
            dt, limits, newton_steps = step, [position_limit, speed_limit], iters
        for _ in range(newton_steps):
            velocity = mink.solve_ik(configuration, [*tasks, posture], dt, "daqp", damping=1e-1, limits=limits)
            configuration.integrate_inplace(velocity, dt)
        base_pos[t], base_quat[t] = configuration.q[:3], configuration.q[3:7]
        joint_pos[t] = configuration.q[joint_addrs]
    base_quat[base_quat[:, 0] < 0] *= -1.0
    return base_pos, base_quat, joint_pos


def _matrices(quat: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternions ``(..., 4)`` to rotation matrices ``(..., 3, 3)``."""
    w, x, y, z = np.moveaxis(quat.astype(np.float64), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)
