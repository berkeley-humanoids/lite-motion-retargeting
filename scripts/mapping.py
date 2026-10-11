"""Map human motion onto Lite: where each Lite link should be, and the IK that puts it there.

The mapping is the part every retargeting backend shares. It turns the human bodies of
``human.py`` into a goal pose for each Lite link in :data:`LITE_PAIRS`:

* **Calibration.** Both skeletons stand in the T-pose, facing +x. The scale is the ratio of
  their pelvis heights, and the rest pose of each Lite link is measured on the model.
* **Goal.** Each human body turns about a joint, and :data:`ANCHORS` names the Lite link at
  the same joint. The pelvis goes to the human pelvis, scaled. Every other joint hangs off
  the joint above it by the segment between them in the Lite T-pose, turned by the
  rest-normalized rotation of the human segment that joins them. A Lite link then turns
  with its human body about that joint. The goals therefore form a skeleton with the
  proportions of Lite and the joint rotations of the human, which Lite can reach wherever
  its joints allow.
* **Planting.** A skeleton with other proportions does not step where the human steps, so
  a foot the human plants would slide. :func:`plant` moves the whole goal skeleton, frame
  by frame, so that the sole point a planted foot touches the floor with stays where it is
  and on the floor. The goals stay reachable, since a shift of all of them is. A foot that
  would still dip below the floor rises out of it on its own.

The IK backend solves for the joints that reach these goals, frame by frame. An RL backend
such as ReActor tracks the same goals in simulation.
"""

from typing import NamedTuple

import mujoco
import numpy as np
from common import FPS, Sole, body_id, lowest_points, quat_mul, sole_points
from human import HUMAN_BODIES, HUMAN_PARENTS
from lite_description import get_mjcf_path
from scipy.ndimage import uniform_filter1d

#: Human body -> the Lite link at the joint it turns about. A toe has no Lite joint.
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
    """The human body a Lite link turns with, and the IK cost of each half of its pose."""

    human: str
    position_cost: float
    orientation_cost: float


#: Lite link -> the human body it turns with, the root first. Every link the IK aims at.
#: A cost weighs metres against radians: a position cost of 10 weighs 1 cm like an
#: orientation cost of 1 weighs 0.1 rad. The thigh, shin, upper arm and forearm carry an
#: orientation goal, so no limb can twist about itself while its end still lands on its goal.
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

#: The Lite bodies that stand on the floor.
FEET: tuple[str, ...] = ("left_foot", "right_foot")

#: Per-DoF posture cost. It only settles what the goals leave open, except at a finger,
#: which no human body drives and which it holds straight, and at an ankle yaw, which
#: it keeps from turning the foot in place of the hip.
_POSTURE_COST: float = 1e-2
_FINGER_COST: float = 1e3
_ANKLE_YAW_COST: float = 1.0

#: rad/s. The IK turns no joint faster. A human motion rarely asks for more, except where
#: the arm passes the T-pose, in which the shoulder pitch and yaw of Lite share an axis, and
#: the solution would swap one for the other in a single frame.
_JOINT_SPEED: float = 20.0

#: Frames. :func:`plant` averages its height shift over this window, so the body does not
#: jump when the foot it stands on changes.
_HEIGHT_WINDOW: int = 5

#: Newton steps on the first frame, which starts from the T-pose rather than the last frame.
_FIRST_FRAME_ITERS: int = 50


class Calibration(NamedTuple):
    """The constants that take one human skeleton onto Lite."""

    scale: float
    #: Each link of :data:`LITE_PAIRS` in the Lite T-pose: standing on the floor at the
    #: origin, facing +x. Its position, and its world rotation, wxyz.
    rest_pos: np.ndarray
    rest_quat: np.ndarray
    #: Human body -> the position of its :data:`ANCHORS` link in that T-pose.
    joints: dict[str, np.ndarray]


def load_floating_model() -> mujoco.MjModel:
    """The Lite Pro MJCF with a free joint on the pelvis, for an IK that moves the base."""
    spec = mujoco.MjSpec.from_file(str(get_mjcf_path("lite_pro")))
    spec.body("pelvis").add_freejoint()
    return spec.compile()


def calibrate(lite: mujoco.MjModel, human_rest: np.ndarray) -> Calibration:
    """Measure the Lite T-pose, which is its zero pose, against ``human_rest`` (B, 3).

    ``lite`` is the model with the pelvis welded to the origin.
    """
    data = mujoco.MjData(lite)
    mujoco.mj_kinematics(lite, data)
    soles = sole_points(lite, np.zeros(0, dtype=np.int32), np.zeros((1, 0)), np.zeros((1, 3)),
                        np.array([[1.0, 0.0, 0.0, 0.0]]), FEET)
    lift = np.array([0.0, 0.0, -soles[0, :, 2].min()])
    ids = [body_id(lite, link) for link in LITE_PAIRS]
    return Calibration(
        scale=float(lift[2] / human_rest[HUMAN_BODIES.index("pelvis"), 2]),
        rest_pos=data.xpos[ids] + lift,
        rest_quat=data.xquat[ids].copy(),
        joints={body: data.xpos[body_id(lite, link)] + lift for body, link in ANCHORS.items()},
    )


def goals(pos: np.ndarray, quat: np.ndarray, calibration: Calibration) -> tuple[np.ndarray, np.ndarray]:
    """World goal pose of each link of :data:`LITE_PAIRS`, per frame.

    ``pos`` ``(T, B, 3)`` and ``quat`` ``(T, B, 4)`` are human bodies as ``human.py`` defines
    them. Returns ``(T, P, 3)`` positions and ``(T, P, 4)`` wxyz rotations.
    """
    rotation = _matrices(quat)
    joints = {"pelvis": calibration.scale * pos[:, HUMAN_BODIES.index("pelvis")]}
    for body in HUMAN_BODIES[1:]:
        if body in ANCHORS:
            parent = HUMAN_PARENTS[body]
            segment = calibration.joints[body] - calibration.joints[parent]
            joints[body] = joints[parent] + rotation[:, HUMAN_BODIES.index(parent)] @ segment

    human = [HUMAN_BODIES.index(pair.human) for pair in LITE_PAIRS.values()]
    goal_pos = np.stack([
        joints[pair.human] + rotation[:, index] @ (rest - calibration.joints[pair.human])
        for pair, index, rest in zip(LITE_PAIRS.values(), human, calibration.rest_pos, strict=True)
    ], axis=1)
    goal_quat = quat_mul(quat[:, human], calibration.rest_quat)
    return goal_pos.astype(np.float32), goal_quat.astype(np.float32)


def plant(goal_pos: np.ndarray, goal_quat: np.ndarray, feet_planted: np.ndarray, foot_sole: Sole) -> np.ndarray:
    """Shift the goals of :func:`goals` so a planted foot neither slides nor leaves the floor.

    ``feet_planted`` ``(T, 2)`` says which of :data:`FEET` the source stands on. Between two
    frames where a foot is planted, the sole point it touches with moves as a point of the
    foot, and the goals shift back by that slide, averaged over the planted feet. In a frame
    with a foot planted, the goals shift down by the height of its lowest sole point; across
    frames with none, such as the flight of a jump, the shift runs straight from the one
    before to the one after. A foot whose sole is still below the floor then rises out of
    it. Returns the shifted ``goal_pos``.
    """
    feet = [list(LITE_PAIRS).index(foot) for foot in FEET]
    foot_pos = goal_pos[:, feet].astype(np.float64)
    foot_rot = _matrices(goal_quat[:, feet])
    lowest = lowest_points(foot_sole, foot_pos, foot_rot)

    local = np.einsum("tfji,tfj->tfi", foot_rot[:-1], lowest[:-1] - foot_pos[:-1])
    moved = foot_pos[1:] + np.einsum("tfij,tfj->tfi", foot_rot[1:], local)
    both = feet_planted[1:] & feet_planted[:-1]
    slide = np.sum((moved[..., :2] - lowest[:-1, :, :2]) * both[..., None], axis=1)
    step = -slide / np.maximum(both.sum(axis=1), 1)[:, None]
    shift_xy = np.concatenate([np.zeros((1, 2)), np.cumsum(step, axis=0)])

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
    # A foot that still dips below the floor, such as one that lands while it is still
    # moving, rises out of it on its own.
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
    """Per-frame IK of the floating-base ``model`` onto the goals of :func:`goals`.

    Each frame starts from the last one, so the solution moves continuously, and no joint
    turns faster than :data:`_JOINT_SPEED` between two frames. Returns the base position
    ``(T, 3)``, the base rotation ``(T, 4)`` wxyz, and ``(T, J)`` joint positions in the
    order of ``joint_names``.
    """
    import mink

    configuration = mink.Configuration(model)
    tasks = [
        mink.FrameTask(link, "body", position_cost=pair.position_cost,
                       orientation_cost=pair.orientation_cost, lm_damping=1.0)
        for link, pair in LITE_PAIRS.items()
    ]
    posture = mink.PostureTask(model, cost=_posture_costs(model))
    position_limit = mink.ConfigurationLimit(model)
    speed_limit = mink.VelocityLimit(model, dict.fromkeys(joint_names, _JOINT_SPEED))
    # A Newton step takes this long, so the steps of one frame add up to one frame, and the
    # speed limit holds between frames.
    step = 1.0 / (FPS * iters)
    joint_addrs = np.asarray(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names]
    )

    # Joints at zero are the T-pose. The posture task ignores the base it holds.
    start = np.zeros(model.nq)
    start[:3], start[3:7] = goal_pos[0, 0], goal_quat[0, 0]
    posture.set_target(start.copy())
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
        # The first frame moves from the T-pose in no time, so no speed limit applies to it.
        dt, limits = (1.0, [position_limit]) if t == 0 else (step, [position_limit, speed_limit])
        for _ in range(_FIRST_FRAME_ITERS if t == 0 else iters):
            velocity = mink.solve_ik(configuration, [*tasks, posture], dt, "daqp", damping=1e-1, limits=limits)
            configuration.integrate_inplace(velocity, dt)
        base_pos[t], base_quat[t] = configuration.q[:3], configuration.q[3:7]
        joint_pos[t] = configuration.q[joint_addrs]
    base_quat[base_quat[:, 0] < 0] *= -1.0
    return base_pos, base_quat, joint_pos


def _posture_costs(model: mujoco.MjModel) -> np.ndarray:
    """Posture cost per DoF: none on the free joint, see :data:`_POSTURE_COST` for the rest."""
    cost = np.full(model.nv, _POSTURE_COST)
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        dof = model.jnt_dofadr[joint]
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            cost[dof:dof + 6] = 0.0
        elif "_finger" in name:
            cost[dof] = _FINGER_COST
        elif "ankle_yaw" in name:
            cost[dof] = _ANKLE_YAW_COST
    return cost


def _matrices(quat: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(..., 3, 3)`` from wxyz quaternions ``(..., 4)``."""
    w, x, y, z = np.moveaxis(quat.astype(np.float64), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)
