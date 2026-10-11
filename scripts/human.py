"""Human motion in one skeleton, whatever format a source ships it in.

A reader turns one clip into world poses of :data:`HUMAN_BODIES`. Every array is world
frame, metres, z up, and the floor of the source at ``z = 0``:

    pos   (T, B, 3)
    quat  (T, B, 4)  wxyz, rest-normalized
    rest  (B, 3)     each body in the T-pose: standing on the floor, facing +x, left +y

A rest-normalized rotation is the world rotation of the body times the inverse of its
world rotation in the T-pose. Identity therefore means "as in the T-pose" for every
source, whatever axes its joints use, and a robot mapping needs no offsets per source.

A body is the frame at a joint of the skeleton, turned with the segment that starts
there: ``left_knee`` sits at the knee and turns with the shin.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

#: The skeleton every source maps onto, after the keypoints of the Lite conventions.
HUMAN_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "left_knee",
    "left_foot",
    "left_toe",
    "right_hip",
    "right_knee",
    "right_foot",
    "right_toe",
    "waist",
    "chest",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_hand",
)

#: The parent of each body of :data:`HUMAN_BODIES`, the root first.
HUMAN_PARENTS: dict[str, str] = {
    "left_hip": "pelvis",
    "left_knee": "left_hip",
    "left_foot": "left_knee",
    "left_toe": "left_foot",
    "right_hip": "pelvis",
    "right_knee": "right_hip",
    "right_foot": "right_knee",
    "right_toe": "right_foot",
    "waist": "pelvis",
    "chest": "waist",
    "neck": "chest",
    "head": "neck",
    "left_shoulder": "chest",
    "left_elbow": "left_shoulder",
    "left_hand": "left_elbow",
    "right_shoulder": "chest",
    "right_elbow": "right_shoulder",
    "right_hand": "right_elbow",
}


class HumanClip(NamedTuple):
    """One clip at its own rate: the skeleton of the source, and the same motion in ours.

    The ``source_*`` arrays are every joint of the source skeleton as published, in world
    frame, metres, z up, wxyz. The rest are :data:`HUMAN_BODIES`, as the module docstring
    defines them.
    """

    fps: int
    source_bodies: tuple[str, ...]
    source_pos: np.ndarray
    source_quat: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    rest: np.ndarray


# ── LAFAN1 ────────────────────────────────────────────────────────────────────

#: BVH is y up. This takes a BVH vector (x, y, z) to (z, x, y), z up. A LAFAN1 actor then
#: stands in the T-pose that opens every take facing +x, with the left arm along +y.
_BVH_TO_WORLD: np.ndarray = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

#: Our body -> the LAFAN1 joint at the same place. ``Spine1`` and the clavicles
#: (``LeftShoulder``, ``RightShoulder``) have no body of ours.
LAFAN1_JOINTS: dict[str, str] = {
    "pelvis": "Hips",
    "left_hip": "LeftUpLeg",
    "left_knee": "LeftLeg",
    "left_foot": "LeftFoot",
    "left_toe": "LeftToe",
    "right_hip": "RightUpLeg",
    "right_knee": "RightLeg",
    "right_foot": "RightFoot",
    "right_toe": "RightToe",
    "waist": "Spine",
    "chest": "Spine2",
    "neck": "Neck",
    "head": "Head",
    "left_shoulder": "LeftArm",
    "left_elbow": "LeftForeArm",
    "left_hand": "LeftHand",
    "right_shoulder": "RightArm",
    "right_elbow": "RightForeArm",
    "right_hand": "RightHand",
}

#: Radians. In a flat stance a LAFAN1 foot bone, which runs from the ankle to the ball of
#: the foot, points this far below the horizontal (median over the T-pose of every take).
_LAFAN1_FOOT_PITCH: float = np.radians(20.0)


def _frame(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The rotation whose x axis is ``x`` and whose y axis is ``y`` made orthogonal to it."""
    x = x / np.linalg.norm(x)
    y = y - x * (x @ y)
    y = y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def _lafan1_rest() -> dict[str, np.ndarray]:
    """World rotation of each LAFAN1 joint in the T-pose, facing +x.

    A LAFAN1 joint points its bone along its own +x. Its +y points forward along the
    trunk and legs, and backward along the arms. Every take opens in a T-pose that matches
    this table to within the few degrees an actor holds it.
    """
    forward, left, up = np.eye(3)
    foot = np.array([np.cos(_LAFAN1_FOOT_PITCH), 0.0, -np.sin(_LAFAN1_FOOT_PITCH)])
    rest = {name: _frame(up, forward) for name in ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head")}
    for side, out in (("Left", left), ("Right", -left)):
        rest[f"{side}UpLeg"] = rest[f"{side}Leg"] = _frame(-up, forward)
        rest[f"{side}Foot"] = _frame(foot, up)
        rest[f"{side}Toe"] = _frame(forward, up)
        for bone in ("Shoulder", "Arm", "ForeArm", "Hand"):
            rest[f"{side}{bone}"] = _frame(out, -forward)
    return rest


_LAFAN1_REST: dict[str, np.ndarray] = _lafan1_rest()


def load_lafan1_bvh(path: Path) -> HumanClip:
    """Read a LAFAN1 BVH take: 22 joints, 30 Hz, y up, centimetres."""
    names, parents, offsets, channels, values, frame_time = _read_bvh(path)
    rotations, positions = _bvh_world(parents, offsets, channels, values)

    # The T-pose: an offset is in the frame of the parent joint, which the rest table turns
    # into the world. The toe joints stand on the floor, as they do in the takes.
    rest_pos = np.zeros((len(names), 3))
    for joint, parent in enumerate(parents):
        if parent >= 0:
            rest_pos[joint] = rest_pos[parent] + _LAFAN1_REST[names[parent]] @ offsets[joint]
    rest_pos[:, 2] -= rest_pos[[names.index("LeftToe"), names.index("RightToe")], 2].min()

    world_rot = _BVH_TO_WORLD @ rotations
    world_pos = positions @ _BVH_TO_WORLD.T
    index = [names.index(LAFAN1_JOINTS[body]) for body in HUMAN_BODIES]
    rest_rot = np.stack([_LAFAN1_REST[LAFAN1_JOINTS[body]] for body in HUMAN_BODIES])
    normalized = world_rot[:, index] @ np.swapaxes(rest_rot, -1, -2)
    return HumanClip(
        fps=round(1.0 / frame_time),
        source_bodies=tuple(names),
        source_pos=world_pos.astype(np.float32),
        source_quat=_wxyz(world_rot),
        pos=world_pos[:, index].astype(np.float32),
        quat=_wxyz(normalized),
        rest=rest_pos[index].astype(np.float32),
    )


# ── BVH ───────────────────────────────────────────────────────────────────────


def _read_bvh(path: Path) -> tuple[list[str], list[int], np.ndarray, list[list[str]], np.ndarray, float]:
    """Parse a BVH file into its joints and its frames, in the units of the file.

    Returns the joint names, the parent index of each (``-1`` for the root), the offset of
    each from its parent, the channels of each, a ``(T, channels)`` array of the frames,
    and the frame time in seconds. End sites carry no channels and are dropped.
    """
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []
    stack: list[int] = []
    in_end_site = False
    lines = path.read_text().splitlines()
    line = 0
    while not lines[line].strip().startswith("MOTION"):
        tokens = lines[line].split()
        line += 1
        if not tokens:
            continue
        if tokens[0] in ("ROOT", "JOINT"):
            names.append(tokens[1])
            parents.append(stack[-1] if stack else -1)
        elif tokens[0] == "End":
            in_end_site = True
        elif tokens[0] == "{" and not in_end_site:
            stack.append(len(names) - 1)
        elif tokens[0] == "}":
            if in_end_site:
                in_end_site = False
            else:
                stack.pop()
        elif tokens[0] == "OFFSET" and not in_end_site:
            offsets.append([float(value) for value in tokens[1:4]])
        elif tokens[0] == "CHANNELS":
            channels.append(tokens[2:])
    frames = int(lines[line + 1].split()[1])
    frame_time = float(lines[line + 2].split()[2])
    values = np.loadtxt(lines[line + 3:line + 3 + frames], dtype=np.float64, ndmin=2)
    return names, parents, np.asarray(offsets) / 100.0, channels, values, frame_time


def _bvh_world(
    parents: list[int], offsets: np.ndarray, channels: list[list[str]], values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World rotation ``(T, J, 3, 3)`` and position ``(T, J, 3)`` of every joint, in BVH axes.

    Positions are in metres: a position channel is centimetres, like the offsets.
    """
    frames = values.shape[0]
    rotations = np.empty((frames, len(parents), 3, 3))
    positions = np.empty((frames, len(parents), 3))
    column = 0
    for joint, parent in enumerate(parents):
        local_pos = np.broadcast_to(offsets[joint], (frames, 3)).copy()
        axes, angles = "", []
        for channel in channels[joint]:
            if channel.endswith("position"):
                local_pos[:, "XYZ".index(channel[0])] = values[:, column] / 100.0
            else:
                axes += channel[0]
                angles.append(values[:, column])
            column += 1
        # Channels list the rotations outermost first, which is scipy's intrinsic order.
        local_rot = Rotation.from_euler(axes, np.stack(angles, axis=1), degrees=True).as_matrix()
        if parent < 0:
            rotations[:, joint], positions[:, joint] = local_rot, local_pos
        else:
            rotations[:, joint] = rotations[:, parent] @ local_rot
            positions[:, joint] = positions[:, parent] + np.einsum(
                "tij,tj->ti", rotations[:, parent], local_pos
            )
    return rotations, positions


def _wxyz(matrices: np.ndarray) -> np.ndarray:
    """Unit wxyz quaternions with ``w >= 0`` from rotation matrices of any leading shape."""
    shape = matrices.shape[:-2]
    xyzw = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat()
    xyzw[xyzw[:, 3] < 0] *= -1.0
    return xyzw[:, [3, 0, 1, 2]].reshape(*shape, 4).astype(np.float32)
