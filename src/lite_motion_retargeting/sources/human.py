"""Human motion in one skeleton, whatever format a source ships it in.

A reader turns one clip into world poses of :data:`HUMAN_BODIES`. Every array is in world
frame and metres, with z up and the floor of the source at ``z = 0``:

    pos   (T, B, 3)
    quat  (T, B, 4)  wxyz, rest-normalized
    rest  (B, 3)     each body in the T-pose: on the floor, facing +x, left along +y

A rest-normalized rotation is the world rotation of the body times the inverse of its
world rotation in the T-pose. Thus identity means "as in the T-pose" for every source,
whatever axes its joints use, and a robot mapping needs no offsets per source.

A body is the frame at a joint of the skeleton, and it turns with the segment that starts
at that joint: ``left_knee`` sits at the knee and turns with the shin.
"""

import tarfile
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from .. import CACHE_ROOT

#: The skeleton that every source maps onto, named after the keypoints of the Lite
#: conventions. A parent comes before its children.
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

#: The parent of each body of :data:`HUMAN_BODIES` except the root, in the same order.
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
    """One clip at its own rate, in the skeleton of the source and in :data:`HUMAN_BODIES`.

    The ``source_*`` arrays hold every joint of the source skeleton as published, in world
    frame, metres, z up, and wxyz. The other arrays hold :data:`HUMAN_BODIES`, as the module
    docstring defines them.
    """

    fps: int
    source_bodies: tuple[str, ...]
    source_pos: np.ndarray
    source_quat: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    rest: np.ndarray


# ── LAFAN1 ────────────────────────────────────────────────────────────────────

#: Rotation from BVH axes (y up) to world axes (z up): a BVH vector (x, y, z) becomes
#: (z, x, y). In world axes, the LAFAN1 actor in the T-pose at the start of every take faces
#: +x, with the left arm along +y.
_BVH_TO_WORLD: np.ndarray = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

#: Body of :data:`HUMAN_BODIES` -> the LAFAN1 joint at the same place. No body maps to
#: ``Spine1`` or to the clavicles (``LeftShoulder``, ``RightShoulder``).
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

#: Radians. In a flat stance, a LAFAN1 foot bone points this far below the horizontal. The
#: bone runs from the ankle to the ball of the foot. The value is the median over the
#: T-pose of every take.
_LAFAN1_FOOT_PITCH: float = np.radians(20.0)


def _frame(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the rotation whose x axis is ``x`` and whose y axis is ``y`` made orthogonal to it."""
    x = x / np.linalg.norm(x)
    y = y - x * (x @ y)
    y = y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def _lafan1_rest() -> dict[str, np.ndarray]:
    """Return the world rotation of each LAFAN1 joint in the T-pose, facing +x.

    A LAFAN1 joint points its bone along its own +x. Its +y points forward along the trunk
    and the legs, and backward along the arms. The T-pose at the start of every take matches
    this table to within a few degrees, because an actor does not hold the pose exactly.
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

    # Build the T-pose from the offsets. An offset is in the frame of the parent joint, and
    # the rest table turns that frame into the world. The toes stand on the floor, as they
    # do in the takes.
    rest_pos = np.zeros((len(names), 3))
    for joint, parent in enumerate(parents):
        if parent >= 0:
            rest_pos[joint] = rest_pos[parent] + _LAFAN1_REST[names[parent]] @ offsets[joint]
    rest_pos[:, 2] -= rest_pos[[names.index("LeftToe"), names.index("RightToe")], 2].min()

    world_rot = _BVH_TO_WORLD @ rotations
    world_pos = positions @ _BVH_TO_WORLD.T
    joints = [names.index(LAFAN1_JOINTS[body]) for body in HUMAN_BODIES]
    rest_rot = np.stack([_LAFAN1_REST[names[joint]] for joint in joints])
    normalized = world_rot[:, joints] @ np.swapaxes(rest_rot, -1, -2)
    return HumanClip(
        fps=round(1.0 / frame_time),
        source_bodies=tuple(names),
        source_pos=world_pos.astype(np.float32),
        source_quat=_wxyz(world_rot),
        pos=world_pos[:, joints].astype(np.float32),
        quat=_wxyz(normalized),
        rest=rest_pos[joints].astype(np.float32),
    )


# ── BONES-SEED ────────────────────────────────────────────────────────────────

#: The SOMA rig of BONES-SEED, which the download holds. Its first frame is a T-pose: y up,
#: facing +z, the left arm along +x, and the ``Root`` joint on the floor.
SEED_RIG: Path = CACHE_ROOT / "seed" / "soma_shapes" / "soma_base_rig" / "soma_base_skel_minimal.bvh"

#: Body of :data:`HUMAN_BODIES` -> the SOMA joint at the same place. No body maps to
#: ``Spine2``, ``Neck2``, the clavicles (``LeftShoulder``, ``RightShoulder``) or the fingers.
SEED_JOINTS: dict[str, str] = {
    "pelvis": "Hips",
    "left_hip": "LeftLeg",
    "left_knee": "LeftShin",
    "left_foot": "LeftFoot",
    "left_toe": "LeftToeBase",
    "right_hip": "RightLeg",
    "right_knee": "RightShin",
    "right_foot": "RightFoot",
    "right_toe": "RightToeBase",
    "waist": "Spine1",
    "chest": "Chest",
    "neck": "Neck1",
    "head": "Head",
    "left_shoulder": "LeftArm",
    "left_elbow": "LeftForeArm",
    "left_hand": "LeftHand",
    "right_shoulder": "RightArm",
    "right_elbow": "RightForeArm",
    "right_hand": "RightHand",
}

#: The archive of the SOMA Proportional takes: each skeleton keeps the proportions of its
#: actor, and thus the contacts of the capture. SOMA Uniform moves every actor onto one
#: skeleton.
SEED_ARCHIVE: str = "soma_proportional.tar.gz"


def unpack_seed(root: Path) -> None:
    """Extract the SOMA Proportional takes into ``clips/<category>/<take>.bvh``.

    The archive files each take under the date of its capture. The metadata gives each take
    one of 20 categories, such as ``Basic Locomotion Neutral``, which the motion browser
    shows. Half of the takes are mirrors of the others, and their names end in ``_M``. A run
    that stopped part way can run again: it extracts only if no take is in ``clips/`` yet,
    and moves the takes that are still in the archive layout.
    """
    if not any((root / "clips").glob("*/*.bvh")):
        with tarfile.open(root / SEED_ARCHIVE) as archive:
            archive.extractall(root, filter="data")
    metadata = max((root / "metadata").glob("seed_metadata_v*.parquet"))
    table = pq.read_table(metadata, columns=["filename", "category", "move_soma_proportional_path"]).to_pydict()
    for name, category, path in zip(
        table["filename"], table["category"], table["move_soma_proportional_path"], strict=True
    ):
        if (root / path).exists():
            target = root / "clips" / category.replace(" ", "_") / f"{name}.bvh"
            target.parent.mkdir(parents=True, exist_ok=True)
            (root / path).rename(target)
    for folder in sorted((root / SEED_ARCHIVE.removesuffix(".tar.gz")).glob("**/"), reverse=True):
        if not any(folder.iterdir()):
            folder.rmdir()


@cache
def _seed_rest() -> dict[str, np.ndarray]:
    """Return the world rotation of each SOMA joint in the T-pose, facing +x.

    The rotations come from the first frame of :data:`SEED_RIG`. Every BONES-SEED skeleton
    shares the joints and their axes of this rig, and only the bone lengths differ.
    """
    if not SEED_RIG.exists():
        raise FileNotFoundError(f"{SEED_RIG} is missing. Run `uv run download --source seed`.")
    names, parents, offsets, channels, values, _ = _read_bvh(SEED_RIG)
    rotations, _ = _bvh_world(parents, offsets, channels, values[:1])
    return {name: _BVH_TO_WORLD @ rotations[0, joint] for joint, name in enumerate(names)}


def load_seed_bvh(path: Path) -> HumanClip:
    """Read a BONES-SEED take: the 78 joints of SOMA, 120 Hz, y up, centimetres.

    About 0.5% of the takes have frames with NaN channels. The reader drops such frames at
    the start and at the end of a take. A take with such frames between valid frames raises
    a ValueError, because no part of it is whole. So does a take whose skeleton has a NaN
    offset.
    """
    rest = _seed_rest()
    names, parents, offsets, channels, values, frame_time = _read_bvh(path)
    if np.isnan(offsets).any():
        raise ValueError(f"{path.name}: the skeleton has a NaN offset.")
    broken = np.isnan(values).any(axis=1)
    if broken.any():
        valid = np.flatnonzero(~broken)
        if len(valid) == 0 or broken[valid[0]:valid[-1]].any():
            raise ValueError(f"{path.name}: {broken.sum()} of {len(values)} frames hold NaN, between valid frames.")
        values = values[valid[0]:valid[-1] + 1]
    rotations, positions = _bvh_world(parents, offsets, channels, values)

    # Build the T-pose from the offsets of this skeleton and the rotations of the rig. The
    # Root joint stands on the floor.
    rest_pos = np.zeros((len(names), 3))
    for joint, parent in enumerate(parents):
        if parent >= 0:
            rest_pos[joint] = rest_pos[parent] + rest[names[parent]] @ offsets[joint]
    rest_pos[:, 2] -= rest_pos[names.index("Root"), 2]

    world_rot = _BVH_TO_WORLD @ rotations
    world_pos = positions @ _BVH_TO_WORLD.T
    joints = [names.index(SEED_JOINTS[body]) for body in HUMAN_BODIES]
    rest_rot = np.stack([rest[names[joint]] for joint in joints])
    normalized = world_rot[:, joints] @ np.swapaxes(rest_rot, -1, -2)
    return HumanClip(
        fps=round(1.0 / frame_time),
        source_bodies=tuple(names),
        source_pos=world_pos.astype(np.float32),
        source_quat=_wxyz(world_rot),
        pos=world_pos[:, joints].astype(np.float32),
        quat=_wxyz(normalized),
        rest=rest_pos[joints].astype(np.float32),
    )


# ── AMASS SMPL-X ──────────────────────────────────────────────────────────────

#: The SMPL-X body model that AMASS fits its SMPL-X clips with: the neutral model with the
#: locked head, ``neutral/model.npz`` of AMASS. Its license forbids redistribution, so
#: ``uv run download`` cannot fetch it. Copy it here yourself.
SMPLX_MODEL: Path = CACHE_ROOT / "smplx" / "SMPLX_NEUTRAL.npz"

#: The shape coefficients that build the body. AMASS fits 16. UMR builds its bodies with the
#: first 10, and so does this reader, so that both retarget one body.
SMPLX_BETAS: int = 10

#: The 22 body joints of SMPL-X, in the order of the model. The face and the fingers follow
#: them in the model. No body of :data:`HUMAN_BODIES` maps to those.
SMPLX_JOINTS: tuple[str, ...] = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
    "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
)

#: Body of :data:`HUMAN_BODIES` -> the SMPL-X joint at the same place. No body maps to
#: ``spine2`` or to the collars. The SMPL-X ``left_foot`` is the ball of the foot.
SMPLX_BODIES: dict[str, str] = {
    "pelvis": "pelvis",
    "left_hip": "left_hip",
    "left_knee": "left_knee",
    "left_foot": "left_ankle",
    "left_toe": "left_foot",
    "right_hip": "right_hip",
    "right_knee": "right_knee",
    "right_foot": "right_ankle",
    "right_toe": "right_foot",
    "waist": "spine1",
    "chest": "spine3",
    "neck": "neck",
    "head": "head",
    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_hand": "left_wrist",
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_hand": "right_wrist",
}

#: Rotation from the axes of the SMPL-X model to world axes. The model stands in the T-pose
#: along its +y, faces its +z, and holds its left arm along its +x. Every joint of the model
#: has these axes in the T-pose, so this is the rest rotation of every joint, facing +x.
_SMPLX_TO_WORLD: np.ndarray = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@cache
def _smplx_model() -> dict[str, np.ndarray]:
    """Return the arrays of :data:`SMPLX_MODEL` that place the joints of a body."""
    if not SMPLX_MODEL.exists():
        raise FileNotFoundError(f"{SMPLX_MODEL} is missing. Copy the SMPL-X model of AMASS there.")
    with np.load(SMPLX_MODEL) as model:
        return {key: model[key] for key in ("v_template", "shapedirs", "J_regressor", "kintree_table")}


def load_amass_smplx(path: Path) -> HumanClip:
    """Read an AMASS SMPL-X clip: 22 body joints at the rate of the clip, z up, metres."""
    model = _smplx_model()
    joints = len(SMPLX_JOINTS)
    with np.load(path) as data:
        betas = data["betas"][:SMPLX_BETAS]
        poses = data["poses"][:, : 3 * joints].reshape(-1, joints, 3)
        trans = data["trans"]
        fps = round(float(data["mocap_frame_rate"]))

    # The body of these betas in the T-pose, in model axes.
    vertices = model["v_template"] + model["shapedirs"][:, :, :SMPLX_BETAS] @ betas
    rest_joints = model["J_regressor"][:joints] @ vertices
    parents = model["kintree_table"][0, :joints]

    # Forward kinematics. The pelvis turns about its own joint, and ``trans`` moves it.
    local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(*poses.shape, 3)
    world_rot = np.empty_like(local)
    world_pos = np.empty(poses.shape)
    world_rot[:, 0], world_pos[:, 0] = local[:, 0], rest_joints[0] + trans
    for joint in range(1, joints):
        parent = parents[joint]
        world_rot[:, joint] = world_rot[:, parent] @ local[:, joint]
        world_pos[:, joint] = world_pos[:, parent] + world_rot[:, parent] @ (rest_joints[joint] - rest_joints[parent])

    # The T-pose faces +x, with the pelvis above the origin and the lowest vertex on the floor.
    rest_pos = (rest_joints - rest_joints[0]) @ _SMPLX_TO_WORLD.T
    rest_pos[:, 2] -= ((vertices - rest_joints[0]) @ _SMPLX_TO_WORLD.T)[:, 2].min()

    bodies = [SMPLX_JOINTS.index(SMPLX_BODIES[body]) for body in HUMAN_BODIES]
    return HumanClip(
        fps=fps,
        source_bodies=SMPLX_JOINTS,
        source_pos=world_pos.astype(np.float32),
        source_quat=_wxyz(world_rot),
        pos=world_pos[:, bodies].astype(np.float32),
        quat=_wxyz(world_rot[:, bodies] @ _SMPLX_TO_WORLD.T),
        rest=rest_pos[bodies].astype(np.float32),
    )


# ── BVH ───────────────────────────────────────────────────────────────────────


def _read_bvh(path: Path) -> tuple[list[str], list[int], np.ndarray, list[list[str]], np.ndarray, float]:
    """Parse a BVH file in centimetres into its joints and its frames.

    Returns the joint names, the parent index of each joint (``-1`` for the root), the
    offset of each joint from its parent in metres, the channels of each joint, the
    ``(T, C)`` channel values as the file stores them, and the frame time in seconds. The
    parser skips end sites, because they have no channels.
    """
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []
    stack: list[int] = []
    in_end_site = False
    lines = path.read_text().splitlines()
    motion = next((index for index, line in enumerate(lines) if line.strip().startswith("MOTION")), None)
    if motion is None:
        raise ValueError(f"{path} has no MOTION section.")
    for tokens in map(str.split, lines[:motion]):
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
    frames = int(lines[motion + 1].split()[1])
    frame_time = float(lines[motion + 2].split()[2])
    values = np.loadtxt(lines[motion + 3:motion + 3 + frames], dtype=np.float64, ndmin=2)
    return names, parents, np.asarray(offsets) / 100.0, channels, values, frame_time


def _bvh_world(
    parents: list[int], offsets: np.ndarray, channels: list[list[str]], values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the world rotation ``(T, J, 3, 3)`` and position ``(T, J, 3)`` of every joint.

    The result is in BVH axes and metres. ``offsets`` are in metres, and the position
    channels in ``values`` are in centimetres.
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
        # The channels list the rotations outermost first: the intrinsic order of scipy.
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
    """Convert rotation matrices ``(..., 3, 3)`` to float32 wxyz quaternions with ``w >= 0``."""
    shape = matrices.shape[:-2]
    xyzw = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat()
    xyzw[xyzw[:, 3] < 0] *= -1.0
    return xyzw[:, [3, 0, 1, 2]].reshape(*shape, 4).astype(np.float32)
