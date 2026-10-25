"""Readers of BVH takes: LAFAN1 and BONES-SEED."""

import tarfile
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from .. import CACHE_ROOT
from .human import HUMAN_BODIES, HumanClip, wxyz

#: Rotation from BVH axes (y up) to world axes (z up): a BVH vector (x, y, z) becomes
#: (z, x, y). In world axes, the LAFAN1 actor in the T-pose at the start of every take faces
#: +x, with the left arm along +y.
_BVH_TO_WORLD: np.ndarray = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class Bvh(NamedTuple):
    """A BVH file: its joints and its frames."""

    names: list[str]
    #: The index of the parent of each joint, ``-1`` for the root.
    parents: list[int]
    #: The offset of each joint from its parent, in metres.
    offsets: np.ndarray
    channels: list[list[str]]
    #: ``(T, C)`` channel values as the file stores them.
    values: np.ndarray
    #: Seconds.
    frame_time: float


# ── LAFAN1 ────────────────────────────────────────────────────────────────────

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
    """Read a LAFAN1 BVH take: 22 joints, 30 Hz, y up, centimetres. The toes stand on the floor."""
    return _human_clip(_read_bvh(path), _LAFAN1_REST, LAFAN1_JOINTS, ("LeftToe", "RightToe"))


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
    rig = _read_bvh(SEED_RIG)
    rotations, _ = _bvh_world(rig.parents, rig.offsets, rig.channels, rig.values[:1])
    return {name: _BVH_TO_WORLD @ rotations[0, joint] for joint, name in enumerate(rig.names)}


def load_seed_bvh(path: Path) -> HumanClip:
    """Read a BONES-SEED take: the 78 joints of SOMA, 120 Hz, y up, centimetres.

    About 0.5% of the takes have frames with NaN channels. The reader drops such frames at
    the start and at the end of a take. A take with such frames between valid frames raises
    a ValueError, because no part of it is whole. So does a take whose skeleton has a NaN
    offset.
    """
    rest = _seed_rest()
    bvh = _read_bvh(path)
    if np.isnan(bvh.offsets).any():
        raise ValueError(f"{path.name}: the skeleton has a NaN offset.")
    broken = np.isnan(bvh.values).any(axis=1)
    if broken.any():
        valid = np.flatnonzero(~broken)
        if len(valid) == 0 or broken[valid[0]:valid[-1]].any():
            raise ValueError(f"{path.name}: {broken.sum()} of {len(broken)} frames hold NaN, between valid frames.")
        bvh = bvh._replace(values=bvh.values[valid[0]:valid[-1] + 1])
    return _human_clip(bvh, rest, SEED_JOINTS, ("Root",))


# ── BVH ───────────────────────────────────────────────────────────────────────


def _human_clip(bvh: Bvh, rest: dict[str, np.ndarray], joints: dict[str, str], floor: tuple[str, ...]) -> HumanClip:
    """Return a take as a :class:`HumanClip`.

    ``rest`` is the world rotation of each joint in the T-pose, facing +x, and ``joints``
    maps each body of :data:`HUMAN_BODIES` to its joint. The T-pose comes from the offsets
    of the take: an offset is in the frame of the parent joint, and ``rest`` turns that frame
    into the world. The lowest of the ``floor`` joints stands on the floor.
    """
    rotations, positions = _bvh_world(bvh.parents, bvh.offsets, bvh.channels, bvh.values)
    rest_pos = np.zeros((len(bvh.names), 3))
    for joint, parent in enumerate(bvh.parents):
        if parent >= 0:
            rest_pos[joint] = rest_pos[parent] + rest[bvh.names[parent]] @ bvh.offsets[joint]
    rest_pos[:, 2] -= rest_pos[[bvh.names.index(name) for name in floor], 2].min()

    world_rot = _BVH_TO_WORLD @ rotations
    world_pos = positions @ _BVH_TO_WORLD.T
    bodies = [bvh.names.index(joints[body]) for body in HUMAN_BODIES]
    rest_rot = np.stack([rest[bvh.names[joint]] for joint in bodies])
    return HumanClip(
        fps=round(1.0 / bvh.frame_time),
        source_bodies=tuple(bvh.names),
        source_pos=world_pos.astype(np.float32),
        source_quat=wxyz(world_rot),
        pos=world_pos[:, bodies].astype(np.float32),
        quat=wxyz(world_rot[:, bodies] @ np.swapaxes(rest_rot, -1, -2)),
        rest=rest_pos[bodies].astype(np.float32),
    )


def _read_bvh(path: Path) -> Bvh:
    """Parse a BVH file in centimetres. The parser skips end sites, because they have no channels."""
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
    return Bvh(names, parents, np.asarray(offsets) / 100.0, channels, values, frame_time)


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
