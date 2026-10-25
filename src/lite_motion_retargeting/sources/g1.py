"""Loaders of the G1 sources.

The authors of a G1 source retargeted it to the 29-DoF G1. Every G1 source stores the same
36 numbers per frame: the base position, the base rotation, and the joints in the order of
:data:`robots.G1_JOINT_NAMES`. The sources differ only in file format, quaternion order,
and rate. A loader reads one clip into a :class:`G1Clip`, at its own rate.

A source that does not ship one file per clip, such as Kimodo with its LeRobot shards,
unpacks itself after download into one file per clip in that layout.

The height of the base is a convention of each source, and AMASS puts the pelvis at the
floor. Thus the retargeting grounds every clip itself, and no loader corrects the height.
"""

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..robots import G1_JOINT_NAMES


class G1Clip(NamedTuple):
    """One G1 clip at its own rate."""

    fps: int
    #: ``(T, 3)``, world, metres.
    base_pos: np.ndarray
    #: ``(T, 4)``, world, wxyz, ``w >= 0``.
    base_quat: np.ndarray
    #: ``(T, 29)``, radians, in the order of :data:`robots.G1_JOINT_NAMES`.
    joint_pos: np.ndarray


def load_lafan1(path: Path) -> G1Clip:
    """Read a CSV with no header, at 30 Hz, with the base rotation in xyzw."""
    return _clip(path, np.loadtxt(path, delimiter=",", dtype=np.float32), "xyzw", 30)


def load_amass(path: Path) -> G1Clip:
    """Read a ``(T, 36)`` array with the base rotation in xyzw.

    The rate is that of the AMASS capture, which differs between subsets. The file name
    holds it: ``<take>_poses_<fps>_jpos.npy``.
    """
    fps = int(re.fullmatch(r".*_(\d+)_jpos", path.stem).group(1))
    return _clip(path, np.load(path).astype(np.float32), "xyzw", fps)


def load_motiondecode(path: Path) -> G1Clip:
    """Read a CSV with a header row, at 120 Hz, with the base rotation in wxyz.

    The header names every column, so the loader checks the joint order and does not
    assume it.
    """
    with path.open() as file:
        header = file.readline().strip().split(",")
    if header[7:] != [f"dof_{name}(rad)" for name in G1_JOINT_NAMES]:
        raise ValueError(f"{path}: the joint columns are not in the G1 order")
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    return _clip(path, raw, "wxyz", 120)


def load_kimodo(path: Path) -> G1Clip:
    """Read a ``(T, 36)`` array from :func:`unpack_kimodo`, at 50 Hz, with the rotation in wxyz."""
    return _clip(path, np.load(path), "wxyz", 50)


def unpack_kimodo(root: Path) -> None:
    """Split the LeRobot shards into ``episodes/<group>/episode_<index>.npy``.

    A row of ``observation.state`` holds the base position, the base rotation in wxyz, the
    29 joints in G1 order, and then their 29 velocities, which a clip does not keep.

    The group comes from the text prompt of the episode. It is the motion that the prompt
    asks for, so ``perform: a person bowing forward`` goes to ``bowing_forward``. A
    long-horizon prompt chains motions after the name of its family, so
    ``combat drill: kick → walk`` goes to ``combat_drill``.
    """
    tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pydict()
    groups = {}
    for index, text in zip(tasks["task_index"], tasks["task"], strict=True):
        if text.startswith("perform:"):
            group = re.sub(r"^perform: (a person )?", "", text)
        else:
            group = text.split(":")[0]
        groups[index] = group.replace(" ", "_")
    columns = ["episode_index", "task_index", "frame_index", "observation.state"]
    for shard in sorted((root / "data").glob("**/*.parquet")):
        try:
            table = pq.read_table(shard, columns=columns)
        except pa.ArrowInvalid as error:
            # The last shard of each Kimodo set on the Hub has no footer, and no reader
            # can open it.
            print(f"Skipped {shard.relative_to(root)}, because pyarrow cannot read it: {error}")
            continue
        episodes = table.column("episode_index").to_numpy()
        task = table.column("task_index").to_numpy()
        frame = table.column("frame_index").to_numpy()
        state = np.asarray(
            table.column("observation.state").combine_chunks().flatten(), dtype=np.float32
        ).reshape(len(table), -1)
        for episode in np.unique(episodes):
            rows = np.flatnonzero(episodes == episode)
            rows = rows[np.argsort(frame[rows])]
            path = root / "episodes" / groups[int(task[rows[0]])] / f"episode_{episode:05d}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, state[rows, :36])


def _clip(path: Path, raw: np.ndarray, quat_order: str, fps: int) -> G1Clip:
    """Split ``(T, 36)`` rows into a G1 clip, with the rotation in wxyz.

    ``quat_order`` is the order of the base rotation in ``raw``: ``"wxyz"`` or ``"xyzw"``.
    """
    columns = 7 + len(G1_JOINT_NAMES)
    if raw.ndim != 2 or raw.shape[1] != columns:
        raise ValueError(f"{path}: expected {columns} columns, got shape {raw.shape}")
    quat = raw[:, 3:7] if quat_order == "wxyz" else raw[:, [6, 3, 4, 5]]
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    quat[quat[:, 0] < 0] *= -1.0
    return G1Clip(fps, raw[:, 0:3], quat, raw[:, 7:])
