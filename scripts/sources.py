"""The motion sets this pipeline reads, each already retargeted to the 29-DoF Unitree G1.

Every source stores the same 36 numbers per frame: the base position, the base rotation,
and the joints in the order of :data:`common.G1_JOINT_NAMES`. The sources differ only in
file format, quaternion order, and rate, so a loader is all a new source needs. A loader
reads one clip and returns it at its own rate:

    base_pos        (T, 3)   world, metres
    base_quat_wxyz  (T, 4)   world, wxyz, w >= 0
    g1_joint_pos    (T, 29)  radians

A source that does not ship one file per clip, such as Kimodo with its LeRobot shards,
unpacks itself after download into one file per clip in that layout.

The height of the base is the convention of each source, and AMASS puts the pelvis at the
floor. ``retarget.py`` therefore grounds every clip itself, and no loader corrects it.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from common import G1_JOINT_NAMES

Motion = dict[str, np.ndarray]


class Source(NamedTuple):
    """One motion set on the Hub, and how to read a clip of it."""

    #: What people call the set.
    title: str
    #: The terms its clips, and so its motions, come under.
    license: str
    repo_id: str
    #: What ``download.py`` fetches by default, as Hub glob patterns.
    allow_patterns: tuple[str, ...]
    #: The directory of the clips, under the cache of the source. A motion is named
    #: after the path of its clip below this directory.
    clip_dir: str
    clip_glob: str
    #: Reads one clip, and returns it with its rate in Hz.
    load: Callable[[Path], tuple[Motion, int]]
    #: Turns the download, at the path it takes, into the clip files. None if it already is.
    unpack: Callable[[Path], None] | None = None


def load_lafan1(path: Path) -> tuple[Motion, int]:
    """A headerless CSV at 30 Hz, with the base rotation in xyzw."""
    return _motion(path, np.loadtxt(path, delimiter=",", dtype=np.float32), "xyzw"), 30


def load_amass(path: Path) -> tuple[Motion, int]:
    """A ``(T, 36)`` array, with the base rotation in xyzw.

    The rate is that of the AMASS capture, which differs between subsets, and the file
    name carries it: ``<take>_poses_<fps>_jpos.npy``.
    """
    fps = int(re.fullmatch(r".*_(\d+)_jpos", path.stem).group(1))
    return _motion(path, np.load(path).astype(np.float32), "xyzw"), fps


def load_motiondecode(path: Path) -> tuple[Motion, int]:
    """A CSV at 120 Hz with a header row, with the base rotation in wxyz.

    The header names every column, so the joint order is checked rather than assumed.
    """
    with path.open() as file:
        header = file.readline().strip().split(",")
    joints = [f"dof_{name}(rad)" for name in G1_JOINT_NAMES]
    if header[7:] != joints:
        raise ValueError(f"{path}: the joint columns are not in the G1 order")
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    return _motion(path, raw, "wxyz"), 120


def load_kimodo(path: Path) -> tuple[Motion, int]:
    """A ``(T, 36)`` array that :func:`unpack_kimodo` wrote, at 50 Hz, rotation in wxyz."""
    return _motion(path, np.load(path), "wxyz"), 50


def _task_group(text: str) -> str:
    """The folder of a Kimodo prompt: the motion it asks for, or the family of a chain.

    ``perform: a person bowing forward`` gives ``bowing_forward``. A long-horizon prompt
    chains motions after the name of its family, so ``combat drill: kick → walk`` gives
    ``combat_drill``.
    """
    if text.startswith("perform:"):
        group = re.sub(r"^perform: (a person )?", "", text)
    else:
        group = text.split(":")[0]
    return group.replace(" ", "_")


def unpack_kimodo(root: Path) -> None:
    """Split the LeRobot shards into ``episodes/<group>/episode_<index>.npy``.

    A row of ``observation.state`` is the base position, the base rotation in wxyz, the 29
    joints in G1 order, and then their 29 velocities, which a clip does not keep. The group
    comes from the text prompt of the episode, by :func:`_task_group`.
    """
    tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pydict()
    folders = {
        index: _task_group(text)
        for index, text in zip(tasks["task_index"], tasks["task"], strict=True)
    }
    columns = ["episode_index", "task_index", "frame_index", "observation.state"]
    for shard in sorted((root / "data").glob("**/*.parquet")):
        try:
            table = pq.read_table(shard, columns=columns)
        except pa.ArrowInvalid as error:
            # The last shard of each Kimodo set on the Hub ends without its footer, and
            # no reader opens it.
            print(f"Skipped {shard.relative_to(root)}, which cannot be read: {error}")
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
            path = root / "episodes" / folders[int(task[rows[0]])] / f"episode_{episode:05d}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, state[rows, :36])


# A key holds no "_": a motion is named "<key>_<clip>", and the site reads the key back
# from the name up to its first "_".
SOURCES: dict[str, Source] = {
    "lafan1": Source(
        title="LAFAN1",
        license="CC BY-NC-ND 4.0",
        repo_id="lvhaidong/LAFAN1_Retargeting_Dataset",
        allow_patterns=("g1/*.csv", "meta_data/**", "README.md", "LICENSE"),
        clip_dir="g1",
        clip_glob="*.csv",
        load=load_lafan1,
    ),
    "amass": Source(
        title="AMASS",
        license="AMASS: non-commercial research, no redistribution",
        repo_id="fleaven/Retargeted_AMASS_for_robotics",
        allow_patterns=("g1/**", "README.md"),
        clip_dir="g1",
        clip_glob="**/*.npy",
        load=load_amass,
    ),
    "motiondecode": Source(
        title="MotionDecode",
        license="ChingMu: non-commercial research, not to be hosted as a competing dataset",
        repo_id="CMRobot/MotionDecode",
        allow_patterns=("samples/**", "metadata/**", "README.md", "LICENSE.md"),
        clip_dir="samples",
        clip_glob="**/*.csv",
        load=load_motiondecode,
    ),
    "kimodo": Source(
        title="Kimodo G1",
        license="Apache-2.0",
        repo_id="cagataydev/kimodo-g1-motion-v1",
        allow_patterns=("data/**", "meta/**", "README.md"),
        clip_dir="episodes",
        clip_glob="*/*.npy",
        load=load_kimodo,
        unpack=unpack_kimodo,
    ),
    "kimodolh2": Source(
        title="Kimodo G1 long-horizon v2",
        license="Apache-2.0",
        repo_id="cagataydev/kimodo-g1-longhorizon-v2",
        allow_patterns=("data/**", "meta/**", "README.md"),
        clip_dir="episodes",
        clip_glob="*/*.npy",
        load=load_kimodo,
        unpack=unpack_kimodo,
    ),
    "kimodolh3": Source(
        title="Kimodo G1 long-horizon v3",
        license="Apache-2.0",
        repo_id="cagataydev/kimodo-g1-longhorizon-v3",
        allow_patterns=("data/**", "meta/**", "README.md"),
        clip_dir="episodes",
        clip_glob="*/*.npy",
        load=load_kimodo,
        unpack=unpack_kimodo,
    ),
}


def clip_key(clip_dir: Path, path: Path) -> str:
    """The path of a clip below the clip directory of its source, without the suffix."""
    return path.relative_to(clip_dir).with_suffix("").as_posix()


def motion_name(source: str, clip_dir: Path, path: Path) -> str:
    """Name a motion ``<source>_<path of its clip below clip_dir>``.

    Every run of characters other than letters and digits becomes ``_``. The path, and
    not the file name alone, keeps apart two takes of one name in two directories, which
    AMASS and MotionDecode both have. A minus sign before a number becomes ``m`` instead,
    because AMASS CNRS holds both ``-01_L_1`` and ``01_L_1``, and the sign is all that
    tells them apart.
    """
    key = re.sub(r"(^|[/_ ])-(?=\d)", r"\1m", clip_key(clip_dir, path))
    return f"{source}_{re.sub(r'[^A-Za-z0-9]+', '_', key).strip('_')}"


def _motion(path: Path, raw: np.ndarray, quat_order: str) -> Motion:
    """Split ``(T, 36)`` rows into the arrays of a motion, with the rotation in wxyz."""
    columns = 7 + len(G1_JOINT_NAMES)
    if raw.ndim != 2 or raw.shape[1] != columns:
        raise ValueError(f"{path}: expected {columns} columns, got shape {raw.shape}")
    quat = raw[:, 3:7] if quat_order == "wxyz" else raw[:, [6, 3, 4, 5]]
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    quat[quat[:, 0] < 0] *= -1.0
    return {"base_pos": raw[:, 0:3], "base_quat_wxyz": quat, "g1_joint_pos": raw[:, 7:]}
