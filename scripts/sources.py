"""The motion sets that this pipeline reads: human motion, or motion already on the Unitree G1.

A human source, such as the LAFAN1 BVH takes, has a loader that returns a
:class:`human.HumanClip`: the bodies of one shared human skeleton, which ``human.py``
defines. A new human format needs only a reader into that skeleton.

The authors of a G1 source retargeted it to the 29-DoF G1. Every G1 source stores the same
36 numbers per frame: the base position, the base rotation, and the joints in the order of
:data:`common.G1_JOINT_NAMES`. The sources differ only in file format, quaternion order,
and rate. A loader reads one clip and returns it at its own rate:

    base_pos        (T, 3)   world, metres
    base_quat_wxyz  (T, 4)   world, wxyz, w >= 0
    g1_joint_pos    (T, 29)  radians

A source that does not ship one file per clip, such as Kimodo with its LeRobot shards,
unpacks itself after download into one file per clip in that layout.

The height of the base is a convention of each source, and AMASS puts the pelvis at the
floor. Thus ``retarget.py`` grounds every clip itself, and no loader corrects the height.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from common import G1_JOINT_NAMES
from human import HumanClip, load_lafan1_bvh

#: A G1 clip: the arrays that the module docstring lists, by name.
Motion = dict[str, np.ndarray]


class Source(NamedTuple):
    """One motion set, where to download it, and how to read a clip of it."""

    #: The name that people use for the set.
    title: str
    #: The terms of its clips, and thus of its motions.
    license: str
    #: The skeleton that its clips move, and thus what of Lite they can drive.
    skeleton: str
    #: The directory of the clips in the cache of the source, and the glob of a clip in that
    #: directory. A motion gets its name from the path of its clip below this directory.
    clip_dir: str
    clip_glob: str
    #: Reads one clip: a human clip, or a G1 clip with its rate in Hz.
    load: Callable[[Path], HumanClip | tuple[Motion, int]]
    #: The Hub dataset repo of the set, and the Hub glob patterns that ``download.py``
    #: downloads from it by default.
    repo_id: str | None = None
    allow_patterns: tuple[str, ...] = ()
    #: A zip of the clips, for a set that is not on the Hub, and the page of the set.
    zip_url: str | None = None
    page: str | None = None
    #: Turns the download, at its path, into the clip files. None if the download is clips.
    unpack: Callable[[Path], None] | None = None


def load_lafan1(path: Path) -> tuple[Motion, int]:
    """Read a CSV with no header, at 30 Hz, with the base rotation in xyzw."""
    return _motion(path, np.loadtxt(path, delimiter=",", dtype=np.float32), "xyzw"), 30


def load_amass(path: Path) -> tuple[Motion, int]:
    """Read a ``(T, 36)`` array with the base rotation in xyzw.

    The rate is that of the AMASS capture, which differs between subsets. The file name
    holds it: ``<take>_poses_<fps>_jpos.npy``.
    """
    fps = int(re.fullmatch(r".*_(\d+)_jpos", path.stem).group(1))
    return _motion(path, np.load(path).astype(np.float32), "xyzw"), fps


def load_motiondecode(path: Path) -> tuple[Motion, int]:
    """Read a CSV with a header row, at 120 Hz, with the base rotation in wxyz.

    The header names every column, so the loader checks the joint order and does not
    assume it.
    """
    with path.open() as file:
        header = file.readline().strip().split(",")
    if header[7:] != [f"dof_{name}(rad)" for name in G1_JOINT_NAMES]:
        raise ValueError(f"{path}: the joint columns are not in the G1 order")
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    return _motion(path, raw, "wxyz"), 120


def load_kimodo(path: Path) -> tuple[Motion, int]:
    """Read a ``(T, 36)`` array from :func:`unpack_kimodo`, at 50 Hz, with the rotation in wxyz."""
    return _motion(path, np.load(path), "wxyz"), 50


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


#: Source key -> source. A key has no "_": a motion is named "<key>_<clip>", and the site
#: reads the key back from the name up to its first "_".
SOURCES: dict[str, Source] = {
    "lafan1bvh": Source(
        title="LAFAN1 BVH",
        license="CC BY-NC-ND 4.0",
        skeleton="Human, 22 joints, no fingers",
        zip_url="https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/master/lafan1/lafan1.zip",
        page="https://github.com/ubisoft/ubisoft-laforge-animation-dataset",
        clip_dir="bvh",
        clip_glob="*.bvh",
        load=load_lafan1_bvh,
    ),
    "lafan1": Source(
        title="LAFAN1",
        license="CC BY-NC-ND 4.0",
        skeleton="Unitree G1, 29 joints",
        repo_id="lvhaidong/LAFAN1_Retargeting_Dataset",
        allow_patterns=("g1/*.csv", "meta_data/**", "README.md", "LICENSE"),
        clip_dir="g1",
        clip_glob="*.csv",
        load=load_lafan1,
    ),
    "amass": Source(
        title="AMASS",
        license="AMASS: non-commercial research, no redistribution",
        skeleton="Unitree G1, 29 joints",
        repo_id="fleaven/Retargeted_AMASS_for_robotics",
        allow_patterns=("g1/**", "README.md"),
        clip_dir="g1",
        clip_glob="**/*.npy",
        load=load_amass,
    ),
    "motiondecode": Source(
        title="MotionDecode",
        license="ChingMu: non-commercial research, not to be hosted as a competing dataset",
        skeleton="Unitree G1, 29 joints",
        repo_id="CMRobot/MotionDecode",
        allow_patterns=("samples/**", "metadata/**", "README.md", "LICENSE.md"),
        clip_dir="samples",
        clip_glob="**/*.csv",
        load=load_motiondecode,
    ),
    "kimodo": Source(
        title="Kimodo G1",
        license="Apache-2.0",
        skeleton="Unitree G1, 29 joints",
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
        skeleton="Unitree G1, 29 joints",
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
        skeleton="Unitree G1, 29 joints",
        repo_id="cagataydev/kimodo-g1-longhorizon-v3",
        allow_patterns=("data/**", "meta/**", "README.md"),
        clip_dir="episodes",
        clip_glob="*/*.npy",
        load=load_kimodo,
        unpack=unpack_kimodo,
    ),
}


def clip_key(clip_dir: Path, path: Path) -> str:
    """Return the path of a clip below the clip directory of its source, without the suffix."""
    return path.relative_to(clip_dir).with_suffix("").as_posix()


def motion_name(source: str, clip_dir: Path, path: Path) -> str:
    """Name a motion ``<source>_<path of its clip below clip_dir>``.

    Every run of characters other than letters and digits becomes ``_``. The full path, and
    not only the file name, keeps two takes with one name in two directories apart. AMASS
    and MotionDecode both have such takes. A minus sign before a number becomes ``m``,
    because AMASS CNRS has both ``-01_L_1`` and ``01_L_1``, and only the sign tells them
    apart.
    """
    key = re.sub(r"(^|[/_ ])-(?=\d)", r"\1m", clip_key(clip_dir, path))
    return f"{source}_{re.sub(r'[^A-Za-z0-9]+', '_', key).strip('_')}"


def _motion(path: Path, raw: np.ndarray, quat_order: str) -> Motion:
    """Split ``(T, 36)`` rows into the arrays of a G1 clip, with the rotation in wxyz.

    ``quat_order`` is the order of the base rotation in ``raw``: ``"wxyz"`` or ``"xyzw"``.
    """
    columns = 7 + len(G1_JOINT_NAMES)
    if raw.ndim != 2 or raw.shape[1] != columns:
        raise ValueError(f"{path}: expected {columns} columns, got shape {raw.shape}")
    quat = raw[:, 3:7] if quat_order == "wxyz" else raw[:, [6, 3, 4, 5]]
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    quat[quat[:, 0] < 0] *= -1.0
    return {"base_pos": raw[:, 0:3], "base_quat_wxyz": quat, "g1_joint_pos": raw[:, 7:]}
