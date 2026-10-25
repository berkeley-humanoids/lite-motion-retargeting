"""The motion sets that this pipeline reads: human motion, or motion already on the Unitree G1.

A human source, such as the LAFAN1 BVH takes, has a loader that returns a
:class:`human.HumanClip`: the bodies of one shared human skeleton, which ``human.py``
defines. A new human format needs only a reader into that skeleton: ``bvh.py`` reads BVH
takes, and ``smplx.py`` reads SMPL-X clips. ``g1.py`` reads the G1 sources, which share one
layout per frame.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from .. import CACHE_ROOT
from .bvh import SEED_ARCHIVE, load_lafan1_bvh, load_seed_bvh, unpack_seed
from .g1 import (
    Motion,
    load_amass,
    load_kimodo,
    load_lafan1,
    load_motiondecode,
    unpack_kimodo,
)
from .human import HumanClip
from .smplx import load_amass_smplx


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
    #: The Hub dataset repo of the set, and the Hub glob patterns that ``uv run download``
    #: downloads from it by default.
    repo_id: str | None = None
    allow_patterns: tuple[str, ...] = ()
    #: A zip of the clips, for a set that is not on the Hub, and the page of the set.
    zip_url: str | None = None
    page: str | None = None
    #: Turns the download, at its path, into the clip files. None if the download is clips.
    unpack: Callable[[Path], None] | None = None


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
    "seed": Source(
        title="BONES-SEED",
        license="BONES-SEED: academic or startup (revenue under $1M) use only, no redistribution, credit Bones Studio",
        skeleton="Human, SOMA, 78 joints with fingers",
        repo_id="bones-studio/seed",
        allow_patterns=(
            SEED_ARCHIVE, "metadata/*.parquet", "soma_shapes/soma_base_rig/*", "README.md", "LICENSE.md",
        ),
        page="https://bones.studio/datasets/seed",
        clip_dir="clips",
        clip_glob="*/*.bvh",
        load=load_seed_bvh,
        unpack=unpack_seed,
    ),
    "amassx": Source(
        title="AMASS SMPL-X",
        license="AMASS: non-commercial research, no redistribution",
        skeleton="Human, SMPL-X, 22 body joints",
        page="https://amass.is.tue.mpg.de",
        # UMR takes a folder named "motions" for BONES-SEED, so the clips go in "smplx".
        clip_dir="smplx",
        clip_glob="**/*_stageii.npz",
        load=load_amass_smplx,
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


def clip_dir(source: str) -> Path:
    """Return the directory of the clips of ``source`` in the cache."""
    return CACHE_ROOT / source / SOURCES[source].clip_dir


def clip_key(source: str, path: Path) -> str:
    """Return the path of a clip below the clip directory of its source, without the suffix."""
    return path.relative_to(clip_dir(source)).with_suffix("").as_posix()


def motion_name(source: str, path: Path) -> str:
    """Name a motion ``<source>_<clip key>``.

    Every run of characters other than letters and digits becomes ``_``. The full path, and
    not only the file name, keeps two takes with one name in two directories apart. AMASS
    and MotionDecode both have such takes. A minus sign before a number becomes ``m``,
    because AMASS CNRS has both ``-01_L_1`` and ``01_L_1``, and only the sign tells them
    apart.
    """
    key = re.sub(r"(^|[/_ ])-(?=\d)", r"\1m", clip_key(source, path))
    return f"{source}_{re.sub(r'[^A-Za-z0-9]+', '_', key).strip('_')}"


def clips(source: str, pattern: str | None = None) -> list[Path]:
    """Return the clips of ``source`` in the cache whose :func:`clip_key` matches the regex ``pattern``.

    Stops if the source is unknown, if no clip matches, or if two clips get one motion name.
    """
    if source not in SOURCES:
        raise SystemExit(f"Unknown source {source!r}. Expected one of {sorted(SOURCES)}.")
    paths = sorted(clip_dir(source).glob(SOURCES[source].clip_glob))
    if pattern is not None:
        paths = [path for path in paths if re.search(pattern, clip_key(source, path))]
    if not paths:
        raise SystemExit(f"No clip under {clip_dir(source)} matches {pattern!r}. Run `uv run download --source {source}`.")
    names = [motion_name(source, path) for path in paths]
    if len(set(names)) != len(names):
        raise SystemExit(f"Two clips of {source} get one motion name.")
    return paths


def source_of(path: Path) -> str:
    """Return the source key of a motion file, which is named ``<source>_<clip>``."""
    return path.stem.split("_", 1)[0]
