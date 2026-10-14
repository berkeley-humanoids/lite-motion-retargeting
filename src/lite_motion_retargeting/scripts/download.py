"""Download a motion source, and the G1 model that the G1 sources use.

The step puts the G1 URDF and its meshes in ``.cache/g1/``, because every G1 source
shares them. It puts the clips in ``.cache/<source>/``, from the Hub, or from the zip of a
source that is not on the Hub. To download only part of a large set from the Hub, give Hub
glob patterns with ``--include``. They replace the default patterns of the source.

Usage:
    uv run download
    uv run download --source lafan1bvh
    uv run download --source amass --include 'g1/CMU/**' 'g1/KIT/**'
    uv run download --source motiondecode --include 'samples/1.3.Basic_Gait_Category/**'
    uv run download --source kimodo
    uv run download --force
"""

import urllib.request
import zipfile

import tyro
from huggingface_hub import snapshot_download

from .. import CACHE_ROOT
from ..robots import G1_REPO_ID, G1_URDF_PATH
from ..sources import SOURCES
from ..sources.human import HumanClip

#: The number of files that download in parallel. AMASS and MotionDecode have tens of
#: thousands of small files, so the time per request, not the bandwidth, sets the download
#: time.
DOWNLOAD_WORKERS: int = 32


def download(source: str = "lafan1", include: list[str] | None = None, force: bool = False) -> None:
    """Download ``source`` into ``.cache/<source>/``, and the G1 model into ``.cache/g1/``.

    Args:
        source: A key of ``sources.SOURCES``.
        include: Hub glob patterns to download instead of the default patterns of the source.
        force: Download the files again, even if they exist.
    """
    if source not in SOURCES:
        raise SystemExit(f"Unknown source {source!r}. Expected one of {sorted(SOURCES)}.")
    spec = SOURCES[source]
    local_path = CACHE_ROOT / source
    if spec.repo_id is None and spec.zip_url is None:
        raise SystemExit(f"{spec.title} has no public download. Copy its clips to {local_path / spec.clip_dir} "
                         f"yourself, from {spec.page}.")

    snapshot_download(
        repo_id=G1_REPO_ID,
        repo_type="dataset",
        local_dir=str(CACHE_ROOT / "g1"),
        allow_patterns=["robot_description/g1/**"],
        force_download=force,
        max_workers=DOWNLOAD_WORKERS,
    )
    if spec.zip_url is not None:
        archive = local_path / spec.zip_url.rsplit("/", 1)[1]
        if force or not archive.exists():
            local_path.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(spec.zip_url, archive)
        with zipfile.ZipFile(archive) as files:
            files.extractall(local_path / spec.clip_dir)
    else:
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            local_dir=str(local_path),
            allow_patterns=include or list(spec.allow_patterns),
            force_download=force,
            max_workers=DOWNLOAD_WORKERS,
        )
    if spec.unpack is not None:
        spec.unpack(local_path)

    clips = sorted((local_path / spec.clip_dir).glob(spec.clip_glob))
    if not clips:
        raise SystemExit(f"No clips under {local_path / spec.clip_dir} (include={include}).")
    loaded = spec.load(clips[0])
    if isinstance(loaded, HumanClip):
        frames, fps = loaded.pos.shape[0], loaded.fps
    else:
        frames, fps = loaded[0]["base_pos"].shape[0], loaded[1]
    print(f"G1 model → {G1_URDF_PATH}")
    print(f"{source} → {local_path}")
    print(f"  clips  : {len(clips)}")
    print(f"  sample : {clips[0].relative_to(local_path)}  ({frames} frames @ {fps} Hz)")


def main() -> None:
    """Run :func:`download` with the options on the command line."""
    tyro.cli(download)
