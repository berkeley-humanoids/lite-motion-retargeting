"""Download a motion source, and the G1 model it was retargeted to, from the Hub.

The G1 URDF and its meshes go to ``.cache/g1/``, because every source shares them. The
clips go to ``.cache/<source>/``. ``--include`` takes Hub glob patterns in place of the
default ones of the source, to fetch part of a large set.

Usage:
    uv run scripts/download.py
    uv run scripts/download.py --source amass --include 'g1/CMU/**' 'g1/KIT/**'
    uv run scripts/download.py --source motiondecode --include 'samples/1.3.Basic_Gait_Category/**'
    uv run scripts/download.py --source kimodo
    uv run scripts/download.py --force
"""

import sys
from pathlib import Path

import tyro
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_ROOT, G1_REPO_ID, G1_URDF_PATH  # noqa: E402
from sources import SOURCES  # noqa: E402

#: Files fetched at once. AMASS and MotionDecode are tens of thousands of small files, so
#: the time per request, not the bandwidth, sets how long a download takes.
DOWNLOAD_WORKERS: int = 32


def main(source: str = "lafan1", include: list[str] | None = None, force: bool = False) -> None:
    """Download ``source`` into ``.cache/<source>/``, and the G1 model into ``.cache/g1/``.

    Args:
        source: One of the keys of ``sources.SOURCES``.
        include: Hub glob patterns to download in place of the default ones of the source.
        force: Re-download even if files already exist.
    """
    if source not in SOURCES:
        raise SystemExit(f"Unknown source {source!r}. Expected one of {sorted(SOURCES)}.")
    spec = SOURCES[source]

    snapshot_download(
        repo_id=G1_REPO_ID,
        repo_type="dataset",
        local_dir=str(CACHE_ROOT / "g1"),
        allow_patterns=["robot_description/g1/**"],
        force_download=force,
        max_workers=DOWNLOAD_WORKERS,
    )
    local_path = Path(snapshot_download(
        repo_id=spec.repo_id,
        repo_type="dataset",
        local_dir=str(CACHE_ROOT / source),
        allow_patterns=include or list(spec.allow_patterns),
        force_download=force,
        max_workers=DOWNLOAD_WORKERS,
    ))
    if spec.unpack is not None:
        spec.unpack(local_path)

    clips = sorted((local_path / spec.clip_dir).glob(spec.clip_glob))
    if not clips:
        raise SystemExit(f"No clips under {local_path / spec.clip_dir} (include={include}).")
    motion, fps = spec.load(clips[0])
    frames = motion["base_pos"].shape[0]
    print(f"G1 model → {G1_URDF_PATH}")
    print(f"{source} → {local_path}")
    print(f"  clips  : {len(clips)}")
    print(f"  sample : {clips[0].relative_to(local_path)}  ({frames} frames @ {fps} Hz)")


if __name__ == "__main__":
    tyro.cli(main)
