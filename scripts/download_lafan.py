"""Download the LAFAN1 G1 subset from the Hugging Face Hub.

Pulls only the G1 motion CSVs, the G1 URDF and its meshes (which ``retarget.py``
compiles to run the kinematics of the source), and dataset metadata.

Usage:
    uv run scripts/download_lafan.py
    uv run scripts/download_lafan.py --clip 'walk.*subject1'
    uv run scripts/download_lafan.py --force
"""

import re
import sys
from pathlib import Path

import numpy as np
import tyro
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import G1_LAFAN_JOINT_NAMES, LAFAN_REPO_ID  # noqa: E402

CACHE_ROOT: Path = Path(__file__).resolve().parent.parent / ".cache" / "lafan1_g1"
ALLOW_PATTERNS: list[str] = [
    "g1/*.csv",
    "robot_description/g1/**",
    "meta_data/**",
    "README.md",
    "LICENSE",
]


def main(clip: str | None = None, force: bool = False) -> None:
    """Download the LAFAN1 G1 subset into ``.cache/lafan1_g1/``.

    Args:
        clip: Regex; if given, the smoke check only inspects matching CSVs.
            The download itself always pulls every G1 CSV — restricting at
            ``snapshot_download`` would defeat the cache for later runs.
        force: Re-download even if files already exist.
    """
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    local_path = Path(snapshot_download(
        repo_id=LAFAN_REPO_ID,
        repo_type="dataset",
        local_dir=str(CACHE_ROOT),
        allow_patterns=ALLOW_PATTERNS,
        force_download=force,
    ))

    csvs = sorted((local_path / "g1").glob("*.csv"))
    if clip is not None:
        csvs = [p for p in csvs if re.search(clip, p.stem)]
    if not csvs:
        raise SystemExit(f"No G1 CSVs found under {local_path / 'g1'} (clip={clip!r}).")

    raw = np.loadtxt(csvs[0], delimiter=",", dtype=np.float32)
    expected = 7 + len(G1_LAFAN_JOINT_NAMES)
    if raw.ndim != 2 or raw.shape[1] != expected:
        raise SystemExit(
            f"Schema check failed on {csvs[0]}: expected {expected} cols, got {raw.shape}"
        )
    print(f"LAFAN1 G1 subset → {local_path}")
    print(f"  csvs   : {len(csvs)}")
    print(f"  sample : {csvs[0].name}  ({raw.shape[0]} frames @ 30 FPS, {raw.shape[1]} cols)")


if __name__ == "__main__":
    tyro.cli(main)
