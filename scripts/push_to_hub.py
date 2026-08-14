"""Push the local LeRobotDataset to the Hugging Face Hub.

Authenticate first (``hf auth login`` or ``export HF_TOKEN=hf_...``).
Uses :meth:`LeRobotDataset.push_to_hub` so the repo gets a LeRobot
auto-generated dataset card and a version tag.

Usage:
    uv run scripts/push_to_hub.py
    uv run scripts/push_to_hub.py --repo-id your-org/your-dataset --private
    uv run scripts/push_to_hub.py --branch v0.1
"""

import sys
from pathlib import Path

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LITE_DATASET_REPO_ID  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
TAGS: list[str] = [
    "lerobot",
    "humanoid",
    "motion-capture",
    "motion-tracking",
    "berkeley-humanoids",
    "lite",
    "lafan1",
]


def main(
    repo_id: str = LITE_DATASET_REPO_ID,
    private: bool = False,
    branch: str | None = None,
) -> None:
    """Upload the local dataset to the HF Hub.

    Args:
        repo_id: Target dataset repo id ``{user_or_org}/{name}``. Must be a
            location your authenticated account can write to.
        private: If True, create the repo as private.
        branch: Optional branch to push to (created from main if missing).
    """
    from lerobot.datasets import LeRobotDataset

    if not (REPO_ROOT / "meta").exists() or not (REPO_ROOT / "data").exists():
        raise SystemExit(
            f"Expected meta/ and data/ under {REPO_ROOT}. "
            f"Run scripts/retarget.py first."
        )

    dataset = LeRobotDataset(repo_id=repo_id, root=REPO_ROOT)
    print(
        f"Uploading {dataset.meta.total_episodes} episodes "
        f"({dataset.meta.total_frames} frames @ {dataset.meta.fps} FPS) "
        f"→ https://huggingface.co/datasets/{repo_id}"
    )
    # The dataset root *is* the repo root, so the folder also contains the
    # LAFAN1 download cache, venv, build artefacts, etc. ``allow_patterns``
    # restricts the upload to dataset content + reproduction sources
    # (scripts, pyproject, instructions). ``scripts/*.py`` is intentionally
    # not ``scripts/**`` so we don't accidentally publish ``__pycache__/``.
    dataset.push_to_hub(
        tags=TAGS,
        license="cc-by-nc-4.0",
        private=private,
        branch=branch,
        push_videos=False,
        allow_patterns=[
            "meta/**",
            "data/**",
            "scripts/*.py",
            "pyproject.toml",
            "INSTRUCTIONS.md",
        ],
    )
    print(f"Done. View at https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
