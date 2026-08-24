"""Upload the motion store to the Hugging Face Hub.

The Hub repo mirrors ``dataset/`` exactly: a directory per layer, a file per motion, and
``DATASET_CARD.md`` as the card the repo opens with. ``scripts/retarget.py`` writes the
store, and this script only moves it.

The card is a template. This script fills its ``<!-- motions -->`` marker with an index
of the motions that go up in the same commit, so the index cannot name a file the repo
does not hold.

Usage:
    uv run scripts/publish.py --dry-run
    uv run scripts/publish.py
    uv run scripts/publish.py --clear   # and drop whatever else the repo holds

A write needs a token for the namespace:

    hf auth login          # or: export HF_TOKEN=hf_...
"""

import sys
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import tyro
from huggingface_hub import DatasetCard, HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASET_REPO_ID  # noqa: E402
from store import BASE_LAYER  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATASET_ROOT: Path = REPO_ROOT / "dataset"
CARD_PATH: Path = REPO_ROOT / "DATASET_CARD.md"

#: Where the card's motion index goes.
MOTIONS_MARKER: str = "<!-- motions -->"

#: The Rerun web viewer, which takes the URL of one recording.
VIEWER: str = "https://rerun.io/viewer?url="


def _motion_index(motions: list[Path], repo_id: str) -> str:
    """A table of ``motions``, each row a link into the Rerun web viewer.

    A row links the robot layer, which carries the geometry and embeds the viewer
    layout, so the one file opens as the robot and not as a bare entity tree.
    """
    rows = ["| Motion | View |", "|---|---|"]
    for path in motions:
        url = (f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
               f"{path.parent.name}/{path.name}")
        rows.append(f"| `{path.stem}` | [Open in Rerun]({VIEWER}{quote(url, safe='')}) |")
    return "\n".join(rows)


def main(
    repo_id: str = DATASET_REPO_ID,
    clear: bool = False,
    dry_run: bool = False,
) -> None:
    """Upload ``dataset/`` and the dataset card to ``repo_id``.

    Args:
        repo_id: The Hub dataset repo to write to.
        clear: Delete every file the repo holds that this store does not replace. Use it
            when a rename or a dropped clip would otherwise leave the old file behind.
        dry_run: Print what would go up, and upload nothing. The card is still
            rendered into ``dataset/README.md``, so you can read it first.
    """
    files = sorted(
        path for path in DATASET_ROOT.rglob("*") if path.suffix in (".rrd", ".rbl")
    )
    if not files:
        raise SystemExit(
            f"No motions under {DATASET_ROOT}. Run scripts/retarget.py first."
        )

    # The card rides in the same commit as the data, so the repo never describes itself
    # with a card its files do not match. `DatasetCard` parses the YAML front matter, so
    # a broken card fails here and not on the Hub.
    motions = [
        path for path in files
        if path.suffix == ".rrd" and path.parent.name != BASE_LAYER
    ]
    card = CARD_PATH.read_text().replace(
        MOTIONS_MARKER, _motion_index(motions, repo_id)
    )
    DatasetCard(card).save(DATASET_ROOT / "README.md")

    layers = Counter(path.parent.name for path in files)
    size_mb = sum(path.stat().st_size for path in files) / 1e6
    print(f"{len(files)} files, {size_mb:.1f} MB -> {repo_id}")
    for layer, count in sorted(layers.items()):
        print(f"  {layer}/  {count}")
    print(f"  card indexes {len(motions)} motions")
    if dry_run:
        return

    HfApi().upload_folder(
        folder_path=str(DATASET_ROOT),
        repo_id=repo_id,
        repo_type="dataset",
        # The store holds nothing else, and this keeps a stray file out of the repo.
        allow_patterns=["*/*.rrd", "*/*.rbl", "README.md"],
        # A repo needs its `.gitattributes`, and the Hub keeps that one whatever the
        # pattern matches.
        delete_patterns="*" if clear else None,
        commit_message=f"Publish {len(files)} layer files from Lite-Motion-Retargeting",
    )
    print(f"Pushed to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
