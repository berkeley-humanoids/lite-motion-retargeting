"""Upload the motion store to the Hugging Face Hub, and the motion browser that plays it.

The Hub repo mirrors the LAFAN1 motions of ``dataset/``: a directory per layer, a file per
motion, and ``DATASET_CARD.md`` as the card the repo opens with. ``scripts/retarget.py``
writes the store, and this script only moves it. Beside the layers, ``web/`` holds the
same motions as the browser plays them, which ``scripts/build_site.py`` writes from the
store.

The card is a template. This script fills its ``<!-- motions -->`` marker with an index
of the motions that go up in the same commit, so the index cannot name a file the repo
does not hold.

The browser itself, ``site/``, goes to a static Space, with a ``collections.json`` that
points at ``web/`` as of the commit just made. The Space keeps playing the motions it was
published with until the next publish.

Usage:
    uv run scripts/publish.py --dry-run
    uv run scripts/publish.py
    uv run scripts/publish.py --clear   # and drop whatever else the repo holds

A write needs a token for the namespace:

    hf auth login          # or: export HF_TOKEN=hf_...
"""

import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import tyro
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_site import main as build_site  # noqa: E402
from common import DATASET_REPO_ID, PUBLIC_SOURCE, SPACE_REPO_ID  # noqa: E402
from store import BASE_LAYER  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATASET_ROOT: Path = REPO_ROOT / "dataset"
CARD_PATH: Path = REPO_ROOT / "DATASET_CARD.md"
SPACE_CARD_PATH: Path = REPO_ROOT / "SPACE_CARD.md"
SITE_ROOT: Path = REPO_ROOT / "site"

#: Where the browser data goes, under ``dataset/`` and in the repo.
WEB_DIR: str = "web"

#: The files of the page. The Space gets its own ``collections.json``.
PAGE_FILES: tuple[str, ...] = ("index.html", "app.js", "style.css")

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
    space_id: str = SPACE_REPO_ID,
    clear: bool = False,
    dry_run: bool = False,
) -> None:
    """Upload ``dataset/``, its browser data and the dataset card to ``repo_id``, and the page to ``space_id``.

    Args:
        repo_id: The Hub dataset repo to write to.
        space_id: The static Space to write the page to. It is created if missing.
        clear: Delete every file the repo holds that this store does not replace. Use it
            when a rename or a dropped clip would otherwise leave the old file behind.
        dry_run: Print what would go up, and upload nothing. The card is still
            rendered into ``dataset/README.md``, and the browser data built into
            ``dataset/web/``, so you can look at both first.
    """
    files = sorted(
        path for path in DATASET_ROOT.rglob("*")
        if path.suffix == ".rbl"
        or (path.suffix == ".rrd" and path.stem.startswith(f"{PUBLIC_SOURCE}_"))
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

    # So does the browser data, built from the motions it goes up with.
    build_site(sources=[PUBLIC_SOURCE], out=DATASET_ROOT / WEB_DIR)
    web_files = [path for path in (DATASET_ROOT / WEB_DIR).rglob("*") if path.is_file()]

    layers = Counter(path.parent.name for path in files)
    size_mb = sum(path.stat().st_size for path in files + web_files) / 1e6
    print(f"{len(files) + len(web_files)} files, {size_mb:.1f} MB -> {repo_id}")
    for layer, count in sorted(layers.items()):
        print(f"  {layer}/  {count}")
    print(f"  {WEB_DIR}/  {len(web_files)}")
    print(f"  card indexes {len(motions)} motions")
    print(f"  page: {', '.join(PAGE_FILES)} -> {space_id}")
    if dry_run:
        return

    api = HfApi()
    api.upload_folder(
        folder_path=str(DATASET_ROOT),
        repo_id=repo_id,
        repo_type="dataset",
        # This keeps the other sources, and any stray file, out of the repo.
        allow_patterns=[f"*/{PUBLIC_SOURCE}_*.rrd", "*/*.rbl", f"{WEB_DIR}/*", "README.md"],
        # A repo needs its `.gitattributes`, and the Hub keeps that one whatever the
        # pattern matches.
        delete_patterns="*" if clear else None,
        commit_message=f"Publish {len(files)} layer files from Lite-Motion-Retargeting",
    )
    print(f"Pushed to https://huggingface.co/datasets/{repo_id}")

    # The page reads the data as of this commit, so a later publish cannot hand it packs
    # that its index does not describe.
    revision = api.dataset_info(repo_id).sha
    publish_page(api, space_id, f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{WEB_DIR}/")


def publish_page(api: HfApi, space_id: str, data_url: str) -> None:
    """Upload ``site/`` to the static Space ``space_id``, with ``data_url`` as its one collection."""
    api.create_repo(space_id, repo_type="space", space_sdk="static", exist_ok=True)
    collections = [{"url": data_url}]
    operations = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(SPACE_CARD_PATH)),
        CommitOperationAdd(path_in_repo="collections.json", path_or_fileobj=json.dumps(collections).encode()),
    ]
    for name in PAGE_FILES:
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(SITE_ROOT / name)))
    api.create_commit(
        repo_id=space_id,
        repo_type="space",
        operations=operations,
        commit_message="Publish the motion browser from Lite-Motion-Retargeting",
    )
    print(f"Pushed the page to https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    tyro.cli(main)
