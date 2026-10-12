"""Upload the motion store to the Hugging Face Hub, and the motion browser that plays it.

The Hub repo mirrors the LAFAN1 motions of ``dataset/``: a directory per layer, a file per
motion, and ``DATASET_CARD.md`` as the card of the repo. ``uv run retarget`` writes the
store, and this step only uploads it. Beside the layers, ``web/`` holds the same motions
in the form that the browser plays. ``build_site.py`` writes ``web/`` from the store.

The card is a template. This step replaces its ``<!-- motions -->`` marker with an index
of the motions in the same commit, so the index cannot name a file that the repo does not
hold.

The browser page, ``site/``, goes to a static Space. Its ``collections.json`` points at
``web/`` as of the commit that this step just made. Thus the Space plays the same
motions until the next publish.

Usage:
    uv run publish --dry-run
    uv run publish
    uv run publish --clear   # also delete the other files in the repo

To upload, log in with a token that can write to the namespace:

    hf auth login          # or: export HF_TOKEN=hf_...
"""

import json
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import tyro
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi

from .build_site import VIEWER
from .build_site import main as build_site
from .common import (
    DATASET_REPO_ID,
    DATASET_ROOT,
    PUBLIC_SOURCE,
    REPO_ROOT,
    SITE_ROOT,
    SPACE_REPO_ID,
)
from .store import BASE_LAYER

CARD_PATH: Path = REPO_ROOT / "DATASET_CARD.md"
SPACE_CARD_PATH: Path = REPO_ROOT / "SPACE_CARD.md"

#: The directory of the browser data, in ``dataset/`` and in the repo.
WEB_DIR: str = "web"

#: The files of the page. The Space gets its own ``collections.json``.
PAGE_FILES: tuple[str, ...] = ("index.html", "app.js", "style.css")

#: The marker in the card that the motion index replaces.
MOTIONS_MARKER: str = "<!-- motions -->"


def _motion_index(motions: list[Path], repo_id: str) -> str:
    """Return a Markdown table of ``motions``, each row a link into the Rerun web viewer.

    A row links the robot layer, which holds the geometry and the viewer layout. Thus the
    one file opens as the robot, not as a bare entity tree.
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
    """Upload the store, its browser data and the card to ``repo_id``, and the page to ``space_id``.

    Args:
        repo_id: The Hub dataset repo to write to.
        space_id: The static Space to write the page to. The step creates it if it does
            not exist.
        clear: Delete each file in the repo that this upload does not replace. Use this flag
            after you rename or remove a clip, to delete the old file.
        dry_run: Print a summary of the upload, and upload nothing. The step still
            writes the card to ``dataset/README.md`` and the browser data to
            ``dataset/web/``, so you can examine both first.
    """
    files = sorted(
        path for path in DATASET_ROOT.rglob("*")
        if path.suffix == ".rbl"
        or (path.suffix == ".rrd" and path.stem.startswith(f"{PUBLIC_SOURCE}_"))
    )
    if not files:
        raise SystemExit(f"No motions under {DATASET_ROOT}. Run `uv run retarget` first.")

    # The card goes up in the same commit as the data, so the card always matches the files
    # of the repo. `DatasetCard` parses the YAML front matter, so a broken card fails here,
    # not on the Hub.
    motions = [
        path for path in files
        if path.suffix == ".rrd" and path.parent.name != BASE_LAYER
    ]
    card = CARD_PATH.read_text().replace(MOTIONS_MARKER, _motion_index(motions, repo_id))
    DatasetCard(card).save(DATASET_ROOT / "README.md")

    # The browser data goes up in the same commit too, built from the same motions.
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
        # The Hub keeps `.gitattributes`, which a repo needs, though the pattern matches it.
        delete_patterns="*" if clear else None,
        commit_message=f"Publish {len(files)} layer files from Lite-Motion-Retargeting",
    )
    print(f"Pushed to https://huggingface.co/datasets/{repo_id}")

    # The page reads the data at this commit, so a later publish cannot give it packs that
    # its index does not describe.
    revision = api.dataset_info(repo_id).sha
    publish_page(api, space_id, f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{WEB_DIR}/")


def publish_page(api: HfApi, space_id: str, data_url: str) -> None:
    """Upload ``site/`` to the static Space ``space_id``, with ``data_url`` as its one collection."""
    api.create_repo(space_id, repo_type="space", space_sdk="static", exist_ok=True)
    files = {
        "README.md": str(SPACE_CARD_PATH),
        "collections.json": json.dumps([{"url": data_url}]).encode(),
        **{name: str(SITE_ROOT / name) for name in PAGE_FILES},
    }
    api.create_commit(
        repo_id=space_id,
        repo_type="space",
        operations=[
            CommitOperationAdd(path_in_repo=path, path_or_fileobj=data) for path, data in files.items()
        ],
        commit_message="Publish the motion browser from Lite-Motion-Retargeting",
    )
    print(f"Pushed the page to https://huggingface.co/spaces/{space_id}")


def cli() -> None:
    """Run :func:`main` with the options on the command line."""
    tyro.cli(main)
