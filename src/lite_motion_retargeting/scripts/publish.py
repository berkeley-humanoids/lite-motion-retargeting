"""Upload Hub datasets from the store, and the motion browser that plays them.

``hub.py`` lists the Hub datasets. Each Hub dataset repo mirrors the motions of its
sources in ``dataset/``, with a directory per source in each layer, and holds its card.
Beside the layers, ``web/`` holds the same motions in the form that the browser plays.
``uv run retarget`` writes the store, and ``browser.py`` writes ``web/`` from it. This
step only uploads them.

The step first writes the repo as it will be into ``dataset/hub/<name>/``: hard links to
the files of the store, the card, and ``web/``. Then it uploads that directory. A large
upload goes up in several commits, and running the step again resumes it.

The card is a template in ``cards/``. If the card has a ``<!-- motions -->`` marker, this
step replaces the marker with a table that links each motion into the Rerun web viewer.

The browser page, ``site/``, goes to a static Space. Its ``collections.json`` lists the
``web/`` of each Hub dataset, as of the last commit to that repo. Thus the Space plays the
same motions until the next publish.

Usage:
    uv run publish --dry-run
    uv run publish                    # every Hub dataset, then the page
    uv run publish --datasets lafan1  # one Hub dataset, then the page
    uv run publish --datasets         # only the page
    uv run publish --clear            # also delete the other files in each repo

To upload, log in with a token that can write to the namespace:

    hf auth login          # or: export HF_TOKEN=hf_...
"""

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import tyro
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, SpaceCard
from huggingface_hub.errors import RepositoryNotFoundError
from rerun.chunk import RrdReader

from .. import DATASET_ROOT, SITE_ROOT
from ..browser import build as build_site
from ..hub import HUB_DATASETS, SPACE_CARD, SPACE_REPO_ID, VIEWER, repo_dir
from ..sources import source_of
from ..store import BASE_LAYER

#: The directory of the browser data in a Hub dataset repo.
WEB_DIR: str = "web"

#: Where this step writes each Hub dataset repo before the upload, a directory per Hub
#: dataset. It is in the store, so that it can hold hard links to the files of the store.
STAGE_ROOT: Path = DATASET_ROOT / "hub"

#: The Hub takes at most this many files in one directory of a repo.
FOLDER_FILES: int = 10_000

#: The files of the page. The Space gets its own ``collections.json``.
PAGE_FILES: tuple[str, ...] = ("index.html", "app.js", "style.css")

#: The marker in a card that the motion index replaces.
MOTIONS_MARKER: str = "<!-- motions -->"


def publish(
    datasets: tuple[str, ...] = tuple(HUB_DATASETS),
    clear: bool = False,
    dry_run: bool = False,
) -> None:
    """Upload ``datasets`` to their Hub repos, then the motion browser to its Space.

    Args:
        datasets: The names in ``hub.HUB_DATASETS`` to upload. The default is every Hub
            dataset. With ``--datasets`` and no name, only the page goes up.
        clear: Delete each file in a Hub dataset repo that this upload does not replace. Use
            this flag after you rename or remove a clip, to delete the old file.
        dry_run: Print a summary of each upload, and upload nothing. The step still writes
            the card and ``web/`` of each Hub dataset to ``dataset/hub/<name>/``, so you can
            examine them first.
    """
    unknown = sorted(set(datasets) - set(HUB_DATASETS))
    if unknown:
        raise SystemExit(f"Unknown Hub datasets {unknown}. Choose from {sorted(HUB_DATASETS)}.")
    api = HfApi()
    for name in datasets:
        publish_dataset(api, name, clear, dry_run)
    publish_page(api, dry_run)


def publish_dataset(api: HfApi, name: str, clear: bool, dry_run: bool) -> None:
    """Upload the motions of the Hub dataset ``name``, their browser data, and the card."""
    dataset = HUB_DATASETS[name]
    layers = sorted(path for path in DATASET_ROOT.glob("*/*.rrd") if source_of(path) in dataset.sources)
    motions = [path for path in layers if path.parent.name != BASE_LAYER]
    if not motions:
        raise SystemExit(f"No motions of {', '.join(dataset.sources)} in {DATASET_ROOT}. Run `uv run retarget` first.")
    if dataset.geometry and not _has_geometry(motions[0]):
        raise SystemExit(
            f"{motions[0]} has no robot geometry, so the Rerun web viewer cannot draw it. "
            f"Retarget the sources of {name!r} again: `uv run retarget` writes the geometry for a "
            "source of a Hub dataset with geometry."
        )

    files = {f"{repo_dir(path.parent.name, source_of(path))}/{path.name}": path for path in layers}
    files |= {f"blueprints/{path.name}": path for path in DATASET_ROOT.glob("blueprints/*.rbl")}
    if not dataset.geometry:
        files |= {f"assets/{path.name}": path for path in DATASET_ROOT.glob("assets/*.rrd")}
    folders = Counter(repo_path.rsplit("/", 1)[0] for repo_path in files if "/" in repo_path)
    full = [folder for folder, count in folders.items() if count > FOLDER_FILES]
    if full:
        raise SystemExit(f"The Hub takes at most {FOLDER_FILES} files in a directory, and {full} hold more.")

    stage = STAGE_ROOT / name
    shutil.rmtree(stage, ignore_errors=True)
    for repo_path, path in files.items():
        (stage / repo_path).parent.mkdir(parents=True, exist_ok=True)
        os.link(path, stage / repo_path)
    # `DatasetCard` parses the YAML front matter, so a broken card fails here, not on the Hub.
    card = dataset.card.read_text().replace(MOTIONS_MARKER, _motion_index(motions, dataset.repo_id))
    DatasetCard(card).save(stage / "README.md")
    build_site(sources=list(dataset.sources), out=stage / WEB_DIR)

    staged = [path for path in stage.rglob("*") if path.is_file()]
    size_gb = sum(path.stat().st_size for path in staged) / 1e9
    print(f"{name}: {len(staged)} files, {size_gb:.2f} GB -> {dataset.repo_id}"
          f"{' (private)' if dataset.private else ''}")
    for folder, count in sorted(folders.items()):
        print(f"  {folder}/  {count}")
    print(f"  {WEB_DIR}/  {sum(1 for path in staged if path.relative_to(stage).parts[0] == WEB_DIR)}")
    if dry_run:
        return

    api.create_repo(dataset.repo_id, repo_type="dataset", private=dataset.private, exist_ok=True)
    # `create_repo` keeps the visibility of a repo that exists, so set it before the upload.
    api.update_repo_settings(dataset.repo_id, repo_type="dataset", private=dataset.private)
    api.upload_folder(
        repo_id=dataset.repo_id,
        repo_type="dataset",
        folder_path=stage,
        # The Hub keeps `.gitattributes`, which a repo needs, though the pattern matches it.
        delete_patterns="*" if clear else None,
        commit_message=f"Publish {len(layers)} layer files from Lite-Motion-Retargeting",
    )
    print(f"Pushed to https://huggingface.co/datasets/{dataset.repo_id}")


def publish_page(api: HfApi, dry_run: bool) -> None:
    """Upload ``site/`` to the Space, with the ``web/`` of each Hub dataset as a collection.

    A collection reads its Hub dataset at the last commit to the repo, so a later publish
    cannot give the page packs that its index does not describe. A Hub dataset that is not
    on the Hub yet has no collection.
    """
    collections = []
    for name, dataset in HUB_DATASETS.items():
        try:
            revision = api.dataset_info(dataset.repo_id).sha
        except RepositoryNotFoundError:
            print(f"  {name}: {dataset.repo_id} is not on the Hub, so the page leaves it out")
            continue
        url = f"https://huggingface.co/datasets/{dataset.repo_id}/resolve/{revision}/{WEB_DIR}/"
        collections.append({"url": url, "private": True} if dataset.private else {"url": url})

    # The card links the public Hub datasets, so that the Hub lists the Space on their pages.
    card = SpaceCard(SPACE_CARD.read_text())
    card.data.datasets = [dataset.repo_id for dataset in HUB_DATASETS.values() if not dataset.private]
    files = {
        "README.md": str(card).encode(),
        "collections.json": json.dumps(collections, indent=2).encode(),
        **{name: str(SITE_ROOT / name) for name in PAGE_FILES},
    }
    print(f"page: {', '.join(files)} -> {SPACE_REPO_ID}")
    print(f"  {len(collections)} collections: " + ", ".join(c["url"] for c in collections))
    if dry_run:
        return

    api.create_repo(SPACE_REPO_ID, repo_type="space", space_sdk="static", exist_ok=True)
    api.create_commit(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        operations=[
            CommitOperationAdd(path_in_repo=path, path_or_fileobj=data) for path, data in files.items()
        ],
        commit_message="Publish the motion browser from Lite-Motion-Retargeting",
    )
    print(f"Pushed the page to https://huggingface.co/spaces/{SPACE_REPO_ID}")


def _motion_index(motions: list[Path], repo_id: str) -> str:
    """Return a Markdown table of ``motions``, each row a link into the Rerun web viewer.

    A row links the robot layer, which holds the geometry and the viewer layout. Thus the
    one file opens as the robot, not as a bare entity tree.
    """
    rows = ["| Motion | View |", "|---|---|"]
    for path in motions:
        url = (f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
               f"{repo_dir(path.parent.name, source_of(path))}/{path.name}")
        rows.append(f"| `{path.stem}` | [Open in Rerun]({VIEWER}{quote(url, safe='')}) |")
    return "\n".join(rows)


def _has_geometry(path: Path) -> bool:
    """Return whether the robot layer in ``path`` holds the geometry of its robot."""
    return any("/visual_geometries/" in chunk.entity_path for chunk in RrdReader(path).stream())


def main() -> None:
    """Run :func:`publish` with the options on the command line."""
    tyro.cli(publish)
