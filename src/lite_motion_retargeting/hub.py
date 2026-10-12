"""The Hub datasets that ``uv run publish`` uploads, and the Space that plays them.

A Hub dataset is one Hub dataset repo, with its own card. It holds the motions of one or
more sources of the store: a directory per layer, a directory per source in each layer,
and a file per motion. ``web/`` holds the same motions in the form that the motion browser
plays. A source goes to one Hub dataset at most.

The terms of a source decide whether a Hub dataset can hold it. AMASS bars any
redistribution, and ChingMu bars hosting MotionDecode as a competing dataset. Thus their
motions stay in the local store, for training. LAFAN1 is CC BY-NC-ND, which bars sharing a
retarget with the public, so its Hub dataset is private.

The Space plays the ``web/`` of every Hub dataset on one page. A private Hub dataset shows
there only to a visitor who signs in with an account that can read it.

A motion of a Hub dataset with ``geometry`` carries the geometry of the robot, so that the
motion opens on its own, in the Rerun web viewer or after a download. The geometry takes
9 MB, more than a short motion. Thus the motions of a large Hub dataset share the
geometry in ``assets/<robot>.rrd``, and its card does not link each motion into the web
viewer. ``uv run retarget`` writes the geometry only for a source of a Hub dataset with
``geometry``. After you add a source here, retarget it again before you publish it.
"""

from pathlib import Path
from typing import NamedTuple

from . import REPO_ROOT

#: The Rerun web viewer. It opens the recording at the URL that follows this prefix.
VIEWER: str = "https://rerun.io/viewer?url="

#: The templates of the cards, one per Hub dataset, and one for the Space.
CARD_ROOT: Path = REPO_ROOT / "cards"


class HubDataset(NamedTuple):
    """One Hub dataset repo, and the sources of the store that it holds."""

    #: The Hub dataset repo.
    repo_id: str
    #: The keys of ``sources.SOURCES`` that the repo holds.
    sources: tuple[str, ...]
    #: The template of the card of the repo. See ``scripts/publish.py``.
    card: Path
    #: Make the repo private. Only the accounts that can read the repo see its motions, on
    #: the Hub and in the Space.
    private: bool = False
    #: Write the geometry of the robot into each motion. See the module docstring.
    geometry: bool = True


#: Name -> Hub dataset. The Space lists the motions of the Hub datasets in this order.
HUB_DATASETS: dict[str, HubDataset] = {
    "lafan1": HubDataset(
        repo_id="Berkeley-Humanoids/Lite-LAFAN1-Dataset",
        sources=("lafan1bvh",),
        card=CARD_ROOT / "lafan1.md",
        private=True,
    ),
    "kimodo": HubDataset(
        repo_id="Berkeley-Humanoids/Lite-Kimodo-Dataset",
        sources=("kimodo", "kimodolh2", "kimodolh3"),
        card=CARD_ROOT / "kimodo.md",
        geometry=False,
    ),
}

#: The static Space that hosts the motion browser, and the template of its card.
SPACE_REPO_ID: str = "Berkeley-Humanoids/Lite-Motions"
SPACE_CARD: Path = CARD_ROOT / "space.md"


def hub_dataset_of(source: str) -> HubDataset | None:
    """Return the Hub dataset that holds ``source``, or None if no Hub dataset holds it."""
    holders = [dataset for dataset in HUB_DATASETS.values() if source in dataset.sources]
    if len(holders) > 1:
        raise ValueError(f"{len(holders)} Hub datasets hold {source!r}. A source goes to one at most.")
    return holders[0] if holders else None


def repo_dir(layer: str, source: str) -> str:
    """Return the directory of the motions of ``source`` in ``layer``, in a Hub dataset repo.

    A layer has a directory per source, because the Hub takes at most 10,000 files in one
    directory, and a source has fewer motions than that.
    """
    return f"{layer}/{source}"


def layer_url(source: str, robot: str) -> str | None:
    """Return the URL of the robot layer of ``source`` on the Hub, or None if it has none.

    The Rerun web viewer opens a motion at this URL plus ``<motion>.rrd``. The viewer
    cannot read a private repo, and cannot draw a motion without its geometry, so such a
    source has no URL.
    """
    dataset = hub_dataset_of(source)
    if dataset is None or dataset.private or not dataset.geometry:
        return None
    return f"https://huggingface.co/datasets/{dataset.repo_id}/resolve/main/{repo_dir(robot, source)}/"
