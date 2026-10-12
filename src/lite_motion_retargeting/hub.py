"""The Hub datasets that ``uv run publish`` uploads, and the Space that plays them.

A Hub dataset is one Hub dataset repo, with its own card. It holds the motions of one or
more sources of the store: a directory per layer, a file per motion, and in ``web/`` the
same motions in the form that the motion browser plays. A source goes to one Hub dataset
at most.

The terms of a source decide whether a Hub dataset can hold it. AMASS bars any
redistribution, and ChingMu bars hosting MotionDecode as a competing dataset. Thus their
motions stay in the local store, for training.

The Space plays the ``web/`` of every Hub dataset on one page. A private Hub dataset shows
there only to a visitor who signs in with an account that can read it.

A motion of a Hub dataset carries the geometry of the robot, so that the motion opens on
its own, in the Rerun web viewer or after a download. ``uv run retarget`` writes the
geometry only for a source of a Hub dataset. Thus, after you add a source here, retarget
it again before you publish it.
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


#: Name -> Hub dataset. The Space lists the motions of the Hub datasets in this order.
HUB_DATASETS: dict[str, HubDataset] = {
    "lafan1": HubDataset(
        repo_id="Berkeley-Humanoids/Lite-LAFAN1-Dataset",
        sources=("lafan1",),
        card=CARD_ROOT / "lafan1.md",
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


def layer_url(source: str, robot: str) -> str | None:
    """Return the URL of the robot layer of ``source`` on the Hub, or None if it has none.

    The Rerun web viewer opens a motion at this URL. The viewer cannot read a private repo,
    so a source of a private Hub dataset has no URL.
    """
    dataset = hub_dataset_of(source)
    if dataset is None or dataset.private:
        return None
    return f"https://huggingface.co/datasets/{dataset.repo_id}/resolve/main/{robot}/"
