"""Retarget motion datasets to the Berkeley Lite humanoid, and publish them.

The layout follows mjlab. ``scripts/`` holds the steps of the pipeline, a module per
command: ``uv run retarget`` runs ``scripts/retarget.py``. The other modules are the
library that the steps share:

    sources/       the motion sets, a loader for each, and the human skeleton
    retargeting/   the methods that retarget a clip, IK, ReActor and UMR, and the metrics
    robots.py      the Lite and G1 models
    kinematics.py  resampling, forward kinematics, soles and foot contacts
    store.py       the layers of a motion, as Rerun files
    browser.py     the data that the motion browser plays
    hub.py         the Hub datasets, and the Space that plays them

Conventions:
- Quaternions are scalar-first wxyz, unit-norm, with ``w >= 0``. A source file can store
  xyzw, and its loader in ``sources/`` converts it exactly once.
- MuJoCo sets the column order of the Lite ``joint_pos`` from the MJCF
  (:func:`robots.joint_names`). No code hardcodes it.
"""

from pathlib import Path

#: The clone of this repo. The package runs as an editable install of its clone, and keeps
#: its data in the clone, beside the code.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: The local copy of each source and of the G1 model, one directory each.
CACHE_ROOT: Path = REPO_ROOT / ".cache"

#: The store: a directory per layer, and a file per motion. See ``store.py``.
DATASET_ROOT: Path = REPO_ROOT / "dataset"

#: The motion browser, and in ``data/`` the motions that it plays.
SITE_ROOT: Path = REPO_ROOT / "site"
