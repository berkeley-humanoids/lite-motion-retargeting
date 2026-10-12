"""Write the data of the motion browser, ``site/data/``, from the store.

``browser.py`` documents the format.

Usage:
    uv run build-site
    uv run build-site --sources lafan1 kimodo kimodolh2 kimodolh3  # what others may see
    uv run serve                            # then open http://localhost:8000
"""

import tyro

from ..browser import build


def main() -> None:
    """Run :func:`browser.build` with the options on the command line."""
    tyro.cli(build)
