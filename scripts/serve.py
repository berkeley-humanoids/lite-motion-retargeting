"""Serve the motion browser in ``site/`` on this machine.

The page reads each motion as a byte range of a pack. ``python -m http.server`` answers
a range request with the whole file, but this server answers it with the range.

Usage:
    uv run scripts/serve.py              # then open http://localhost:8000
    uv run scripts/serve.py --port 8080
"""

import io
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import tyro

SITE_ROOT: Path = Path(__file__).resolve().parent.parent / "site"


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """A static file handler that also answers ``Range: bytes=<start>-<end>``."""

    def send_head(self) -> io.BufferedIOBase | None:
        """Send the response headers, and return the body to copy after them."""
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        path = self.translate_path(self.path)
        if match is None or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start = int(match[1])
        stop = min(int(match[2]) + 1, size) if match[2] else size
        if start >= stop:
            self.send_error(416, "Range Not Satisfiable")
            return None
        # A range is one motion of a few hundred kB, so one read gets all of it.
        with open(path, "rb") as file:
            file.seek(start)
            body = file.read(stop - start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{stop - 1}/{size}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        return io.BytesIO(body)


def main(port: int = 8000) -> None:
    """Serve ``site/`` until you stop the server with Ctrl+C.

    Args:
        port: The port to listen on. The server listens on all interfaces.
    """
    server = ThreadingHTTPServer(("", port), partial(RangeRequestHandler, directory=str(SITE_ROOT)))
    print(f"Serving {SITE_ROOT} at http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    tyro.cli(main)
