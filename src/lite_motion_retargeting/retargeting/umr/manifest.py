"""Write ``manifest.json`` beside the motions of a UMR run, to record what made them.

The manifest holds the ``robots.model_hash`` of the robot model that UMR read, the commit of
UMR, and both configs. UMR names its caches after the robot and the betas, not the geometry,
so the manifest is the only record of the model. ``uv run compare`` stops if the run used
another model of Lite Pro than the installed one.

Usage, in the clone of UMR, after the run:
    uv run --project <this repo> python -m lite_motion_retargeting.retargeting.umr.manifest \\
        <robot config> <batch config> <motion dir>
"""

import json
import subprocess
import sys
from pathlib import Path

from ...robots import model_hash


def main() -> None:
    robot_config, batch_config, motion_dir = (Path(arg) for arg in sys.argv[1:4])
    robot = json.loads(robot_config.read_text())
    xml = (robot_config.parent / robot["robot"]["xml"]).resolve()
    manifest = {
        "robot_model": model_hash(xml),
        "umr_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "robot_config": robot,
        "batch_config": json.loads(batch_config.read_text()),
        "motions": len(list(motion_dir.glob("*.npz"))),
    }
    (motion_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{motion_dir / 'manifest.json'}: model {manifest['robot_model']}, {manifest['motions']} motions")


if __name__ == "__main__":
    main()
