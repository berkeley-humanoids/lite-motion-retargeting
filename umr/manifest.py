"""Write ``manifest.json`` beside the motions of a UMR run, to record what made them.

The manifest holds the hash of the robot model that UMR read, the commit of UMR, and both
configs. The hash is the ``robots.model_hash`` of this package, over the MJCF of the robot
and its ``meshes`` folder. ``uv run compare`` reads it, and stops if the run used
another model of Lite Pro than the installed one. UMR names its caches after the robot and
the betas, not the geometry, so the manifest is the only record of the model.

Usage, in the clone of UMR, after the run:
    python manifest.py <robot config> <batch config> <motion dir>
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def model_hash(mjcf_path: Path) -> str:
    """Return a short hash of a robot MJCF and of the ``meshes`` folder beside its folder."""
    digest = hashlib.sha256()
    meshes = mjcf_path.parent.parent / "meshes"
    for path in [mjcf_path, *sorted(path for path in meshes.rglob("*") if path.is_file())]:
        digest.update(path.relative_to(mjcf_path.parent.parent).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


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
