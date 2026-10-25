"""Convert LAFAN1 BVH takes to SMPL-X with ``lafan_to_smplx``, so that UMR can read them.

`lafan_to_smplx <https://github.com/jaraujo98/lafan_to_smplx>`_ imports its BVH readers from
GMR. The ``__init__`` of GMR imports the whole retargeter of GMR, with mink and its viewers,
so an empty package module stands in for it.

``lafan_to_smplx`` passes 16 betas, and places each frame with the joints of that body. UMR
builds the body again with 10 betas, which moves it by about 1.4 mm. Thus ``smplx.create``
keeps only the first 10 betas here, so that every step uses the same body.

``lafan_to_smplx`` moves its betas to the GPU, but not its body model, so it runs only on
the CPU. This script hides the GPU from it. A take that the output directory already holds
is skipped.

Usage, in the environment of UMR:
    python lafan1_to_smplx.py <lafan_to_smplx.py> <BVH dir> <out dir> <SMPL-X model dir> <take>...
"""

import os
import pathlib
import runpy
import sys
import sysconfig
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""

root = pathlib.Path(sysconfig.get_paths()["purelib"]) / "general_motion_retargeting"
package = types.ModuleType("general_motion_retargeting")
package.__path__ = [str(root)]
sys.modules["general_motion_retargeting"] = package

import smplx  # noqa: E402

#: The betas of the body that UMR builds.
BETAS: int = 10

_create = smplx.create


def _create_with_umr_betas(*args, **kwargs):
    if kwargs.get("betas") is not None:
        kwargs["betas"] = kwargs["betas"][..., :BETAS]
    kwargs["num_betas"] = BETAS
    return _create(*args, **kwargs)


smplx.create = _create_with_umr_betas

script = sys.argv[1]
bvh_dir, out_dir, model_dir = (pathlib.Path(arg) for arg in sys.argv[2:5])
out_dir.mkdir(parents=True, exist_ok=True)
for take in sys.argv[5:]:
    out = out_dir / f"{take}.npz"
    if out.exists():
        continue
    # A take that stops halfway leaves no file under the final name.
    part = out_dir / f"{take}.part.npz"
    sys.argv = [script, "--bvh_file", str(bvh_dir / f"{take}.bvh"),
                "--smplx_model_path", str(model_dir), "--output_file", str(part)]
    runpy.run_path(script, run_name="__main__")
    part.rename(out)
    print("converted", take, flush=True)
