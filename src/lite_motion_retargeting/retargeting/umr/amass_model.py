"""Make the SMPL-X body model of AMASS loadable by the ``smplx`` package, and thus by UMR.

AMASS fits its SMPL-X clips with its own neutral model, ``neutral/model.npz``. Its body
differs from that of SMPL-X v1.1 by up to 1.2 cm, but its feet do not. The ``smplx``
package also needs the hand PCA and the face landmarks, which this file lacks. This script
copies them from SMPL-X v1.1. With ``use_pca=False`` and ``flat_hand_mean=True``, as UMR
and ``prepare_amass.py`` build the body, they do not move it.

Usage, in the environment of UMR:
    python amass_model.py <AMASS neutral/model.npz> <SMPL-X v1.1 SMPLX_NEUTRAL.npz> <out dir>

The script writes ``<out dir>/SMPLX_NEUTRAL.npz``, where UMR looks for a neutral model.
"""

import sys
from pathlib import Path

import numpy as np


def main() -> None:
    amass, v11, out = (Path(arg) for arg in sys.argv[1:4])
    model = dict(np.load(amass, allow_pickle=True))
    with np.load(v11, allow_pickle=True) as extra:
        added = [key for key in extra.files if key not in model]
        model |= {key: extra[key] for key in added}
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "SMPLX_NEUTRAL.npz", **model)
    print(f"{out / 'SMPLX_NEUTRAL.npz'}: added {', '.join(added)}")


if __name__ == "__main__":
    main()
