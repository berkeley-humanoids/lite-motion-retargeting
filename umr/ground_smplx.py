"""Shift each SMPL-X take up or down, so that its feet stand on the floor.

``lafan1_to_smplx.py`` puts the LAFAN1 bodies 2 to 6 cm below the floor. UMR keeps a source
where it is, because its ``source_ground_align`` is "none". This script adds one height to
each take, as the authors of UMR do (`UMR issue 5 <https://github.com/hanyang9/UMR/issues/5>`_):
every LAFAN1 take starts and ends in a T-pose, so the height puts the lowest foot vertex of
the first and the last frame on or above the floor. Two obstacle takes end on a box, so the
lower of the two frames sets the height. The ``dance1_subject2`` file of UMR has this
height, to 1 mm. The rest of that mm comes from the betas: UMR builds the body with 10
betas, and so does this script.

Usage, in the environment of UMR:
    python ground_smplx.py <SMPL-X model dir> <in dir> <out dir> <take>...
"""

import sys
from pathlib import Path

import numpy as np
import smplx
import torch

#: The betas of the body that UMR builds.
BETAS: int = 10

#: The SMPL-X joints of the ankles and the feet. A vertex that one of them moves most is a
#: foot vertex.
FOOT_JOINTS: tuple[int, ...] = (7, 8, 10, 11)


def body_vertices(model: smplx.SMPLX, poses: np.ndarray, trans: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Return the vertices ``(T, V, 3)`` of the body in ``poses`` ``(T, J*3)``, with the hands and the face at rest."""
    pose = torch.from_numpy(poses.astype(np.float32))
    n = len(pose)
    with torch.no_grad():
        body = model(global_orient=pose[:, :3], body_pose=pose[:, 3:66],
                     betas=torch.from_numpy(betas.astype(np.float32)).expand(n, -1),
                     transl=torch.from_numpy(trans.astype(np.float32)),
                     left_hand_pose=torch.zeros(n, 45), right_hand_pose=torch.zeros(n, 45),
                     jaw_pose=torch.zeros(n, 3), leye_pose=torch.zeros(n, 3), reye_pose=torch.zeros(n, 3),
                     expression=torch.zeros(n, 10))
    return body.vertices.numpy()


def lowest_foot_z(model_dir: Path, data: dict, frames: list[int]) -> np.ndarray:
    """Return the height of the lowest foot vertex in ``frames`` of a take."""
    model = smplx.create(str(model_dir), model_type="smplx", gender="neutral", use_pca=False,
                         num_betas=BETAS, batch_size=len(frames), flat_hand_mean=True)
    poses = data["poses"].reshape(len(data["poses"]), -1)[frames]
    vertices = body_vertices(model, poses, data["trans"][frames], data["betas"][:BETAS])
    feet = torch.isin(model.lbs_weights.argmax(dim=1), torch.tensor(FOOT_JOINTS)).numpy()
    return vertices[:, feet, 2].min(axis=1)


def main() -> None:
    model_dir, src, dst = (Path(arg) for arg in sys.argv[1:4])
    dst.mkdir(parents=True, exist_ok=True)
    for take in sys.argv[4:]:
        data = dict(np.load(src / f"{take}.npz", allow_pickle=True))
        first, last = lowest_foot_z(model_dir, data, [0, len(data["trans"]) - 1])
        shift = -float(min(first, last))
        data["trans"] = data["trans"].copy()
        data["trans"][:, 2] += shift
        np.savez(dst / f"{take}.npz", **data)
        print(f"{take} {shift * 100:+.2f} cm, first and last frame {(first - last) * 100:+.2f} cm apart", flush=True)


if __name__ == "__main__":
    main()
