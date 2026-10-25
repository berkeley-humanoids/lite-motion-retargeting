"""UMR on Lite Pro: its config, the scripts that prepare its input, and a reader of its motions.

`UMR <https://hanyang9.github.io/UMR/>`_ is a kinematic retargeter. It matches points on the
surface of an SMPL-X body to points on the surface of the robot, so it reads only SMPL-X
motion. It pins its own torch, MuJoCo and NumPy, so it runs in an environment of its own.
The scripts beside this module run in that environment, not in this package.

UMR writes ``<clip>_<robot>.npz`` into a folder named after its robot, at the rate of its
input, and can place the motion in a world of its own. For LAFAN1, ``lafan_to_smplx`` turns
the BVH axes into (x, -z, y), and ``sources/bvh.py`` into (z, x, y), so the world of UMR is
that of the IK turned by -90 degrees about z. AMASS SMPL-X reaches UMR and the IK in one
world.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ...kinematics import quat_mul, resample_motion
from ...store import FPS

#: Source -> the turn about z, in rad, from the world of a UMR motion to the world of the IK.
UMR_TURNS: dict[str, float] = {"lafan1bvh": np.pi / 2, "amassx": 0.0}


def umr_path(folder: Path, key: str) -> Path:
    """Return the UMR motion of the clip ``key`` in ``folder``, the output of UMR for one robot.

    UMR names a motion after its input file and its robot, and the SMPL-X files keep the
    names of the clips.
    """
    return folder / f"{key}_{folder.name}.npz"


def read_umr(
    path: Path, joint_names: tuple[str, ...], ref_pos: np.ndarray, ref_quat: np.ndarray, turn: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Read one motion of UMR at :data:`store.FPS`, and move it into the world of the IK.

    ``ref_pos`` and ``ref_quat`` are the base of the IK. The motion turns by ``turn`` rad
    about z, and one shift per motion puts the mean of the two roots together. A fit of the
    turn would hide an error of heading, and a fit per frame would hide a drift. Returns the
    base, the joints of ``joint_names``, and the median heading of the UMR base from the IK
    base after the turn, in degrees.
    """
    data = np.load(path, allow_pickle=True)
    qpos = data["qpos"].astype(np.float64)
    names = [str(name) for name in data["robot_joint_names"]]
    columns = [7 + names.index(name) for name in joint_names]
    base_pos, base_quat, joint_pos = resample_motion(
        qpos[:, :3], qpos[:, 3:7], qpos[:, columns], round(float(data["fps"][0])), FPS
    )
    pos = Rotation.from_euler("z", turn).apply(base_pos)
    quat = quat_mul(np.array([np.cos(turn / 2), 0.0, 0.0, np.sin(turn / 2)]), base_quat)
    n = min(len(pos), len(ref_pos))
    pos[:, :2] += (ref_pos[:n, :2] - pos[:n, :2]).mean(axis=0)
    heading = np.degrees(np.angle(np.median(np.exp(1j * (_yaw(quat[:n]) - _yaw(ref_quat[:n]))))))
    return pos, quat, joint_pos, float(heading)


def umr_model(folder: Path) -> str | None:
    """Return the hash of the robot model of the UMR run in ``folder``, or None without a manifest.

    ``manifest.py`` writes the manifest after the run.
    """
    manifest = folder / "manifest.json"
    return json.loads(manifest.read_text())["robot_model"] if manifest.exists() else None


def _yaw(quat: np.ndarray) -> np.ndarray:
    """Return the heading of each wxyz rotation, in rad."""
    return Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_euler("ZYX")[:, 0]
