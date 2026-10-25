"""The robots: the Lite Pro that every motion is for, and the Unitree G1 of the G1 sources.

Lite comes from ``lite-description``, the asset that the training repo also loads. The G1
is the model that the authors of every G1 source retargeted to. ``uv run download`` puts
it in the cache.
"""

import hashlib
import re
from functools import cache
from pathlib import Path

import mujoco
from lite_description import get_mjcf_path, get_urdf_path

from . import CACHE_ROOT

#: The upstream authors retargeted every G1 source to this G1 model. The LAFAN1 repo
#: bundles it.
G1_REPO_ID: str = "lvhaidong/LAFAN1_Retargeting_Dataset"
G1_URDF_PATH: Path = CACHE_ROOT / "g1" / "robot_description" / "g1" / "g1_29dof_rev_1_0.urdf"

#: The body that both robots are rooted at. Neither model gives it a joint, so it is the
#: frame that forward kinematics runs in, not a body of the result. The base trajectory of
#: a motion is its world pose.
BASE_BODY: str = "pelvis"

#: The Lite bodies that stand on the floor.
FEET: tuple[str, ...] = ("left_foot", "right_foot")

#: The joints of the 29-DoF G1, in the order that every G1 source stores them after the
#: base position and rotation.
G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def load_lite_model(variant: str = "lite_pro") -> mujoco.MjModel:
    """Load the MJCF of a Lite variant, the same asset that the training repo loads."""
    return mujoco.MjModel.from_xml_path(str(get_mjcf_path(variant)))


def lite_urdf_path() -> Path:
    """Return the path of the Lite Pro URDF, whose geometry goes into the robot layer.

    The MJCF drives the retarget and this URDF only draws it, so the two must describe
    one robot. They come from one asset package, and their joint and link names match.
    """
    return get_urdf_path("lite_pro")


@cache
def model_hash(mjcf_path: Path) -> str:
    """Return a short hash of a robot MJCF and of the ``meshes`` folder beside its folder.

    The hash names one version of a robot model. ``mjcf_path`` is ``<robot>/mjcf/<name>.xml``,
    as ``lite-description`` lays it out, or the copy of it that UMR reads. UMR writes files
    of its own into the ``mjcf`` folder, so only the MJCF itself counts. ``umr/manifest.py``
    computes the same hash.
    """
    digest = hashlib.sha256()
    meshes = mjcf_path.parent.parent / "meshes"
    for path in [mjcf_path, *sorted(path for path in meshes.rglob("*") if path.is_file())]:
        digest.update(path.relative_to(mjcf_path.parent.parent).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def lite_model_hash() -> str:
    """Return the :func:`model_hash` of the Lite Pro that ``lite-description`` installs."""
    return model_hash(get_mjcf_path("lite_pro"))


def load_g1_model() -> mujoco.MjModel:
    """Load the Unitree G1 URDF that ``uv run download`` puts in the cache.

    The URDF declares ``<compiler meshdir="meshes"/>`` and also gives each mesh as
    ``meshes/foo.STL``, so MuJoCo would look for ``meshes/meshes/foo.STL``. The loader sets
    meshdir to ``"."`` and gives MuJoCo the mesh bytes through ``assets``, so MuJoCo does
    not read the mesh files a second time.
    """
    urdf = G1_URDF_PATH
    if not urdf.exists():
        raise FileNotFoundError(
            f"Cannot find the G1 URDF at {urdf}. Run `uv run download` first."
        )
    text = re.sub(r'meshdir="[^"]*"', 'meshdir="."', urdf.read_text())
    assets = {f"meshes/{p.name}": p.read_bytes() for p in (urdf.parent / "meshes").glob("*.STL")}
    return mujoco.MjModel.from_xml_string(text, assets=assets)


def joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
    """Return the ``qpos`` address of the 1-DoF joint ``joint_name``."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise KeyError(f"The model has no joint {joint_name!r}.")
    return int(model.jnt_qposadr[jid])


def body_id(model: mujoco.MjModel, body_name: str) -> int:
    """Return the id of the body ``body_name``."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise KeyError(f"The model has no body {body_name!r}.")
    return bid


def joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """Return the names of the joints of ``model``, except a free joint, in the order of MuJoCo.

    MuJoCo orders the joints depth first. The names give the column order of ``joint_pos``,
    and the motion file stores them, so a loader maps a column to a robot joint by name.
    """
    names: list[str] = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None:
            raise RuntimeError(f"Joint {jid} has no name.")
        names.append(name)
    return tuple(names)
