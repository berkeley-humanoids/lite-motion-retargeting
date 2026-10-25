"""Reader of AMASS clips in their SMPL-X form."""

from functools import cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .. import CACHE_ROOT
from .human import HUMAN_BODIES, HumanClip, wxyz

#: The SMPL-X body model that AMASS fits its SMPL-X clips with: the neutral model with the
#: locked head, ``neutral/model.npz`` of AMASS. Its license forbids redistribution, so
#: ``uv run download`` cannot fetch it. Copy it here yourself.
SMPLX_MODEL: Path = CACHE_ROOT / "smplx" / "SMPLX_NEUTRAL.npz"

#: The shape coefficients that build the body. AMASS fits 16. UMR builds its bodies with the
#: first 10, and so does this reader, so that both retarget one body.
SMPLX_BETAS: int = 10

#: The 22 body joints of SMPL-X, in the order of the model. The face and the fingers follow
#: them in the model. No body of :data:`HUMAN_BODIES` maps to those.
SMPLX_JOINTS: tuple[str, ...] = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
    "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
)

#: Body of :data:`HUMAN_BODIES` -> the SMPL-X joint at the same place. No body maps to
#: ``spine2`` or to the collars. The SMPL-X ``left_foot`` is the ball of the foot.
SMPLX_BODIES: dict[str, str] = {
    "pelvis": "pelvis",
    "left_hip": "left_hip",
    "left_knee": "left_knee",
    "left_foot": "left_ankle",
    "left_toe": "left_foot",
    "right_hip": "right_hip",
    "right_knee": "right_knee",
    "right_foot": "right_ankle",
    "right_toe": "right_foot",
    "waist": "spine1",
    "chest": "spine3",
    "neck": "neck",
    "head": "head",
    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_hand": "left_wrist",
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_hand": "right_wrist",
}

#: Rotation from the axes of the SMPL-X model to world axes. The model stands in the T-pose
#: along its +y, faces its +z, and holds its left arm along its +x. Every joint of the model
#: has these axes in the T-pose, so this is the rest rotation of every joint, facing +x.
_SMPLX_TO_WORLD: np.ndarray = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@cache
def _smplx_model() -> dict[str, np.ndarray]:
    """Return the arrays of :data:`SMPLX_MODEL` that place the joints of a body."""
    if not SMPLX_MODEL.exists():
        raise FileNotFoundError(f"{SMPLX_MODEL} is missing. Copy the SMPL-X model of AMASS there.")
    with np.load(SMPLX_MODEL) as model:
        return {key: model[key] for key in ("v_template", "shapedirs", "J_regressor", "kintree_table")}


def load_amass_smplx(path: Path) -> HumanClip:
    """Read an AMASS SMPL-X clip: 22 body joints at the rate of the clip, z up, metres."""
    model = _smplx_model()
    joints = len(SMPLX_JOINTS)
    with np.load(path) as data:
        betas = data["betas"][:SMPLX_BETAS]
        poses = data["poses"][:, : 3 * joints].reshape(-1, joints, 3)
        trans = data["trans"]
        fps = round(float(data["mocap_frame_rate"]))

    # The body of these betas in the T-pose, in model axes.
    vertices = model["v_template"] + model["shapedirs"][:, :, :SMPLX_BETAS] @ betas
    rest_joints = model["J_regressor"][:joints] @ vertices
    parents = model["kintree_table"][0, :joints]

    # Forward kinematics. The pelvis turns about its own joint, and ``trans`` moves it.
    local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(*poses.shape, 3)
    world_rot = np.empty_like(local)
    world_pos = np.empty(poses.shape)
    world_rot[:, 0], world_pos[:, 0] = local[:, 0], rest_joints[0] + trans
    for joint in range(1, joints):
        parent = parents[joint]
        world_rot[:, joint] = world_rot[:, parent] @ local[:, joint]
        world_pos[:, joint] = world_pos[:, parent] + world_rot[:, parent] @ (rest_joints[joint] - rest_joints[parent])

    # The T-pose faces +x, with the pelvis above the origin and the lowest vertex on the floor.
    rest_pos = (rest_joints - rest_joints[0]) @ _SMPLX_TO_WORLD.T
    rest_pos[:, 2] -= ((vertices - rest_joints[0]) @ _SMPLX_TO_WORLD.T)[:, 2].min()

    bodies = [SMPLX_JOINTS.index(SMPLX_BODIES[body]) for body in HUMAN_BODIES]
    return HumanClip(
        fps=fps,
        source_bodies=SMPLX_JOINTS,
        source_pos=world_pos.astype(np.float32),
        source_quat=wxyz(world_rot),
        pos=world_pos[:, bodies].astype(np.float32),
        quat=wxyz(world_rot[:, bodies] @ _SMPLX_TO_WORLD.T),
        rest=rest_pos[bodies].astype(np.float32),
    )
