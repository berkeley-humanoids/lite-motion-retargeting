"""Human motion in one skeleton, whatever format a source ships it in.

A reader turns one clip into world poses of :data:`HUMAN_BODIES`. Every array is in world
frame and metres, with z up and the floor of the source at ``z = 0``:

    pos   (T, B, 3)
    quat  (T, B, 4)  wxyz, rest-normalized
    rest  (B, 3)     each body in the T-pose: on the floor, facing +x, left along +y

A rest-normalized rotation is the world rotation of the body times the inverse of its
world rotation in the T-pose. Thus identity means "as in the T-pose" for every source,
whatever axes its joints use, and a robot mapping needs no offsets per source.

A body is the frame at a joint of the skeleton, and it turns with the segment that starts
at that joint: ``left_knee`` sits at the knee and turns with the shin.
"""

from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

#: The skeleton that every source maps onto, named after the keypoints of the Lite
#: conventions. A parent comes before its children.
HUMAN_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "left_knee",
    "left_foot",
    "left_toe",
    "right_hip",
    "right_knee",
    "right_foot",
    "right_toe",
    "waist",
    "chest",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_hand",
)

#: The parent of each body of :data:`HUMAN_BODIES` except the root, in the same order.
HUMAN_PARENTS: dict[str, str] = {
    "left_hip": "pelvis",
    "left_knee": "left_hip",
    "left_foot": "left_knee",
    "left_toe": "left_foot",
    "right_hip": "pelvis",
    "right_knee": "right_hip",
    "right_foot": "right_knee",
    "right_toe": "right_foot",
    "waist": "pelvis",
    "chest": "waist",
    "neck": "chest",
    "head": "neck",
    "left_shoulder": "chest",
    "left_elbow": "left_shoulder",
    "left_hand": "left_elbow",
    "right_shoulder": "chest",
    "right_elbow": "right_shoulder",
    "right_hand": "right_elbow",
}


class HumanClip(NamedTuple):
    """One clip at its own rate, in the skeleton of the source and in :data:`HUMAN_BODIES`.

    The ``source_*`` arrays hold every joint of the source skeleton as published, in world
    frame, metres, z up, and wxyz. The other arrays hold :data:`HUMAN_BODIES`, as the module
    docstring defines them.
    """

    fps: int
    source_bodies: tuple[str, ...]
    source_pos: np.ndarray
    source_quat: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    rest: np.ndarray


def wxyz(matrices: np.ndarray) -> np.ndarray:
    """Convert rotation matrices ``(..., 3, 3)`` to float32 wxyz quaternions with ``w >= 0``."""
    shape = matrices.shape[:-2]
    xyzw = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat()
    xyzw[xyzw[:, 3] < 0] *= -1.0
    return xyzw[:, [3, 0, 1, 2]].reshape(*shape, 4).astype(np.float32)
