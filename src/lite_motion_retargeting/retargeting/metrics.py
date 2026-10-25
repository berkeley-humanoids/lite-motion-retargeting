"""Measure a retargeted motion: how well it tracks its goals, how it treats the floor, and how
close it stays to its source.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from ..kinematics import lowest_points, sole, world_poses
from ..robots import FEET
from ..store import FPS

if TYPE_CHECKING:
    from . import Models

#: m. A sole point this close to the floor is near it, for ``near_floor_slide_cm_s``.
CONTACT_GAP: float = 0.01

#: Metric of :func:`motion_metrics` -> the samples of :func:`motion_samples` that it sums up,
#: and the :func:`statistic` that sums them up.
SUMMARY: dict[str, tuple[str, str]] = {
    "foot_error_cm": ("foot_error_cm", "mean"),
    "foot_angle_deg": ("foot_angle_deg", "mean"),
    "hand_error_cm": ("hand_error_cm", "mean"),
    "hand_angle_deg": ("hand_angle_deg", "mean"),
    "foot_slide_cm_s": ("foot_slide_cm_s", "mean"),
    "foot_slide_median_cm_s": ("foot_slide_cm_s", "median"),
    "near_floor_slide_cm_s": ("near_floor_slide_cm_s", "mean"),
    "near_floor_slide_median_cm_s": ("near_floor_slide_cm_s", "median"),
    "planted_gap_cm": ("planted_gap_cm", "mean"),
    "penetration_cm": ("penetration_cm", "p95"),
    "joint_speed_max": ("joint_speed", "max"),
}

#: The limbs that ``limb_angle_deg`` compares, as pairs of links.
LIMBS: tuple[tuple[str, str], ...] = (
    ("left_hip_yaw", "left_knee_pitch"),
    ("left_knee_pitch", "left_foot"),
    ("right_hip_yaw", "right_knee_pitch"),
    ("right_knee_pitch", "right_foot"),
    ("left_shoulder_yaw", "left_elbow_pitch"),
    ("left_elbow_pitch", "left_hand"),
    ("right_shoulder_yaw", "right_elbow_pitch"),
    ("right_elbow_pitch", "right_hand"),
)

#: s. ``root_error_cm`` compares the move of the root over this time.
ROOT_WINDOW: float = 1.0


def motion_samples(
    models: "Models",
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    joint_pos: np.ndarray,
    goal_links: tuple[str, ...],
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
    feet_planted: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the per-frame quantities that :func:`motion_metrics` sums up.

    The arguments are those of :func:`motion_metrics`. ``uv run compare`` pools these
    samples over many motions, so that a long motion counts more than a short one, and a
    motion with no contact adds no slide. Each array has one row per frame, or per pair of
    frames in a row, or per sample where a mask holds:

    * ``foot_error_cm``, ``foot_angle_deg``, ``hand_error_cm``, ``hand_angle_deg``: ``(T, 2)``.
    * ``foot_slide_cm_s``: the slide speeds while the source stands on a foot.
    * ``near_floor_slide_cm_s``: the slide speeds while a sole is near the floor.
    * ``planted_gap_cm``: the heights of the soles while the source stands on them.
    * ``penetration_cm``: ``(T,)``, the depth of the deepest sole below the floor.
    * ``joint_speed``: ``(T - 1,)``, the speed of the fastest joint, in rad/s.
    * ``near_floor``: ``(T, 2)`` bool, where a sole is within :data:`CONTACT_GAP` of the floor.
    * ``link_pos``: ``(T, P, 3)``, where each of ``goal_links`` is, the root first.
    """
    reached_pos, reached_quat = world_poses(
        models.lite, models.joint_addrs, joint_pos, base_pos, base_quat, goal_links[1:]
    )
    error = np.linalg.norm(reached_pos - goal_pos, axis=-1) * 100.0
    dot = np.abs(np.sum(reached_quat * goal_quat, axis=-1))
    angle = np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))
    feet = [goal_links.index(link) for link in FEET]
    hands = [goal_links.index(link) for link in ("left_hand", "right_hand")]

    both = feet_planted[1:] & feet_planted[:-1]
    foot_pos = reached_pos[:, feet]
    foot_rot = Rotation.from_quat(reached_quat[:, feet].reshape(-1, 4)[:, [1, 2, 3, 0]])
    foot_rot = foot_rot.as_matrix().reshape(*foot_pos.shape[:2], 3, 3)
    soles = lowest_points(sole(models.lite, FEET), foot_pos, foot_rot)
    # Where the touching point of frame t is one frame later.
    local = np.einsum("tfji,tfj->tfi", foot_rot[:-1], soles[:-1] - foot_pos[:-1])
    moved = foot_pos[1:] + np.einsum("tfij,tfj->tfi", foot_rot[1:], local)
    speed = np.linalg.norm(moved[..., :2] - soles[:-1, :, :2], axis=-1) * FPS * 100.0
    near = soles[..., 2] < CONTACT_GAP
    return {
        "foot_error_cm": error[:, feet],
        "foot_angle_deg": angle[:, feet],
        "hand_error_cm": error[:, hands],
        "hand_angle_deg": angle[:, hands],
        "foot_slide_cm_s": speed[both],
        "near_floor_slide_cm_s": speed[near[1:] & near[:-1]],
        "planted_gap_cm": np.abs(soles[..., 2][feet_planted]) * 100.0,
        "penetration_cm": np.maximum(-soles[:, :, 2].min(axis=1), 0.0) * 100.0,
        "joint_speed": np.abs(np.diff(joint_pos, axis=0)).max(axis=1) * FPS,
        "near_floor": near,
        "link_pos": reached_pos,
    }


def summarize(samples: dict[str, np.ndarray]) -> dict[str, float]:
    """Return the metrics of :func:`motion_metrics` from the samples of :func:`motion_samples`.

    A metric over no samples, such as the slide of a motion that never touches the floor, is
    NaN, not 0. So is a metric whose samples are missing, such as a goal error of UMR.
    """
    return {metric: statistic(samples.get(key, np.zeros(0)), how) for metric, (key, how) in SUMMARY.items()}


def motion_metrics(
    models: "Models",
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    joint_pos: np.ndarray,
    goal_links: tuple[str, ...],
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
    feet_planted: np.ndarray,
) -> dict[str, float]:
    """Return how well a retargeted motion tracks its goals, and how it treats the floor.

    ``goal_links`` starts with the root link. ``feet_planted`` ``(T, 2)`` tells which of
    :data:`robots.FEET` the source stands on.

    * ``foot_error_cm``, ``hand_error_cm``: the mean distance from a goal to its link.
    * ``foot_angle_deg``, ``hand_angle_deg``: the mean angle between a goal and its link.
    * ``foot_slide_cm_s``: the mean horizontal speed of the lowest sole point, taken as a
      point of the foot, while the source stands on that foot in two frames in a row. A
      foot that rolls over its sole does not slide. ``foot_slide_median_cm_s`` is the median.
    * ``near_floor_slide_cm_s``: the same speed, while the lowest sole point is within
      :data:`CONTACT_GAP` of the floor in two frames in a row, whatever the source does.
      Height alone decides it, so a swing foot that passes low counts too. The IK plants
      the feet where the source stands, so ``foot_slide_cm_s`` measures what the IK aims
      for, and this metric does not. ``near_floor_slide_median_cm_s`` is the median.
    * ``planted_gap_cm``: the mean distance from the lowest sole point to the floor, while
      the source stands on that foot.
    * ``penetration_cm``: the 95th percentile, over frames, of the deepest sole point below
      the floor.
    * ``joint_speed_max``: the highest speed of any joint between two frames, in rad/s. An
      IK that flips a limb shows here.

    A metric over an empty set is NaN.
    """
    return summarize(motion_samples(
        models, base_pos, base_quat, joint_pos, goal_links, goal_pos, goal_quat, feet_planted
    ))


def statistic(values: np.ndarray, how: str) -> float:
    """Return the ``mean``, ``median`` or ``max`` of ``values``, or the percentile ``p<n>``.

    The statistic of no values is NaN.
    """
    if values.size == 0:
        return float("nan")
    if how == "mean":
        return float(values.mean())
    if how == "median":
        return float(np.median(values))
    if how == "max":
        return float(values.max())
    return float(np.percentile(values, int(how[1:])))


def contact_iou(near: np.ndarray, planted: np.ndarray) -> float:
    """Return the intersection over the union of two masks, in percent, or NaN if both are empty.

    ``near`` is where the sole of the robot is near the floor, and ``planted`` where the
    source stands.
    """
    union = (near | planted).sum()
    return float((near & planted).sum() / union * 100.0) if union else float("nan")


def fidelity(
    link_pos: np.ndarray, base_pos: np.ndarray, links: tuple[str, ...], ref_pos: np.ndarray | None
) -> dict[str, np.ndarray]:
    """Return the samples that compare a motion with its source, whatever the method aims for.

    ``link_pos`` ``(T, P, 3)`` is where the robot puts ``links``, and ``ref_pos`` the goals of
    ``ik.human.goals`` before ``ik.human.plant`` moves them, or None for a source with no
    such goals, such as a G1 source:

    * ``limb_angle_deg``: the angle between each limb of :data:`LIMBS` and the same limb of
      the source.
    * ``root_error_cm``: how far the horizontal move of the root over :data:`ROOT_WINDOW`
      differs from that of the source root, which ``ik.human.goals`` scales to Lite.
    """
    if ref_pos is None:
        return {"limb_angle_deg": np.zeros(0), "root_error_cm": np.zeros(0)}
    pairs = [(links.index(a), links.index(b)) for a, b in LIMBS if a in links and b in links]
    robot = np.stack([link_pos[:, b] - link_pos[:, a] for a, b in pairs], axis=1)
    source = np.stack([ref_pos[:, b] - ref_pos[:, a] for a, b in pairs], axis=1)
    cos = np.sum(robot * source, axis=-1) / (
        np.linalg.norm(robot, axis=-1) * np.linalg.norm(source, axis=-1)
    )
    window = round(ROOT_WINDOW * FPS)
    move = base_pos[window:, :2] - base_pos[:-window, :2]
    ref_move = ref_pos[window:, 0, :2] - ref_pos[:-window, 0, :2]
    return {
        "limb_angle_deg": np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).ravel(),
        "root_error_cm": np.linalg.norm(move - ref_move, axis=-1) * 100.0,
    }
