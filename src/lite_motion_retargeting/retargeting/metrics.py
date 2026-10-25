"""Measure a retargeted motion: how well it tracks its goals, and how it treats the floor."""

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

    A metric over an empty set, such as the slide of a motion that never touches the floor,
    is NaN, not 0.
    """

    def mean(key: str) -> float:
        return float(samples[key].mean()) if samples[key].size else float("nan")

    def median(key: str) -> float:
        return float(np.median(samples[key])) if samples[key].size else float("nan")

    return {
        "foot_error_cm": mean("foot_error_cm"),
        "foot_angle_deg": mean("foot_angle_deg"),
        "hand_error_cm": mean("hand_error_cm"),
        "hand_angle_deg": mean("hand_angle_deg"),
        "foot_slide_cm_s": mean("foot_slide_cm_s"),
        "foot_slide_median_cm_s": median("foot_slide_cm_s"),
        "near_floor_slide_cm_s": mean("near_floor_slide_cm_s"),
        "near_floor_slide_median_cm_s": median("near_floor_slide_cm_s"),
        "planted_gap_cm": mean("planted_gap_cm"),
        "penetration_cm": float(np.percentile(samples["penetration_cm"], 95)),
        "joint_speed_max": float(samples["joint_speed"].max()),
    }


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
