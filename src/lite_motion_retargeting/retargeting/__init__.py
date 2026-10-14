"""Retarget one clip to the Berkeley Lite humanoid.

A source is either human motion, such as the LAFAN1 BVH takes, or motion that its
authors already retargeted to the 29-DoF Unitree G1. The loader of the source decides
which path a clip takes, after the pipeline resamples the source to the output rate:

* **Human**: ``human.goals`` poses a skeleton with the proportions of Lite and the
  joint rotations of the human, and ``human.solve_ik`` reaches it frame by frame with a
  floating base.
* **G1, step 1**: a direct joint copy, ``lite_q = sign * g1_q + offset``, from the static
  :data:`g1.G1_TO_LITE` table.
* **G1, step 2**: a per-frame ``mink`` IK that refines step 1 to match the feet and hands
  of the G1 relative to the pelvis, in position and orientation. A posture cost per DoF
  pulls the solution toward step 1, so the IK changes step 1 only a little.

Both paths then share:

* **Grounding**: a constant pelvis-z shift per clip that puts the lowest point of the Lite
  soles on the floor in the median frame.
* **Metrics**: how far each goal is from the link that tracks it, how fast a foot on the
  floor slides, how deep a sole sinks into the floor, and the fastest joint. The robot
  layer keeps them, and a run prints their mean.

``g1.py`` holds the G1 path, and ``human.py`` the human path. :func:`retarget_clip` runs
the path of the source on one clip, then grounds the result and measures it.
"""

from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..kinematics import (
    lowest_points,
    planted,
    quat_mul,
    resample_motion,
    resample_poses,
    sole,
    sole_points,
    world_poses,
)
from ..robots import (
    FEET,
    G1_JOINT_NAMES,
    joint_qpos_addr,
    lite_joint_names,
    load_g1_model,
    load_lite_model,
)
from ..sources import SOURCES
from ..sources.human import HUMAN_BODIES, HumanClip
from ..store import FPS
from .g1 import (
    G1_DRIVEN,
    ROBOT_BODIES,
    SOURCE_BODIES,
    goal_conversions,
    step1_direct_remap,
    step2_ik_refine,
)
from .human import (
    LITE_PAIRS,
    calibrate,
    driven_joints,
    goals,
    load_floating_model,
    plant,
    solve_ik,
)


class Models(NamedTuple):
    """Both robots, and the constants that a retarget derives from them."""

    g1: mujoco.MjModel
    lite: mujoco.MjModel
    floating: mujoco.MjModel
    g1_addrs: np.ndarray
    joint_addrs: np.ndarray
    joint_names: tuple[str, ...]
    goal_conv: np.ndarray


def load_models() -> Models:
    """Compile both robots, and Lite again with a free base for the human IK."""
    g1 = load_g1_model()
    lite = load_lite_model()
    names = lite_joint_names(lite)
    return Models(
        g1=g1,
        lite=lite,
        floating=load_floating_model(),
        g1_addrs=np.asarray([joint_qpos_addr(g1, n) for n in G1_JOINT_NAMES], dtype=np.int32),
        joint_addrs=np.asarray([joint_qpos_addr(lite, n) for n in names], dtype=np.int32),
        joint_names=names,
        goal_conv=goal_conversions(g1, lite),
    )


# ── Grounding and metrics ─────────────────────────────────────────────────────

#: m. A sole point this close to the floor is near it, for ``near_floor_slide_cm_s``.
CONTACT_GAP: float = 0.01


def ground_height(
    models: Models, joint_pos: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray
) -> float:
    """Return the pelvis-z shift that puts the lowest sole point of Lite on the floor.

    A source puts the base at its own height, AMASS at the floor, and Lite stands on shorter
    legs and thicker soles than the G1. Thus no constant offset grounds every clip. The
    shift comes from the Lite result instead, in the median frame. A single frame would let
    one sole that dips through the floor lift the whole clip, and a foot is on the floor in
    most frames of a clip.
    """
    soles = sole_points(models.lite, models.joint_addrs, joint_pos, base_pos, base_quat, FEET)
    return -float(np.median(soles[:, :, 2].min(axis=1)))


def motion_samples(
    models: Models,
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
    models: Models,
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


# ── One clip ──────────────────────────────────────────────────────────────────


class Clip(NamedTuple):
    """One retargeted clip: the arrays that the two layers of its motion come from.

    The ``source_*`` arrays are the clip as published, at ``source_fps``: the joints of the
    source, if it has joint angles to keep, and the world pose of its bodies. The other
    arrays are the retarget, at :data:`store.FPS`. ``goal_sources`` names the source body that
    each goal follows. ``active_joints`` are the joints that the source drives. Every other
    joint holds 0.
    """

    source_fps: int
    source_joint_names: tuple[str, ...]
    source_joint_pos: np.ndarray
    source_bodies: tuple[str, ...]
    source_pos: np.ndarray
    source_quat: np.ndarray
    joint_pos: np.ndarray
    goal_links: tuple[str, ...]
    goal_sources: tuple[str, ...]
    goal_pos: np.ndarray
    goal_quat: np.ndarray
    metrics: dict[str, float]
    active_joints: tuple[str, ...]


def retarget_clip(
    models: Models,
    source: str,
    path: Path,
    do_ik: bool,
    ik_iters: int,
    show_progress: bool = True,
    plant_feet: bool = True,
) -> Clip:
    """Run every step on one clip of ``source``, and pose the bodies that each layer holds.

    The base layer keeps the source as it is, at its own rate and in its own frame
    convention. The robot layer holds the goal of each link in the convention of that link,
    so the goal compares directly against the link. The goal of the root is where the
    retarget put the base, so the robot and its goals share one root.

    If ``plant_feet`` is False, a human clip skips :func:`human.plant`: the IK then reaches
    the goals as the source gives them, as a start for a method that fixes the feet itself,
    such as ReActor.
    """
    loaded = SOURCES[source].load(path)
    if isinstance(loaded, HumanClip):
        return _retarget_human(models, loaded, ik_iters, plant_feet)

    clip, source_fps = loaded
    source_pos, source_quat = world_poses(
        models.g1, models.g1_addrs, clip["g1_joint_pos"],
        clip["base_pos"], clip["base_quat_wxyz"], SOURCE_BODIES[1:],
    )

    motion = resample_motion(clip, source_fps, FPS)
    joint_pos = step1_direct_remap(motion["g1_joint_pos"], models.joint_names)
    if do_ik:
        joint_pos = step2_ik_refine(
            models, motion["g1_joint_pos"], joint_pos, ik_iters, show_progress
        )
    joint_pos[:, [name not in G1_DRIVEN for name in models.joint_names]] = 0.0

    base_pos = motion["base_pos"].astype(np.float32)
    base_quat = motion["base_quat_wxyz"].astype(np.float32)
    base_pos[:, 2] += ground_height(models, joint_pos, base_pos, base_quat)

    # The goal rides the base trajectory of the retarget, not that of the source, so it
    # sits where the Lite pelvis sits and the two are directly comparable.
    goal_pos, goal_quat = world_poses(
        models.g1, models.g1_addrs, motion["g1_joint_pos"], base_pos, base_quat, SOURCE_BODIES[1:]
    )
    goal_quat = quat_mul(goal_quat, models.goal_conv).astype(np.float32)
    feet_planted = planted(goal_pos[:, [ROBOT_BODIES.index(foot) for foot in FEET], None], FPS)
    return Clip(
        source_fps=source_fps,
        source_joint_names=G1_JOINT_NAMES,
        source_joint_pos=clip["g1_joint_pos"],
        source_bodies=SOURCE_BODIES,
        source_pos=source_pos,
        source_quat=source_quat,
        joint_pos=joint_pos,
        goal_links=ROBOT_BODIES,
        goal_sources=SOURCE_BODIES,
        goal_pos=goal_pos,
        goal_quat=goal_quat,
        metrics=motion_metrics(
            models, base_pos, base_quat, joint_pos, ROBOT_BODIES, goal_pos, goal_quat, feet_planted
        ),
        active_joints=G1_DRIVEN,
    )


def human_planted(pos: np.ndarray) -> np.ndarray:
    """Return which feet of a human clip stand on the floor, ``(T, 2)``, left first.

    ``pos`` holds the world positions of :data:`HUMAN_BODIES` at :data:`store.FPS`. A foot is
    planted while its ankle or its toe is, as :func:`kinematics.planted` decides.
    """
    points = [[HUMAN_BODIES.index(f"{side}_{point}") for point in ("foot", "toe")] for side in ("left", "right")]
    return planted(pos[:, points], FPS)


def _retarget_human(models: Models, clip: HumanClip, ik_iters: int, plant_feet: bool) -> Clip:
    """Retarget one human clip: goals from ``human.py``, then the floating-base IK."""
    pos, quat = resample_poses(clip.pos, clip.quat, clip.fps, FPS)
    feet_planted = human_planted(pos)
    goal_pos, goal_quat = goals(pos, quat, calibrate(models.lite, clip.rest))
    if plant_feet:
        goal_pos = plant(goal_pos, goal_quat, feet_planted, sole(models.lite, FEET))
    base_pos, base_quat, joint_pos = solve_ik(
        models.floating, models.joint_names, goal_pos, goal_quat, ik_iters
    )
    active = driven_joints(models.lite)
    joint_pos[:, [name not in active for name in models.joint_names]] = 0.0

    # The planted goals stand on the floor, and the IK leaves the soles within a few
    # millimetres of them. Thus the shift is small, and it moves the goals with the robot.
    # Goals that are not planted can float or sink, and the shift grounds the median frame.
    shift = ground_height(models, joint_pos, base_pos, base_quat)
    base_pos[:, 2] += shift
    goal_pos[:, :, 2] += shift

    goal_pos[:, 0], goal_quat[:, 0] = base_pos, base_quat
    links = tuple(LITE_PAIRS)
    return Clip(
        source_fps=clip.fps,
        source_joint_names=(),
        source_joint_pos=np.zeros((clip.pos.shape[0], 0), dtype=np.float32),
        source_bodies=clip.source_bodies,
        source_pos=clip.source_pos,
        source_quat=clip.source_quat,
        joint_pos=joint_pos,
        goal_links=links,
        goal_sources=tuple(pair.human for pair in LITE_PAIRS.values()),
        goal_pos=goal_pos,
        goal_quat=goal_quat,
        metrics=motion_metrics(
            models, base_pos, base_quat, joint_pos, links, goal_pos, goal_quat, feet_planted
        ),
        active_joints=active,
    )
