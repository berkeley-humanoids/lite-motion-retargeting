"""Retarget motion to the Berkeley Lite humanoid, by one of three methods.

* ``ik/``: the IK, the default. ``uv run retarget`` runs it on every clip of a source, and
  writes the store.
* ``reactor/``: ReActor, which tracks the goals of the IK store with a policy in physics.
  ``uv run reactor`` trains it, and exports a new store.
* ``umr/``: UMR, an outside kinematic retargeter, which runs in an environment of its own.
  ``umr/`` holds its config for Lite Pro, and the scripts that prepare its input.

``metrics.py`` measures a motion of any method, and ``uv run compare`` measures the methods
on the same frames.

:func:`retarget_clip` runs the IK on one clip: the path of its source, then a constant
pelvis-z shift that puts the lowest point of the Lite soles on the floor in the median
frame, then the metrics.
"""

from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np

from ..kinematics import (
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
    joint_names,
    joint_qpos_addr,
    load_g1_model,
    load_lite_model,
)
from ..sources import SOURCES
from ..sources.human import HUMAN_BODIES, HumanClip
from ..store import FPS
from .ik.g1 import (
    G1_DRIVEN,
    ROBOT_BODIES,
    SOURCE_BODIES,
    goal_conversions,
    step1_direct_remap,
    step2_ik_refine,
)
from .ik.human import (
    LITE_PAIRS,
    calibrate,
    driven_joints,
    goals,
    load_floating_model,
    plant,
    solve_ik,
)
from .metrics import motion_metrics


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
    names = joint_names(lite)
    return Models(
        g1=g1,
        lite=lite,
        floating=load_floating_model(),
        g1_addrs=np.asarray([joint_qpos_addr(g1, n) for n in G1_JOINT_NAMES], dtype=np.int32),
        joint_addrs=np.asarray([joint_qpos_addr(lite, n) for n in names], dtype=np.int32),
        joint_names=names,
        goal_conv=goal_conversions(g1, lite),
    )


# ── Grounding ─────────────────────────────────────────────────────────────────

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


def retarget_clip(models: Models, source: str, path: Path) -> Clip:
    """Run every step on one clip of ``source``, and pose the bodies that each layer holds.

    The base layer keeps the source as it is, at its own rate and in its own frame
    convention. The robot layer holds the goal of each link in the convention of that link,
    so the goal compares directly against the link. The goal of the root is where the
    retarget put the base, so the robot and its goals share one root.
    """
    loaded = SOURCES[source].load(path)
    if isinstance(loaded, HumanClip):
        return _retarget_human(models, loaded)
    return _retarget_g1(models, *loaded)


def _retarget_g1(models: Models, clip: dict[str, np.ndarray], source_fps: int) -> Clip:
    """Retarget one G1 clip: the joint copy of ``ik/g1.py``, then the IK that refines it."""
    source_pos, source_quat = world_poses(
        models.g1, models.g1_addrs, clip["g1_joint_pos"],
        clip["base_pos"], clip["base_quat_wxyz"], SOURCE_BODIES[1:],
    )

    motion = resample_motion(clip, source_fps, FPS)
    joint_pos = step1_direct_remap(motion["g1_joint_pos"], models.joint_names)
    joint_pos = step2_ik_refine(models, motion["g1_joint_pos"], joint_pos)
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


def _retarget_human(models: Models, clip: HumanClip) -> Clip:
    """Retarget one human clip: the planted goals of ``ik/human.py``, then the floating-base IK."""
    pos, quat = resample_poses(clip.pos, clip.quat, clip.fps, FPS)
    feet_planted = human_planted(pos)
    goal_pos, goal_quat = goals(pos, quat, calibrate(models.lite, clip.rest))
    goal_pos = plant(goal_pos, goal_quat, feet_planted, sole(models.lite, FEET))
    base_pos, base_quat, joint_pos = solve_ik(models.floating, models.joint_names, goal_pos, goal_quat)
    active = driven_joints(models.lite)
    joint_pos[:, [name not in active for name in models.joint_names]] = 0.0

    # The planted goals stand on the floor, and the IK leaves the soles within a few
    # millimetres of them. Thus the shift is small, and it moves the goals with the robot.
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
