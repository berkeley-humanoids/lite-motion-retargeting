"""Retarget motion to the Berkeley Lite humanoid.

A source is either human motion, such as the LAFAN1 BVH takes, or motion that its
authors already retargeted to the 29-DoF Unitree G1. The loader of the source decides
which path a clip takes, after the pipeline resamples the source to the output rate:

* **Human**: ``mapping.goals`` poses a skeleton with the proportions of Lite and the
  joint rotations of the human, and ``mapping.solve_ik`` reaches it frame by frame with a
  floating base.
* **G1, step 1**: a direct joint copy, ``lite_q = sign * g1_q + offset``, from the static
  :data:`common.G1_TO_LITE` table.
* **G1, step 2**: a per-frame ``mink`` IK that refines step 1 to match the feet and hands
  of the G1 relative to the pelvis, in position and orientation. A posture cost per DoF
  pulls the solution toward step 1, so the IK changes step 1 only a little.

Both paths then share:

* **Grounding**: a constant pelvis-z shift per clip that puts the lowest point of the Lite
  soles on the floor in the median frame.
* **Metrics**: how far each goal is from the link that tracks it, how fast a foot on the
  floor slides, how deep a sole sinks into the floor, and the fastest joint. The robot
  layer keeps them, and a run prints their mean.

One clip becomes two layers of one motion in ``dataset/``: ``base/``, the source clip
as published at its own rate, and ``lite_pro/``, at the 50 Hz control rate, with the
joints that the IK reached and the goal that it aimed at. ``sources.py`` lists the
sources, ``store.py`` defines the store, and ``uv run publish`` uploads it.

Usage:
    uv run retarget
    uv run retarget --clip 'walk1_subject1'
    uv run retarget --source lafan1bvh --clip 'walk1_subject1' --validate-only
    uv run retarget --source amass --clip '^CMU/'
    uv run retarget --workers -1          # every clip, every core
"""

import os
import re
from multiprocessing import get_context
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .common import (
    CACHE_ROOT,
    DATASET_ROOT,
    FPS,
    G1_DRIVEN,
    G1_JOINT_NAMES,
    G1_TO_LITE,
    IK_BODY_PAIRS,
    ROBOT_BODIES,
    SOURCE_BODIES,
    body_id,
    joint_qpos_addr,
    lite_joint_names,
    lite_urdf_path,
    load_g1_model,
    load_lite_model,
    lowest_points,
    planted,
    quat_mul,
    resample_motion,
    resample_poses,
    sole,
    sole_points,
    world_poses,
)
from .hub import hub_dataset_of
from .human import HUMAN_BODIES, HumanClip
from .mapping import (
    FEET,
    LITE_PAIRS,
    calibrate,
    driven_joints,
    goals,
    load_floating_model,
    plant,
    solve_ik,
)
from .sources import SOURCES, clip_key, motion_name
from .store import write_asset, write_base_layer, write_blueprint, write_robot_layer

#: The number of clips that a worker retargets before a new process replaces it. Rerun 0.38
#: keeps two threads alive for every recording that it opened, and a motion opens two. A
#: worker that wrote about 1,200 motions could not start a thread.
WORKER_CLIPS: int = 100


# ── The models, compiled once per process ─────────────────────────────────────


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
        goal_conv=_goal_conversions(g1, lite),
    )


def _rest_frame_conversions(g1: mujoco.MjModel, lite: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Return a constant ``R_conv`` per IK link, such that ``R_target = R_g1_actual @ R_conv``.

    The world-frame motion of a Lite link relative to its matched rest must equal the
    world-frame motion of its G1 link relative to the G1 rest. Solve for the target rotation:
    ``R_target = R_g1_actual @ (R_g1_rest^-1 @ R_lite_matched_rest)``. The matched rest of
    Lite is the pose that step 1 makes from the G1 rest.
    """
    g1_data = mujoco.MjData(g1)
    mujoco.mj_kinematics(g1, g1_data)
    lite_data = mujoco.MjData(lite)
    for lite_name, _, offset in G1_TO_LITE.values():
        lite_data.qpos[joint_qpos_addr(lite, lite_name)] = offset
    mujoco.mj_kinematics(lite, lite_data)

    out: dict[str, np.ndarray] = {}
    for lite_name, g1_name in IK_BODY_PAIRS.items():
        R_g1 = g1_data.xmat[body_id(g1, g1_name)].reshape(3, 3)
        R_lite = lite_data.xmat[body_id(lite, lite_name)].reshape(3, 3)
        out[lite_name] = R_g1.T @ R_lite
    return out


def _goal_conversions(g1: mujoco.MjModel, lite: mujoco.MjModel) -> np.ndarray:
    """Return the wxyz rotation per body from the G1 frame convention to Lite, root body first.

    A G1 link and the Lite link that it drives disagree about which way their axes point at
    rest, by up to 120 degrees. Thus a goal in the raw G1 convention is not comparable to
    the link that it is the goal for, and a tracking reward on orientation would never reach
    zero. The IK aims through the same conversion, so the goal that a motion stores is the
    goal that the IK got.

    The root body is not an IK target, and its goal is the base trajectory, so it converts
    by the identity.
    """
    conversions = _rest_frame_conversions(g1, lite)
    quats = np.zeros((len(ROBOT_BODIES), 4))
    quats[0] = (1.0, 0.0, 0.0, 0.0)
    for index, lite_name in enumerate(IK_BODY_PAIRS, start=1):
        mujoco.mju_mat2Quat(quats[index], conversions[lite_name].reshape(9))
    return quats


# ── G1, step 1: direct copy with sign and offset ──────────────────────────────


def step1_direct_remap(g1_joint_pos: np.ndarray, joint_names: tuple[str, ...]) -> np.ndarray:
    """Copy each G1 joint to its Lite joint as ``lite_q = sign * g1_q + offset``.

    Return ``(T, J)`` joint positions in the order of ``joint_names``. A Lite joint with no
    G1 joint (the neck, the fingers, and the ankle yaw) holds 0.
    """
    lite_names, signs, offsets = zip(*(G1_TO_LITE[name] for name in G1_JOINT_NAMES), strict=True)
    joint_pos = np.zeros((g1_joint_pos.shape[0], len(joint_names)), dtype=np.float32)
    joint_pos[:, [joint_names.index(name) for name in lite_names]] = (
        np.asarray(signs, dtype=np.float32) * g1_joint_pos + np.asarray(offsets, dtype=np.float32)
    )
    return joint_pos


# ── G1, step 2: per-frame IK refinement ───────────────────────────────────────


def _posture_cost(lite: mujoco.MjModel) -> np.ndarray:
    """Return the posture cost of each DoF, in three tiers, so that step 2 only adjusts step 1.

    The posture task pulls each joint toward step 1 with this cost. The frame task of each
    foot and hand pulls the joints away with cost 1.0 for position and 1.0 for orientation.

    * ``1e3`` locks a joint. The neck, the ankle yaw and the fingers have no G1 joint, so
      they stay at the 0 of step 1. The motion holds them at 0, so the IK must not use them.
    * ``10.0`` makes a joint stiff but adjustable. The waist turns the torso and the head,
      so a high cost keeps the IK from twisting the whole upper body to reach a hand goal.
      The IK moves the arm joints first.
    * ``1.0`` matches the frame tasks. The arms and the legs get a balanced trade-off, and
      the correction in each frame is small.
    """
    cost = np.full(lite.nv, 1e3, dtype=np.float64)
    for jid in range(lite.njnt):
        name = mujoco.mj_id2name(lite, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name not in G1_DRIVEN:
            continue
        dof = int(lite.jnt_dofadr[jid])
        if any(limb in name for limb in ("hip", "knee", "ankle", "shoulder", "elbow", "wrist")):
            cost[dof] = 1.0
        elif "waist" in name:
            cost[dof] = 10.0
    return cost


def step2_ik_refine(
    models: Models,
    g1_joint_pos: np.ndarray,
    step1_joint_pos: np.ndarray,
    iters: int,
    show_progress: bool,
) -> np.ndarray:
    """Refine the step-1 joints to match the feet and hands of the G1 relative to the pelvis.

    The IK solves each frame on its own, from the step-1 seed. ``show_progress`` shows a
    progress bar for the frames. Workers turn it off, so that their bars do not mix in the
    terminal.
    """
    import mink

    g1_data = mujoco.MjData(models.g1)
    conversions = _rest_frame_conversions(models.g1, models.lite)
    configuration = mink.Configuration(models.lite)
    frame_tasks = {
        name: mink.FrameTask(name, "body", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
        for name in IK_BODY_PAIRS
    }
    posture_task = mink.PostureTask(models.lite, cost=_posture_cost(models.lite))
    tasks = [*frame_tasks.values(), posture_task]
    limits = [mink.ConfigurationLimit(models.lite)]

    out = step1_joint_pos.copy()
    seed = np.zeros(models.lite.nq, dtype=np.float64)
    frames = range(step1_joint_pos.shape[0])
    for t in tqdm(frames, desc="    IK", leave=False, unit="frame", disable=not show_progress):
        g1_data.qpos[models.g1_addrs] = g1_joint_pos[t]
        mujoco.mj_kinematics(models.g1, g1_data)
        for lite_name, g1_name in IK_BODY_PAIRS.items():
            bid = body_id(models.g1, g1_name)
            target = np.eye(4)
            target[:3, :3] = g1_data.xmat[bid].reshape(3, 3) @ conversions[lite_name]
            target[:3, 3] = g1_data.xpos[bid]
            frame_tasks[lite_name].set_target(mink.SE3.from_matrix(target))

        seed[models.joint_addrs] = step1_joint_pos[t]
        configuration.q[:] = seed
        posture_task.set_target(seed.copy())
        for _ in range(iters):
            vel = mink.solve_ik(
                configuration, tasks, 1.0, solver="daqp", damping=1e-1, limits=limits
            )
            configuration.integrate_inplace(vel, 1.0)
        out[t] = configuration.q[models.joint_addrs]
    return out


# ── Grounding and metrics ─────────────────────────────────────────────────────


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
    :data:`mapping.FEET` the source stands on.

    * ``foot_error_cm``, ``hand_error_cm``: the mean distance from a goal to its link.
    * ``foot_angle_deg``, ``hand_angle_deg``: the mean angle between a goal and its link.
    * ``foot_slide_cm_s``: the mean horizontal speed of the lowest sole point, taken as a
      point of the foot, while the source stands on that foot in two frames in a row. A
      foot that rolls over its sole does not slide.
    * ``planted_gap_cm``: the mean distance from the lowest sole point to the floor, while
      the source stands on that foot.
    * ``penetration_cm``: the 95th percentile, over frames, of the deepest sole point below
      the floor.
    * ``joint_speed_max``: the highest speed of any joint between two frames, in rad/s. An
      IK that flips a limb shows here.
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
    depth = np.maximum(-soles[:, :, 2].min(axis=1), 0.0) * 100.0
    gap = np.abs(soles[..., 2][feet_planted])
    return {
        "foot_error_cm": float(error[:, feet].mean()),
        "foot_angle_deg": float(angle[:, feet].mean()),
        "hand_error_cm": float(error[:, hands].mean()),
        "hand_angle_deg": float(angle[:, hands].mean()),
        "foot_slide_cm_s": float(speed[both].mean()) if both.any() else 0.0,
        "planted_gap_cm": float(gap.mean() * 100.0) if feet_planted.any() else 0.0,
        "penetration_cm": float(np.percentile(depth, 95)),
        "joint_speed_max": float(np.abs(np.diff(joint_pos, axis=0)).max() * FPS),
    }


# ── One clip, and the process that retargets it ───────────────────────────────


class Clip(NamedTuple):
    """One retargeted clip: the arrays that the two layers of its motion come from.

    The ``source_*`` arrays are the clip as published, at ``source_fps``: the joints of the
    source, if it has joint angles to keep, and the world pose of its bodies. The other
    arrays are the retarget, at :data:`FPS`. ``goal_sources`` names the source body that
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
) -> Clip:
    """Run every step on one clip of ``source``, and pose the bodies that each layer holds.

    The base layer keeps the source as it is, at its own rate and in its own frame
    convention. The robot layer holds the goal of each link in the convention of that link,
    so the goal compares directly against the link. The goal of the root is where the
    retarget put the base, so the robot and its goals share one root.
    """
    loaded = SOURCES[source].load(path)
    if isinstance(loaded, HumanClip):
        return _retarget_human(models, loaded, ik_iters)

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


def _retarget_human(models: Models, clip: HumanClip, ik_iters: int) -> Clip:
    """Retarget one human clip: goals from ``mapping``, then the floating-base IK."""
    pos, quat = resample_poses(clip.pos, clip.quat, clip.fps, FPS)
    foot_points = [
        [HUMAN_BODIES.index(f"{side}_{point}") for point in ("foot", "toe")]
        for side in ("left", "right")
    ]
    feet_planted = planted(pos[:, foot_points], FPS)
    goal_pos, goal_quat = goals(pos, quat, calibrate(models.lite, clip.rest))
    goal_pos = plant(goal_pos, goal_quat, feet_planted, sole(models.lite, FEET))
    base_pos, base_quat, joint_pos = solve_ik(
        models.floating, models.joint_names, goal_pos, goal_quat, ik_iters
    )
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


def write_motion(
    models: Models, name: str, clip: Clip, urdf: Path, geometry: bool, out: Path
) -> list[Path]:
    """Write the base layer and the robot layer of one motion under ``out``."""
    return [
        write_base_layer(
            out, name, clip.source_fps, clip.source_joint_names,
            clip.source_joint_pos, clip.source_bodies, clip.source_pos, clip.source_quat,
        ),
        write_robot_layer(
            out, name, urdf, FPS, models.joint_names, clip.joint_pos,
            clip.goal_links, clip.goal_sources, clip.goal_pos, clip.goal_quat, geometry,
            clip.metrics, clip.active_joints,
        ),
    ]


_MODELS: Models | None = None


def _worker_init() -> None:
    """Compile the models once per worker."""
    global _MODELS
    _MODELS = load_models()


def _worker_retarget(
    job: tuple[str, Path, str, bool, int, Path, bool, Path],
) -> tuple[list[Path], dict[str, float], str | None]:
    """Retarget one clip in a worker, write its layers, and return their paths and metrics.

    A motion has files of its own, so no two workers write one file. Writing here keeps
    the parent from being the bottleneck, because a short clip takes longer to write than
    its share of the IK. A clip that fails returns its error in place of paths, so one bad
    file among thousands does not stop the rest.
    """
    assert _MODELS is not None
    source, path, name, ik, ik_iters, urdf, geometry, out = job
    try:
        clip = retarget_clip(_MODELS, source, path, ik, ik_iters, show_progress=False)
        written = write_motion(_MODELS, name, clip, urdf, geometry, out)
    except Exception as error:  # noqa: BLE001 -- the parent lists every failed clip
        return [], {}, f"{path}: {type(error).__name__}: {error}"
    return written, clip.metrics, None


def print_metrics(rows: list[dict[str, float]]) -> None:
    """Print the mean of each metric over ``rows``, one clip per row."""
    print(f"\nMetrics, mean over {len(rows)} clips:")
    for key in rows[0]:
        print(f"  {key:<16s} {np.mean([row[key] for row in rows]):8.2f}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(
    source: str = "lafan1",
    clip: str | None = None,
    ik: bool = True,
    ik_iters: int = 15,
    workers: int = 1,
    validate_only: bool = False,
    out: Path = DATASET_ROOT,
) -> None:
    """Retarget the clips of a source to Lite, and write the two layers of each motion.

    Args:
        source: The key of the source in ``sources.SOURCES``.
        clip: A regex. Retarget only the clips whose path below the clip directory of the
            source matches it, such as ``CMU/07/07_01_poses_120_jpos`` for AMASS.
        ik: Refine step 1 of a G1 source with the step-2 IK. Use ``--no-ik`` to keep only
            step 1. A human source always uses its IK.
        ik_iters: The number of Newton steps per frame, for both IKs.
        workers: The number of processes that retarget clips in parallel. ``-1`` uses every
            core.
        validate_only: Retarget only the first matching clip, print its metrics, and write
            nothing.
        out: The store to write. Use another directory to keep two runs apart and compare
            them.
    """
    if source not in SOURCES:
        raise SystemExit(f"Unknown source {source!r}. Expected one of {sorted(SOURCES)}.")
    if workers == -1:
        workers = os.cpu_count() or 1

    spec = SOURCES[source]
    clip_dir = CACHE_ROOT / source / spec.clip_dir
    paths = sorted(clip_dir.glob(spec.clip_glob))
    if clip is not None:
        paths = [p for p in paths if re.search(clip, clip_key(clip_dir, p))]
    if not paths:
        raise SystemExit(
            f"No clips to retarget under {clip_dir} (clip={clip!r}). "
            f"Run `uv run download --source {source}` first."
        )
    names = [motion_name(source, clip_dir, p) for p in paths]
    if len(set(names)) != len(names):
        raise SystemExit(f"Two clips of {source} map to one motion name.")

    print("Loading models …")
    models = load_models()
    flipped = sum(1 for _, s, _ in G1_TO_LITE.values() if s < 0)
    nonzero = sum(1 for _, _, off in G1_TO_LITE.values() if abs(off) > 1e-6)
    print(f"  G1 nq={models.g1.nq}, Lite nq={models.lite.nq}, "
          f"joints={len(models.joint_names)}")
    print(f"  {len(G1_TO_LITE)} joint pairs, {flipped} sign flips, {nonzero} nonzero offsets")
    print(f"  {len(paths)} clips of {source}, resampled to {FPS} Hz")

    if validate_only:
        result = retarget_clip(models, source, paths[0], ik, ik_iters)
        print(f"\nClip: {paths[0].name} ({result.joint_pos.shape[0]} frames at {FPS} Hz, "
              f"from {result.source_fps} Hz)")
        print_metrics([result.metrics])
        return

    urdf = lite_urdf_path()
    blueprint = write_blueprint(out, urdf)
    asset = write_asset(out, urdf)
    written = [blueprint, asset]
    # The Hub card links each published motion into the web viewer, which cannot load the
    # asset, so a published motion carries its own geometry.
    geometry = hub_dataset_of(source) is not None

    failed = []
    metrics = []
    if workers > 1:
        jobs = [
            (source, path, name, ik, ik_iters, urdf, geometry, out)
            for path, name in zip(paths, names, strict=True)
        ]
        # Spawn, not fork: the parent runs threads by now, and a forked child can inherit a
        # held lock. ProcessPoolExecutor stops replacing workers that reach their task limit
        # once many jobs are queued (CPython 3.13), so this uses multiprocessing.Pool.
        with get_context("spawn").Pool(workers, _worker_init, maxtasksperchild=WORKER_CLIPS) as pool:
            results = pool.imap(_worker_retarget, jobs)
            for files, clip_metrics, error in tqdm(
                results, total=len(jobs), desc=f"Clips (workers={workers})", unit="clip"
            ):
                written += files
                if error is not None:
                    failed.append(error)
                else:
                    metrics.append(clip_metrics)
        if failed:
            print(f"\n{len(failed)} of {len(jobs)} clips failed and were skipped:")
            for error in failed:
                print(f"  {error}")
    else:
        for path, name in tqdm(zip(paths, names, strict=True), total=len(names), desc="Clips",
                               unit="clip"):
            result = retarget_clip(models, source, path, ik, ik_iters)
            written += write_motion(models, name, result, urdf, geometry, out)
            metrics.append(result.metrics)

    if metrics:
        print_metrics(metrics)
    total_mb = sum(path.stat().st_size for path in written) / 1e6
    print(f"\nWrote {len(names) - len(failed)} motions as {len(written)} files "
          f"({total_mb:.1f} MB) under {out}")
    # The blueprint goes last: blueprints do not merge, and the last one loaded wins.
    print(f"  view: uv run rerun {out}/*/{names[0]}.rrd --asset {asset} {blueprint}")


def cli() -> None:
    """Run :func:`main` with the options on the command line."""
    tyro.cli(main)
