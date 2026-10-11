"""Retarget G1 motions to the Berkeley Lite humanoid.

Pipeline per clip, after the source is resampled to the output rate:

* **Step 1** — direct joint copy ``lite_q = sign * g1_q + offset`` using the
  static :data:`common.G1_TO_LITE` table.
* **Step 2** — per-frame ``mink`` IK that refines step 1 to match G1's
  pelvis-local EE poses (feet + hands, position + orientation). The
  per-DOF posture cost biases the solution toward step 1 so the refinement
  is a *tweak*, not a rearrangement.
* **Grounding** — a constant pelvis-z shift per clip that puts the lowest point of
  Lite's soles on the floor in the median frame.

One clip becomes two layers of one motion in ``dataset/``: ``base/``, the source clip
as published at its own rate, and ``lite_pro/``, at the 50 Hz control rate, holding the
joints the IK reached and the goal it aimed at. ``scripts/sources.py`` lists the sources,
``scripts/store.py`` defines the store, and ``scripts/publish.py`` uploads it.

Usage:
    uv run scripts/retarget.py
    uv run scripts/retarget.py --clip 'walk1_subject1'
    uv run scripts/retarget.py --source amass --clip '^CMU/'
    uv run scripts/retarget.py --validate-only
    uv run scripts/retarget.py --workers -1          # every clip, every core
"""

import os
import re
import sys
from multiprocessing import get_context
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np
import tyro
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CACHE_ROOT,
    FPS,
    G1_JOINT_NAMES,
    G1_TO_LITE,
    IK_BODY_PAIRS,
    PUBLIC_SOURCE,
    ROBOT_BODIES,
    SOURCE_BODIES,
    body_id,
    joint_qpos_addr,
    lite_joint_names,
    lite_urdf_path,
    load_g1_model,
    load_lite_model,
    quat_mul,
    resample_motion,
    sole_heights,
    world_poses,
)
from sources import SOURCES, clip_key, motion_name  # noqa: E402
from store import write_asset, write_base_layer, write_blueprint, write_robot_layer  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATASET_ROOT: Path = REPO_ROOT / "dataset"

#: The Lite bodies that stand on the floor.
FEET: tuple[str, ...] = ("left_foot", "right_foot")

# Validation frame stride for the EE-error summary.
SAMPLE_STRIDE: int = 50

# Clips a worker retargets before a fresh process replaces it. rerun 0.38 keeps two threads
# alive for every recording it ever opened, and a motion opens two, so a worker that wrote
# about 1,200 motions could no longer start a thread.
WORKER_CLIPS: int = 100


# ── Step 1: direct copy with sign + offset ────────────────────────────────────


def _build_remap_indices(
    lite_model: mujoco.MjModel,
    lite_joint_addrs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parallel arrays ``(lite_col, g1_col, sign, offset)`` over mapped joints."""
    addr_to_col = {int(a): i for i, a in enumerate(lite_joint_addrs.tolist())}
    lite_cols, g1_cols, signs, offsets = [], [], [], []
    for g1_col, g1_name in enumerate(G1_JOINT_NAMES):
        entry = G1_TO_LITE.get(g1_name)
        if entry is None:
            continue
        lite_name, sign, offset = entry
        lite_cols.append(addr_to_col[joint_qpos_addr(lite_model, lite_name)])
        g1_cols.append(g1_col)
        signs.append(float(sign))
        offsets.append(float(offset))
    return (
        np.asarray(lite_cols, dtype=np.int32),
        np.asarray(g1_cols, dtype=np.int32),
        np.asarray(signs, dtype=np.float32),
        np.asarray(offsets, dtype=np.float32),
    )


def step1_direct_remap(
    motion: dict[str, np.ndarray],
    lite_joint_addrs: np.ndarray,
    lite_model: mujoco.MjModel,
) -> dict[str, np.ndarray]:
    """Apply ``lite_q = sign * g1_q + offset`` to every mapped joint.

    Returns ``base_pos`` and ``base_quat`` (WXYZ), both unchanged from the source, and a
    ``(T, 74)`` ``joint_pos`` in Lite MJCF order. Joints with no G1 source (neck,
    fingers, ankle_yaw) stay at 0.
    """
    lite_cols, g1_cols, signs, offsets = _build_remap_indices(lite_model, lite_joint_addrs)
    frames = motion["base_pos"].shape[0]
    joint_pos = np.zeros((frames, lite_joint_addrs.shape[0]), dtype=np.float32)
    joint_pos[:, lite_cols] = signs * motion["g1_joint_pos"][:, g1_cols] + offsets

    return {
        "base_pos": motion["base_pos"].astype(np.float32, copy=True),
        "base_quat": motion["base_quat_wxyz"].astype(np.float32),
        "joint_pos": joint_pos,
    }


# ── Step 2: per-frame IK refinement ───────────────────────────────────────────

_LIMB_TOKENS: tuple[str, ...] = ("hip", "knee", "ankle", "shoulder", "elbow", "wrist")
_TRUNK_TOKENS: tuple[str, ...] = ("waist",)


def _posture_cost_vector(lite_model: mujoco.MjModel) -> np.ndarray:
    """Per-DOF posture cost — three tiers so step 2 only *tweaks* step 1.

    The PostureTask pulls each joint toward step 1 with cost ``c``; the EE
    FrameTasks pull joints away with cost 1.0 + 1.0 (position + orientation).
    Costs:

    * ``1e3`` — locked. Neck, fingers, and ankle_yaw have no G1 source, so
      we keep them at step 1's zero.
    * ``10.0`` — stiff but adjustable. Waist rotates the torso (and head),
      so a high cost prevents the IK from twisting the whole upper body to
      satisfy hand targets; arm joints are recruited first.
    * ``1.0`` — same magnitude as the EE tasks. Arms + legs get a balanced
      trade-off; per-frame corrections come out small.
    """
    cost = np.full(lite_model.nv, 1e3, dtype=np.float64)
    for jid in range(lite_model.njnt):
        name = mujoco.mj_id2name(lite_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name:
            continue
        dof = int(lite_model.jnt_dofadr[jid])
        if any(tok in name for tok in _LIMB_TOKENS):
            cost[dof] = 1.0
        elif any(tok in name for tok in _TRUNK_TOKENS):
            cost[dof] = 10.0
    return cost


def _rest_frame_conversions(
    g1_model: mujoco.MjModel,
    lite_model: mujoco.MjModel,
) -> dict[str, np.ndarray]:
    """Per-body constant ``R_conv`` such that ``R_target = R_g1_actual @ R_conv``.

    Derivation: we want Lite's world-frame motion delta (relative to its
    matched rest) to equal G1's world-frame motion delta (relative to G1
    rest). Solving for the target orientation gives
    ``R_target = R_g1_actual @ (R_g1_rest^-1 @ R_lite_matched_rest)``.
    """
    g1_data = mujoco.MjData(g1_model)
    mujoco.mj_kinematics(g1_model, g1_data)
    lite_data = mujoco.MjData(lite_model)
    for _, (lite_name, _, offset) in G1_TO_LITE.items():
        lite_data.qpos[joint_qpos_addr(lite_model, lite_name)] = offset
    mujoco.mj_kinematics(lite_model, lite_data)

    out: dict[str, np.ndarray] = {}
    for lite_name, g1_name in IK_BODY_PAIRS.items():
        R_g1 = g1_data.xmat[body_id(g1_model, g1_name)].reshape(3, 3)
        R_lite = lite_data.xmat[body_id(lite_model, lite_name)].reshape(3, 3)
        out[lite_name] = R_g1.T @ R_lite
    return out


def step2_ik_refine(
    motion: dict[str, np.ndarray],
    step1_joint_pos: np.ndarray,
    g1_model: mujoco.MjModel,
    lite_model: mujoco.MjModel,
    lite_joint_addrs: np.ndarray,
    iters: int = 15,
    show_progress: bool = True,
) -> np.ndarray:
    """Refine step-1 joint positions to match G1's pelvis-local EE poses.

    Per-frame IK is independent — each frame seeds from step 1 and converges
    on its own. ``show_progress`` controls the inner tqdm bar; worker
    processes set it to False so their bars don't interleave in the terminal.
    """
    import mink

    g1_data = mujoco.MjData(g1_model)
    g1_addrs = np.asarray(
        [joint_qpos_addr(g1_model, n) for n in G1_JOINT_NAMES], dtype=np.int32
    )
    R_conv = _rest_frame_conversions(g1_model, lite_model)

    configuration = mink.Configuration(lite_model)
    ee_tasks = {
        name: mink.FrameTask(
            name, "body", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0
        )
        for name in IK_BODY_PAIRS
    }
    posture_task = mink.PostureTask(lite_model, cost=_posture_cost_vector(lite_model))
    all_tasks = [*ee_tasks.values(), posture_task]
    limits = [mink.ConfigurationLimit(lite_model)]

    out = step1_joint_pos.copy()
    seed_qpos = np.zeros(lite_model.nq, dtype=np.float64)
    frames = step1_joint_pos.shape[0]
    frame_iter = tqdm(range(frames), desc="    IK", leave=False, unit="frame") if show_progress else range(frames)

    for t in frame_iter:
        g1_data.qpos[g1_addrs] = motion["g1_joint_pos"][t]
        mujoco.mj_kinematics(g1_model, g1_data)
        for lite_name, g1_name in IK_BODY_PAIRS.items():
            bid = body_id(g1_model, g1_name)
            mat = np.eye(4)
            mat[:3, :3] = g1_data.xmat[bid].reshape(3, 3) @ R_conv[lite_name]
            mat[:3, 3] = g1_data.xpos[bid]
            ee_tasks[lite_name].set_target(mink.SE3.from_matrix(mat))

        seed_qpos[lite_joint_addrs] = step1_joint_pos[t]
        configuration.q[:] = seed_qpos
        posture_task.set_target(seed_qpos.copy())
        for _ in range(iters):
            vel = mink.solve_ik(
                configuration, all_tasks, 1.0, solver="daqp", damping=1e-1, limits=limits,
            )
            configuration.integrate_inplace(vel, 1.0)
        out[t] = configuration.q[lite_joint_addrs]

    return out


# ── Step 3: grounding ─────────────────────────────────────────────────────────


def ground_height(
    lite_model: mujoco.MjModel,
    lite_joint_addrs: np.ndarray,
    joint_pos: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
) -> float:
    """The pelvis-z shift that puts Lite's lowest sole point on the floor.

    A source places the base at its own height, AMASS at the floor, and Lite stands on
    shorter legs and thicker soles than the G1, so no constant offset grounds every clip.
    The shift is measured on the Lite result instead, in the median frame: a single
    frame would let one sole that dips through the floor lift the whole clip, and a
    foot stands on the floor in most frames of a clip.
    """
    heights = sole_heights(lite_model, lite_joint_addrs, joint_pos, base_pos, base_quat, FEET)
    return -float(np.median(heights))


# ── Validation ────────────────────────────────────────────────────────────────


def _rotation_angle_error(R_a: np.ndarray, R_b: np.ndarray) -> float:
    cos_theta = np.clip((np.trace(R_a @ R_b.T) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def validate_ee_tracking(
    motion: dict[str, np.ndarray],
    lite_joint_pos: np.ndarray,
    g1_model: mujoco.MjModel,
    lite_model: mujoco.MjModel,
    lite_joint_addrs: np.ndarray,
) -> None:
    """Print per-EE position + rotation error vs. G1 at every SAMPLE_STRIDE frames.

    Rotation error is measured as the angle of "motion delta from matched
    rest" — each robot's EE rotation relative to its own matched rest.
    Lite's matched rest is step 1 applied at G1 ``q = 0`` (arms at offsets).
    """
    g1_data = mujoco.MjData(g1_model)
    lite_data = mujoco.MjData(lite_model)
    g1_addrs = np.asarray(
        [joint_qpos_addr(g1_model, n) for n in G1_JOINT_NAMES], dtype=np.int32
    )

    mujoco.mj_kinematics(g1_model, g1_data)
    rest_g1 = {
        name: g1_data.xmat[body_id(g1_model, name)].reshape(3, 3).copy()
        for name in IK_BODY_PAIRS.values()
    }

    matched_rest = np.zeros(lite_model.nq, dtype=np.float64)
    for _, (lite_name, _, offset) in G1_TO_LITE.items():
        matched_rest[joint_qpos_addr(lite_model, lite_name)] = offset
    lite_data.qpos[:] = matched_rest
    mujoco.mj_kinematics(lite_model, lite_data)
    rest_lite = {
        name: lite_data.xmat[body_id(lite_model, name)].reshape(3, 3).copy()
        for name in IK_BODY_PAIRS
    }

    frames = lite_joint_pos.shape[0]
    indices = list(range(0, frames, SAMPLE_STRIDE))
    stats: dict[str, dict[str, list[float]]] = {
        name: {"pos": [], "rot": []} for name in IK_BODY_PAIRS
    }

    for t in indices:
        lite_data.qpos[lite_joint_addrs] = lite_joint_pos[t]
        mujoco.mj_kinematics(lite_model, lite_data)
        g1_data.qpos[g1_addrs] = motion["g1_joint_pos"][t]
        mujoco.mj_kinematics(g1_model, g1_data)
        for lite_name, g1_name in IK_BODY_PAIRS.items():
            lite_body = body_id(lite_model, lite_name)
            g1_body = body_id(g1_model, g1_name)
            stats[lite_name]["pos"].append(
                float(np.linalg.norm(lite_data.xpos[lite_body] - g1_data.xpos[g1_body]))
            )
            R_lite = lite_data.xmat[lite_body].reshape(3, 3) @ rest_lite[lite_name].T
            R_g1 = g1_data.xmat[g1_body].reshape(3, 3) @ rest_g1[g1_name].T
            stats[lite_name]["rot"].append(_rotation_angle_error(R_lite, R_g1))

    print(f"\nEE tracking error across {len(indices)} frames "
          f"(stride={SAMPLE_STRIDE}, total={frames}):")
    print(f"  {'body':<14s} {'pos mean':>9s} {'pos max':>9s}   "
          f"{'rot mean':>9s} {'rot max':>9s}")
    for lite_name in IK_BODY_PAIRS:
        pos = np.asarray(stats[lite_name]["pos"])
        rot = np.asarray(stats[lite_name]["rot"])
        print(
            f"  {lite_name:<14s} {pos.mean():>8.3f}m {pos.max():>8.3f}m   "
            f"{np.degrees(rot.mean()):>7.2f}° {np.degrees(rot.max()):>7.2f}°"
        )


# ── The models, compiled once per process ─────────────────────────────────────


class Models(NamedTuple):
    """Both robots, and the constants a retarget derives from them."""

    g1: mujoco.MjModel
    lite: mujoco.MjModel
    g1_addrs: np.ndarray
    joint_addrs: np.ndarray
    joint_names: tuple[str, ...]
    goal_conv: np.ndarray


def load_models() -> Models:
    """Compile both robots."""
    g1 = load_g1_model()
    lite = load_lite_model()
    names = lite_joint_names(lite)
    return Models(
        g1=g1,
        lite=lite,
        g1_addrs=np.asarray(
            [joint_qpos_addr(g1, n) for n in G1_JOINT_NAMES], dtype=np.int32
        ),
        joint_addrs=np.asarray(
            [joint_qpos_addr(lite, n) for n in names], dtype=np.int32
        ),
        joint_names=names,
        goal_conv=_goal_conversions(g1, lite),
    )


def _goal_conversions(
    g1_model: mujoco.MjModel, lite_model: mujoco.MjModel
) -> np.ndarray:
    """Per-body wxyz rotation from the G1 frame convention into Lite's, root body first.

    A G1 link and the Lite link it drives disagree about which way their axes point at
    rest, by up to 120 degrees. A goal written in the raw G1 convention is therefore not
    comparable to the link it is the goal for, and a tracking reward on orientation
    would never reach zero. This is the same conversion the IK aims through, so the goal
    a motion stores is the goal the IK was given.

    The root body is not an IK target, and its goal is the base trajectory, so it
    converts by the identity.
    """
    conversions = _rest_frame_conversions(g1_model, lite_model)
    quats = np.zeros((len(ROBOT_BODIES), 4))
    quats[0] = (1.0, 0.0, 0.0, 0.0)
    for index, lite_name in enumerate(IK_BODY_PAIRS, start=1):
        mujoco.mju_mat2Quat(quats[index], conversions[lite_name].reshape(9))
    return quats


# ── One clip, and the process that retargets it ───────────────────────────────


class Clip(NamedTuple):
    """One retargeted clip: the arrays the two layers of its motion are written from.

    The ``source_*`` arrays are the clip as published, at ``source_fps``. The rest are
    the retarget, at :data:`FPS`.
    """

    source_fps: int
    source_joint_pos: np.ndarray
    source_pos: np.ndarray
    source_quat: np.ndarray
    joint_pos: np.ndarray
    goal_pos: np.ndarray
    goal_quat: np.ndarray


def retarget_clip(
    models: Models,
    source: str,
    path: str,
    do_ik: bool,
    ik_iters: int,
    show_progress: bool = True,
) -> Clip:
    """Run every step on one clip of ``source``, and pose the bodies each layer carries.

    The base layer keeps the source untouched, at its own rate and in the frame
    convention of the G1. The robot layer carries the goal, which is the same G1 pose
    turned into the convention of the Lite link it drives, so it compares directly
    against that link.
    """
    clip, source_fps = SOURCES[source].load(Path(path))
    source_pos, source_quat = world_poses(
        models.g1, models.g1_addrs, clip["g1_joint_pos"],
        clip["base_pos"], clip["base_quat_wxyz"], tuple(IK_BODY_PAIRS.values()),
    )

    motion = resample_motion(clip, source_fps, FPS)
    step1 = step1_direct_remap(motion, models.joint_addrs, models.lite)
    joint_pos = step1["joint_pos"]
    if do_ik:
        joint_pos = step2_ik_refine(
            motion, joint_pos, models.g1, models.lite, models.joint_addrs,
            iters=ik_iters, show_progress=show_progress,
        )

    base_pos, base_quat = step1["base_pos"], step1["base_quat"]
    base_pos[:, 2] += ground_height(models.lite, models.joint_addrs, joint_pos, base_pos, base_quat)

    # The goal rides the base trajectory of the retarget, not that of the source, so it
    # sits where the Lite pelvis sits and the two are directly comparable.
    goal_pos, goal_quat = world_poses(
        models.g1, models.g1_addrs, motion["g1_joint_pos"],
        base_pos, base_quat, tuple(IK_BODY_PAIRS.values()),
    )
    return Clip(
        source_fps, clip["g1_joint_pos"], source_pos, source_quat,
        joint_pos, goal_pos, quat_mul(goal_quat, models.goal_conv).astype(np.float32),
    )


_MODELS: Models | None = None


def _worker_init() -> None:
    """Pool initializer: compile the models once per worker."""
    global _MODELS
    _MODELS = load_models()


def write_motion(models: Models, name: str, clip: Clip, urdf: Path, geometry: bool) -> list[Path]:
    """Write the base and the robot layer of one motion under ``DATASET_ROOT``."""
    return [
        write_base_layer(
            DATASET_ROOT, name, clip.source_fps, G1_JOINT_NAMES,
            clip.source_joint_pos, SOURCE_BODIES, clip.source_pos, clip.source_quat,
        ),
        write_robot_layer(
            DATASET_ROOT, name, urdf, FPS, models.joint_names, clip.joint_pos,
            ROBOT_BODIES, SOURCE_BODIES, clip.goal_pos, clip.goal_quat, geometry,
        ),
    ]


def _worker_retarget(job: tuple[str, str, str, bool, int, str, bool]) -> tuple[list[str], str | None]:
    """Retarget one clip in a worker, write its layers, and return their paths.

    A motion has files of its own, so no two workers write one file, and writing here
    keeps the parent from being the bottleneck: a short clip takes longer to write than
    its share of the IK. A clip that fails returns its error in place of paths, so one
    bad file among thousands does not stop the rest.
    """
    assert _MODELS is not None
    source, path, name, ik, ik_iters, urdf, geometry = job
    try:
        clip = retarget_clip(_MODELS, source, path, ik, ik_iters, show_progress=False)
        written = write_motion(_MODELS, name, clip, Path(urdf), geometry)
    except Exception as error:  # noqa: BLE001 -- the parent lists every failed clip
        return [], f"{path}: {type(error).__name__}: {error}"
    return [str(path) for path in written], None


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(
    source: str = "lafan1",
    clip: str | None = None,
    ik: bool = True,
    ik_iters: int = 15,
    workers: int = 1,
    validate_only: bool = False,
) -> None:
    """Retarget the G1 clips of a source to Lite, and write the two layers of each motion.

    Args:
        source: One of the keys of ``sources.SOURCES``.
        clip: A regex. Retarget the clips whose path below the clip directory of the
            source matches it, as in ``CMU/07/07_01_poses_120_jpos`` for AMASS.
        ik: Refine step 1 with the step-2 IK. Set False to keep step 1 only.
        ik_iters: Newton steps per frame in step 2.
        workers: Processes to retarget clips in parallel. ``-1`` uses every core.
        validate_only: Retarget the first matching clip, print the end-effector
            error table, and write nothing.
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
            f"Run scripts/download.py --source {source} first."
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
        clip_motion, source_fps = spec.load(paths[0])
        motion = resample_motion(clip_motion, source_fps, FPS)
        step1 = step1_direct_remap(motion, models.joint_addrs, models.lite)
        print(f"\nClip: {paths[0].name} ({step1['joint_pos'].shape[0]} frames, "
              f"{source_fps} Hz source)")
        print("\n=== Step 1 (direct copy with sign + offset) ===")
        validate_ee_tracking(
            motion, step1["joint_pos"], models.g1, models.lite, models.joint_addrs
        )
        if ik:
            step2 = step2_ik_refine(
                motion, step1["joint_pos"], models.g1, models.lite, models.joint_addrs,
                iters=ik_iters,
            )
            print("\n=== Step 1 + Step 2 (per-frame IK refinement) ===")
            validate_ee_tracking(
                motion, step2, models.g1, models.lite, models.joint_addrs
            )
        return

    urdf = lite_urdf_path()
    blueprint = write_blueprint(DATASET_ROOT, urdf)
    asset = write_asset(DATASET_ROOT, urdf)
    written = [blueprint, asset]
    # The Hub card links each published motion into the web viewer, which cannot load the
    # asset, so a published motion carries its own geometry.
    geometry = source == PUBLIC_SOURCE

    failed = []
    if workers > 1:
        jobs = [
            (source, str(path), name, ik, ik_iters, str(urdf), geometry)
            for path, name in zip(paths, names, strict=True)
        ]
        # Spawn, not fork: the parent runs threads by now, and a forked child can inherit a
        # held lock. ProcessPoolExecutor stops replacing workers that reach their task limit
        # once many jobs are queued (CPython 3.13), so this uses multiprocessing.Pool.
        with get_context("spawn").Pool(workers, _worker_init, maxtasksperchild=WORKER_CLIPS) as pool:
            results = pool.imap(_worker_retarget, jobs)
            for files, error in tqdm(
                results, total=len(jobs), desc=f"Clips (workers={workers})", unit="clip"
            ):
                written += [Path(path) for path in files]
                if error is not None:
                    failed.append(error)
        if failed:
            print(f"\n{len(failed)} of {len(jobs)} clips failed and were skipped:")
            for error in failed:
                print(f"  {error}")
    else:
        for path, name in tqdm(zip(paths, names, strict=True), total=len(names), desc="Clips",
                               unit="clip"):
            result = retarget_clip(models, source, str(path), ik, ik_iters)
            written += write_motion(models, name, result, urdf, geometry)

    total_mb = sum(path.stat().st_size for path in written) / 1e6
    print(f"\nWrote {len(names) - len(failed)} motions as {len(written)} files "
          f"({total_mb:.1f} MB) under {DATASET_ROOT}")
    # The blueprint goes last: blueprints do not merge, and the last one loaded wins.
    print(f"  view: uv run rerun {DATASET_ROOT}/*/{names[0]}.rrd --asset {asset} {blueprint}")


if __name__ == "__main__":
    tyro.cli(main)
