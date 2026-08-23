"""Retarget LAFAN1 G1 motions to the Berkeley Lite humanoid.

Two-step pipeline per clip:

* **Step 1** — direct joint copy ``lite_q = sign * g1_q + offset`` using the
  static :data:`common.G1_TO_LITE` table. Adds a constant pelvis-z shift so
  Lite's feet stand on the ground.
* **Step 2** — per-frame ``mink`` IK that refines step 1 to match G1's
  pelvis-local EE poses (feet + hands, position + orientation). The
  per-DOF posture cost biases the solution toward step 1 so the refinement
  is a *tweak*, not a rearrangement.

One clip becomes two layers of one motion in ``dataset/``: ``base/``, the source clip
as published at 30 Hz, and ``lite_pro/``, at the 50 Hz control rate, holding the joints
the IK reached and the goal it aimed at. ``scripts/store.py`` defines that store, and
``scripts/publish.py`` uploads it.

Usage:
    uv run scripts/retarget.py
    uv run scripts/retarget.py --clip 'walk1_subject1'
    uv run scripts/retarget.py --validate-only
    uv run scripts/retarget.py --workers -1          # every clip, every core
"""

import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np
import tyro
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FPS,
    G1_LAFAN_JOINT_NAMES,
    G1_TO_LITE,
    IK_BODY_PAIRS,
    ROBOT_BODIES,
    SOURCE_BODIES,
    SOURCE_FPS,
    body_id,
    joint_qpos_addr,
    lite_joint_names,
    lite_urdf_path,
    load_g1_model,
    load_lafan_csv,
    load_lite_model,
    quat_mul,
    resample_motion,
    world_poses,
)
from store import write_base_layer, write_blueprint, write_robot_layer  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
LAFAN_ROOT: Path = REPO_ROOT / ".cache" / "lafan1_g1"
DATASET_ROOT: Path = REPO_ROOT / "dataset"

#: Prefixes every motion name, so one store can hold a second source later.
MOTION_PREFIX: str = "lafan1"

# Validation frame stride for the EE-error summary.
SAMPLE_STRIDE: int = 50


# ── Step 1: direct copy with sign + offset ────────────────────────────────────


def _build_remap_indices(
    lite_model: mujoco.MjModel,
    lite_joint_addrs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parallel arrays ``(lite_col, g1_col, sign, offset)`` over mapped joints."""
    addr_to_col = {int(a): i for i, a in enumerate(lite_joint_addrs.tolist())}
    lite_cols, g1_cols, signs, offsets = [], [], [], []
    for g1_col, g1_name in enumerate(G1_LAFAN_JOINT_NAMES):
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


def _pelvis_z_offset(g1_model: mujoco.MjModel, lite_model: mujoco.MjModel) -> float:
    """Difference in standing leg length between Lite and G1 (welded pelvis).

    Both robots default with the pelvis at z=0 and feet hanging at negative z,
    so ``-min(foot_z)`` is each robot's standing leg length; the difference
    is the z-shift to apply to LAFAN1's base trajectory so Lite stays grounded.
    """

    def _leg_length(model: mujoco.MjModel, foot_names: tuple[str, str]) -> float:
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        return -min(float(data.xpos[body_id(model, n), 2]) for n in foot_names)

    feet = ("left_foot", "right_foot")
    return (_leg_length(lite_model, feet)
            - _leg_length(g1_model, tuple(IK_BODY_PAIRS[foot] for foot in feet)))


def step1_direct_remap(
    motion: dict[str, np.ndarray],
    lite_joint_addrs: np.ndarray,
    lite_model: mujoco.MjModel,
    z_offset: float,
) -> dict[str, np.ndarray]:
    """Apply ``lite_q = sign * g1_q + offset`` to every mapped joint.

    Returns ``base_pos`` (with the pelvis-z shift), ``base_quat`` (WXYZ,
    unchanged from the source), and a ``(T, 74)`` ``joint_pos`` in Lite MJCF
    order. Joints with no G1 source (neck, fingers, ankle_yaw) stay at 0.
    """
    lite_cols, g1_cols, signs, offsets = _build_remap_indices(lite_model, lite_joint_addrs)
    frames = motion["base_pos"].shape[0]
    joint_pos = np.zeros((frames, lite_joint_addrs.shape[0]), dtype=np.float32)
    joint_pos[:, lite_cols] = signs * motion["g1_joint_pos"][:, g1_cols] + offsets

    base_pos = motion["base_pos"].astype(np.float32, copy=True)
    base_pos[:, 2] += z_offset
    return {
        "base_pos": base_pos,
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
        [joint_qpos_addr(g1_model, n) for n in G1_LAFAN_JOINT_NAMES], dtype=np.int32
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
        [joint_qpos_addr(g1_model, n) for n in G1_LAFAN_JOINT_NAMES], dtype=np.int32
    )

    mujoco.mj_kinematics(g1_model, g1_data)
    pairs = IK_BODY_PAIRS.items()
    rest_g1 = {g: g1_data.xmat[body_id(g1_model, g)].reshape(3, 3).copy() for _, g in pairs}

    matched_rest = np.zeros(lite_model.nq, dtype=np.float64)
    for _, (lite_name, _, offset) in G1_TO_LITE.items():
        matched_rest[joint_qpos_addr(lite_model, lite_name)] = offset
    lite_data.qpos[:] = matched_rest
    mujoco.mj_kinematics(lite_model, lite_data)
    rest_lite = {l: lite_data.xmat[body_id(lite_model, l)].reshape(3, 3).copy() for l, _ in pairs}

    frames = lite_joint_pos.shape[0]
    indices = list(range(0, frames, SAMPLE_STRIDE))
    stats: dict[str, dict[str, list[float]]] = {l: {"pos": [], "rot": []} for l, _ in pairs}

    for t in indices:
        lite_data.qpos[lite_joint_addrs] = lite_joint_pos[t]
        mujoco.mj_kinematics(lite_model, lite_data)
        g1_data.qpos[g1_addrs] = motion["g1_joint_pos"][t]
        mujoco.mj_kinematics(g1_model, g1_data)
        for lname, gname in pairs:
            lbid, gbid = body_id(lite_model, lname), body_id(g1_model, gname)
            stats[lname]["pos"].append(float(np.linalg.norm(lite_data.xpos[lbid] - g1_data.xpos[gbid])))
            R_l = lite_data.xmat[lbid].reshape(3, 3) @ rest_lite[lname].T
            R_g = g1_data.xmat[gbid].reshape(3, 3) @ rest_g1[gname].T
            stats[lname]["rot"].append(_rotation_angle_error(R_l, R_g))

    print(f"\nEE tracking error across {len(indices)} frames (stride={SAMPLE_STRIDE}, total={frames}):")
    print(f"  {'body':<14s} {'pos mean':>9s} {'pos max':>9s}   {'rot mean':>9s} {'rot max':>9s}")
    for lname, _ in pairs:
        pos = np.asarray(stats[lname]["pos"])
        rot = np.asarray(stats[lname]["rot"])
        print(
            f"  {lname:<14s} {pos.mean():>8.3f}m {pos.max():>8.3f}m   "
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
    z_offset: float
    goal_conv: np.ndarray


def load_models() -> Models:
    """Compile both robots and measure the pelvis shift between them."""
    g1 = load_g1_model(LAFAN_ROOT)
    lite = load_lite_model()
    names = lite_joint_names(lite)
    return Models(
        g1=g1,
        lite=lite,
        g1_addrs=np.asarray(
            [joint_qpos_addr(g1, n) for n in G1_LAFAN_JOINT_NAMES], dtype=np.int32
        ),
        joint_addrs=np.asarray(
            [joint_qpos_addr(lite, n) for n in names], dtype=np.int32
        ),
        joint_names=names,
        z_offset=_pelvis_z_offset(g1, lite),
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

    The ``source_*`` arrays are the clip as published, at :data:`SOURCE_FPS`. The rest
    are the retarget, at :data:`FPS`.
    """

    source_joint_pos: np.ndarray
    source_pos: np.ndarray
    source_quat: np.ndarray
    joint_pos: np.ndarray
    goal_pos: np.ndarray
    goal_quat: np.ndarray


def retarget_clip(
    models: Models,
    csv_path: str,
    do_ik: bool,
    ik_iters: int,
    show_progress: bool = True,
) -> Clip:
    """Run both steps on one clip, and pose the bodies each layer carries.

    The base layer keeps the source untouched, at its own rate and in the frame
    convention of the G1. The robot layer carries the goal, which is the same G1 pose
    turned into the convention of the Lite link it drives, so it compares directly
    against that link.
    """
    source = load_lafan_csv(Path(csv_path))
    source_pos, source_quat = world_poses(
        models.g1, models.g1_addrs, source["g1_joint_pos"],
        source["base_pos"], source["base_quat_wxyz"], tuple(IK_BODY_PAIRS.values()),
    )

    motion = resample_motion(source, SOURCE_FPS, FPS)
    step1 = step1_direct_remap(motion, models.joint_addrs, models.lite, models.z_offset)
    joint_pos = step1["joint_pos"]
    if do_ik:
        joint_pos = step2_ik_refine(
            motion, joint_pos, models.g1, models.lite, models.joint_addrs,
            iters=ik_iters, show_progress=show_progress,
        )

    # The goal rides the base trajectory of the retarget, not that of the source, so it
    # sits where the Lite pelvis sits and the two are directly comparable.
    goal_pos, goal_quat = world_poses(
        models.g1, models.g1_addrs, motion["g1_joint_pos"],
        step1["base_pos"], step1["base_quat"], tuple(IK_BODY_PAIRS.values()),
    )
    return Clip(
        source["g1_joint_pos"], source_pos, source_quat,
        joint_pos, goal_pos, quat_mul(goal_quat, models.goal_conv).astype(np.float32),
    )


_MODELS: Models | None = None


def _worker_init() -> None:
    """ProcessPoolExecutor initializer: compile the models once per worker."""
    global _MODELS
    _MODELS = load_models()


def _worker_retarget(job: tuple[str, bool, int]) -> Clip:
    """Retarget one clip in a worker. The parent writes, so the store has one writer."""
    assert _MODELS is not None
    return retarget_clip(_MODELS, *job, show_progress=False)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(
    clip: str | None = None,
    ik: bool = True,
    ik_iters: int = 15,
    workers: int = 1,
    validate_only: bool = False,
) -> None:
    """Retarget LAFAN1 G1 clips to Lite, and write the two layers of each motion.

    Args:
        clip: A regex. Retarget the clips whose name matches it.
        ik: Refine step 1 with the step-2 IK. Set False to keep step 1 only.
        ik_iters: Newton steps per frame in step 2.
        workers: Processes to retarget clips in parallel. ``-1`` uses every core.
        validate_only: Retarget the first matching clip, print the end-effector
            error table, and write nothing.
    """
    if workers == -1:
        workers = os.cpu_count() or 1

    csvs = sorted((LAFAN_ROOT / "g1").glob("*.csv"))
    if clip is not None:
        csvs = [p for p in csvs if re.search(clip, p.stem)]
    if not csvs:
        raise SystemExit(
            f"No CSVs to retarget under {LAFAN_ROOT / 'g1'} (clip={clip!r}). "
            f"Run scripts/download_lafan.py first."
        )

    print("Loading models …")
    models = load_models()
    flipped = sum(1 for _, s, _ in G1_TO_LITE.values() if s < 0)
    nonzero = sum(1 for _, _, off in G1_TO_LITE.values() if abs(off) > 1e-6)
    print(f"  G1 nq={models.g1.nq}, Lite nq={models.lite.nq}, "
          f"joints={len(models.joint_names)}")
    print(
        f"  {len(G1_TO_LITE)} joint pairs, {flipped} sign flips, "
        f"{nonzero} nonzero offsets, pelvis z-offset={models.z_offset * 1000:.2f} mm"
    )
    print(f"  resampling {SOURCE_FPS} Hz -> {FPS} Hz")

    if validate_only:
        motion = resample_motion(load_lafan_csv(csvs[0]), SOURCE_FPS, FPS)
        step1 = step1_direct_remap(
            motion, models.joint_addrs, models.lite, models.z_offset
        )
        print(f"\nClip: {csvs[0].name} ({step1['joint_pos'].shape[0]} frames)")
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
    written = [blueprint]
    jobs = [(str(path), ik, ik_iters) for path in csvs]

    def write_all(clips, desc: str) -> None:
        for csv_path, result in tqdm(
            zip(csvs, clips), total=len(csvs), desc=desc, unit="clip"
        ):
            name = f"{MOTION_PREFIX}_{csv_path.stem}"
            written.append(write_base_layer(
                DATASET_ROOT, name, SOURCE_FPS, G1_LAFAN_JOINT_NAMES,
                result.source_joint_pos, SOURCE_BODIES,
                result.source_pos, result.source_quat,
            ))
            written.append(write_robot_layer(
                DATASET_ROOT, name, urdf, FPS, models.joint_names, result.joint_pos,
                ROBOT_BODIES, SOURCE_BODIES, result.goal_pos, result.goal_quat,
            ))

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
            write_all(
                pool.map(_worker_retarget, jobs, chunksize=1),
                f"Clips (workers={workers})",
            )
    else:
        write_all((retarget_clip(models, *job) for job in jobs), "Clips")

    total_mb = sum(path.stat().st_size for path in written) / 1e6
    print(f"\nWrote {len(csvs)} motions as {len(written)} files "
          f"({total_mb:.1f} MB) under {DATASET_ROOT}")
    # The blueprint goes last: blueprints do not merge, and the last one loaded wins.
    print(f"  view: uv run rerun {DATASET_ROOT}/*/{MOTION_PREFIX}_{csvs[0].stem}.rrd "
          f"{blueprint}")


if __name__ == "__main__":
    tyro.cli(main)
