"""Retarget LAFAN1 G1 motions to the Berkeley Lite humanoid.

Two-step pipeline per clip:

* **Step 1** — direct joint copy ``lite_q = sign * g1_q + offset`` using the
  static :data:`common.G1_TO_LITE` table. Adds a constant pelvis-z shift so
  Lite's feet stand on the ground.
* **Step 2** — per-frame ``mink`` IK that refines step 1 to match G1's
  pelvis-local EE poses (feet + hands, position + orientation). The
  per-DOF posture cost biases the solution toward step 1 so the refinement
  is a *tweak*, not a rearrangement.

Output is a HuggingFace LeRobotDataset written at the repository root.

Usage:
    uv run scripts/retarget.py
    uv run scripts/retarget.py --clip 'walk1_subject1'
    uv run scripts/retarget.py --validate-only
    uv run scripts/retarget.py --workers 8        # parallel across clips
    uv run scripts/retarget.py --workers -1       # use all CPU cores
"""

import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import tyro
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FPS,
    G1_FOOT_BODIES,
    G1_HAND_BODIES,
    G1_LAFAN_JOINT_NAMES,
    G1_TO_LITE,
    LITE_DATASET_REPO_ID,
    LITE_FOOT_BODIES,
    LITE_HAND_BODIES,
    LITE_TASK_NAME,
    angular_velocity_from_quat,
    body_id,
    dataset_features,
    finite_diff,
    joint_qpos_addr,
    lite_joint_names,
    load_g1_model,
    load_lafan_csv,
    load_lite_model,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
LAFAN_ROOT: Path = REPO_ROOT / ".cache" / "lafan1_g1"
BUILD_ROOT: Path = REPO_ROOT / ".lerobot_build"
# LeRobotDataset.create refuses an existing destination, so the writer drops
# its meta/ + data/ tree into BUILD_ROOT and we move it up to REPO_ROOT on
# finalize.

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

    return _leg_length(lite_model, LITE_FOOT_BODIES) - _leg_length(g1_model, G1_FOOT_BODIES)


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
    for lite_name, g1_name in zip(
        (*LITE_FOOT_BODIES, *LITE_HAND_BODIES),
        (*G1_FOOT_BODIES, *G1_HAND_BODIES),
        strict=True,
    ):
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
    foot_tasks = [
        mink.FrameTask(name, "body", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
        for name in LITE_FOOT_BODIES
    ]
    hand_tasks = [
        mink.FrameTask(name, "body", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
        for name in LITE_HAND_BODIES
    ]
    posture_task = mink.PostureTask(lite_model, cost=_posture_cost_vector(lite_model))
    all_tasks = [*foot_tasks, *hand_tasks, posture_task]
    limits = [mink.ConfigurationLimit(lite_model)]

    ee_pairs = tuple(zip(
        (*LITE_FOOT_BODIES, *LITE_HAND_BODIES),
        (*G1_FOOT_BODIES, *G1_HAND_BODIES),
        strict=True,
    ))
    out = step1_joint_pos.copy()
    seed_qpos = np.zeros(lite_model.nq, dtype=np.float64)
    frames = step1_joint_pos.shape[0]
    frame_iter = tqdm(range(frames), desc="    IK", leave=False, unit="frame") if show_progress else range(frames)

    for t in frame_iter:
        g1_data.qpos[g1_addrs] = motion["g1_joint_pos"][t]
        mujoco.mj_kinematics(g1_model, g1_data)
        for task, (lite_name, g1_name) in zip([*foot_tasks, *hand_tasks], ee_pairs, strict=True):
            bid = body_id(g1_model, g1_name)
            mat = np.eye(4)
            mat[:3, :3] = g1_data.xmat[bid].reshape(3, 3) @ R_conv[lite_name]
            mat[:3, 3] = g1_data.xpos[bid]
            task.set_target(mink.SE3.from_matrix(mat))

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
    pairs = tuple(zip(
        (*LITE_FOOT_BODIES, *LITE_HAND_BODIES),
        (*G1_FOOT_BODIES, *G1_HAND_BODIES),
        strict=True,
    ))
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


# ── Per-clip pipeline + LeRobotDataset writer ─────────────────────────────────


def _frame_records(
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    joint_pos: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compose the six dataset-feature arrays from a per-clip trajectory."""
    base_pos = base_pos.astype(np.float32, copy=False)
    base_quat = base_quat.astype(np.float32, copy=False)
    joint_pos = joint_pos.astype(np.float32, copy=False)
    return {
        "base_pos": base_pos,
        "base_quat": base_quat,
        "base_lin_vel": finite_diff(base_pos, FPS),
        "base_ang_vel": angular_velocity_from_quat(base_quat, FPS).astype(np.float32),
        "joint_pos": joint_pos,
        "joint_vel": finite_diff(joint_pos, FPS),
    }


# ── Multiprocess worker (across-clip parallelism) ─────────────────────────────

_WORKER_STATE: dict[str, object] = {}


def _worker_init() -> None:
    """ProcessPoolExecutor initializer: compile MuJoCo models once per worker."""
    g1_model = load_g1_model(LAFAN_ROOT)
    lite_model = load_lite_model()
    addrs = np.asarray(
        [joint_qpos_addr(lite_model, n) for n in lite_joint_names(lite_model)], dtype=np.int32
    )
    _WORKER_STATE.update(
        g1_model=g1_model,
        lite_model=lite_model,
        lite_joint_addrs=addrs,
        z_offset=_pelvis_z_offset(g1_model, lite_model),
    )


def _worker_retarget(args: tuple[str, bool, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retarget a single clip; returns ``(base_pos, base_quat, joint_pos)``."""
    csv_path_str, do_ik, ik_iters = args
    g1_model: mujoco.MjModel = _WORKER_STATE["g1_model"]  # type: ignore[assignment]
    lite_model: mujoco.MjModel = _WORKER_STATE["lite_model"]  # type: ignore[assignment]
    lite_joint_addrs: np.ndarray = _WORKER_STATE["lite_joint_addrs"]  # type: ignore[assignment]
    z_offset: float = _WORKER_STATE["z_offset"]  # type: ignore[assignment]

    motion = load_lafan_csv(Path(csv_path_str))
    step1 = step1_direct_remap(motion, lite_joint_addrs, lite_model, z_offset)
    joint_pos = step1["joint_pos"]
    if do_ik:
        joint_pos = step2_ik_refine(
            motion, joint_pos, g1_model, lite_model, lite_joint_addrs,
            iters=ik_iters, show_progress=False,
        )
    return step1["base_pos"], step1["base_quat"], joint_pos


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(
    repo_id: str = LITE_DATASET_REPO_ID,
    clip: str | None = None,
    ik: bool = True,
    ik_iters: int = 15,
    workers: int = 1,
    validate_only: bool = False,
) -> None:
    """Retarget LAFAN1 G1 clips to Lite and write a LeRobotDataset.

    Args:
        repo_id: HF dataset repo id, recorded in dataset metadata.
        clip: Optional regex to retarget only matching CSVs.
        ik: If True, run step 2 IK to refine step 1. If False, output step 1 only.
        ik_iters: Newton-step iterations per frame in step 2.
        workers: Worker processes for across-clip parallelism. ``1`` (default)
            keeps the sequential path with per-frame tqdm. ``-1`` uses every
            CPU core. ``>1`` spawns a ``ProcessPoolExecutor`` and suppresses
            inner tqdm bars to keep the terminal readable.
        validate_only: Run on the first matching clip and stop without writing
            the dataset. Prints the step-1 (and step-2 if ``ik=True``) EE error
            table.
    """
    if workers == -1:
        workers = os.cpu_count() or 1
    # Each save_episode runs an HFDataset.map pass that prints its own bar
    # — 218 of those interleave badly with our outer clip bar.
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

    csvs = sorted((LAFAN_ROOT / "g1").glob("*.csv"))
    if clip is not None:
        csvs = [p for p in csvs if re.search(clip, p.stem)]
    if not csvs:
        raise SystemExit(
            f"No CSVs to retarget under {LAFAN_ROOT / 'g1'} (clip={clip!r}). "
            f"Run scripts/download_lafan.py first."
        )

    print("Loading models …")
    g1_model = load_g1_model(LAFAN_ROOT)
    lite_model = load_lite_model()
    lite_jnames = lite_joint_names(lite_model)
    lite_joint_addrs = np.asarray(
        [joint_qpos_addr(lite_model, n) for n in lite_jnames], dtype=np.int32
    )
    z_offset = _pelvis_z_offset(g1_model, lite_model)
    flipped = sum(1 for _, s, _ in G1_TO_LITE.values() if s < 0)
    nonzero = sum(1 for _, _, off in G1_TO_LITE.values() if abs(off) > 1e-6)
    print(f"  G1 nq={g1_model.nq}, Lite nq={lite_model.nq}, joints={len(lite_jnames)}")
    print(
        f"  {len(G1_TO_LITE)} joint pairs, {flipped} sign flips, "
        f"{nonzero} nonzero offsets, pelvis z-offset={z_offset * 1000:.2f} mm"
    )

    if validate_only:
        motion = load_lafan_csv(csvs[0])
        step1 = step1_direct_remap(motion, lite_joint_addrs, lite_model, z_offset)
        print(f"\nClip: {csvs[0].name} ({step1['joint_pos'].shape[0]} frames)")
        print("\n=== Step 1 (direct copy with sign + offset) ===")
        validate_ee_tracking(motion, step1["joint_pos"], g1_model, lite_model, lite_joint_addrs)
        if ik:
            step2 = step2_ik_refine(
                motion, step1["joint_pos"], g1_model, lite_model, lite_joint_addrs,
                iters=ik_iters,
            )
            print("\n=== Step 1 + Step 2 (per-frame IK refinement) ===")
            validate_ee_tracking(motion, step2, g1_model, lite_model, lite_joint_addrs)
        return

    from lerobot.datasets import LeRobotDataset  # deferred: heavy import

    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=dataset_features(joint_count=len(lite_jnames)),
        root=BUILD_ROOT,
        robot_type="lite",
        use_videos=False,
    )

    def _write_clip(base_pos: np.ndarray, base_quat: np.ndarray, joint_pos: np.ndarray) -> None:
        records = _frame_records(base_pos, base_quat, joint_pos)
        for t in range(records["base_pos"].shape[0]):
            dataset.add_frame({"task": LITE_TASK_NAME, **{k: v[t] for k, v in records.items()}})
        dataset.save_episode()

    if workers > 1:
        args_list = [(str(p), bool(ik), int(ik_iters)) for p in csvs]
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
            for base_pos, base_quat, joint_pos in tqdm(
                executor.map(_worker_retarget, args_list, chunksize=1),
                total=len(csvs), desc=f"Clips (workers={workers})", unit="clip",
            ):
                _write_clip(base_pos, base_quat, joint_pos)
    else:
        for csv_path in tqdm(csvs, desc="Clips", unit="clip"):
            motion = load_lafan_csv(csv_path)
            step1 = step1_direct_remap(motion, lite_joint_addrs, lite_model, z_offset)
            joint_pos = step1["joint_pos"]
            if ik:
                joint_pos = step2_ik_refine(
                    motion, joint_pos, g1_model, lite_model, lite_joint_addrs, iters=ik_iters,
                )
            _write_clip(step1["base_pos"], step1["base_quat"], joint_pos)

    dataset.finalize()
    for sub in ("meta", "data"):
        src = BUILD_ROOT / sub
        if src.exists():
            dst = REPO_ROOT / sub
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
    shutil.rmtree(BUILD_ROOT, ignore_errors=True)
    print(f"\nWrote dataset to {REPO_ROOT} ({len(csvs)} episodes)")


if __name__ == "__main__":
    tyro.cli(main)
