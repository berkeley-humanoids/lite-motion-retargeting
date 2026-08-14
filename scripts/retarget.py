"""Retarget LAFAN1 G1 motions to the Berkeley Lite humanoid.

Two-step pipeline per clip:

* **Step 1** — direct joint copy ``lite_q = sign * g1_q + offset`` using the
  static :data:`common.G1_TO_LITE` table. Adds a constant pelvis-z shift so
  Lite's feet stand on the ground.
* **Step 2** — per-frame ``mink`` IK that refines step 1 to match G1's
  pelvis-local EE poses (feet + hands, position + orientation). The
  per-DOF posture cost biases the solution toward step 1 so the refinement
  is a *tweak*, not a rearrangement.

One clip becomes one Rerun ``.rrd`` motion under ``dataset/tracking/``, which
``--push`` uploads to the Hub. ``scripts/rrd.py`` defines that file.

Usage:
    uv run scripts/retarget.py
    uv run scripts/retarget.py --clip 'walk1_subject1'
    uv run scripts/retarget.py --validate-only
    uv run scripts/retarget.py --workers -1 --push   # every core, then upload
"""

import os
import re
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
    finite_diff,
    joint_qpos_addr,
    lite_joint_names,
    load_g1_model,
    load_lafan_csv,
    load_lite_model,
)
from rrd import LAYER_TRACKING, write_motion  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
LAFAN_ROOT: Path = REPO_ROOT / ".cache" / "lafan1_g1"
DATASET_ROOT: Path = REPO_ROOT / "dataset"

#: The asset the motion drives, which also prefixes every file name.
ROBOT_TYPE: str = "lite_pro"

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


# ── Per-clip writer ───────────────────────────────────────────────────────────


def write_clip(
    clip: str,
    joint_names: tuple[str, ...],
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    joint_pos: np.ndarray,
    ik: bool,
    ik_iters: int,
    dataset_root: Path,
) -> Path:
    """Derive the velocities of one clip and write it as a motion."""
    base_pos = base_pos.astype(np.float32, copy=False)
    base_quat = base_quat.astype(np.float32, copy=False)
    joint_pos = joint_pos.astype(np.float32, copy=False)
    return write_motion(
        dataset_root / LAYER_TRACKING / f"{ROBOT_TYPE}_lafan1_{clip}.rrd",
        joint_names=joint_names,
        fps=FPS,
        base_pos=base_pos,
        base_quat=base_quat,
        base_lin_vel=finite_diff(base_pos, FPS),
        base_ang_vel=angular_velocity_from_quat(base_quat, FPS).astype(np.float32),
        joint_pos=joint_pos,
        joint_vel=finite_diff(joint_pos, FPS),
        task=LITE_TASK_NAME,
        source={
            "retargeted_from": "lvhaidong/LAFAN1_Retargeting_Dataset",
            "clip": clip,
            "source_robot": "unitree_g1",
            "source_rate_hz": FPS,
            "pipeline": "Berkeley-Humanoids/Lite-Motion-Retargeting",
            "ik_refined": ik,
            "ik_iters": ik_iters if ik else 0,
        },
        robot_type=ROBOT_TYPE,
    )


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
    push: bool = False,
) -> None:
    """Retarget LAFAN1 G1 clips to Lite, and write one .rrd motion per clip.

    Args:
        repo_id: The Hub dataset repo that ``--push`` uploads to.
        clip: A regex. Retarget the clips whose name matches it.
        ik: Refine step 1 with the step-2 IK. Set False to keep step 1 only.
        ik_iters: Newton steps per frame in step 2.
        workers: Processes to retarget clips in parallel. ``-1`` uses every core.
        validate_only: Retarget the first matching clip, print the end-effector
            error table, and write nothing.
        push: Upload the motions to ``repo_id`` after the last clip.
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

    written = []
    if workers > 1:
        args_list = [(str(p), bool(ik), int(ik_iters)) for p in csvs]
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
            for csv_path, (base_pos, base_quat, joint_pos) in tqdm(
                zip(csvs, executor.map(_worker_retarget, args_list, chunksize=1)),
                total=len(csvs), desc=f"Clips (workers={workers})", unit="clip",
            ):
                written.append(write_clip(
                    csv_path.stem, lite_jnames, base_pos, base_quat, joint_pos,
                    ik, ik_iters, DATASET_ROOT,
                ))
    else:
        for csv_path in tqdm(csvs, desc="Clips", unit="clip"):
            motion = load_lafan_csv(csv_path)
            step1 = step1_direct_remap(motion, lite_joint_addrs, lite_model, z_offset)
            joint_pos = step1["joint_pos"]
            if ik:
                joint_pos = step2_ik_refine(
                    motion, joint_pos, g1_model, lite_model, lite_joint_addrs, iters=ik_iters,
                )
            written.append(write_clip(
                csv_path.stem, lite_jnames, step1["base_pos"], step1["base_quat"],
                joint_pos, ik, ik_iters, DATASET_ROOT,
            ))

    total_mb = sum(p.stat().st_size for p in written) / 1e6
    print(f"\nWrote {len(written)} motions to {DATASET_ROOT / LAYER_TRACKING} "
          f"({total_mb:.1f} MB)")

    if push:
        from huggingface_hub import HfApi  # deferred: only the upload needs it

        HfApi().upload_folder(
            folder_path=str(DATASET_ROOT),
            path_in_repo=".",
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["*.rrd"],
            commit_message=f"Retarget {len(written)} LAFAN1 clips",
        )
        print(f"Pushed to {repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
