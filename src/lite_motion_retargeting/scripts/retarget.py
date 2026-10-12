"""Retarget the clips of a source to Lite, and write two layers per motion into the store.

``retargeting/`` documents how a clip is retargeted. One clip becomes two layers of one
motion in ``dataset/``: ``base/``, the source clip as published at its own rate, and
``lite_pro/``, at the 50 Hz control rate, with the joints that the IK reached and the goal
that it aimed at. ``store.py`` defines the store, and ``uv run publish`` uploads it.

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

import numpy as np
import tyro
from tqdm import tqdm

from .. import CACHE_ROOT, DATASET_ROOT
from ..hub import hub_dataset_of
from ..retargeting import Clip, Models, load_models, retarget_clip
from ..retargeting.g1 import G1_TO_LITE
from ..robots import lite_urdf_path
from ..sources import SOURCES, clip_key, motion_name
from ..store import (
    FPS,
    write_asset,
    write_base_layer,
    write_blueprint,
    write_robot_layer,
)

#: The number of clips that a worker retargets before a new process replaces it. Rerun 0.38
#: keeps two threads alive for every recording that it opened, and a motion opens two. A
#: worker that wrote about 1,200 motions could not start a thread.
WORKER_CLIPS: int = 100


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


def retarget(
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
    # The card of a Hub dataset with geometry links each motion into the web viewer, which
    # cannot load the asset, so such a motion carries its own geometry.
    dataset = hub_dataset_of(source)
    geometry = dataset is not None and dataset.geometry

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


def main() -> None:
    """Run :func:`retarget` with the options on the command line."""
    tyro.cli(retarget)
