"""Measure the IK, UMR and ReActor on the same frames of one source, with one planted mask.

``docs/comparison.md`` explains the method and gives the results. Each method writes its
motions in its own form: ``uv run retarget`` the IK store, ``reactor export`` a store in
which a motion can be cut into pieces, and UMR a folder of ``<clip>_<robot>.npz``. This step
reads the base and the joints of each method, and measures them with the stance mask of the
source, on the frames of each ReActor piece. The tables pool every sample of every piece.

UMR scales the human by its height, not by the height of its pelvis as the IK does. Thus a
UMR link is not where the goal of the IK is, and this step leaves out the goal errors of
UMR. A joint that the source does not drive holds 0 in every method, as it does in the IK.
The step stops if a layer or a UMR run records another model of Lite than the installed one.

Usage:
    uv run compare --source lafan1bvh --umr ../umr/output/lite_pro
    uv run compare --source lafan1bvh --umr ../umr/output/lite_pro --reactor dataset_reactor
    uv run compare --source kimodo --reactor dataset_reactor --out kimodo.json
"""

import json
import os
import re
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import tyro
from tqdm import tqdm

from .. import DATASET_ROOT
from ..kinematics import planted, resample_poses
from ..retargeting import Models, human_planted, load_models
from ..retargeting.ik.human import calibrate, goals
from ..retargeting.metrics import contact_iou, fidelity, motion_samples, statistic, summarize
from ..retargeting.umr import UMR_TURNS, read_umr, umr_model, umr_path
from ..robots import FEET, lite_model_hash
from ..sources import SOURCES, clip_key, clips, motion_name
from ..sources.human import HumanClip
from ..store import FPS, read_robot_layer

#: The robot layer that every method writes.
ROBOT: str = "lite_pro"

#: The rows of a table: a label, the samples, and how to sum them up.
ROWS: tuple[tuple[str, str, str], ...] = (
    ("Slide, source stands, mean (cm/s)", "foot_slide_cm_s", "mean"),
    ("Slide, source stands, median (cm/s)", "foot_slide_cm_s", "median"),
    ("Slide, source stands, p95 (cm/s)", "foot_slide_cm_s", "p95"),
    ("Slide, sole near floor, mean (cm/s)", "near_floor_slide_cm_s", "mean"),
    ("Slide, sole near floor, median (cm/s)", "near_floor_slide_cm_s", "median"),
    ("Slide, sole near floor, p95 (cm/s)", "near_floor_slide_cm_s", "p95"),
    ("Sole near floor (% of foot frames)", "near_floor", "share"),
    ("Contact IoU with the source (%)", "near_floor", "iou"),
    ("Gap, source stands, mean (cm)", "planted_gap_cm", "mean"),
    ("Penetration, p95 of frames (cm)", "penetration_cm", "p95"),
    ("Fastest joint, p99 of frames (rad/s)", "joint_speed", "p99"),
    ("Fastest joint, max (rad/s)", "joint_speed", "max"),
    ("Limb direction error, mean (deg)", "limb_angle_deg", "mean"),
    ("Root error over 1 s, mean (cm)", "root_error_cm", "mean"),
    ("Foot goal error, mean (cm)", "foot_error_cm", "mean"),
    ("Hand goal error, mean (cm)", "hand_error_cm", "mean"),
)

#: The samples that compare a motion with the goals of the IK.
GOAL_KEYS: tuple[str, ...] = ("foot_error_cm", "foot_angle_deg", "hand_error_cm", "hand_angle_deg")


def reactor_pieces(folder: Path, name: str) -> list[tuple[str, int, dict]]:
    """Return the pieces of motion ``name`` in the ReActor store ``folder``, with their first frames."""
    pieces = []
    for path in sorted((folder / ROBOT).glob(f"{name}*.rrd")):
        match = re.fullmatch(rf"{re.escape(name)}(?:_f(\d{{5}}))?", path.stem)
        if match:
            pieces.append((path.stem, int(match.group(1) or 0), read_robot_layer(path, ROBOT)))
    return pieces


_MODELS: Models | None = None


def _worker_init() -> None:
    """Compile the models once per worker."""
    global _MODELS
    _MODELS = load_models()


def _worker_measure(job: tuple[str, Path, str, Path, Path | None, Path | None]) -> list[dict]:
    """Measure every method on one clip, and return a row per piece.

    ``job`` is the source, the path of the clip, its motion name, the IK store, the ReActor
    store, and the UMR motion. A row holds the frames of the piece, the hash of the model of
    each layer, and per method the samples and their summary.
    """
    assert _MODELS is not None
    source, path, name, store, reactor, umr = job
    ik = read_robot_layer(store / ROBOT / f"{name}.rrd", ROBOT)
    links = tuple(ik["bodies"])
    loaded = SOURCES[source].load(path)
    ref_pos = None
    if isinstance(loaded, HumanClip):
        pos, quat = resample_poses(loaded.pos, loaded.quat, loaded.fps, FPS)
        mask = human_planted(pos)
        ref_pos, _ = goals(pos, quat, calibrate(_MODELS.lite, loaded.rest))
    else:
        mask = planted(ik["goal_pos"][:, [links.index(foot) for foot in FEET], None], FPS)
    idle = [joint not in ik["active"] for joint in _MODELS.joint_names]

    motions = {"ik": (ik["goal_pos"][:, 0], ik["goal_quat"][:, 0], ik["joint_pos"])}
    heading = None
    if umr is not None:
        base_pos, base_quat, joint_pos, heading = read_umr(
            umr, _MODELS.joint_names, ik["goal_pos"][:, 0], ik["goal_quat"][:, 0], UMR_TURNS[source]
        )
        motions["umr"] = (base_pos, base_quat, joint_pos)
    frames = min(len(mask), *(len(joints) for _, _, joints in motions.values()))
    pieces = [(name, 0, None)] if reactor is None else reactor_pieces(reactor, name)
    layers = [ik] + [layer for _, _, layer in pieces if layer is not None]
    if any(tuple(layer["joint_names"]) != _MODELS.joint_names for layer in layers):
        raise ValueError(f"A layer of {name} does not hold the joints of Lite.")

    rows = []
    for piece, begin, layer in pieces:
        span = slice(begin, frames if layer is None else min(begin + len(layer["joint_pos"]), frames))
        methods = {method: (p[span], q[span], j[span]) for method, (p, q, j) in motions.items()}
        if layer is not None:
            n = span.stop - begin
            methods["reactor"] = (layer["goal_pos"][:n, 0], layer["goal_quat"][:n, 0], layer["joint_pos"][:n])
        row = {
            "motion": name, "piece": piece, "whole": piece == name, "frames": span.stop - begin,
            "models": {"ik": ik["model"]} | ({"reactor": layer["model"]} if layer is not None else {}),
            "umr_heading_deg": heading, "methods": {},
        }
        for method, (base_pos, base_quat, joint_pos) in methods.items():
            joint_pos = joint_pos.astype(np.float64)
            joint_pos[:, idle] = 0.0
            samples = motion_samples(
                _MODELS, base_pos.astype(np.float64), base_quat.astype(np.float64), joint_pos,
                links, ik["goal_pos"][span], ik["goal_quat"][span], mask[span],
            )
            samples |= fidelity(
                samples.pop("link_pos"), base_pos, links, None if ref_pos is None else ref_pos[span]
            )
            samples["planted"] = mask[span]
            if method == "umr":
                samples = {k: v for k, v in samples.items() if k not in GOAL_KEYS}
            summary = summarize(samples) | {
                "limb_angle_deg": statistic(samples["limb_angle_deg"], "mean"),
                "root_error_cm": statistic(samples["root_error_cm"], "mean"),
                "contact_iou": contact_iou(samples["near_floor"], samples["planted"]),
            }
            row["methods"][method] = {"samples": samples, "summary": summary}
        rows.append(row)
    return rows


def print_table(title: str, rows: list[dict], methods: list[str]) -> None:
    """Print each metric per method, over the samples of every row pooled together."""
    frames = sum(row["frames"] for row in rows)
    print(f"\n{title}: {len(rows)} pieces, {frames / FPS / 60:.1f} min, every sample pooled")
    print(f"  {'metric':<40}" + "".join(f"{method:>10}" for method in methods))
    for label, key, how in ROWS:
        cells = []
        for method in methods:
            parts = [row["methods"][method]["samples"] for row in rows]
            if key not in parts[0]:
                cells.append(f"{'-':>10}")
                continue
            values = np.concatenate([part[key] for part in parts])
            if how == "share":
                value = float(values.mean() * 100.0)
            elif how == "iou":
                value = contact_iou(values, np.concatenate([part["planted"] for part in parts]))
            else:
                value = statistic(values, how)
            cells.append(f"{value:10.2f}" if np.isfinite(value) else f"{'-':>10}")
        print(f"  {label:<40}" + "".join(cells))


def check_models(rows: list[dict], umr: Path | None) -> None:
    """Stop if a layer or the UMR run was made for another model of Lite than the installed one."""
    installed = lite_model_hash()
    seen = {(method, model) for row in rows for method, model in row["models"].items()}
    if umr is not None:
        seen.add(("umr", umr_model(umr)))
    wrong = sorted({(method, model) for method, model in seen if model is not None and model != installed})
    if wrong:
        raise SystemExit(f"The installed Lite is {installed}, but these methods used another model: {wrong}.")
    unknown = sorted({method for method, model in seen if model is None})
    if unknown:
        print(f"Warning: {', '.join(unknown)} do not record their model of Lite, so this step cannot check it.")


def compare(
    source: str = "lafan1bvh",
    clip: str | None = None,
    store: Path = DATASET_ROOT,
    reactor: Path | None = None,
    umr: Path | None = None,
    workers: int = -1,
    out: Path | None = None,
) -> None:
    """Measure the methods on the clips of a source, and print a table of their metrics.

    Args:
        source: The key of the source in ``sources.SOURCES``.
        clip: A regex. Measure only the clips whose path below the clip directory of the
            source matches it.
        store: The store of the IK, which ``uv run retarget`` wrote.
        reactor: A store that ``reactor export`` wrote.
        umr: The folder of UMR motions for one robot, such as ``output/<run>/lite_pro``.
        workers: The number of processes. ``-1`` uses every core.
        out: Also write the summary of every piece, and its sample counts, a piece per row,
            to this JSON file.
    """
    paths = clips(source, clip)
    names = [motion_name(source, path) for path in paths]
    missing = [name for name in names if not (store / ROBOT / f"{name}.rrd").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(names)} motions are not in {store}. Run `uv run retarget`.")
    jobs = [(source, path, name, store, reactor, None) for path, name in zip(paths, names, strict=True)]
    if umr is not None:
        if source not in UMR_TURNS:
            raise SystemExit(f"The world of a UMR motion of {source} is unknown. Add its turn to UMR_TURNS.")
        jobs = [(*job[:5], umr_path(umr, clip_key(source, job[1]))) for job in jobs]
        jobs = [job for job in jobs if job[5].exists()]
        print(f"UMR has {len(jobs)} of {len(paths)} clips in {umr}.")
    if not jobs:
        raise SystemExit(f"No clip of {source} to measure (clip={clip!r}).")

    if workers == -1:
        workers = os.cpu_count() or 1
    with get_context("spawn").Pool(min(workers, len(jobs)), _worker_init) as pool:
        rows = [
            row
            for result in tqdm(pool.imap(_worker_measure, jobs), total=len(jobs), desc="Clips", unit="clip")
            for row in result
        ]
    if not rows:
        raise SystemExit(f"{reactor} holds no piece of these clips.")
    check_models(rows, umr)
    if out is not None:
        # The sample counts let a reader pool the means of any group of pieces.
        out.write_text(json.dumps([
            {k: v for k, v in row.items() if k != "methods"}
            | {
                method: data["summary"] | {"samples": {k: int(v.size) for k, v in data["samples"].items()}}
                for method, data in row["methods"].items()
            }
            for row in rows
        ]))

    methods = [method for method in ("ik", "umr", "reactor") if method in rows[0]["methods"]]
    measured = sum(row["frames"] for row in rows if row["whole"] or reactor is not None)
    print(f"\n{len(jobs)} clips of {source}, {measured / FPS / 60:.1f} min measured.")
    if umr is not None:
        headings = list({row["motion"]: row["umr_heading_deg"] for row in rows}.values())
        print(f"The UMR base heads {np.median(headings):+.1f} deg from the IK base in the median clip, "
              f"{min(headings):+.1f} to {max(headings):+.1f} deg over the clips.")
    if reactor is not None:
        print_table("Motions that ReActor played whole", [row for row in rows if row["whole"]], methods)
        print_table("Every piece", rows, methods)
    else:
        print_table("Every motion", rows, methods)


def main() -> None:
    """Run :func:`compare` with the options on the command line."""
    tyro.cli(compare)
