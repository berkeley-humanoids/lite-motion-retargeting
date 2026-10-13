"""Measure the IK, UMR and ReActor on the same frames of one source, with one planted mask.

Each method writes its motions in its own form. ``uv run retarget`` writes the IK store, and
``reactor export`` writes a store in which a motion can be cut into pieces. UMR writes
``<clip>_<robot>.npz`` into a folder named after its robot, as the README shows. This step
reads the base and the joints of each method, and measures them with
``retargeting.motion_samples``.

The metrics that a store keeps do not compare across methods. The IK measures the foot slide
on the frames where the source plants a foot, and ``reactor export`` on the frames where its
learned goals plant one. Here every method uses the mask of the source. With a ReActor
store, every method is measured on the frames of each ReActor piece.

The tables pool the samples of every piece: a mean is over every sample, and a median or a
percentile too. Thus a long motion counts more than a short one, a motion with no contact
adds no slide, and the cut of a motion into pieces changes nothing.

Three metrics compare a motion with its source, whatever the method aims for:

* ``limb_angle_deg``: the angle between each limb of the robot, such as the thigh, and the
  same limb of the source. The source limb is the goal of ``human.goals`` before
  ``human.plant`` moves it. Only a human source has these goals.
* ``root_error_cm``: how far the horizontal move of the root over :data:`ROOT_WINDOW`
  differs from that of the source root, which ``human.goals`` scales to Lite.
* ``contact_iou``: where the sole of the robot is near the floor, against where the source
  stands, as the intersection over the union of the two.

UMR places a motion in a world of its own. ``lafan_to_smplx`` turns the BVH axes into
(x, -z, y), and ``sources/human.py`` into (z, x, y), so the world of UMR is that of the IK
turned by -90 degrees about z. This step turns it back by :data:`UMR_TURN`, and shifts it by
one offset per motion. A fit of the turn would hide an error of heading, and a fit per frame
would hide a drift. UMR also scales the human by its height, not by the height of its pelvis
as the IK does. Thus a UMR link is not where the goal of the IK
is, and this step leaves out the goal errors of UMR. A joint that the source does not drive
holds 0 in every method, as it does in the IK.

Every layer records the hash of the model of Lite that it was written for, and a UMR run
records it in ``manifest.json``. This step stops if a hash differs from that of the
installed Lite.

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
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .. import CACHE_ROOT, DATASET_ROOT
from ..kinematics import planted, quat_mul, resample_motion, resample_poses
from ..retargeting import Models, human_planted, load_models, motion_samples, summarize
from ..retargeting.human import calibrate, goals
from ..robots import FEET, lite_model_hash
from ..sources import SOURCES, clip_key, motion_name
from ..sources.human import HumanClip
from ..store import FPS, read_robot_layer

#: The robot layer that every method writes.
ROBOT: str = "lite_pro"

#: The limbs that ``limb_angle_deg`` compares, as pairs of links.
LIMBS: tuple[tuple[str, str], ...] = (
    ("left_hip_yaw", "left_knee_pitch"),
    ("left_knee_pitch", "left_foot"),
    ("right_hip_yaw", "right_knee_pitch"),
    ("right_knee_pitch", "right_foot"),
    ("left_shoulder_yaw", "left_elbow_pitch"),
    ("left_elbow_pitch", "left_hand"),
    ("right_shoulder_yaw", "right_elbow_pitch"),
    ("right_elbow_pitch", "right_hand"),
)

#: rad. The turn about z from the world of a UMR motion of LAFAN1 to the world of the IK.
UMR_TURN: float = np.pi / 2

#: s. ``root_error_cm`` compares the move of the root over this time.
ROOT_WINDOW: float = 1.0

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

#: The other samples that ``retargeting.summarize`` needs.
SUMMARY_KEYS: tuple[str, ...] = (
    "foot_slide_cm_s", "near_floor_slide_cm_s", "planted_gap_cm", "penetration_cm", "joint_speed",
)


def _yaw(quat: np.ndarray) -> np.ndarray:
    """Return the heading of each wxyz rotation, in rad."""
    return Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_euler("ZYX")[:, 0]


def read_umr(
    path: Path, joint_names: tuple[str, ...], ref_pos: np.ndarray, ref_quat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Read one motion of UMR at :data:`store.FPS`, and move it into the world of the IK.

    ``ref_pos`` and ``ref_quat`` are the base of the IK. The motion turns by
    :data:`UMR_TURN`, and the shift puts the mean of the two roots together. Returns the
    base, the joints of ``joint_names``, and the median heading of the UMR base from the IK
    base after the turn, in degrees.
    """
    data = np.load(path, allow_pickle=True)
    qpos = data["qpos"].astype(np.float64)
    names = [str(name) for name in data["robot_joint_names"]]
    columns = [7 + names.index(name) for name in joint_names]
    motion = resample_motion(
        {"base_pos": qpos[:, :3], "base_quat_wxyz": qpos[:, 3:7], "g1_joint_pos": qpos[:, columns]},
        round(float(data["fps"][0])),
        FPS,
    )
    pos = Rotation.from_euler("z", UMR_TURN).apply(motion["base_pos"])
    quat = quat_mul(np.array([np.cos(UMR_TURN / 2), 0.0, 0.0, np.sin(UMR_TURN / 2)]), motion["base_quat_wxyz"])
    n = min(len(pos), len(ref_pos))
    pos[:, :2] += (ref_pos[:n, :2] - pos[:n, :2]).mean(axis=0)
    heading = np.degrees(np.angle(np.median(np.exp(1j * (_yaw(quat[:n]) - _yaw(ref_quat[:n]))))))
    return pos, quat, motion["g1_joint_pos"], float(heading)


def reactor_pieces(folder: Path, name: str) -> list[tuple[str, int, dict]]:
    """Return the pieces of motion ``name`` in the ReActor store ``folder``, with their first frames."""
    pieces = []
    for path in sorted((folder / ROBOT).glob(f"{name}*.rrd")):
        match = re.fullmatch(rf"{re.escape(name)}(?:_f(\d{{5}}))?", path.stem)
        if match:
            pieces.append((path.stem, int(match.group(1) or 0), read_robot_layer(path, ROBOT)))
    return pieces


def fidelity(
    link_pos: np.ndarray, base_pos: np.ndarray, links: tuple[str, ...], ref_pos: np.ndarray | None
) -> dict[str, np.ndarray]:
    """Return the samples of ``limb_angle_deg`` and ``root_error_cm`` of one motion.

    ``link_pos`` ``(T, P, 3)`` is where the robot puts ``links``, and ``ref_pos`` the goals
    of the source before planting, or None for a source with no such goals.
    """
    if ref_pos is None:
        return {"limb_angle_deg": np.zeros(0), "root_error_cm": np.zeros(0)}
    pairs = [(links.index(a), links.index(b)) for a, b in LIMBS if a in links and b in links]
    robot = np.stack([link_pos[:, b] - link_pos[:, a] for a, b in pairs], axis=1)
    source = np.stack([ref_pos[:, b] - ref_pos[:, a] for a, b in pairs], axis=1)
    cos = np.sum(robot * source, axis=-1) / (
        np.linalg.norm(robot, axis=-1) * np.linalg.norm(source, axis=-1)
    )
    window = round(ROOT_WINDOW * FPS)
    move = base_pos[window:, :2] - base_pos[:-window, :2]
    ref_move = ref_pos[window:, 0, :2] - ref_pos[:-window, 0, :2]
    return {
        "limb_angle_deg": np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).ravel(),
        "root_error_cm": np.linalg.norm(move - ref_move, axis=-1) * 100.0,
    }


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
            umr, _MODELS.joint_names, ik["goal_pos"][:, 0], ik["goal_quat"][:, 0]
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
            summary = summarize({k: samples.get(k, np.zeros(0)) for k in (*GOAL_KEYS, *SUMMARY_KEYS)})
            summary |= {
                "limb_angle_deg": _stat(samples["limb_angle_deg"], "mean"),
                "root_error_cm": _stat(samples["root_error_cm"], "mean"),
                "contact_iou": _iou(samples["near_floor"], samples["planted"]),
            }
            row["methods"][method] = {"samples": samples, "summary": summary}
        rows.append(row)
    return rows


def _stat(values: np.ndarray, how: str) -> float:
    """Return the mean, median, or a percentile of ``values``, or NaN if there are none."""
    if values.size == 0:
        return float("nan")
    if how == "mean":
        return float(values.mean())
    if how == "median":
        return float(np.median(values))
    if how == "max":
        return float(values.max())
    return float(np.percentile(values, int(how[1:])))


def _iou(near: np.ndarray, planted: np.ndarray) -> float:
    """Return the intersection over the union of two masks, in percent."""
    union = (near | planted).sum()
    return float((near & planted).sum() / union * 100.0) if union else float("nan")


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
                value = _iou(values, np.concatenate([part["planted"] for part in parts]))
            else:
                value = _stat(values, how)
            cells.append(f"{value:10.2f}" if np.isfinite(value) else f"{'-':>10}")
        print(f"  {label:<40}" + "".join(cells))


def check_models(rows: list[dict], umr: Path | None) -> None:
    """Stop if a layer or the UMR run was made for another model of Lite than the installed one."""
    installed = lite_model_hash()
    seen = {(method, model) for row in rows for method, model in row["models"].items()}
    if umr is not None:
        manifest = umr / "manifest.json"
        model = json.loads(manifest.read_text())["robot_model"] if manifest.exists() else None
        seen.add(("umr", model))
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
    if source not in SOURCES:
        raise SystemExit(f"Unknown source {source!r}. Expected one of {sorted(SOURCES)}.")
    spec = SOURCES[source]
    clip_dir = CACHE_ROOT / source / spec.clip_dir
    paths = [
        path for path in sorted(clip_dir.glob(spec.clip_glob))
        if clip is None or re.search(clip, clip_key(clip_dir, path))
    ]
    names = [motion_name(source, clip_dir, path) for path in paths]
    missing = [name for name in names if not (store / ROBOT / f"{name}.rrd").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(names)} motions are not in {store}. Run `uv run retarget`.")
    jobs = [(source, path, name, store, reactor, None) for path, name in zip(paths, names, strict=True)]
    if umr is not None:
        # UMR names a motion after its input file and its robot, and SMPL-X files keep the clip names.
        jobs = [(*job[:5], umr / f"{clip_key(clip_dir, job[1])}_{umr.name}.npz") for job in jobs]
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
