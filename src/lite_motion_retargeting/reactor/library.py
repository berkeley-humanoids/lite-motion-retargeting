"""The motion library: the robot layers of a store, end to end, as the environment reads them."""

import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from mjlab.utils.lab_api.math import quat_inv, quat_mul
from rerun.chunk import RrdReader

from .. import CACHE_ROOT
from ..sources import SOURCES, clip_key, motion_name
from ..store import FPS
from .env_cfg import ROBOT, robot_joints


def motion_paths(source: str, clip: str | None, store: Path) -> list[Path]:
    """Return the robot layers in ``store`` of the clips of ``source`` that match ``clip``."""
    spec = SOURCES[source]
    clip_dir = CACHE_ROOT / source / spec.clip_dir
    paths = [
        store / ROBOT / f"{motion_name(source, clip_dir, path)}.rrd"
        for path in sorted(clip_dir.glob(spec.clip_glob))
        if clip is None or re.search(clip, clip_key(clip_dir, path))
    ]
    if not paths:
        raise SystemExit(f"No clip in {clip_dir} matches {clip!r}.")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(paths)} motions are not in {store}. Run `uv run retarget`.")
    return paths


def read_motion(path: Path) -> dict:
    """Read one robot layer: its joints, its goals, and the joints that its source drives."""
    columns: dict[str, list[pa.RecordBatch]] = {}
    properties = None
    for chunk in RrdReader(path).stream():
        entity = chunk.entity_path
        if entity == f"/__properties/{ROBOT}":
            properties = chunk.to_record_batch().to_pydict()
        elif entity == f"/{ROBOT}/joint/positions" or entity.startswith(f"/{ROBOT}/goal/poses/"):
            columns.setdefault(entity, []).append(chunk.to_record_batch())
    if properties is None:
        raise SystemExit(f"{path} has no /__properties/{ROBOT}, so it is not a robot layer.")

    def values(entity: str, component: str, width: int) -> np.ndarray:
        table = pa.Table.from_batches(columns[entity]).sort_by("frame")
        flat = table.column(component).combine_chunks().flatten()
        if pa.types.is_fixed_size_list(flat.type):
            flat = flat.flatten()
        return flat.to_numpy().astype(np.float32).reshape(-1, width)

    joint_names = properties["joint_names"][0]
    bodies = properties["body_names"][0]
    poses = [f"/{ROBOT}/goal/poses/{body}" for body in bodies]
    goal_quat = np.stack([values(entity, "Transform3D:quaternion", 4) for entity in poses], axis=1)
    return {
        "fps": properties["fps"][0][0],
        "joint_names": joint_names,
        "active": properties["active_joint_names"][0],
        "bodies": bodies,
        "sources": properties["source_body_names"][0],
        "joint_pos": values(f"/{ROBOT}/joint/positions", "Scalars:scalars", len(joint_names)),
        "goal_pos": np.stack([values(entity, "Transform3D:translation", 3) for entity in poses], axis=1),
        "goal_quat": goal_quat[..., [3, 0, 1, 2]],  # Rerun stores xyzw.
    }


def build_library(paths: list[Path], out: Path) -> Path:
    """Write the motions of ``paths`` to one ``library.npz`` in ``out``, and return its path.

    The motions run end to end. The joints are those of the simulated robot, in its order:
    every joint that some motion drives. The velocities that a reset needs are finite
    differences of the positions.
    """
    motions = [read_motion(path) for path in paths]
    bodies = motions[0]["bodies"]
    for path, motion in zip(paths, motions, strict=True):
        if motion["fps"] != FPS or motion["bodies"] != bodies:
            raise SystemExit(f"{path} is not at {FPS} Hz, or its goals differ from those of {paths[0].name}.")
    joints = robot_joints(set().union(*(motion["active"] for motion in motions)))

    joint_pos = [motion["joint_pos"][:, [motion["joint_names"].index(name) for name in joints]] for motion in motions]
    lengths = np.array([len(pos) for pos in joint_pos])
    library = out / "library.npz"
    np.savez(
        library,
        # Export copies the base layers from this store.
        store=str(paths[0].parent.parent),
        names=np.array([path.stem for path in paths]),
        lengths=lengths,
        starts=np.concatenate([[0], np.cumsum(lengths)[:-1]]),
        bodies=np.array(bodies),
        sources=np.array(motions[0]["sources"]),
        joints=np.array(joints),
        active=np.array([",".join(motion["active"]) for motion in motions]),
        goal_pos=np.concatenate([motion["goal_pos"] for motion in motions]),
        goal_quat=np.concatenate([motion["goal_quat"] for motion in motions]),
        joint_pos=np.concatenate(joint_pos),
        joint_vel=np.concatenate([np.gradient(q, axis=0) * FPS for q in joint_pos]),
        root_lin_vel=np.concatenate([np.gradient(motion["goal_pos"][:, 0], axis=0) * FPS for motion in motions]),
        root_ang_vel=np.concatenate([_angular_velocity(motion["goal_quat"][:, 0]) for motion in motions]),
    )
    return library


def _angular_velocity(quat: np.ndarray) -> np.ndarray:
    """Return the world angular velocity ``(T, 3)`` of wxyz rotations ``(T, 4)``, by central differences."""
    q = torch.from_numpy(quat)
    ahead, behind = torch.cat([q[1:], q[-1:]]), torch.cat([q[:1], q[:-1]])
    step = quat_mul(ahead, quat_inv(behind))
    # q and -q are one rotation. Take the short way.
    step = torch.where(step[:, :1] < 0, -step, step)
    span = np.r_[1, np.full(len(q) - 2, 2), 1][: len(q)] / FPS
    return (2.0 * step[:, 1:].numpy() / span[:, None]).astype(np.float32)
