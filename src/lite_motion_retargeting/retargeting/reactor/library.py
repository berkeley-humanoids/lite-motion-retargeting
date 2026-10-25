"""The motion library: the robot layers of a store, end to end, as the environment reads them."""

from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_inv, quat_mul

from ...robots import joint_names, load_lite_model
from ...sources import clips, motion_name
from ...store import FPS, read_robot_layer
from .env_cfg import ROBOT


def motion_paths(source: str, clip: str | None, store: Path) -> list[Path]:
    """Return the robot layers in ``store`` of the clips of ``source`` that match ``clip``."""
    paths = [store / ROBOT / f"{motion_name(source, path)}.rrd" for path in clips(source, clip)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(paths)} motions are not in {store}. Run `uv run retarget`.")
    return paths


def build_library(paths: list[Path], out: Path) -> Path:
    """Write the motions of ``paths`` to one ``library.npz`` in ``out``, and return its path.

    The motions run end to end. The joints are those of the simulated robot, in its order:
    every joint that some motion drives. The velocities that a reset needs are finite
    differences of the positions.
    """
    motions = [read_robot_layer(path, ROBOT) for path in paths]
    bodies = motions[0]["bodies"]
    for path, motion in zip(paths, motions, strict=True):
        if motion["fps"] != FPS or motion["bodies"] != bodies:
            raise SystemExit(f"{path} is not at {FPS} Hz, or its goals differ from those of {paths[0].name}.")
    # The joints of the simulated robot are in the order of the MJCF.
    active = set().union(*(motion["active"] for motion in motions))
    joints = tuple(name for name in joint_names(load_lite_model(ROBOT)) if name in active)

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
