"""Read and write a retargeted motion as a Rerun ``.rrd`` file.

The file format belongs to the training repo. Its definition is
``lite_motion_tracking/tasks/tracking/reference_motion.py`` in
[Lite-Motion-Tracking][1], and the constants below repeat it. Keep them equal, or
the training loader rejects what this script writes.

The definition is not imported, because that package pulls mjlab and torch. This
module needs ``rerun`` and ``numpy`` only.

A motion holds the base pose and the joint positions on one ``frame`` timeline.
The properties hold the joint names, so a loader maps a column to a robot joint
by name and not by position.

[1]: https://github.com/Berkeley-Humanoids/Lite-Motion-Tracking
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rerun as rr

#: Shared by every motion, so a directory of them registers as one dataset.
APPLICATION_ID = "lite-motion-tracking"

#: The layer that holds a training reference.
LAYER_TRACKING = "tracking"

TIMELINE_FRAME = "frame"
TIMELINE_TIME = "time"

EP_JOINT_POS = "/reference/joint_positions"
EP_JOINT_VEL = "/reference/joint_velocities"
EP_BASE = "/reference/base"
EP_BASE_LIN_VEL = "/reference/base_linear_velocity"
EP_BASE_ANG_VEL = "/reference/base_angular_velocity"

PROP_MOTION = "motion"
PROP_SOURCE = "source"

#: Rerun stores a rotation as xyzw. MuJoCo and this repo use wxyz.
QUAT_WXYZ_TO_XYZW = [1, 2, 3, 0]
QUAT_XYZW_TO_WXYZ = [3, 0, 1, 2]


def write_motion(
    path: str | Path,
    joint_names: tuple[str, ...],
    fps: int,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    base_lin_vel: np.ndarray,
    base_ang_vel: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    task: str,
    source: dict,
    robot_type: str = "lite_pro",
) -> Path:
    """Write one motion to ``path``, and return that path.

    Every array starts at the same frame and holds ``T`` rows. ``base_quat`` is
    wxyz.
    """
    path = Path(path)
    arrays = {
        "base_pos": (base_pos, 3),
        "base_quat": (base_quat, 4),
        "base_lin_vel": (base_lin_vel, 3),
        "base_ang_vel": (base_ang_vel, 3),
        "joint_pos": (joint_pos, len(joint_names)),
        "joint_vel": (joint_vel, len(joint_names)),
    }
    num_frames = joint_pos.shape[0]
    wrong = [
        name for name, (a, width) in arrays.items() if a.shape != (num_frames, width)
    ]
    if wrong:
        raise ValueError(
            f"{wrong} do not match joint_pos, which has shape "
            f"({num_frames}, {len(joint_names)})"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = rr.RecordingStream(APPLICATION_ID, recording_id=path.stem)
    stream.save(str(path))
    stream.send_recording_name(path.stem)
    stream.send_property(PROP_MOTION, rr.AnyValues(
        fps=fps,
        joint_names=list(joint_names),
        num_joints=len(joint_names),
        robot_type=robot_type,
        task=task,
        layer=LAYER_TRACKING,
        duration_s=num_frames / float(fps),
    ))
    # The source dict is nested and its keys vary, so it goes in as one JSON
    # string. A column per key would change the schema for each script.
    stream.send_property(
        PROP_SOURCE, rr.AnyValues(json=json.dumps(source, sort_keys=True))
    )

    frames = np.arange(num_frames, dtype=np.int64)
    indexes = [
        rr.TimeColumn(TIMELINE_FRAME, sequence=frames),
        rr.TimeColumn(TIMELINE_TIME, duration=frames / float(fps)),
    ]
    for entity, array in (
        (EP_JOINT_POS, joint_pos),
        (EP_JOINT_VEL, joint_vel),
        (EP_BASE_LIN_VEL, base_lin_vel),
        (EP_BASE_ANG_VEL, base_ang_vel),
    ):
        stream.send_columns(
            entity, indexes=indexes,
            columns=rr.Scalars.columns(scalars=np.ascontiguousarray(
                array, dtype=np.float32)),
        )
    stream.send_columns(EP_BASE, indexes=indexes, columns=rr.Transform3D.columns(
        translation=np.ascontiguousarray(base_pos, dtype=np.float32),
        quaternion=np.ascontiguousarray(
            base_quat[:, QUAT_WXYZ_TO_XYZW], dtype=np.float32),
    ))
    stream.flush()
    return path


def read_motion(path: str | Path) -> dict:
    """Read a motion back, for the viewer.

    Returns the base pose, the joint positions, and the joint names. Chunks
    arrive in any order, so this sorts the rows by the ``frame`` index.
    """
    import pyarrow as pa

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"motion not found: {path}")

    parts: dict[tuple[str, str], list] = {}
    props: dict[str, object] = {}
    for chunk in rr.experimental.RrdReader(str(path)).stream():
        batch = chunk.to_record_batch()
        for column in batch.schema.names:
            if column.startswith("rerun.controls") or column in (
                TIMELINE_FRAME, TIMELINE_TIME
            ):
                continue
            if chunk.entity_path == f"/__properties/{PROP_MOTION}":
                value = batch.column(column).to_pylist()[0]
                # A property holds one row whose value is a list. `fps` arrives
                # as [30] and `joint_names` as the whole list, so unwrap only a
                # single-element list.
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                props[column] = value
            elif TIMELINE_FRAME in batch.schema.names:
                values = batch.column(column)
                inner = values.flatten()
                if pa.types.is_fixed_size_list(inner.type):
                    width = inner.type.list_size
                    flat = inner.flatten().to_numpy(zero_copy_only=False)
                else:
                    flat = inner.to_numpy(zero_copy_only=False)
                    width = flat.size // batch.num_rows
                parts.setdefault((chunk.entity_path, column), []).append((
                    batch.column(TIMELINE_FRAME).to_numpy(zero_copy_only=False),
                    flat.reshape(batch.num_rows, width),
                ))

    columns = {}
    for key, chunks in parts.items():
        frames = np.concatenate([c[0] for c in chunks])
        rows = np.concatenate([c[1] for c in chunks])
        columns[key] = rows[np.argsort(frames, kind="stable")]

    return {
        "joint_names": tuple(props["joint_names"]),
        "fps": int(props["fps"]),
        "joint_pos": columns[EP_JOINT_POS, "Scalars:scalars"].astype(np.float64),
        "base_pos": columns[EP_BASE, "Transform3D:translation"].astype(np.float64),
        "base_quat": columns[EP_BASE, "Transform3D:quaternion"][
            :, QUAT_XYZW_TO_WXYZ
        ].astype(np.float64),
    }
