"""Prepare AMASS SMPL-X clips so that UMR and the IK retarget the same input.

For each clip, this script:

* resamples the clip to 30 Hz, the rate at which UMR retargets LAFAN1. A joint turns by
  slerp between two frames, and the translation moves linearly.
* keeps the first 10 betas, the body that UMR builds. The IK reader builds the same body.
* shifts the clip up or down, so that its feet stand on the floor. AMASS does not put its
  floor at ``z = 0``: a standing foot sinks 1 to 2 cm in most clips, and floats in some.
  UMR snaps a source point to the floor only within 5 mm of it, so the height decides
  whether its ground anchors act. A foot stands while its lowest vertex moves slower than
  :data:`STANCE_SPEED`, the speed of ``kinematics.CONTACT_SPEED``. The shift puts the
  median height of a standing foot at ``z = 0``. A low percentile of every frame would
  not do: in a run, a foot sinks up to 7 cm for a few frames, and that percentile lifts the
  whole clip off the floor. The authors' rule for LAFAN1 (``ground_smplx.py``) needs a
  T-pose at the ends of a take, which AMASS lacks.
* skips a clip whose feet stand on more than one level, such as stairs, a beam or a box.
  Neither the IK nor UMR knows the terrain, so such a clip tests neither.
* writes the clip to one flat folder, named after its path in AMASS. UMR reads one folder
  and names its output after the file, so a flat name keeps two takes of one name apart.

Usage, in the environment of UMR:
    python prepare_amass.py <SMPL-X model file> <AMASS dir> <out dir> <clip list>

The clip list holds one path per line, below the AMASS dir, such as
``ACCAD/Female1Walking_c3d/B10_-_walk_turn_left_(45)_stageii.npz``. ``amass_clips.txt``
is the list of the fair test: a random sample of 4.2 h from 47 subjects of the 8 AMASS
subsets that use the neutral body, drawn in proportion to the hours of each subset. It
leaves out the clips on several levels, and those named after a beam or stairs.
"""

import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import smplx
import torch
from ground_smplx import BETAS, body_vertices
from scipy.spatial.transform import Rotation

#: Hz. UMR retargets LAFAN1 at this rate.
FPS: float = 30.0

#: m/s. A foot stands while its lowest vertex moves slower than this.
STANCE_SPEED: float = 0.2

#: m. A clip stands on more than one level if the heights of its standing feet spread more
#: than this, from their 5th to their 95th percentile.
MAX_LEVEL_SPREAD: float = 0.1

#: The SMPL-X joints of each foot, left first: the ankle and the ball of the foot.
FEET: tuple[tuple[int, int], ...] = ((7, 10), (8, 11))

#: Frames per batch of the body model.
CHUNK: int = 2000

#: Processes. Each holds one body model.
WORKERS: int = 32

_MODEL: smplx.SMPLX | None = None


def resample(poses: np.ndarray, trans: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``poses`` ``(T, J*3)`` and ``trans`` ``(T, 3)`` of a clip at ``fps`` resampled to :data:`FPS`."""
    times = np.arange(len(poses)) / fps
    out_times = np.arange(int(times[-1] * FPS + 1e-9) + 1) / FPS
    index = np.clip(np.searchsorted(times, out_times, side="right") - 1, 0, len(poses) - 2)
    alpha = np.clip((out_times - times[index]) * fps, 0.0, 1.0)
    joints = poses.shape[1] // 3
    start = Rotation.from_rotvec(poses[index].reshape(-1, 3))
    end = Rotation.from_rotvec(poses[index + 1].reshape(-1, 3))
    step = (start.inv() * end).as_rotvec() * np.repeat(alpha, joints)[:, None]
    out_poses = (start * Rotation.from_rotvec(step)).as_rotvec().reshape(len(out_times), -1)
    out_trans = np.stack([np.interp(out_times, times, trans[:, axis]) for axis in range(3)], axis=1)
    return out_poses, out_trans


def stance_heights(model: smplx.SMPLX, poses: np.ndarray, trans: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Return the height of the lowest vertex of a foot, over the frames where that foot stands.

    The speed of a frame is that of its lowest vertex, as a point of the foot, to the next frame.
    """
    owner = model.lbs_weights.argmax(dim=1)
    feet = [torch.isin(owner, torch.tensor(joints)).numpy() for joints in FEET]
    lowest = []
    for start in range(0, len(poses), CHUNK):
        # One frame of overlap gives the last frame of a chunk its next frame.
        frames = slice(start, min(start + CHUNK + 1, len(poses)))
        vertices = body_vertices(model, poses[frames], trans[frames], betas)
        n = len(vertices)
        for foot in feet:
            points = vertices[:, foot]
            index = points[:-1, :, 2].argmin(axis=1)
            now = points[np.arange(n - 1), index]
            later = points[np.arange(1, n), index]
            speed = np.linalg.norm(later[:, :2] - now[:, :2], axis=1) * FPS
            lowest.append(now[speed < STANCE_SPEED, 2])
    return np.concatenate(lowest)


def flat_name(clip: str) -> str:
    """Return the file name of a clip in the flat folder: its AMASS path, one ``_`` per run of other characters."""
    return re.sub(r"[^A-Za-z0-9]+", "_", clip.removesuffix(".npz")).strip("_")


def _init(model_file: Path) -> None:
    global _MODEL
    torch.set_num_threads(1)
    _MODEL = smplx.SMPLX(str(model_file), gender="neutral", use_pca=False, flat_hand_mean=True,
                         num_betas=BETAS, ext=model_file.suffix.lstrip("."))


def prepare(job: tuple[Path, Path, str]) -> str:
    amass, out, clip = job
    with np.load(amass / clip) as data:
        poses, trans = resample(data["poses"], data["trans"], float(data["mocap_frame_rate"]))
        betas = data["betas"][:BETAS]
        gender = str(data["gender"])
    if gender != "neutral":
        raise ValueError(f"{clip} uses the {gender} body. This script builds the neutral one.")
    name = flat_name(clip)
    heights = stance_heights(_MODEL, poses, trans, betas)
    if len(heights) == 0:
        return f"{name} skipped: no foot stands"
    spread = float(np.percentile(heights, 95) - np.percentile(heights, 5))
    if spread > MAX_LEVEL_SPREAD:
        return f"{name} skipped: the standing feet spread {spread * 100:.1f} cm"
    shift = -float(np.median(heights))
    trans[:, 2] += shift
    np.savez(out / f"{name}.npz", poses=poses.astype(np.float32), trans=trans.astype(np.float32),
             betas=betas.astype(np.float32), num_betas=BETAS, gender="neutral", surface_model_type="smplx",
             mocap_frame_rate=FPS, mocap_time_length=len(poses) / FPS, source=clip, ground_shift=shift)
    return f"{name} {len(poses)} frames, {shift * 100:+.2f} cm, standing feet spread {spread * 100:.1f} cm"


def main() -> None:
    model_file, amass, out, clip_list = (Path(arg) for arg in sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=True)
    clips = [clip for clip in clip_list.read_text().splitlines() if clip]
    if len({flat_name(clip) for clip in clips}) != len(clips):
        raise SystemExit("Two clips map to one flat name.")
    jobs = [(amass, out, clip) for clip in clips]
    with Pool(WORKERS, _init, (model_file,)) as pool:
        for line in pool.imap_unordered(prepare, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
