"""Viser-based viewer for the retargeted Lite motion dataset.

Renders Lite (solid) alongside the source LAFAN1 G1 (alpha-blended ghost).
Mesh data comes straight from MuJoCo — no yourdfpy / trimesh dependency.
The GUI exposes an episode dropdown; switching episodes loads the new clip's
motion arrays on demand.

Usage:
    uv run scripts/visualize.py
    uv run scripts/visualize.py --episode-index 3 --port 8080
    uv run scripts/visualize.py --no-ghost
"""

import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import tyro
import viser

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FPS,
    G1_LAFAN_JOINT_NAMES,
    LITE_DATASET_REPO_ID,
    joint_qpos_addr,
    lite_joint_names,
    load_g1_model,
    load_lafan_csv,
    load_lite_model,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
LAFAN_ROOT: Path = REPO_ROOT / ".cache" / "lafan1_g1"


# ── Rotation helpers ──────────────────────────────────────────────────────────


def _rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → WXYZ quaternion with ``w >= 0``."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
            [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# ── Robot rendering ───────────────────────────────────────────────────────────


def _mesh_handles(
    server: viser.ViserServer,
    model: mujoco.MjModel,
    prefix: str,
    color: tuple[int, int, int],
    opacity: float,
) -> list[tuple[viser.MeshHandle, int]]:
    """One viser mesh handle per ``mjGEOM_MESH`` geom in the model."""
    handles: list[tuple[viser.MeshHandle, int]] = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = int(model.geom_dataid[gid])
        if mesh_id < 0:
            continue
        v0, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        f0, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        handle = server.scene.add_mesh_simple(
            name=f"{prefix}/g{gid}",
            vertices=np.asarray(model.mesh_vert[v0 : v0 + vn], dtype=np.float32),
            faces=np.asarray(model.mesh_face[f0 : f0 + fn], dtype=np.int32),
            color=color,
            opacity=opacity,
        )
        handles.append((handle, gid))
    return handles


def _update_pose(
    handles: list[tuple[viser.MeshHandle, int]],
    data: mujoco.MjData,
    base_pos: np.ndarray,
    base_R: np.ndarray,
) -> None:
    """Push current MuJoCo geom poses to viser.

    Both Lite and G1 are welded to world in their shipped descriptions, so
    ``data.geom_xpos`` / ``geom_xmat`` are pelvis-local; we compose with the
    LAFAN1 base transform to recover world coordinates.
    """
    for handle, gid in handles:
        handle.position = base_pos + base_R @ data.geom_xpos[gid]
        handle.wxyz = _rot_to_quat_wxyz(base_R @ data.geom_xmat[gid].reshape(3, 3))


# ── Dataset / CSV loading ─────────────────────────────────────────────────────


def _load_episode_lite(repo_id: str, root: Path, episode_index: int) -> dict[str, np.ndarray]:
    """Load one Lite LeRobotDataset episode as plain arrays."""
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=root, episodes=[episode_index])
    rows = dataset.hf_dataset.with_format("numpy")[:]
    return {
        "base_pos": np.asarray(rows["base_pos"], dtype=np.float64),
        "base_quat": np.asarray(rows["base_quat"], dtype=np.float64),
        "joint_pos": np.asarray(rows["joint_pos"], dtype=np.float64),
    }


def _load_episode_g1(lafan_root: Path, episode_index: int) -> dict[str, np.ndarray]:
    """Load the LAFAN1 G1 CSV paired with ``episode_index`` (sorted file order)."""
    csvs = sorted((lafan_root / "g1").glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No G1 CSVs under {lafan_root / 'g1'}. Run download_lafan.py.")
    if episode_index >= len(csvs):
        raise SystemExit(f"--episode-index {episode_index} out of range ({len(csvs)} clips).")
    return load_lafan_csv(csvs[episode_index])


# ── Main ──────────────────────────────────────────────────────────────────────


def main(
    episode_index: int = 0,
    repo_id: str = LITE_DATASET_REPO_ID,
    port: int = 8080,
    ghost: bool = True,
) -> None:
    """Play one retargeted clip alongside its LAFAN1 source.

    Args:
        episode_index: Initial episode to load (switch later from the GUI).
        repo_id: Repo identifier of the local LeRobotDataset.
        port: viser HTTP port.
        ghost: Render the source G1 alpha-blended on top of the Lite robot.
    """
    print("Loading models …")
    lite_model = load_lite_model()
    lite_data = mujoco.MjData(lite_model)
    lite_addrs = np.asarray(
        [joint_qpos_addr(lite_model, n) for n in lite_joint_names(lite_model)], dtype=np.int32
    )

    g1_model: mujoco.MjModel | None = None
    g1_data: mujoco.MjData | None = None
    g1_addrs: np.ndarray | None = None
    if ghost:
        g1_model = load_g1_model(LAFAN_ROOT)
        g1_data = mujoco.MjData(g1_model)
        g1_addrs = np.asarray(
            [joint_qpos_addr(g1_model, n) for n in G1_LAFAN_JOINT_NAMES], dtype=np.int32
        )

    csvs = sorted((LAFAN_ROOT / "g1").glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No G1 CSVs under {LAFAN_ROOT / 'g1'}. Run download_lafan.py.")
    if not (0 <= episode_index < len(csvs)):
        raise SystemExit(f"--episode-index {episode_index} out of range ({len(csvs)} clips).")
    idx_width = len(str(len(csvs) - 1))
    episode_labels: list[str] = [f"{i:0{idx_width}d}: {p.stem}" for i, p in enumerate(csvs)]

    # Mutable state shared between GUI callbacks and the playback thread; the
    # lock guards the swap during episode reload.
    state: dict[str, object] = {"lite": None, "g1": None, "frames": 0}
    state_lock = threading.Lock()

    def load_episode(idx: int) -> None:
        print(f"Loading episode {episode_labels[idx]} …")
        lite_motion = _load_episode_lite(repo_id, REPO_ROOT, idx)
        g1_motion = _load_episode_g1(LAFAN_ROOT, idx) if ghost else None
        frames = lite_motion["base_pos"].shape[0]
        if g1_motion is not None:
            frames = min(frames, g1_motion["base_pos"].shape[0])
        with state_lock:
            state["lite"] = lite_motion
            state["g1"] = g1_motion
            state["frames"] = frames

    load_episode(episode_index)

    server = viser.ViserServer(port=port)
    server.scene.add_grid("/grid", width=4.0, height=4.0)
    lite_handles = _mesh_handles(server, lite_model, "/lite", (200, 200, 210), opacity=1.0)
    g1_handles: list[tuple[viser.MeshHandle, int]] = []
    if ghost:
        assert g1_model is not None
        g1_handles = _mesh_handles(server, g1_model, "/g1", (100, 180, 255), opacity=0.35)

    episode_dropdown = server.gui.add_dropdown(
        "episode", options=tuple(episode_labels), initial_value=episode_labels[episode_index]
    )
    slider = server.gui.add_slider(
        "frame", min=0, max=int(state["frames"]) - 1, step=1, initial_value=0
    )
    play_btn = server.gui.add_button("play / pause")
    speed = server.gui.add_slider("speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
    playing = threading.Event()

    @play_btn.on_click
    def _(_event):
        playing.clear() if playing.is_set() else playing.set()

    def render_frame(t: int) -> None:
        with state_lock:
            lite_motion = state["lite"]
            g1_motion = state["g1"]
        lite_data.qpos[lite_addrs] = lite_motion["joint_pos"][t]  # type: ignore[index]
        mujoco.mj_kinematics(lite_model, lite_data)
        _update_pose(
            lite_handles, lite_data,
            base_pos=lite_motion["base_pos"][t],          # type: ignore[index]
            base_R=_quat_wxyz_to_rot(lite_motion["base_quat"][t]),  # type: ignore[index]
        )
        if g1_motion is None:
            return
        g1_data.qpos[g1_addrs] = g1_motion["g1_joint_pos"][t]  # type: ignore[index]
        mujoco.mj_kinematics(g1_model, g1_data)
        _update_pose(
            g1_handles, g1_data,
            base_pos=g1_motion["base_pos"][t],
            base_R=_quat_wxyz_to_rot(g1_motion["base_quat_wxyz"][t]),
        )

    render_frame(0)

    def play_loop() -> None:
        while True:
            if playing.is_set():
                frames = int(state["frames"])
                next_frame = (int(slider.value) + 1) % frames
                slider.value = next_frame
                render_frame(next_frame)
                time.sleep(1.0 / (FPS * float(speed.value)))
            else:
                time.sleep(0.05)

    threading.Thread(target=play_loop, daemon=True).start()

    @slider.on_update
    def _(_event):
        t = int(slider.value)
        if t < int(state["frames"]):
            render_frame(t)

    @episode_dropdown.on_update
    def _(_event):
        was_playing = playing.is_set()
        playing.clear()
        load_episode(int(episode_dropdown.value.split(":", 1)[0]))
        slider.max = int(state["frames"]) - 1
        slider.value = 0
        render_frame(0)
        if was_playing:
            playing.set()

    print(f"\nviser listening on http://localhost:{port}")
    print("  ctrl-c to exit")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    tyro.cli(main)
