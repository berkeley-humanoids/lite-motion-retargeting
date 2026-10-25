"""Train ReActor on the motions of a store, or export the motions that it plays.

``retargeting/reactor/`` documents the method.

Usage:
    uv run --extra reactor reactor train --source lafan1bvh --clip '^walk'
    uv run --extra reactor reactor export --run logs/reactor/<run> --out dataset_reactor
"""

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from .. import DATASET_ROOT, REPO_ROOT
from ..kinematics import planted
from ..retargeting import load_models
from ..retargeting.metrics import motion_metrics
from ..retargeting.reactor.env_cfg import agent_cfg, env_cfg
from ..retargeting.reactor.library import build_library, motion_paths
from ..retargeting.reactor.mdp import GoalCommandCfg, lost_track
from ..robots import FEET, lite_urdf_path
from ..store import BASE_LAYER, FPS, write_asset, write_blueprint, write_robot_layer

LOG_ROOT: Path = REPO_ROOT / "logs" / "reactor"
DEVICE: str = "cuda:0"

#: Seconds. Where play lost track, export drops this much motion before that frame, because
#: that motion holds the stumble.
CUT_SECONDS: float = 1.0
#: Seconds. Export drops each piece that is shorter.
PIECE_SECONDS: float = 2.0


def train(
    source: str = "lafan1bvh",
    clip: str | None = None,
    store: Path = DATASET_ROOT,
    num_envs: int = 4096,
    iterations: int = 10_000,
    warmstart: int = 1_000,
    run_name: str = "",
) -> None:
    """Train the policy and the goal offsets on the motions of ``source`` in ``store``.

    Args:
        source: A key of ``sources.SOURCES``. ``uv run retarget`` must first write its motions to ``store``.
        clip: A regex on the path of a clip, as in ``uv run retarget``.
        store: The store to read the motions from.
        num_envs: The number of parallel simulations.
        iterations: The number of PPO iterations.
        warmstart: The number of PPO iterations before the goal offsets start to learn.
        run_name: A suffix for the name of the run directory. The run directory also keeps
            these arguments, in ``train.json``.
    """
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in locals().items()}
    paths = motion_paths(source, clip, store)
    run = LOG_ROOT / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{run_name}" if run_name else ""))
    run.mkdir(parents=True)
    (run / "train.json").write_text(json.dumps(arguments, indent=2))
    library = build_library(paths, run)
    joints = tuple(np.load(library)["joints"])
    print(f"{len(paths)} motions, {len(joints)} joints actuated -> {run}")

    goals = GoalCommandCfg(library=str(library), warmstart=warmstart, save_to=str(run / "goals.pt"))
    env = ManagerBasedRlEnv(cfg=env_cfg(joints, num_envs, goals), device=DEVICE)
    runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env), asdict(agent_cfg(iterations, run_name)), str(run), DEVICE)
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    env.command_manager.get_term("goals").save(run / "goals.pt")
    env.close()


def export(run: Path, out: Path, checkpoint: str | None = None) -> None:
    """Play each motion of a run with its policy and goals, and write the result to ``out``.

    Each motion starts at the IK solution of its first frame and plays to its end. If the
    root gets 0.5 m or 1 rad from its goal, play lost track at that frame. Play starts again
    from the IK solution of that frame, and the piece before it ends :data:`CUT_SECONDS`
    earlier. Export writes each piece of at least :data:`PIECE_SECONDS`. A motion that never
    lost track is one piece and keeps its name. Other pieces get the name
    ``<name>_f<first frame>``. The metrics of ``retargeting.metrics.motion_metrics`` compare each piece
    with the learned goals. Four more metrics measure the wrench, in N and N m:
    ``wrench_force_max``, ``wrench_force_mean``, ``wrench_torque_max`` and
    ``wrench_torque_mean``. A high peak with a low mean is a short push.

    Args:
        run: The run directory that ``train`` wrote.
        out: The store to write.
        checkpoint: A ``model_<n>.pt`` of the run. The default is the last one.
    """
    library_path = run / "library.npz"
    library = np.load(library_path)
    names, lengths, joints = list(library["names"]), library["lengths"], tuple(library["joints"])
    model_path = run / checkpoint if checkpoint else max(run.glob("model_*.pt"), key=lambda p: int(p.stem[6:]))

    goals = GoalCommandCfg(library=str(library_path), learn=False, offsets=str(run / "goals.pt"), playback=True)
    env = ManagerBasedRlEnv(cfg=env_cfg(joints, len(names), goals), device=DEVICE)
    wrapped = RslRlVecEnvWrapper(env)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg(0, "")), None, DEVICE)
    runner.load(str(model_path), map_location=DEVICE)
    policy = runner.get_inference_policy(device=DEVICE)

    command = env.command_manager.get_term("goals")
    robot, wrench = command.robot, env.action_manager.get_term("wrench")
    frames = int(lengths.max())
    root_pos = np.zeros((frames, len(names), 3), dtype=np.float32)
    root_quat = np.zeros((frames, len(names), 4), dtype=np.float32)
    joint_pos = np.zeros((frames, len(names), len(joints)), dtype=np.float32)
    force = np.zeros((frames, len(names)), dtype=np.float32)
    torque = np.zeros((frames, len(names)), dtype=np.float32)
    first = np.zeros(len(names), dtype=int)
    pieces: list[tuple[int, int, int]] = []

    # Frame 0 is the reset, at the IK solution. Each step then reaches the next frame.
    obs, _ = wrapped.reset()
    with torch.no_grad():
        for t in range(frames):
            if t > 0:
                obs, _, _, _ = wrapped.step(policy(obs))
                lost = np.flatnonzero(lost_track(env).cpu().numpy() & (t < lengths))
                if len(lost):
                    pieces += [(i, first[i], t - round(CUT_SECONDS * FPS)) for i in lost]
                    first[lost] = t
                    command.playback_start[lost] = t
                    env.reset(env_ids=torch.as_tensor(lost, device=command.device))
                    obs = wrapped.get_observations()
            root_pos[t] = (robot.data.root_link_pos_w - env.scene.env_origins).cpu().numpy()
            root_quat[t] = robot.data.root_link_quat_w.cpu().numpy()
            joint_pos[t] = robot.data.joint_pos.cpu().numpy()
            force[t] = wrench.force.norm(dim=-1).cpu().numpy()
            torque[t] = wrench.torque.norm(dim=-1).cpu().numpy()
        goal_pos, goal_quat = command.goals(torch.arange(int(lengths.sum()), device=command.device))
    goal_pos, goal_quat = goal_pos.cpu().numpy(), goal_quat.cpu().numpy()
    env.close()
    pieces += [(i, first[i], int(lengths[i])) for i in range(len(names))]

    models = load_models()
    urdf = lite_urdf_path()
    write_blueprint(out, urdf)
    write_asset(out, urdf)
    bodies, sources = tuple(library["bodies"]), tuple(library["sources"])
    columns = [models.joint_names.index(name) for name in joints]
    feet = [bodies.index(foot) for foot in FEET]
    kept = 0
    for i, begin, stop in pieces:
        if stop - begin < round(PIECE_SECONDS * FPS):
            continue
        whole = begin == 0 and stop == lengths[i]
        name = names[i] if whole else f"{names[i]}_f{begin:05d}"
        played = slice(begin, stop)
        span = slice(int(library["starts"][i]) + begin, int(library["starts"][i]) + stop)
        full = np.zeros((stop - begin, len(models.joint_names)), dtype=np.float32)
        full[:, columns] = joint_pos[played, i]
        # In a robot layer, the goal of the root link places the robot, so it takes the simulated root pose.
        pos, quat = goal_pos[span].copy(), goal_quat[span].copy()
        pos[:, 0], quat[:, 0] = root_pos[played, i], root_quat[played, i]
        metrics = motion_metrics(
            models, root_pos[played, i], root_quat[played, i], full, bodies, pos, quat,
            planted(goal_pos[span][:, feet, None], FPS),
        )
        metrics |= {
            "wrench_force_max": float(force[played, i].max()),
            "wrench_force_mean": float(force[played, i].mean()),
            "wrench_torque_max": float(torque[played, i].max()),
            "wrench_torque_mean": float(torque[played, i].mean()),
        }
        write_robot_layer(
            out, name, urdf, FPS, models.joint_names, full, bodies, sources, pos, quat, False, metrics,
            tuple(library["active"][i].split(",")),
        )
        # The base layer holds the whole clip, so only a whole motion takes it.
        base = Path(str(library["store"])) / BASE_LAYER / f"{name}.rrd"
        if whole and base.exists():
            (out / BASE_LAYER).mkdir(parents=True, exist_ok=True)
            shutil.copy(base, out / BASE_LAYER / f"{name}.rrd")
        kept += stop - begin
        print(f"{name}: " + ", ".join(f"{key} {value:.2f}" for key, value in metrics.items()))
    print(f"\nKept {kept / lengths.sum():.0%} of {lengths.sum() / FPS / 60:.1f} min of motion in {out}. "
          f"Play lost track {len(pieces) - len(names)} times.")


def main() -> None:
    """Run :func:`train` or :func:`export`, as the command line says."""
    tyro.extras.subcommand_cli_from_dict({"train": train, "export": export})
