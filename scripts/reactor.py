"""ReActor: retarget by reinforcement learning in physics, after Disney's ReActor.

Paper: "ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting",
https://arxiv.org/abs/2605.06593. Its equation numbers are cited below.

``retarget.py`` gives each Lite link a goal pose and an IK solution that reaches it. The
IK knows nothing of mass, contact or torque, so a motion can float, sink, slide, or ask
more of a motor than it has. ReActor tracks the goals with a policy in MuJoCo Warp, and the
simulated motion is the retarget. The policy sets a PD target per joint around the IK
solution of the frame, and pushes the pelvis with a wrench (eq. 18, 19). The reward
penalizes the wrench, so the policy learns to need little of it.

While the policy trains, the goals learn too: a position and an orientation offset per
link, in the frame of the link, and a height offset per motion (eq. 10, 11, 14). After the
rollout of every PPO iteration, one step of gradient descent moves them toward where the
simulated links went, with the approximate bilevel gradient of eq. 6 at alpha = 0. The
goals then ask for what the robot can do.

The simulation is ideal: a PD position actuator per joint with the armature and the
effort limit of the MJCF, every collision geom of the MJCF, and no noise, delay or domain
randomization. ReActor makes reference motions, not a policy to deploy. A tracking policy
for the robot trains on its output, in Lite-Motion-Tracking.

It reads the store that ``retarget.py`` wrote, and writes another: one robot layer per
motion, with the simulated joints, the learned goals, and the metrics of
``retarget.motion_metrics``. A joint that no motion drives, such as a finger, is welded.

Usage:
    uv run --extra reactor scripts/reactor.py train --source lafan1bvh --clip '^walk'
    uv run --extra reactor scripts/reactor.py export --run logs/reactor/<run> --out dataset_reactor
"""

import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import torch
import tyro
from lite_description import get_mjcf_path
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.envs.mdp.observations import (
    base_ang_vel,
    base_lin_vel,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    projected_gravity,
)
from mjlab.envs.mdp.rewards import (
    action_acc_l2,
    action_rate_l2,
    is_alive,
    joint_acc_l2,
    joint_torques_l2,
)
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
    MjlabOnPolicyRunner,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)
from mjlab.rl.config import RslRlModelCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
)
from rerun.chunk import RrdReader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_ROOT, FPS, lite_urdf_path, planted  # noqa: E402
from mapping import FEET  # noqa: E402
from retarget import DATASET_ROOT, load_models, motion_metrics  # noqa: E402
from sources import SOURCES, clip_key, motion_name  # noqa: E402
from store import BASE_LAYER, write_asset, write_blueprint, write_robot_layer  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
LOG_ROOT: Path = REPO_ROOT / "logs" / "reactor"
ROBOT: str = "lite_pro"

#: Hz. The closed-loop natural frequency of every PD actuator, kp = armature * w^2. The
#: Damiao motors of the tracking task use 5 Hz, and every joint here does, which leaves the
#: policy a wider range of targets than 10 Hz would.
NATURAL_FREQUENCY_HZ: float = 5.0

#: The weights of the upper-level loss, eq. 17 and table 2: position and orientation.
GOAL_POSITION_WEIGHT: float = 10.0
GOAL_ORIENTATION_WEIGHT: float = 1.0

#: The bounds of the learned offsets, eq. 14: metres, radians, metres. Table 2 of the paper
#: sets all three to 0.5. Lite is about 1 m tall and its goals already fit it, so these are
#: tighter.
OFFSET_POSITION_BOUND: float = 0.1
OFFSET_ORIENTATION_BOUND: float = 0.3
OFFSET_HEIGHT_BOUND: float = 0.05

#: Frames ahead, besides the current one, that the policy sees the goals of.
LOOKAHEAD: int = 5

#: Seconds. Export drops the first of these before the frame where play lost track, which
#: holds the stumble, and keeps no piece shorter than the second.
CUT_SECONDS: float = 1.0
PIECE_SECONDS: float = 2.0


# ── The motions ───────────────────────────────────────────────────────────────


def read_motion(path: Path) -> dict:
    """The arrays of one robot layer: joints, goals, and the joints its source drives."""
    columns: dict[str, list[pa.RecordBatch]] = {}
    properties = None
    for chunk in RrdReader(path).stream():
        entity = chunk.entity_path
        if entity == f"/__properties/{ROBOT}":
            properties = chunk.to_record_batch().to_pydict()
        elif entity == f"/{ROBOT}/joint/positions" or entity.startswith(f"/{ROBOT}/goal/poses/"):
            columns.setdefault(entity, []).append(chunk.to_record_batch())
    if properties is None:
        raise SystemExit(f"{path}: no /__properties/{ROBOT}. Is it a robot layer?")

    def values(entity: str, component: str, width: int) -> np.ndarray:
        table = pa.Table.from_batches(columns[entity]).sort_by("frame")
        flat = table.column(component).combine_chunks().flatten()
        if pa.types.is_fixed_size_list(flat.type):
            flat = flat.flatten()
        return flat.to_numpy().astype(np.float32).reshape(-1, width)

    joint_names = properties["joint_names"][0]
    bodies = properties["body_names"][0]
    poses = [f"/{ROBOT}/goal/poses/{body}" for body in bodies]
    return {
        "fps": properties["fps"][0][0],
        "joint_names": joint_names,
        "active": properties["active_joint_names"][0],
        "bodies": bodies,
        "sources": properties["source_body_names"][0],
        "joint_pos": values(f"/{ROBOT}/joint/positions", "Scalars:scalars", len(joint_names)),
        "goal_pos": np.stack([values(entity, "Transform3D:translation", 3) for entity in poses], axis=1),
        # Rerun stores xyzw.
        "goal_quat": np.stack([values(entity, "Transform3D:quaternion", 4) for entity in poses], axis=1)[..., [3, 0, 1, 2]],
    }


def build_library(paths: list[Path], out: Path) -> Path:
    """Read every motion of ``paths`` into one ``library.npz`` under ``out``, and return it.

    The motions run end to end. The joints are those of the simulated robot, in its order:
    every joint some motion drives. The velocities that a reset needs are differenced.
    """
    motions = [read_motion(path) for path in paths]
    bodies = motions[0]["bodies"]
    for path, motion in zip(paths, motions, strict=True):
        if motion["fps"] != FPS or motion["bodies"] != bodies:
            raise SystemExit(f"{path}: not at {FPS} Hz, or not with the goals of {paths[0].name}.")
    active = set().union(*(motion["active"] for motion in motions))
    joints = robot_joints(active)

    columns = [[motion["joint_names"].index(name) for name in joints] for motion in motions]
    joint_pos = [motion["joint_pos"][:, cols] for motion, cols in zip(motions, columns, strict=True)]
    root_pos = [motion["goal_pos"][:, 0] for motion in motions]
    root_quat = [motion["goal_quat"][:, 0] for motion in motions]
    lengths = np.array([len(pos) for pos in joint_pos])
    path = out / "library.npz"
    np.savez(
        path,
        store=str(paths[0].parent.parent),
        names=np.array([p.stem for p in paths]),
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
        root_lin_vel=np.concatenate([np.gradient(p, axis=0) * FPS for p in root_pos]),
        root_ang_vel=np.concatenate([_angular_velocity(q) for q in root_quat]),
    )
    return path


def _angular_velocity(quat: np.ndarray) -> np.ndarray:
    """World angular velocity ``(T, 3)`` of wxyz rotations ``(T, 4)``, central difference."""
    q = torch.from_numpy(quat)
    ahead, behind = torch.cat([q[1:], q[-1:]]), torch.cat([q[:1], q[:-1]])
    step = quat_mul(ahead, quat_inv(behind))
    step = torch.where(step[:, :1] < 0, -step, step)
    span = np.r_[1, np.full(len(q) - 2, 2), 1][: len(q)] / FPS
    return (2.0 * step[:, 1:].numpy() / span[:, None]).astype(np.float32)


# ── The robot ─────────────────────────────────────────────────────────────────


def lite_spec(joints: tuple[str, ...]) -> mujoco.MjSpec:
    """Lite Pro with a free base, no stock actuators or sensors, and every hinge outside
    ``joints`` welded at 0."""
    spec = mujoco.MjSpec.from_file(str(get_mjcf_path(ROBOT)))
    spec.body("pelvis").add_freejoint()
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    for joint in list(spec.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_HINGE and joint.name not in joints:
            spec.delete(joint)
    return spec


def robot_joints(active: set[str]) -> tuple[str, ...]:
    """The joints of ``active`` in the order of the MJCF, which is that of the robot entity."""
    model = mujoco.MjModel.from_xml_path(str(get_mjcf_path(ROBOT)))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    return tuple(name for name in names if name in active)


def robot_cfg(joints: tuple[str, ...]) -> EntityCfg:
    """The robot entity: ``joints`` actuated by ideal PD, the rest welded.

    A joint keeps the armature and the effort limit of the MJCF. Its gains come from
    :data:`NATURAL_FREQUENCY_HZ`, with a damping ratio of 2, as BeyondMimic sets them.
    """
    model = mujoco.MjModel.from_xml_path(str(get_mjcf_path(ROBOT)))
    omega = 2.0 * np.pi * NATURAL_FREQUENCY_HZ
    groups: dict[tuple[float, float], list[str]] = {}
    for actuator in range(model.nu):
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[actuator, 0])
        if joint in joints:
            armature = float(model.dof_armature[model.jnt_dofadr[model.actuator_trnid[actuator, 0]]])
            groups.setdefault((armature, float(model.actuator_forcerange[actuator, 1])), []).append(joint)
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=tuple(names),
            stiffness=armature * omega**2,
            damping=4.0 * armature * omega,
            effort_limit=effort,
        )
        for (armature, effort), names in groups.items()
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
        spec_fn=lambda: lite_spec(joints),
        articulation=EntityArticulationInfoCfg(actuators=actuators),
    )


def action_scales(joints: tuple[str, ...]) -> dict[str, float]:
    """Per joint, the target offset of a unit action: a quarter of the effort limit over kp."""
    model = mujoco.MjModel.from_xml_path(str(get_mjcf_path(ROBOT)))
    omega = 2.0 * np.pi * NATURAL_FREQUENCY_HZ
    scales = {}
    for actuator in range(model.nu):
        joint_id = model.actuator_trnid[actuator, 0]
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint in joints:
            stiffness = model.dof_armature[model.jnt_dofadr[joint_id]] * omega**2
            scales[joint] = float(0.25 * model.actuator_forcerange[actuator, 1] / stiffness)
    return scales


# ── The goals, and the command that steps through them ────────────────────────


@dataclass(kw_only=True)
class GoalCommandCfg(CommandTermCfg):
    """The goals of the motions in ``library``, and the offsets they learn."""

    library: str
    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    #: Learn the offsets. Off, they stay at those of ``offsets``, or at zero.
    learn: bool = True
    #: A ``goals.pt`` to start the offsets from, and where to save them as they learn.
    offsets: str | None = None
    save_to: str | None = None
    #: Env steps per PPO iteration, which is how often the offsets learn.
    steps_per_iteration: int = 24
    #: PPO iterations before the offsets start to learn, so the policy tracks first.
    warmstart: int = 1000
    learning_rate: float = 1e-3
    #: Play motion i in env i from ``GoalCommand.playback_start``, and never resample: for
    #: export.
    playback: bool = False

    def build(self, env: ManagerBasedRlEnv) -> "GoalCommand":
        return GoalCommand(self, env)


class GoalCommand(CommandTerm):
    """Steps each env through a motion of the library, and poses the goals of its frame.

    ``frame`` is the frame the robot should reach at the end of the coming step. A reset
    puts the robot at the IK solution of the frame before it.
    """

    cfg: GoalCommandCfg

    def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        library = np.load(cfg.library)
        tensor = lambda key: torch.tensor(library[key], device=self.device)  # noqa: E731
        self.goal_pos0, self.goal_quat0 = tensor("goal_pos"), tensor("goal_quat")
        self.ref_joint_pos, self.ref_joint_vel = tensor("joint_pos"), tensor("joint_vel")
        self.root_lin_vel, self.root_ang_vel = tensor("root_lin_vel"), tensor("root_ang_vel")
        self.starts, self.lengths = tensor("starts").long(), tensor("lengths").long()
        self.motion_of = torch.repeat_interleave(torch.arange(len(self.starts), device=self.device), self.lengths)
        self.body_ids = self.robot.find_bodies(list(library["bodies"]), preserve_order=True)[0]
        if tuple(library["joints"]) != self.robot.joint_names:
            raise RuntimeError("The library joints are not those of the robot, in its order.")

        bodies, motions = len(self.body_ids), len(self.starts)
        self.offset_pos = torch.zeros(bodies, 3, device=self.device, requires_grad=True)
        self.offset_ori = torch.zeros(bodies, 3, device=self.device, requires_grad=True)
        self.offset_z = torch.zeros(motions, device=self.device, requires_grad=True)
        if cfg.offsets is not None:
            saved = torch.load(cfg.offsets, map_location=self.device)
            with torch.no_grad():
                self.offset_pos.copy_(saved["pos"])
                self.offset_ori.copy_(saved["ori"])
                self.offset_z.copy_(saved["z"])
        self.optimizer = torch.optim.Adam([self.offset_pos, self.offset_ori, self.offset_z], lr=cfg.learning_rate)

        self.motion = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        #: The frame each env starts its motion from, in playback.
        self.playback_start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.trials = torch.zeros(motions, device=self.device)
        self.failures = torch.zeros(motions, device=self.device)
        self.goal_pos = torch.zeros(self.num_envs, bodies, 3, device=self.device)
        self.goal_quat = torch.zeros(self.num_envs, bodies, 4, device=self.device)

        steps = cfg.steps_per_iteration
        self.sample_index = torch.zeros(steps, self.num_envs, dtype=torch.long, device=self.device)
        self.sample_pos = torch.zeros(steps, self.num_envs, bodies, 3, device=self.device)
        self.sample_quat = torch.zeros(steps, self.num_envs, bodies, 4, device=self.device)
        self.sample_valid = torch.zeros(steps, self.num_envs, dtype=torch.bool, device=self.device)
        self.steps = 0
        self.iteration = 0
        self._errors_at = -1
        self.errors: dict[str, torch.Tensor] = {}
        for key in ("root_pos", "root_ori", "body_pos", "body_ori"):
            self.metrics[f"error_{key}"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def index(self) -> torch.Tensor:
        """The library frame of each env."""
        return self.starts[self.motion] + self.frame

    @property
    def command(self) -> torch.Tensor:
        """The goals of the current frame and of :data:`LOOKAHEAD` frames on, in the root frame."""
        ahead = self.starts[self.motion] + torch.minimum(self.frame + LOOKAHEAD, self.lengths[self.motion] - 1)
        with torch.no_grad():
            pos_ahead, quat_ahead = self.goals(ahead)
        root_pos = self.robot.data.root_link_pos_w - self._env.scene.env_origins
        root_quat = self.robot.data.root_link_quat_w
        features = []
        for pos, quat in ((self.goal_pos, self.goal_quat), (pos_ahead, quat_ahead)):
            root_pos_b = root_pos[:, None].expand_as(pos)
            root_quat_b = root_quat[:, None].expand_as(quat)
            local_pos = quat_apply_inverse(root_quat_b, pos - root_pos_b)
            local_rot = matrix_from_quat(quat_mul(quat_inv(root_quat_b), quat))[..., :2]
            features += [local_pos.flatten(1), local_rot.flatten(1)]
        return torch.cat(features, dim=1)

    @property
    def reference_joint_pos(self) -> torch.Tensor:
        return self.ref_joint_pos[self.index]

    def goals(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The goals of library frames ``index``, with the learned offsets, eq. 10 and 11."""
        pos0, quat0 = self.goal_pos0[index], self.goal_quat0[index]
        lift = self.offset_z[self.motion_of[index]][..., None, None].expand(*pos0.shape[:-1], 1)
        pos = pos0 + quat_apply(quat0, self.offset_pos.expand_as(pos0)) + torch.nn.functional.pad(lift, (2, 0))
        return pos, quat_mul(quat0, _exp(self.offset_ori).expand_as(quat0))

    def tracking_errors(self) -> dict[str, torch.Tensor]:
        """The errors of this step, eq. 15 and 16, and a sample of it for the offsets."""
        if self._errors_at == self._env.common_step_counter:
            return self.errors
        self._errors_at = self._env.common_step_counter
        sim_pos = self.robot.data.body_link_pos_w[:, self.body_ids] - self._env.scene.env_origins[:, None]
        sim_quat = self.robot.data.body_link_quat_w[:, self.body_ids]
        delta = self.goal_pos - sim_pos
        angle = quat_error_magnitude(self.goal_quat, sim_quat)
        index = self.index
        self.errors = {
            "root_xy": delta[:, 0, :2].square().sum(-1),
            "root_z": delta[:, 0, 2].square(),
            "root_ori": angle[:, 0].square(),
            "root_lin_vel": (self.root_lin_vel[index] - self.robot.data.root_link_lin_vel_w).square().sum(-1),
            "root_ang_vel": (self.root_ang_vel[index] - self.robot.data.root_link_ang_vel_w).square().sum(-1),
            "body_pos": delta[:, 1:].square().sum(-1).sum(-1),
            "body_ori": angle[:, 1:].square().sum(-1),
            "root_distance": delta[:, 0].norm(dim=-1),
            "root_angle": angle[:, 0],
        }
        if self.cfg.learn:
            slot = self.steps % self.cfg.steps_per_iteration
            self.sample_index[slot] = index
            self.sample_pos[slot] = sim_pos
            self.sample_quat[slot] = sim_quat
            # The first step after a reset starts from the IK, not from the policy.
            self.sample_valid[slot] = self._env.episode_length_buf > 1
        return self.errors

    def learn_offsets(self) -> None:
        """One step of the upper level, eq. 3, 6, 7: move the goals toward the simulation."""
        # The runner steps the env in inference mode, which turns autograd off.
        with torch.inference_mode(False), torch.enable_grad():
            valid = self.sample_valid
            goal_pos, goal_quat = self.goals(self.sample_index[valid])
            # sin^2 of half the angle, times 4, is the squared angle for small angles, and has
            # a gradient at zero, where the angle does not.
            dot = (goal_quat * self.sample_quat[valid]).sum(-1)
            loss = (
                GOAL_POSITION_WEIGHT * (goal_pos - self.sample_pos[valid]).square().sum(-1)
                + GOAL_ORIENTATION_WEIGHT * 4.0 * (1.0 - dot.square())
            ).sum(-1).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        with torch.no_grad():
            for offset, bound in ((self.offset_pos, OFFSET_POSITION_BOUND), (self.offset_ori, OFFSET_ORIENTATION_BOUND)):
                norm = offset.norm(dim=-1, keepdim=True)
                offset.mul_(torch.clamp(bound / norm.clamp_min(1e-9), max=1.0))
            self.offset_z.clamp_(-OFFSET_HEIGHT_BOUND, OFFSET_HEIGHT_BOUND)
        log = self._env.extras.setdefault("log", {})
        log["Goals/loss"] = loss.item()
        log["Goals/offset_pos_max"] = self.offset_pos.norm(dim=-1).max().item()
        log["Goals/offset_ori_max"] = self.offset_ori.norm(dim=-1).max().item()
        log["Goals/offset_z_max"] = self.offset_z.abs().max().item()
        if self.cfg.save_to is not None and self.iteration % 100 == 0:
            self.save(Path(self.cfg.save_to))

    def save(self, path: Path) -> None:
        torch.save({"pos": self.offset_pos.detach(), "ori": self.offset_ori.detach(), "z": self.offset_z.detach()}, path)

    def _update_metrics(self) -> None:
        if self.errors:
            self.metrics["error_root_pos"] = self.errors["root_distance"]
            self.metrics["error_root_ori"] = self.errors["root_angle"]
            self.metrics["error_body_pos"] = self.errors["body_pos"].sqrt()
            self.metrics["error_body_ori"] = self.errors["body_ori"].sqrt()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self.cfg.playback:
            motion, frame = env_ids % len(self.starts), self.playback_start[env_ids]
        else:
            # Each motion is drawn by how often it failed, with a floor for the others.
            if self.steps > 0:
                ended = self.motion[env_ids]
                self.trials.index_add_(0, ended, torch.ones_like(ended, dtype=torch.float))
                self.failures.index_add_(0, ended, self._env.termination_manager.terminated[env_ids].float())
            rate = self.failures / self.trials.clamp_min(1.0)
            weight = rate / rate.sum() if rate.sum() > 0 else torch.full_like(rate, 1.0 / len(rate))
            weight = 0.9 * weight + 0.1 / len(rate)
            motion = torch.multinomial(weight, len(env_ids), replacement=True)
            frame = (torch.rand(len(env_ids), device=self.device) * (self.lengths[motion] - 1)).long()
        self.motion[env_ids], self.frame[env_ids] = motion, frame

        index = self.starts[motion] + frame
        root = torch.cat([
            self.goal_pos0[index, 0] + self._env.scene.env_origins[env_ids], self.goal_quat0[index, 0],
            self.root_lin_vel[index], self.root_ang_vel[index],
        ], dim=-1)
        self.robot.write_joint_state_to_sim(self.ref_joint_pos[index], self.ref_joint_vel[index], env_ids=env_ids)
        self.robot.write_root_state_to_sim(root, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            self.frame += 1
            self.steps += 1
            if self.cfg.learn and self.steps % self.cfg.steps_per_iteration == 0:
                self.iteration += 1
                if self.iteration > self.cfg.warmstart:
                    self.learn_offsets()
        else:
            self.frame[env_ids] += 1
        self.frame = torch.minimum(self.frame, self.lengths[self.motion] - 1)
        with torch.no_grad():
            self.goal_pos, self.goal_quat = self.goals(self.index)


def _exp(rotvec: torch.Tensor) -> torch.Tensor:
    """wxyz quaternions of rotation vectors ``(..., 3)``, with a gradient at zero."""
    angle = (rotvec.square().sum(-1, keepdim=True) + 1e-12).sqrt()
    return torch.cat([torch.cos(0.5 * angle), rotvec * torch.sin(0.5 * angle) / angle], dim=-1)


# ── Actions ───────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class ResidualJointActionCfg(JointPositionActionCfg):
    """Joint position targets around the IK solution of the frame of the command."""

    command_name: str = "goals"

    def build(self, env: ManagerBasedRlEnv) -> "ResidualJointAction":
        return ResidualJointAction(self, env)


class ResidualJointAction(JointPositionAction):
    cfg: ResidualJointActionCfg

    def process_actions(self, actions: torch.Tensor) -> None:
        command = self._env.command_manager.get_term(self.cfg.command_name)
        self._offset = command.reference_joint_pos[:, self._target_ids]
        super().process_actions(actions)


@dataclass(kw_only=True)
class RootWrenchActionCfg(ActionTermCfg):
    """A world-frame force and torque on the root body, behind a deadband, eq. 18 and 19."""

    body_name: str = "pelvis"
    #: N and N m per unit of action past the deadband. Lite weighs 26 kg.
    force_scale: float = 50.0
    torque_scale: float = 10.0
    deadband: float = 0.1

    def build(self, env: ManagerBasedRlEnv) -> "RootWrenchAction":
        return RootWrenchAction(self, env)


class RootWrenchAction(ActionTerm):
    cfg: RootWrenchActionCfg

    def __init__(self, cfg: RootWrenchActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self._body = self._entity.find_bodies(cfg.body_name)[0]
        self._raw = torch.zeros(self.num_envs, 6, device=self.device)
        self.force = torch.zeros(self.num_envs, 3, device=self.device)
        self.torque = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        wrench = torch.sign(actions) * torch.clamp(actions.abs() - self.cfg.deadband, min=0.0)
        self.force = wrench[:, :3] * self.cfg.force_scale
        self.torque = wrench[:, 3:] * self.cfg.torque_scale

    def apply_actions(self) -> None:
        self._entity.write_external_wrench_to_sim(self.force[:, None], self.torque[:, None], body_ids=self._body)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw[env_ids] = 0.0
        self.force[env_ids] = 0.0
        self.torque[env_ids] = 0.0


# ── Observations, rewards and terminations ────────────────────────────────────


def goal_features(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.command_manager.get_term("goals").command


def joint_target_error(env: ManagerBasedRlEnv) -> torch.Tensor:
    command = env.command_manager.get_term("goals")
    return command.reference_joint_pos - command.robot.data.joint_pos


def root_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_link_pos_w[:, 2:3]


def tracking_error(env: ManagerBasedRlEnv, key: str) -> torch.Tensor:
    return env.command_manager.get_term("goals").tracking_errors()[key]


def wrench_l1(env: ManagerBasedRlEnv, part: str) -> torch.Tensor:
    return getattr(env.action_manager.get_term("wrench"), part).abs().sum(-1)


def motion_end(env: ManagerBasedRlEnv) -> torch.Tensor:
    command = env.command_manager.get_term("goals")
    return command.frame >= command.lengths[command.motion] - 1


def lost_track(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The root is 0.5 m or 1 rad from its goal, or the simulation diverged: the motion failed."""
    errors = env.command_manager.get_term("goals").tracking_errors()
    distance, angle = errors["root_distance"], errors["root_angle"]
    return (distance > 0.5) | (angle > 1.0) | ~torch.isfinite(distance) | ~torch.isfinite(angle)


def env_cfg(joints: tuple[str, ...], num_envs: int, goals: GoalCommandCfg) -> ManagerBasedRlEnvCfg:
    """The environment. The terms follow table 1 of the paper, in its weights."""
    observations = {
        "goals": ObservationTermCfg(func=goal_features),
        "joint_target_error": ObservationTermCfg(func=joint_target_error),
        "root_height": ObservationTermCfg(func=root_height),
        "projected_gravity": ObservationTermCfg(func=projected_gravity),
        "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=base_ang_vel),
        "joint_pos": ObservationTermCfg(func=joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel),
        "actions": ObservationTermCfg(func=last_action),
    }
    error = lambda key, weight: RewardTermCfg(func=tracking_error, weight=weight, params={"key": key})  # noqa: E731
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=num_envs,
            entities={"robot": robot_cfg(joints)},
        ),
        observations={
            "actor": ObservationGroupCfg(terms=observations),
            "critic": ObservationGroupCfg(terms=observations),
        },
        actions={
            "joint_pos": ResidualJointActionCfg(
                entity_name="robot", actuator_names=(".*",), scale=action_scales(joints), use_default_offset=False,
            ),
            "wrench": RootWrenchActionCfg(entity_name="robot"),
        },
        commands={"goals": goals},
        events={},
        rewards={
            "root_xy": error("root_xy", -2.0),
            "root_z": error("root_z", -10.0),
            "root_ori": error("root_ori", -2.0),
            "root_lin_vel": error("root_lin_vel", -0.5),
            "root_ang_vel": error("root_ang_vel", -0.5),
            "body_pos": error("body_pos", -2.0),
            "body_ori": error("body_ori", -2.0),
            "alive": RewardTermCfg(func=is_alive, weight=20.0),
            "torque": RewardTermCfg(func=joint_torques_l2, weight=-1e-4),
            "joint_acc": RewardTermCfg(func=joint_acc_l2, weight=-1e-6),
            "action_rate": RewardTermCfg(func=action_rate_l2, weight=-1e-2),
            "action_acc": RewardTermCfg(func=action_acc_l2, weight=-1e-2),
            "wrench_force": RewardTermCfg(func=wrench_l1, weight=-1e-2, params={"part": "force"}),
            "wrench_torque": RewardTermCfg(func=wrench_l1, weight=-1e-2, params={"part": "torque"}),
        },
        terminations={} if goals.playback else {
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "motion_end": TerminationTermCfg(func=motion_end, time_out=True),
            "lost_track": TerminationTermCfg(func=lost_track),
        },
        sim=SimulationCfg(nconmax=128, njmax=512, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
        decimation=4,
        episode_length_s=1.0e6 if goals.playback else 10.0,
        auto_reset=not goals.playback,
    )


def agent_cfg(iterations: int, run_name: str) -> RslRlOnPolicyRunnerCfg:
    """PPO of table 2 of the paper."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=iterations,
        save_interval=500,
        experiment_name="reactor",
        run_name=run_name,
        logger="tensorboard",
        upload_model=False,
        actor=RslRlModelCfg(
            hidden_dims=(512, 512, 512), obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 512, 512), obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.0025, gamma=0.97, lam=0.95, desired_kl=0.009, learning_rate=1e-3,
        ),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


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
        source: A key of ``sources.SOURCES``, whose motions ``retarget.py`` wrote to ``store``.
        clip: A regex on the path of a clip, as ``retarget.py`` takes it.
        store: The store to read the motions from.
        num_envs: Parallel simulations.
        iterations: PPO iterations.
        warmstart: PPO iterations before the goal offsets learn.
        run_name: Appended to the name of the run directory.
    """
    paths = motion_paths(source, clip, store)
    run = LOG_ROOT / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{run_name}" if run_name else ""))
    run.mkdir(parents=True)
    library = build_library(paths, run)
    joints = tuple(np.load(library)["joints"])
    print(f"{len(paths)} motions, {len(joints)} joints actuated -> {run}")

    goals = GoalCommandCfg(library=str(library), warmstart=warmstart, save_to=str(run / "goals.pt"))
    env = ManagerBasedRlEnv(cfg=env_cfg(joints, num_envs, goals), device="cuda:0")
    agent = agent_cfg(iterations, run_name)
    runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env), asdict(agent), str(run), "cuda:0")
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    env.command_manager.get_term("goals").save(run / "goals.pt")
    env.close()


def export(run: Path, out: Path, checkpoint: str | None = None) -> None:
    """Play every motion of a run with its policy and goals, and write the result to ``out``.

    Each motion starts at the IK solution of its first frame and plays to its end. Where
    its root gets 0.5 m or 1 rad from its goal, it lost track: the piece before ends
    :data:`CUT_SECONDS` earlier, and play starts again from the IK solution of that frame.
    A motion that never lost track is written under its name. Otherwise each piece of at
    least :data:`PIECE_SECONDS` is, as ``<name>_f<first frame>``. The metrics of
    ``retarget.motion_metrics`` measure each against the learned goals, and four more
    measure the wrench: ``wrench_force_max``, ``wrench_force_mean``, ``wrench_torque_max``
    and ``wrench_torque_mean``, in N and N m. A high peak with a low mean is a short push.

    Args:
        run: The run directory that ``train`` wrote.
        out: The store to write.
        checkpoint: A ``model_<n>.pt`` of the run. The last one by default.
    """
    library_path = run / "library.npz"
    library = np.load(library_path)
    names, lengths, joints = list(library["names"]), library["lengths"], tuple(library["joints"])
    model_path = run / checkpoint if checkpoint else max(run.glob("model_*.pt"), key=lambda p: int(p.stem[6:]))

    goals = GoalCommandCfg(library=str(library_path), learn=False, offsets=str(run / "goals.pt"), playback=True)
    env = ManagerBasedRlEnv(cfg=env_cfg(joints, len(names), goals), device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg(0, "")), None, "cuda:0")
    runner.load(str(model_path), map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")

    command = env.command_manager.get_term("goals")
    robot, wrench = command.robot, env.action_manager.get_term("wrench")
    frames = int(lengths.max())
    base_pos = np.zeros((frames, len(names), 3), dtype=np.float32)
    base_quat = np.zeros((frames, len(names), 4), dtype=np.float32)
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
                for i in lost:
                    pieces.append((i, first[i], t - round(CUT_SECONDS * FPS)))
                    first[i] = t
                if len(lost):
                    command.playback_start[lost] = t
                    env.reset(env_ids=torch.as_tensor(lost, device=command.device))
                    obs = wrapped.get_observations()
            base_pos[t] = (robot.data.root_link_pos_w - env.scene.env_origins).cpu().numpy()
            base_quat[t] = robot.data.root_link_quat_w.cpu().numpy()
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
        frame = slice(begin, stop)
        span = slice(int(library["starts"][i]) + begin, int(library["starts"][i]) + stop)
        full = np.zeros((stop - begin, len(models.joint_names)), dtype=np.float32)
        full[:, columns] = joint_pos[frame, i]
        pos, quat = goal_pos[span].copy(), goal_quat[span].copy()
        pos[:, 0], quat[:, 0] = base_pos[frame, i], base_quat[frame, i]
        metrics = motion_metrics(
            models, base_pos[frame, i], base_quat[frame, i], full, bodies, pos, quat,
            planted(goal_pos[span][:, feet, None], FPS),
        )
        metrics |= {
            "wrench_force_max": float(force[frame, i].max()),
            "wrench_force_mean": float(force[frame, i].mean()),
            "wrench_torque_max": float(torque[frame, i].max()),
            "wrench_torque_mean": float(torque[frame, i].mean()),
        }
        write_robot_layer(
            out, name, urdf, FPS, models.joint_names, full, bodies, sources, pos, quat, False, metrics,
            tuple(library["active"][i].split(",")),
        )
        # The source layer holds the whole clip, so only a whole motion takes it.
        base = Path(str(library["store"])) / BASE_LAYER / f"{name}.rrd"
        if whole and base.exists():
            (out / BASE_LAYER).mkdir(parents=True, exist_ok=True)
            shutil.copy(base, out / BASE_LAYER / f"{name}.rrd")
        kept += stop - begin
        print(f"{name}: " + ", ".join(f"{key} {value:.2f}" for key, value in metrics.items()))
    lost_count = len(pieces) - len(names)
    print(f"\nKept {kept / lengths.sum():.0%} of {lengths.sum() / FPS / 60:.1f} min of motion in {out}. "
          f"Play lost track {lost_count} times.")


def motion_paths(source: str, clip: str | None, store: Path) -> list[Path]:
    """The robot layers in ``store`` of the clips of ``source`` that match ``clip``."""
    spec = SOURCES[source]
    clip_dir = CACHE_ROOT / source / spec.clip_dir
    paths = [
        store / ROBOT / f"{motion_name(source, clip_dir, path)}.rrd"
        for path in sorted(clip_dir.glob(spec.clip_glob))
        if clip is None or re.search(clip, clip_key(clip_dir, path))
    ]
    missing = [path for path in paths if not path.exists()]
    if missing or not paths:
        raise SystemExit(f"{len(missing)} of {len(paths)} motions are not in {store}. Run scripts/retarget.py.")
    return paths


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"train": train, "export": export})
