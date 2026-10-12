"""The terms of the ReActor environment: the goal command, the actions, and the functions of
the observations, rewards and terminations.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
)

#: The weights of position and orientation in the upper-level loss, eq. 17 and table 2.
GOAL_POSITION_WEIGHT: float = 10.0
GOAL_ORIENTATION_WEIGHT: float = 1.0

#: The bounds of the learned offsets, eq. 14: metres, radians, metres. Table 2 of the paper
#: sets all three to 0.5. Lite is about 1 m tall and its goals already fit it, so these
#: bounds are tighter.
OFFSET_POSITION_BOUND: float = 0.1
OFFSET_ORIENTATION_BOUND: float = 0.3
OFFSET_HEIGHT_BOUND: float = 0.05

#: Env steps per PPO iteration. The offsets learn once per iteration, from its steps.
STEPS_PER_ITERATION: int = 24

#: The policy sees the goals of the current frame and of the frame this many frames later.
LOOKAHEAD: int = 5


# ── The goals, and the command that steps through them ────────────────────────


@dataclass(kw_only=True)
class GoalCommandCfg(CommandTermCfg):
    """The goals of the motions in ``library``, and the offsets that they learn."""

    library: str
    entity_name: str = "robot"
    #: Draw a new motion only at a reset, never on a timer.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    #: Learn the offsets. If this is off, the offsets stay at those of ``offsets``, or at zero.
    learn: bool = True
    #: A ``goals.pt`` to start the offsets from.
    offsets: str | None = None
    #: The ``goals.pt`` to save the offsets to as they learn.
    save_to: str | None = None
    #: PPO iterations before the offsets start to learn, so that the policy learns to track first.
    warmstart: int = 1000
    learning_rate: float = 1e-3
    #: For export: play motion i in env i from ``GoalCommand.playback_start``, and never draw
    #: another motion.
    playback: bool = False

    def build(self, env: ManagerBasedRlEnv) -> "GoalCommand":
        return GoalCommand(self, env)


class GoalCommand(CommandTerm):
    """Steps each env through a motion of the library, and poses the goals of its frame.

    ``frame`` is the frame that the robot should reach at the end of the coming step. A
    reset puts the robot at the IK solution of the frame before it.
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
            raise RuntimeError("The library joints differ from the robot joints, or are in a different order.")

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
        #: The frame that each env starts its motion from, in playback.
        self.playback_start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.trials = torch.zeros(motions, device=self.device)
        self.failures = torch.zeros(motions, device=self.device)
        self.goal_pos = torch.zeros(self.num_envs, bodies, 3, device=self.device)
        self.goal_quat = torch.zeros(self.num_envs, bodies, 4, device=self.device)

        self.sample_index = torch.zeros(STEPS_PER_ITERATION, self.num_envs, dtype=torch.long, device=self.device)
        self.sample_pos = torch.zeros(STEPS_PER_ITERATION, self.num_envs, bodies, 3, device=self.device)
        self.sample_quat = torch.zeros(STEPS_PER_ITERATION, self.num_envs, bodies, 4, device=self.device)
        self.sample_valid = torch.zeros(STEPS_PER_ITERATION, self.num_envs, dtype=torch.bool, device=self.device)
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
        """The goals of the current frame and of a frame ahead, relative to the root.

        The frame ahead is :data:`LOOKAHEAD` frames later, or the last frame of the motion.
        Each goal is a position and the first two columns of a rotation matrix.
        """
        frame = torch.minimum(self.frame + LOOKAHEAD, self.lengths[self.motion] - 1)
        with torch.no_grad():
            ahead = self.goals(self.starts[self.motion] + frame)
        root_pos = (self.robot.data.root_link_pos_w - self._env.scene.env_origins)[:, None]
        root_quat = self.robot.data.root_link_quat_w[:, None]
        features = []
        for pos, quat in ((self.goal_pos, self.goal_quat), ahead):
            root = root_quat.expand_as(quat)
            local_pos = quat_apply_inverse(root, pos - root_pos)
            local_rot = matrix_from_quat(quat_mul(quat_inv(root), quat))[..., :2]
            features += [local_pos.flatten(1), local_rot.flatten(1)]
        return torch.cat(features, dim=1)

    @property
    def reference_joint_pos(self) -> torch.Tensor:
        """The IK solution of the frame of each env."""
        return self.ref_joint_pos[self.index]

    def goals(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the goals of the library frames ``index`` with the learned offsets, eq. 10 and 11."""
        pos0, quat0 = self.goal_pos0[index], self.goal_quat0[index]
        lift = self.offset_z[self.motion_of[index]][..., None, None].expand(*pos0.shape[:-1], 1)
        pos = pos0 + quat_apply(quat0, self.offset_pos.expand_as(pos0)) + torch.nn.functional.pad(lift, (2, 0))
        return pos, quat_mul(quat0, _exp(self.offset_ori).expand_as(quat0))

    def tracking_errors(self) -> dict[str, torch.Tensor]:
        """Return the tracking errors of this step, eq. 15 and 16.

        The rewards and the terminations share one computation per step. If the offsets
        learn, the step also goes into their samples.
        """
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
            slot = self.steps % STEPS_PER_ITERATION
            self.sample_index[slot] = index
            self.sample_pos[slot] = sim_pos
            self.sample_quat[slot] = sim_quat
            # The first step after a reset starts from the IK, not from the policy.
            self.sample_valid[slot] = self._env.episode_length_buf > 1
        return self.errors

    def learn_offsets(self) -> None:
        """Take one step of the upper level, eq. 3, 6 and 7: move the goals toward the simulated links."""
        # The runner steps the env in inference mode, which turns autograd off.
        with torch.inference_mode(False), torch.enable_grad():
            valid = self.sample_valid
            goal_pos, goal_quat = self.goals(self.sample_index[valid])
            # 4 sin^2 of half the angle is about the squared angle for small angles. Unlike the
            # angle, it has a gradient at zero.
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
            # Draw a motion in proportion to how often it failed. A tenth of the draws are
            # uniform, so every motion still plays. The resets before the first step end no
            # motion.
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
            if self.cfg.learn and self.steps % STEPS_PER_ITERATION == 0:
                self.iteration += 1
                if self.iteration > self.cfg.warmstart:
                    self.learn_offsets()
        else:
            self.frame[env_ids] += 1
        self.frame = torch.minimum(self.frame, self.lengths[self.motion] - 1)
        with torch.no_grad():
            self.goal_pos, self.goal_quat = self.goals(self.index)


def _exp(rotvec: torch.Tensor) -> torch.Tensor:
    """Return the wxyz quaternions of rotation vectors ``(..., 3)``, with a gradient at zero."""
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
        # The base class adds ``_offset`` to the scaled action.
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
    """Return which envs lost track of their motion.

    An env lost track if its root is 0.5 m or 1 rad from its goal, or if its simulation diverged.
    """
    errors = env.command_manager.get_term("goals").tracking_errors()
    distance, angle = errors["root_distance"], errors["root_angle"]
    return (distance > 0.5) | (angle > 1.0) | ~torch.isfinite(distance) | ~torch.isfinite(angle)
