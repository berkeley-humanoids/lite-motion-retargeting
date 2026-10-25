"""The ReActor robot, environment and PPO runner, after tables 1 and 2 of the paper."""

import mujoco
import numpy as np
from lite_description import get_mjcf_path
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
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
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.config import RslRlModelCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from ...robots import load_lite_model
from .mdp import (
    STEPS_PER_ITERATION,
    GoalCommandCfg,
    ResidualJointActionCfg,
    RootWrenchActionCfg,
    goal_features,
    joint_target_error,
    lost_track,
    motion_end,
    root_height,
    tracking_error,
    wrench_l1,
)

#: The robot that ReActor simulates.
ROBOT: str = "lite_pro"

#: Hz. The closed-loop natural frequency of every PD actuator: stiffness = armature * w^2.
#: The tracking task uses 5 Hz for its Damiao motors, and so does every joint here. At 5 Hz,
#: the stiffness is a quarter of that at 10 Hz, which gives the policy a wider range of targets.
NATURAL_FREQUENCY_HZ: float = 5.0


def lite_spec(joints: tuple[str, ...]) -> mujoco.MjSpec:
    """Load Lite Pro with a free base and no actuators or sensors.

    This welds each hinge that is not in ``joints`` at 0.
    """
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


def pd_gains(joints: tuple[str, ...]) -> dict[str, tuple[float, float, float]]:
    """Return the stiffness, the damping and the effort limit of each of ``joints``.

    A joint keeps the armature and the effort limit of its actuator in the MJCF. Its gains
    come from :data:`NATURAL_FREQUENCY_HZ`, with a damping ratio of 2, as BeyondMimic sets them.
    """
    model = load_lite_model(ROBOT)
    omega = 2.0 * np.pi * NATURAL_FREQUENCY_HZ
    gains = {}
    for actuator in range(model.nu):
        joint_id = model.actuator_trnid[actuator, 0]
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint in joints:
            armature = float(model.dof_armature[model.jnt_dofadr[joint_id]])
            gains[joint] = armature * omega**2, 4.0 * armature * omega, float(model.actuator_forcerange[actuator, 1])
    return gains


def robot_cfg(joints: tuple[str, ...]) -> EntityCfg:
    """Return the robot entity: ``joints`` with ideal PD actuators, and every other hinge welded."""
    groups: dict[tuple[float, float, float], list[str]] = {}
    for joint, gains in pd_gains(joints).items():
        groups.setdefault(gains, []).append(joint)
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=tuple(names), stiffness=stiffness, damping=damping, effort_limit=effort
        )
        for (stiffness, damping, effort), names in groups.items()
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
        spec_fn=lambda: lite_spec(joints),
        articulation=EntityArticulationInfoCfg(actuators=actuators),
    )


#: The weight of the L1 norm of the root force, in N, and of the root torque, in N m, in
#: table 1 of the paper. The paper calls it its most sensitive weight: a higher weight makes
#: the motion more physical, but fails on more of the hard motions (its fig. 8).
WRENCH_WEIGHT: float = 1e-2


def env_cfg(joints: tuple[str, ...], num_envs: int, goals: GoalCommandCfg) -> ManagerBasedRlEnvCfg:
    """Return the environment. Its terms and their weights follow table 1 of the paper."""
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
    # A unit action moves the target of a joint by a quarter of its effort limit over its stiffness.
    scale = {joint: 0.25 * effort / stiffness for joint, (stiffness, _, effort) in pd_gains(joints).items()}
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
                entity_name="robot", actuator_names=(".*",), scale=scale, use_default_offset=False,
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
            "wrench_force": RewardTermCfg(func=wrench_l1, weight=-WRENCH_WEIGHT, params={"part": "force"}),
            "wrench_torque": RewardTermCfg(func=wrench_l1, weight=-WRENCH_WEIGHT, params={"part": "torque"}),
        },
        # In playback, export decides when an env resets.
        terminations={} if goals.playback else {
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "motion_end": TerminationTermCfg(func=motion_end, time_out=True),
            "lost_track": TerminationTermCfg(func=lost_track),
        },
        # One env step must take one frame, 1 / FPS: four physics steps of 5 ms. A fall or a
        # motion on the ground can need more than 512 constraint rows.
        sim=SimulationCfg(nconmax=128, njmax=1024, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
        decimation=4,
        episode_length_s=1.0e6 if goals.playback else 10.0,
        auto_reset=not goals.playback,
    )


def agent_cfg(iterations: int, run_name: str) -> RslRlOnPolicyRunnerCfg:
    """Return the PPO config of table 2 of the paper."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=STEPS_PER_ITERATION,
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
