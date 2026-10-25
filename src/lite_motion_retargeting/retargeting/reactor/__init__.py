"""ReActor: retarget by reinforcement learning in physics, after Disney's ReActor.

Paper: "ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting",
https://arxiv.org/abs/2605.06593. The equation and table numbers below are those of the paper.

``retargeting`` gives each Lite link a goal pose and an IK solution that reaches it. The
IK knows nothing of mass, contact or torque, so a motion can float, sink, slide, or ask a
motor for more torque than it has. ReActor tracks the goals with a policy in MuJoCo Warp,
and the simulated motion is the retarget. The policy sets a PD target per joint around the
IK solution of the frame, and pushes the pelvis with a wrench (eq. 18, 19). The reward
penalizes the wrench, so the policy learns to use little of it.

The goals learn while the policy trains. Each link has a position offset and an orientation
offset in its own frame, and each motion has a height offset (eq. 10, 11, 14). After the
warm start, one step of gradient descent follows the rollout of each PPO iteration. The step
moves the offsets toward the poses that the simulated links reached, with the approximate
bilevel gradient of eq. 6 at alpha = 0. Thus the goals come to ask for what the robot can do.

The simulation is ideal: a PD position actuator per joint with the armature and the effort
limit of the MJCF, every collision geom of the MJCF, and no noise, delay or domain
randomization. ReActor makes reference motions, not a policy to deploy. A tracking policy
for the robot trains on its output, in Lite-Motion-Tracking.

ReActor reads the store that ``uv run retarget`` wrote, and writes a new store. The new
store has one robot layer per motion, with the simulated joints, the learned goals, and the
metrics of ``retargeting.metrics.motion_metrics``. ReActor welds each joint that no motion drives,
such as a finger joint.

The environment follows the layout of an mjlab task: ``mdp.py`` holds its terms, and
``env_cfg.py`` builds the robot, the environment and the PPO runner from them.
``library.py`` reads the motions of a store for the environment. ``scripts/reactor.py``
trains and exports.
"""
