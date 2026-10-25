"""ReActor: retarget by reinforcement learning in physics, after Disney's ReActor.

Paper: "ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting",
https://arxiv.org/abs/2605.06593. The equation and table numbers below are those of the paper.
``docs/reactor.md`` describes the method, its runs, and how it differs from the paper.

A policy tracks the goals of an IK store in MuJoCo Warp, and the simulated motion is the
retarget. The policy sets a PD target per joint around the IK solution of the frame, and
pushes the pelvis with a wrench (eq. 18, 19). The reward penalizes the wrench, so the
policy learns to use little of it.

The goals learn while the policy trains. Each link has a position offset and an orientation
offset in its own frame, and each motion has a height offset (eq. 10, 11, 14). After the
warm start, one step of gradient descent follows the rollout of each PPO iteration. The step
moves the offsets toward the poses that the simulated links reached, with the approximate
bilevel gradient of eq. 6 at alpha = 0. Thus the goals come to ask for what the robot can do.

The environment follows the layout of an mjlab task: ``mdp.py`` holds its terms, and
``env_cfg.py`` builds the robot, the environment and the PPO runner from them.
``library.py`` reads the motions of a store for the environment. ``scripts/reactor.py``
trains and exports.
"""
