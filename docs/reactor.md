# ReActor

`uv run reactor` retargets the motions of a store again, in physics, with the method of Disney's [ReActor](https://arxiv.org/abs/2605.06593). The IK of `retargeting/ik/` ignores mass, contact and torque. Thus, its motions can float, sink, slide, or ask a motor for more than the motor has. ReActor tracks the same goals with a policy in MuJoCo Warp, through mjlab, and the simulated motion is the retargeted motion:

1. **Policy.** The policy sets a PD target per joint around the IK solution of the frame, and pushes the pelvis with a wrench. The reward follows table 1 of the paper and penalizes the wrench, so the policy learns to use little of it.
2. **Goals.** The goals learn a position offset and an orientation offset per link, and a height offset per motion. After each PPO iteration, one gradient step moves the goals toward the simulated links. The step uses the approximate bilevel gradient of the paper.
3. **Export.** The policy plays each motion from its first frame to its end. The simulated joints become a robot layer of a new store, with the learned goals, the metrics of `retargeting/metrics.py`, and the peak and mean wrench. If the policy loses track of the goals, `export` cuts the motion into pieces, as described below.

The simulation is ideal: a PD position actuator per joint with the armature and effort limit of the MJCF, every collision geom of the MJCF, and no noise, delay or domain randomization. ReActor makes reference motions. The policy that runs on the robot trains on these motions in [Lite-Motion-Tracking](https://github.com/Berkeley-Humanoids/Lite-Motion-Tracking).

ReActor reads the robot layers of a store, so it runs on any source that `uv run retarget` wrote, human or G1. ReActor welds every joint that no motion drives, such as the fingers. The `reactor` extra adds mjlab, which pins MuJoCo to 3.11 for the whole project.

```bash
uv run --extra reactor reactor train --source lafan1bvh --clip '^walk' --iterations 3000
uv run --extra reactor reactor export --run logs/reactor/<run> --out dataset_reactor
```

`train` writes its run to `logs/reactor/`, with its arguments (`train.json`), the motions that it read (`library.npz`), the goal offsets (`goals.pt`), the PPO checkpoints, and TensorBoard logs. With 4,096 environments on one RTX A4500, an iteration takes 1.6 s.

A LAFAN1 take runs for minutes, and a policy that trained on 10 s episodes can stumble in a take. If the policy stumbles, `export` cuts the motion 1 s before the stumble and starts again from the IK solution. It writes each piece of 2 s or more as `<motion>_f<first frame>`. A motion without a stumble keeps its name. The pieces after a stumble track the goals less well, and their metrics show this.

## Runs

Four runs give these results. Two ran on an older Lite Pro model, `lite-description` `99cdef0`, and two on the current one:

| Run | Model | Motions | Iterations | GPU | Time | Kept | Lost track | Played whole |
|---|---|---|---|---|---|---|---|---|
| Walking | `99cdef0` | 12 | 3,000 | RTX A4500 | 1.3 h | 99% of 48 min | 39 times | 6 |
| All, 10k | `99cdef0` | 77 | 10,000 | RTX A4500 | 4.4 h | 98% of 276 min | 354 times | 26 |
| All, 20k | current | 77 | 20,000 | RTX 3090 | 6.7 h | 99% of 276 min | 104 times | 48 |
| Kimodo | current | 9,490 | 10,000 | RTX 3080 Ti | 6.6 h | 100% of 630 min | 71 times | 9,421 |

[Compare the methods](comparison.md) measures the two runs on the current model against the IK and UMR.

The policy depends on the wrench. On the 25 LAFAN1 takes that both All runs played whole, the 20k run pushes the pelvis with a mean wrench of 20 N, against 24 N for the 10k run. But its mean peak is 302 N, against 257 N. Lite Pro weighs 258 N. Over the 48 takes that the 20k run played whole, the mean is 33 N and the mean peak 394 N, because the takes that only this run finishes are the hard ones. The Kimodo motions are short and calm: their mean wrench is 2 N, and their median peak 9 N.

On the walking takes, the All, 10k run does better than the Walking run. It has less foot slide on all 9 motions that both runs wrote, and less goal error on 7 of them. But its peak wrench is higher.

The 20k run trains for the 20,000 iterations of the paper. The paper names the wrench penalty its most sensitive setting. The real test is downstream: a tracking policy without the wrench, trained on these motions.

## Two studies on LAFAN1

Each run of these studies trains on the 77 LAFAN1 takes for 20,000 iterations, as the All, 20k run does, and changes one setting. Two runs with the same settings differ by about 0.1 cm/s of slide and 40 losses of track, so a smaller difference is noise. The options that changed these settings, `train --wrench-weight` and `retarget --no-plant-feet`, were removed after the studies.

**A higher wrench penalty makes the feet slide more.** These runs changed the weight of the force and the torque. The paper uses 0.01, and so does `WRENCH_WEIGHT` in `retargeting/reactor/env_cfg.py`. Each run took about 9 h on an RTX A4500:

| Wrench weight | Lost track | Played whole | Kept | Wrench, mean / mean peak | Slide where the source stands, mean / median | The same, on the 5 takes that every run played whole |
|---|---|---|---|---|---|---|
| 0.01, the All, 20k run | **104 times** | **48** | **99%** | 38 / 340 N | **5.2** / **1.4** cm/s | **2.8 cm/s** |
| 0.01, again | 145 times | 45 | 99% | 38 / 320 N | 5.3 / 1.4 cm/s | 3.1 cm/s |
| 0.03 | 2,374 times | 9 | 89% | 0.1 / 4 N | 13.8 / 2.1 cm/s | 9.6 cm/s |
| 0.1 | 2,537 times | 8 | 88% | 0.0 / 2 N | 14.4 / 2.2 cm/s | 10.5 cm/s |
| 0.3 | 2,777 times | 6 | 86% | **0.0** / **1** N | 16.3 / 2.5 cm/s | 11.8 cm/s |

- **At 3 times the weight of the paper, the policy stops using the wrench.** No run uses only a part of it.
- **Without the wrench, Lite Pro keeps its balance with its feet.** Its feet slide 2.6 to 3.1 times as fast where the source stands, also on the takes that the policy plays whole. The policy loses track 16 times as often, and its gap, penetration and root error grow.
- The wrench thus holds the feet still, because it holds the body up.

**The goals of the source do no better than the planted goals.** This run started from goals as the source gives them, without the planting of `retargeting.ik.human.plant`, and from the IK that reaches them. ReActor cannot start from the planted IK and track the goals of the source: on a long take, the shifts of `plant()` add up, and the planted root ends far from the root of the source. The run took 6.7 h on an RTX 3090:

| Goals | Lost track | Played whole | Slide where the source stands, mean / median | The same, on the 35 takes that the 3 runs played whole | Limb direction error | Root error over 1 s |
|---|---|---|---|---|---|---|
| Planted, the All, 20k run | **104 times** | **48** | **5.2** / **1.4** cm/s | **4.0 cm/s** | 9.1° | 6.6 cm |
| Planted, the 0.01 run above | 145 times | 45 | 5.3 / 1.4 cm/s | 4.1 cm/s | 9.2° | 6.9 cm |
| Of the source | 138 times | 39 | 5.4 / 1.5 cm/s | 4.3 cm/s | **8.8°** | **6.1 cm** |

The goals of the source keep the limbs and the root a little closer to the source. On the feet, the difference is within the noise of a run. Thus the planting does not limit ReActor, and the planted goals stay the only goals.

## Differences from the paper

This implementation differs from the paper in four places. The paper has no public code to compare against. The unofficial reproduction has no license, so this repo uses none of its code.

| Paper | Here | Why |
|---|---|---|
| PD targets, and a reset at the nominal pose with a ramp-in | PD targets around the IK solution, and a reset at the IK state | The IK solution exists, so the policy starts from it |
| Offsets on the source body frames | Offsets on the goals of `retargeting/ik/human.py`, in the frame of each link | The goals already fit Lite: its proportions, and planted feet |
| Offset bounds of 0.5 | 0.1 m, 0.3 rad, 0.05 m | Lite is about 1 m tall |
| Projected SGD on the offsets | Adam, then the projection | The paper gives no step size |
