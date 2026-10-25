# Retargeting

This repo retargets motion to Lite Pro with three methods. Each method is a subpackage of `src/lite_motion_retargeting/retargeting/`:

| Method | Package | Command | What it does |
|---|---|---|---|
| IK | `ik/` | `uv run retarget` | The default. Per-frame inverse kinematics on every clip of a source. It writes the store, `dataset/`. |
| ReActor | `reactor/` | `uv run reactor` | Tracks the goals of an IK store with a policy in physics, and writes a new store. See [ReActor](reactor.md). |
| UMR | `umr/` | Its own scripts | An outside kinematic retargeter that matches surface points of an SMPL-X body to the robot. It runs in an environment of its own. See [UMR](umr.md). |

`retargeting/metrics.py` measures a motion of any method. `uv run compare` measures the methods on the same frames. See [Compare the methods](comparison.md).

## How the IK retargets a clip

A source runs at 30 to 250 Hz. `uv run retarget` writes every motion at 50 Hz, the control rate of the training environment. It resamples the source first, so the IK solves every frame that reaches the file. As a result, the body poses stay the exact kinematics of the joints beside them. The loader of the source decides which of two paths a clip takes: the human path or the G1 path.

### A human clip

1. **One skeleton.** `sources/human.py` defines one human skeleton, and a reader of each format, such as `sources/bvh.py`, reads a clip into the world poses of that skeleton. The skeleton follows the keypoints of the Lite conventions: pelvis, hips, knees, feet, toes, waist, chest, neck, head, shoulders, elbows and hands. Each rotation is rest-normalized: it is identity when the body stands in a T-pose that faces +x. Thus, a new format needs only a reader.
2. **Goals.** `retargeting.ik.human.goals` builds a goal skeleton with the proportions of Lite and the joint rotations of the human. Its pelvis follows the human pelvis, scaled by the ratio of the two pelvis heights. Every other joint hangs from the joint above it by the Lite segment between them, and that segment turns as the human segment turns. Each Lite link in `retargeting.ik.human.LITE_PAIRS` gets a goal pose: the pelvis, thighs, shins, feet, chest, head, upper arms, forearms and hands.
3. **Foot planting.** A skeleton with other proportions does not step where the human steps. `kinematics.planted` finds the frames where a human foot stands on the floor, from the speed and height of its ankle and toe. Then `retargeting.ik.human.plant` shifts the goal skeleton, frame by frame, so that a planted foot keeps its sole point in place and on the floor. It also lifts a foot whose sole would dip through the floor.
4. **IK.** Per-frame [`mink`](https://github.com/kevinzakka/mink) IK with a free base moves Lite onto the goals, in position and orientation. Each frame starts from the solution of the frame before. The thigh, shin, upper arm and forearm goals pin the twist of each limb. Without these goals, the IK drifted along a joint limit onto another branch of the hip, then snapped back in one frame. No joint turns faster than 20 rad/s. This limit binds where an arm passes the T-pose, in which the shoulder pitch and yaw of Lite share an axis.

The goals are the interface that all methods share. [ReActor](reactor.md) tracks them in simulation in place of the IK.

### A G1 clip

1. **Direct remap.** `lite_q = sign * g1_q + offset`, from the table in `retargeting.ik.g1.G1_TO_LITE`.
2. **IK refinement.** Per-frame IK moves the result toward the pelvis-local poses of both feet and both hands of the G1, in position and orientation. A per-DoF posture cost holds the solution near the result of step 1. Thus, this step adjusts the pose but does not rearrange it.

### Both paths

**Grounding.** A constant shift of the pelvis height per clip puts the lowest point of the Lite soles on the floor in the median frame. Each source puts the G1 at its own height, and AMASS puts the pelvis at the floor. Thus, `retargeting/` measures the shift on the Lite result. The median ignores the few frames where a heel or a toe dips through the floor.

**Driven joints.** A source drives only the Lite joints that it has data for, and every other joint holds its default, 0. A G1 source drives the 29 joints of `retargeting.ik.g1.G1_TO_LITE`, and has no neck, ankle yaw or fingers. A human source drives every joint between the pelvis and a link with a goal: 34 joints for LAFAN1, which has no fingers. The robot layer keeps the driven joints as `active_joint_names`. Thus, a motion with fewer driven joints can still train the robot that it covers, such as a G1 motion for the legs of the Lite Biped. The [motion browser](browser.md) shows the driven joints of each motion.

The IK solves against the Lite Pro MJCF of `lite-description`, and the robot layer draws the URDF of the same package. Thus, a new version of `lite-description` gives new motions.

## Metrics

`retargeting.metrics.motion_metrics` measures every motion. The robot layer keeps the numbers as its `metrics` property, and a run prints their mean:

| Metric | What it measures |
|---|---|
| `foot_error_cm`, `hand_error_cm` | Mean distance from a goal to its link |
| `foot_angle_deg`, `hand_angle_deg` | Mean angle between a goal and its link |
| `foot_slide_cm_s` | Mean horizontal speed of the touching sole point, while the source stands on that foot |
| `foot_slide_median_cm_s` | Median of the same speed |
| `near_floor_slide_cm_s`, `near_floor_slide_median_cm_s` | Mean and median of the same speed, while the sole is within 1 cm of the floor, whatever the source does |
| `planted_gap_cm` | Mean distance from the lowest sole point to the floor, while the source stands on that foot |
| `penetration_cm` | 95th percentile, over frames, of the deepest sole point below the floor |
| `joint_speed_max` | Fastest joint between two frames, rad/s |

The lowest sole point is exact: `kinematics.sole` takes the collision capsules of each foot as segments with a radius, and boxes as their corners. A metric over an empty set, such as the slide of a motion that never touches the floor, is NaN.

The IK plants the feet where the source stands, and `foot_slide_cm_s` measures the slide on the same frames. Thus `foot_slide_cm_s` measures what the IK aims for. A method that does not plant the feet, such as ReActor, can still keep the feet still where the source stands, so the metric applies to it too. `near_floor_slide_cm_s` uses the height of the robot sole instead, so it does not depend on the mask of the source. But height alone decides it, so it also counts a swing foot that passes close to the floor.

The human path and the G1 path aim at different goals, so you cannot compare their goal errors. You can compare the floor metrics and the joint speed. This table gives their means over the 40 LAFAN1 takes that both sources hold, with the current Lite Pro model:

| Path | Foot slide | Planted gap | Penetration | Fastest joint |
|---|---|---|---|---|
| `lafan1bvh`, human | 2.6 cm/s | 0.40 cm | 1.05 cm | 19 rad/s |
| `lafan1`, G1 | 9.2 cm/s | 1.52 cm | 4.07 cm | 21 rad/s |

The human path has less foot slide, a smaller planted gap, and less penetration on 39 or more of the 40 takes.

## Code layout

The package, `src/lite_motion_retargeting/`, follows the layout of [mjlab](https://github.com/mujocolab/mjlab). `scripts/` holds a module per command, and `pyproject.toml` lists the commands. The other modules are the library that the commands share:

| Module | What it does |
|---|---|
| `sources/` | The sources, the rule that names a motion, and a reader per format: `g1.py` for the G1 sources, `bvh.py` for LAFAN1 and BONES-SEED, and `smplx.py` for AMASS SMPL-X. `human.py` defines the human skeleton that every human reader maps onto. |
| `retargeting/` | The three methods, and `metrics.py`. `ik/g1.py` is the G1 path, and `ik/human.py` the human path. `reactor/` is the mjlab environment of ReActor: its terms in `mdp.py`, its configs in `env_cfg.py`, and the motions that it reads in `library.py`. `umr/` holds the config of UMR for Lite Pro, the scripts that prepare its input, and the reader of its motions. |
| `robots.py` | Loads the Lite and G1 models, and names their joints and bodies. |
| `kinematics.py` | Resampling, forward kinematics, the soles, and foot contacts. |
| `store.py` | Writes the layers of a motion, and reads a robot layer back. See [The file format](store.md). |
| `browser.py` | Writes the motions and the robot meshes for the [motion browser](browser.md). |
| `hub.py` | Lists the Hub datasets, and names the Space. See [Publish to the Hub](publish.md). |
