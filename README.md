# Lite Motion Retargeting

Retargets motion capture sets to the Berkeley Lite humanoid. A source is either the
original human motion, or motion that its authors already retargeted to the 29-DoF
Unitree G1. One clip becomes one layered [Rerun](https://rerun.io) motion. The LAFAN1 motions are published in
[`Berkeley-Humanoids/Lite-LAFAN1-Dataset`][dataset], which you can
[open in a browser][viewer] or browse in the [Lite Motion Dataset][space] Space.

## Sources

| Source | Clips | Rate | Size | License | On the Hub |
|---|---|---|---|---|---|
| `lafan1bvh`: [LAFAN1 BVH][lafan1bvh], human | 77 | 30 Hz | 0.35 GB | CC BY-NC-ND 4.0 | No |
| `lafan1`: [LAFAN1][lafan1] | 40 | 30 Hz | 0.3 GB | CC BY-NC-ND 4.0 | Yes |
| `amass`: [AMASS][amass], all subsets | 17,717 | 60 to 250 Hz | 7 GB | AMASS: non-commercial research, no redistribution | No |
| `motiondecode`: [ChingMu MotionDecode][motiondecode] | 92,559 | 120 Hz | 148 GB | ChingMu: non-commercial research, not to be hosted as a competing dataset | No |
| `kimodo`: [Kimodo G1][kimodo] | 10,000 | 50 Hz | 1.2 GB | Apache-2.0 | No |
| `kimodolh2`: [Kimodo G1 long-horizon v2][kimodolh2] | 10,000 | 50 Hz | 5.8 GB | Apache-2.0 | No |
| `kimodolh3`: [Kimodo G1 long-horizon v3][kimodolh3] | 10,000 | 50 Hz | 12.8 GB | Apache-2.0 | No |

`scripts/sources.py` defines each source and its loader. The motions of `amass` and
`motiondecode` stay in your local `dataset/`, for training. `publish.py` uploads only
`lafan1`.

`lafan1bvh` is the human motion that `lafan1` was retargeted from: the BVH takes of
Ubisoft's LAFAN1, which come as a zip on GitHub, not on the Hub. Its 40 takes that `lafan1`
also holds have the same names and frames, so the two paths compare clip for clip. The
other sources ship G1 motion only. The raw AMASS takes need an MPI account, MotionDecode
gives its BVH files on request, and Kimodo generated its clips on the G1 directly.

`kimodo` is not motion capture. NVIDIA's Kimodo model generated each clip from a text
prompt, such as "a person throwing boxing punches", and a tracking policy played it on a
simulated G1. Every clip is 4 s long, and its category is its prompt. It ships as LeRobot
shards, which `download.py` splits into a file per clip. `kimodolh2` and `kimodolh3` come
from the same model and policy, but each prompt chains such motions, as in "combat drill:
kick → walk → sidestep". A v2 episode chains 3 to 8 motions over about 20 s, and a v3
episode about 50 over about 43 s. Their category is the family that the prompt names.

AMASS bundles a license per subset, in `.cache/amass/g1/<subset>/license.txt`. Some
subsets, such as CMU, are less restrictive than AMASS as a whole.

## Use

```bash
uv sync
uv run scripts/download.py               # G1 model and LAFAN1 -> .cache/
uv run scripts/retarget.py --workers -1  # every clip, every core
uv run rerun dataset/*/lafan1_walk1_subject1.rrd dataset/blueprints/lite_pro.rbl
uv run scripts/publish.py --clear        # upload LAFAN1 and the browser to the Hub
```

To retarget another source, give `--source` to both scripts. `--include` takes Hub glob
patterns, so you can download part of a large set:

```bash
uv run scripts/download.py --source amass --include 'g1/CMU/**'
uv run scripts/retarget.py --source amass --clip '^CMU/' --workers -1
uv run rerun dataset/*/amass_CMU_07_07_01_poses_120_jpos.rrd \
    --asset dataset/assets/lite_pro.rrd dataset/blueprints/lite_pro.rbl

uv run scripts/download.py --source motiondecode \
    --include 'samples/1.3.Basic_Gait_Category/**' 'samples/5.Dance/**'
uv run scripts/retarget.py --source motiondecode --workers -1

uv run scripts/download.py --source lafan1bvh
uv run scripts/retarget.py --source lafan1bvh --clip '^walk1_subject1$' --validate-only
```

To write to the Hub, run `hf auth login` or set `HF_TOKEN`.

### Browse the motions

`site/` is a static page that lists every motion in `dataset/` as a card. A card plays its
motion when you point at it, and opens onto a stage with a timeline, frame stepping, the
path of the pelvis, and the command that opens the motion in Rerun. `#<motion>` at the end
of the address opens that motion.

```bash
uv run scripts/build_site.py                      # dataset/ -> site/data/
uv run scripts/serve.py                           # then open http://localhost:8000
```

`build_site.py` reads the robot layer of each motion, which is what training reads, and
writes it for the page as float32 rows, end to end in a few packs per source. The page
loads the robot meshes once, and reads a motion as a byte range of its pack only when its
card comes into view. `python -m http.server` answers a range with the whole file, so
serve the page with `serve.py`, or any server that answers ranges, such as Caddy.
`--sources lafan1` builds a page with LAFAN1 alone. Build
`--sources lafan1 kimodo kimodolh2 kimodolh3` for a page others can reach: the licenses
of AMASS and MotionDecode bar sharing their motions.

A motion plays on a character, **Animation**, which the page shows first, or on the robot
of the URDF, **URDF**. The character is the VRM 1.0 model at `site/character.vrm`, which
git does not track. Without it, the page shows the robot only. The character takes the
joint angles as they are: a joint turns its bone about the axis the joint has in the URDF
at rest, where both stand in a T-pose. Where the robot stacks three joints at one point, at
the waist, neck, shoulders, wrists, hips and ankles, the bone is a ball joint and takes all
three, in the order the URDF chains them. A finger takes its two knuckle joints the same
way. The pelvis moves the hips, so that the hip joints of both line up and both stand with
their soles on the floor.

`site/collections.json` lists the folders the page reads, `data/` by default. A folder
can be on the Hub instead, since the Hub answers range requests from any site. That is
how the [Space][space] works: `publish.py` writes the LAFAN1 motions for the page into
`web/` of the dataset repo, then uploads `site/` to the static Space, with a
`collections.json` that points at `web/` as of that commit. A collection marked
`"private": true` loads only for a visitor who signs in with Hugging Face, with the
token of their account, so a private repo shows only to those who can read it.

| Script | What it does |
|---|---|
| `download.py` | Downloads the G1 model into `.cache/g1/`, and a source into `.cache/<source>/`. |
| `retarget.py` | Retargets each clip, measures it, and writes its two layers under `dataset/`. |
| `sources.py` | The sources, the loader that reads a clip of each, and how a motion is named. |
| `human.py` | The human skeleton that every human source maps onto, and the LAFAN1 BVH reader. |
| `mapping.py` | The goal of each Lite link from a human pose, the foot planting, and the IK that reaches the goals. |
| `reactor.py` | Retargets the motions of a store again, by RL in physics, after ReActor. See [ReActor](#reactor). |
| `store.py` | Defines the file format. |
| `publish.py` | Fills the motion index of `DATASET_CARD.md`, uploads it with the store and the browser data, then the page to the Space. |
| `build_site.py` | Writes the motions and the robot meshes that `site/` plays into `site/data/`. |
| `serve.py` | Serves `site/` on this machine, with range requests. |
| `common.py` | The G1 joint table, the model loaders, resampling, forward kinematics, the soles, and foot contacts. |

`retarget.py` takes `--source <name>`, `--clip <regex>`, `--no-ik`, `--ik-iters N`,
`--workers N`, `--out <dir>`, which writes the store to another directory, and
`--validate-only`, which prints the metrics of the first clip and writes nothing. The regex matches the path of a clip below the clip directory of its
source, without the suffix, such as `CMU/07/07_01_poses_120_jpos`. A motion is named
after the same path: `amass_CMU_07_07_01_poses_120_jpos`.
`publish.py` takes `--dry-run`, `--repo-id <id>`, `--space-id <id>`, and `--clear`,
which deletes every file in the Hub repo that this store does not replace.

### Cost

The G1 IK solves about 500 frames per second on one core, and the human IK about 100. A
robot layer holds 12 to 16 MB
for each minute of motion. A base layer keeps the source rate, so it adds 10 to 60 percent
to that. These are the measured totals:

| Source | Motions | Hours | Store | `site/data` |
|---|---|---|---|---|
| `lafan1` | 40 | 2.5 | 2.6 GB | 0.07 GB |
| `lafan1bvh` | 77 | 4.6 | 6.0 GB | |
| `amass`, all subsets | 17,717 | 59 | 66 GB | 2.0 GB |
| `motiondecode` | 92,558 | 991 | 1,146 GB | 29.9 GB |
| `kimodo` | 9,490 | 10 | 11 GB | 0.3 GB |
| `kimodolh2` | 9,947 | 54 | 44 GB | 1.8 GB |
| `kimodolh3` | 9,941 | 118 | 104 GB | 3.5 GB |
| All | 139,693 | 1,234 | 1.37 TB | 37.6 GB |

One MotionDecode clip is empty upstream. On 128 cores, 64,224 MotionDecode clips took
3 h 40 min to retarget, and `build_site.py` takes about 45 min over every source.

The 9 MB robot geometry is written once, to `dataset/assets/lite_pro.rrd`, and
`rerun --asset` attaches it to the motion you open. Only the native viewer loads assets,
so a `lafan1` motion, which the Hub card opens in the web viewer, carries its own copy.

## How a clip is retargeted

A source is 30 to 250 Hz, and a motion is written at 50 Hz, the control rate of the
training environment. `retarget.py` resamples the source before it does anything
else, so the IK solves every frame that reaches the file, and the body poses stay
the exact kinematics of the joints beside them. The loader of the source decides which
of two paths a clip takes.

### A human clip

1. **One skeleton.** `human.py` reads the clip into the world poses of one human
   skeleton, after the keypoints of the Lite conventions: pelvis, hips, knees, feet,
   toes, waist, chest, neck, head, shoulders, elbows and hands. A rotation is
   rest-normalized: it is identity when the body is as in a T-pose that faces +x. A new
   format thus needs a reader and nothing else.
2. **Goals.** `mapping.goals` builds a skeleton with the proportions of Lite and the
   joint rotations of the human. The pelvis follows the human pelvis, scaled by the
   ratio of the two pelvis heights. Every other joint hangs off the joint above it by
   the Lite segment between them, turned as the human segment turns. Each Lite link in
   `mapping.LITE_PAIRS` gets a goal pose: the pelvis, thighs, shins, feet, chest, head,
   upper arms, forearms and hands.
3. **Foot planting.** A skeleton with other proportions does not step where the human
   steps. `common.planted` finds the frames where a human foot stands on the floor, from
   the speed and height of its ankle and toe. `mapping.plant` then shifts the goal
   skeleton, frame by frame, so that a planted foot keeps its sole point in place and
   on the floor. It also lifts a foot whose sole would dip through the floor.
4. **IK.** Per-frame [`mink`](https://github.com/kevinzakka/mink) IK with a free base
   moves Lite onto the goals, position and orientation, from the frame before. The
   thigh, shin, upper arm and forearm goals pin the twist of each limb. Without them, the
   IK drifted along a joint limit onto another branch of the hip, then snapped back in one
   frame. No joint turns faster than 20 rad/s. The limit binds where an arm passes the
   T-pose, in which the shoulder pitch and yaw of Lite share an axis.

The goals are the part that a backend shares. A physics backend, such as an RL policy
that tracks them in simulation, can take the same goals in place of the IK.

### A G1 clip

1. **Direct remap.** `lite_q = sign * g1_q + offset`, from the table in
   `common.G1_TO_LITE`.
2. **IK refinement.** Per-frame IK moves the result toward the pelvis-local pose of both
   feet and both hands of the G1, position and orientation. A per-DOF posture cost holds
   the solution near step 1, so the step adjusts the pose and does not rearrange it.

### Both

**Grounding.** A constant shift of the pelvis height per clip puts the lowest point of
the Lite soles on the floor in the median frame. Each source puts the G1 at its own
height, and AMASS puts the pelvis at the floor, so the shift is measured on the Lite
result. The median ignores the few frames where a heel or a toe dips through the floor.

**Driven joints.** A source drives only the Lite joints it has data for. A G1 source
drives the 29 joints of `common.G1_TO_LITE`, and has no neck, ankle yaw or fingers. A human
source drives every joint between the pelvis and a link with a goal: 34 joints for
LAFAN1, which has no fingers. Every other joint holds its default, 0. The robot layer
keeps the driven joints as `active_joint_names`, so a motion with fewer of them can still
train the robot it covers, such as a G1 motion for the legs of the Lite Biped. The motion
browser shows them, with how many joints of each Lite variant a motion drives, and
filters the motions by the joint groups they drive.

**Metrics.** `retarget.motion_metrics` measures every motion. The robot layer keeps the
numbers as its `metrics` property, and a run prints their mean:

| Metric | What it measures |
|---|---|
| `foot_error_cm`, `hand_error_cm` | Mean distance from a goal to its link |
| `foot_angle_deg`, `hand_angle_deg` | Mean angle between a goal and its link |
| `foot_slide_cm_s` | Mean horizontal speed of the touching sole point, while the source stands on that foot |
| `planted_gap_cm` | Mean distance from the lowest sole point to the floor, while the source stands on that foot |
| `penetration_cm` | 95th percentile, over frames, of the deepest sole point below the floor |
| `joint_speed_max` | Fastest joint between two frames, rad/s |

The two paths do not aim at the same goals, so their goal errors do not compare. The
floor metrics and the joint speed do. These are the means over the 40 LAFAN1 takes that
both sources hold, with the current Lite Pro model:

| Path | Foot slide | Planted gap | Penetration | Fastest joint |
|---|---|---|---|---|
| `lafan1bvh`, human | 2.6 cm/s | 0.40 cm | 1.1 cm | 19 rad/s |
| `lafan1`, G1 | 9.2 cm/s | 1.53 cm | 4.1 cm | 21 rad/s |

The human path has less foot slide, a smaller planted gap, and less penetration on 39 or
more of the 40 takes.

The IK solves against the Lite Pro MJCF of `lite-description`, and the robot layer draws
the URDF of the same package. A new asset version therefore gives new motions.

## ReActor

`reactor.py` retargets the motions of a store again, in physics, after Disney's
[ReActor](https://arxiv.org/abs/2605.06593). The IK of `retarget.py` knows nothing of
mass, contact or torque, so its motions can float, sink, slide, or ask more of a motor
than it has. ReActor tracks the same goals with a policy in MuJoCo Warp, through mjlab,
and the simulated motion is the retarget:

1. **Policy.** It sets a PD target per joint around the IK solution of the frame, and pushes
   the pelvis with a wrench. The reward follows table 1 of the paper and penalizes the
   wrench, so the policy learns to need little of it.
2. **Goals.** They learn a position and an orientation offset per link, and a height offset
   per motion. After each PPO iteration, one gradient step moves them toward where the
   simulated links went, with the approximate bilevel gradient of the paper.
3. **Export.** The policy plays each motion from its first frame to its end. The
   simulated joints become a robot layer of a new store, with the learned goals, the
   metrics of `retarget.py`, and the peak and mean wrench. Where play loses track of the
   goals, export cuts the motion into pieces, as told below.

The simulation is ideal: a PD position actuator per joint with the armature and effort
limit of the MJCF, every collision geom of the MJCF, and no noise, delay or domain
randomization. ReActor makes reference motions. The policy that runs on the robot trains
on them in [Lite-Motion-Tracking](https://github.com/Berkeley-Humanoids/Lite-Motion-Tracking).

ReActor reads the robot layers of a store, so it runs on any source that `retarget.py`
wrote, human or G1. It welds every joint that no motion drives, such as the fingers. The
`reactor` extra adds mjlab, which pins MuJoCo to 3.11 for the whole project.

```bash
uv run --extra reactor scripts/reactor.py train --source lafan1bvh --clip '^walk' --iterations 3000
uv run --extra reactor scripts/reactor.py export --run logs/reactor/<run> --out dataset_reactor
```

`train` writes its run to `logs/reactor/`, with the motions it read (`library.npz`), the
goal offsets (`goals.pt`), the PPO checkpoints, and TensorBoard logs. With 4,096 envs on
one RTX A4500, an iteration takes 1.6 s.

A LAFAN1 take runs for minutes, and a policy trained on 10 s episodes can stumble in one.
`export` then cuts the motion 1 s before the stumble, starts again from the IK solution,
and writes each piece of 2 s or more as `<motion>_f<first frame>`. A motion that never
stumbles keeps its name. The pieces after a stumble track worse, and their metrics say so.

Two runs on LAFAN1 give the first results:

| Run | Takes | Iterations | Time | Kept | Lost track | Played whole |
|---|---|---|---|---|---|---|
| Walking | 12 | 3,000 | 1.3 h | 99% of 48 min | 39 times | 6 takes |
| All | 77 | 10,000 | 4.4 h | 98% of 276 min | 354 times | 26 takes |

On the takes that each run played whole, against the IK, weighted by length:

| Metric | Walking run | IK | All run | IK |
|---|---|---|---|---|
| Foot slide | 2.75 cm/s | 1.97 cm/s | 3.53 cm/s | 2.51 cm/s |
| Planted gap | 0.14 cm | 0.33 cm | 0.18 cm | 0.73 cm |
| Penetration | 0.50 cm | 0.92 cm | 0.61 cm | 1.46 cm |
| Fastest joint | | | 25 rad/s | 19 rad/s |
| Goal error, feet | 6.8 cm | 0.2 cm | 8.3 cm | 0.3 cm |
| Root wrench, mean | | none | 24 N | none |
| Root wrench, peak | 65 to 107 N | none | 265 N | none |

The simulated feet sit on the floor better than the IK feet: the run on all takes has a
smaller planted gap on all 26 takes, and less penetration on 23. But the feet slide more,
the joints move faster, and the policy leans on the wrench. Lite Pro weighs 327 N. The mean
wrench is 7% of that weight, and the peak is 80%. Over all 77 takes, the mean wrench goes
from 17 N on the walking takes to 61 N on the obstacle takes, and no motion has a mean
below 10 N. The falls and the motions on the ground are the worst, with a foot slide of 9
to 13 cm/s.

On the walking takes, the run on all takes does better than the walking run: less foot
slide on all 9 motions that both runs wrote, and less goal error on 7. Its peak wrench is
higher.

The paper trains for 20,000 iterations, and names the wrench penalty its most sensitive
setting. The test that counts is downstream: a tracking policy without the wrench, trained
on either set of motions.

The implementation departs from the paper in four places. There is no public code of the
paper to compare against, and the unofficial reproduction has no license, so none of its
code is used here.

| Paper | Here | Why |
|---|---|---|
| PD targets, and a reset at the nominal pose with a ramp-in | PD targets around the IK solution, and a reset at the IK state | The IK solution exists, so the policy starts from it |
| Offsets on the source body frames | Offsets on the goals of `mapping.py`, in the frame of each link | The goals already fit Lite: its proportions, and planted feet |
| Offset bounds of 0.5 | 0.1 m, 0.3 rad, 0.05 m | Lite is about 1 m tall |
| Projected SGD on the offsets | Adam, then the projection | The paper gives no step size |

## The file format

`store.py` writes `dataset/` as one directory per layer: `base/` for the source clip
as published at its own rate, `lite_pro/` for the 50 Hz retarget, `assets/` for the robot
geometry, and `blueprints/` for the viewer layout. `DATASET_CARD.md` documents that
store, and `store.py` documents every entity path.

The robot layer is what training and deployment read, and it is self-contained: its
joints reset the robot, and its body poses are the goal, turned into the frame
convention of the Lite links so a reward compares them against the robot directly. To
draw the robot, the viewer also needs the geometry, from the asset or from the file.
It stores no reached pose, because that is the forward kinematics of the joints
beside it.

`store.py` is the only module that writes Rerun data, and it uses `rerun-sdk` alone.
The layout on disk matches the motion store of
[MikuMotionTools](https://github.com/T-K-233/MikuMotionTools), so its `MotionStore`
reads these files.

The training repository,
[Lite-Motion-Tracking](https://github.com/Berkeley-Humanoids/Lite-Motion-Tracking),
still reads the older single-layer `tracking/*.rrd`. It needs a reader for this store
before it can train on these motions.

[dataset]: https://huggingface.co/datasets/Berkeley-Humanoids/Lite-LAFAN1-Dataset
[space]: https://huggingface.co/spaces/Berkeley-Humanoids/Lite-Motion-Dataset
[lafan1]: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
[lafan1bvh]: https://github.com/ubisoft/ubisoft-laforge-animation-dataset
[amass]: https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics
[motiondecode]: https://huggingface.co/datasets/CMRobot/MotionDecode
[kimodo]: https://huggingface.co/datasets/cagataydev/kimodo-g1-motion-v1
[kimodolh2]: https://huggingface.co/datasets/cagataydev/kimodo-g1-longhorizon-v2
[kimodolh3]: https://huggingface.co/datasets/cagataydev/kimodo-g1-longhorizon-v3
[viewer]: https://rerun.io/viewer?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FBerkeley-Humanoids%2FLite-LAFAN1-Dataset%2Fresolve%2Fmain%2Flite_pro%2Flafan1_dance1_subject1.rrd
