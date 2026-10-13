# Lite Motion Retargeting

Lite Motion Retargeting retargets motion datasets to the Berkeley Lite humanoid. Each
dataset is a source. A source holds either the original human motion, or motion that its
authors already retargeted to the 29-DoF Unitree G1. Each clip of a source becomes one
layered [Rerun](https://rerun.io) motion.

The Kimodo motions are on the Hub in [`Berkeley-Humanoids/Lite-Kimodo-Dataset`][kimodo-hub],
and the LAFAN1 motions in the private repo [`Berkeley-Humanoids/Lite-LAFAN1-Dataset`][dataset].
Browse them in the [Lite Motions][space] Space. The LAFAN1 motions show there after you
sign in with a Hugging Face account that can read the private repo.

## Sources

| Source | Clips | Rate | Size | License | On the Hub |
|---|---|---|---|---|---|
| `lafan1bvh`: [LAFAN1 BVH][lafan1bvh], human | 77 | 30 Hz | 0.35 GB | CC BY-NC-ND 4.0 | Private |
| `lafan1`: [LAFAN1][lafan1] | 40 | 30 Hz | 0.3 GB | CC BY-NC-ND 4.0 | No |
| `amass`: [AMASS][amass], all subsets | 17,717 | 60 to 250 Hz | 7 GB | AMASS: non-commercial research, no redistribution | No |
| `motiondecode`: [ChingMu MotionDecode][motiondecode] | 92,559 | 120 Hz | 148 GB | ChingMu: non-commercial research, not to be hosted as a competing dataset | No |
| `kimodo`: [Kimodo G1][kimodo] | 10,000 | 50 Hz | 1.2 GB | Apache-2.0 | Yes |
| `kimodolh2`: [Kimodo G1 long-horizon v2][kimodolh2] | 10,000 | 50 Hz | 5.8 GB | Apache-2.0 | Yes |
| `kimodolh3`: [Kimodo G1 long-horizon v3][kimodolh3] | 10,000 | 50 Hz | 12.8 GB | Apache-2.0 | Yes |

`src/lite_motion_retargeting/sources/` defines each source and its loader. The `amass`
and `motiondecode` motions stay in your local store, `dataset/`, for training. `uv run
publish` uploads `lafan1bvh` to a private repo, and the three Kimodo sets to a public one.
See [Publish to the Hub](#publish-to-the-hub).

`lafan1bvh` is the human motion that the authors of `lafan1` retargeted: the BVH takes of
Ubisoft's LAFAN1. These takes come as a zip on GitHub, not on the Hub. `lafan1` holds 40
of them, with the same names and frames, so the two
[retargeting paths](#how-a-clip-is-retargeted) compare clip for clip. The other sources
ship G1 motion only. The raw AMASS takes need an MPI account, MotionDecode gives its BVH
files on request, and Kimodo generated its clips on the G1 directly.

`kimodo` is not motion capture. NVIDIA's Kimodo model generated each clip from a text
prompt, such as "a person throwing boxing punches". Then a tracking policy played the clip
on a simulated G1. Each clip is 4 s long, and its category is its prompt. `kimodo` ships
as LeRobot shards, and `uv run download` splits them into one file per clip.

`kimodolh2` and `kimodolh3` come from the same model and policy, but each prompt chains
several actions, as in "combat drill: kick → walk → sidestep". A v2 clip chains 3 to 8
actions over about 20 s, and a v3 clip chains about 50 actions over about 43 s. The
category of these clips is the family that the prompt names.

AMASS bundles a license per subset, in `.cache/amass/g1/<subset>/license.txt`. Some
subsets, such as CMU, are less restrictive than AMASS as a whole.

## Use

`uv sync` installs the package, `src/lite_motion_retargeting/`, with a command for each
step of the pipeline:

| Command | Reads | Writes |
|---|---|---|
| `uv run download` | a source, on the Hub or the web | `.cache/<source>/` |
| `uv run retarget` | `.cache/<source>/` | the store, `dataset/` |
| `uv run reactor train`, `uv run reactor export` | the store | `logs/reactor/`, then another store |
| `uv run compare` | the store, a ReActor store, the motions of UMR | a table of metrics |
| `uv run build-site` | the store | the browser data, `site/data/` |
| `uv run serve` | `site/` | |
| `uv run publish` | the store, `site/` | the Hub dataset repos, and the Space |

`uv run <command> --help` lists the options of a command. The package runs from its
clone: it keeps `.cache/`, `dataset/` and `logs/` beside the code.

```bash
uv sync
uv run download               # G1 model and LAFAN1 -> .cache/
uv run retarget --workers -1  # every clip, every core
uv run rerun dataset/*/lafan1_walk1_subject1.rrd dataset/blueprints/lite_pro.rbl
uv run publish --clear        # upload the Hub datasets and the browser to the Hub
```

To retarget another source, pass `--source` to both steps. To download part of a large
source, pass Hub glob patterns to `--include`:

```bash
uv run download --source amass --include 'g1/CMU/**'
uv run retarget --source amass --clip '^CMU/' --workers -1
uv run rerun dataset/*/amass_CMU_07_07_01_poses_120_jpos.rrd \
    --asset dataset/assets/lite_pro.rrd dataset/blueprints/lite_pro.rbl

uv run download --source motiondecode \
    --include 'samples/1.3.Basic_Gait_Category/**' 'samples/5.Dance/**'
uv run retarget --source motiondecode --workers -1

uv run download --source lafan1bvh
uv run retarget --source lafan1bvh --clip '^walk1_subject1$' --validate-only
```

### Publish to the Hub

`uv run publish` uploads Hub datasets from the store. A Hub dataset is one Hub dataset
repo with its own card, and it holds the motions of one or more sources. `HUB_DATASETS` in
`src/lite_motion_retargeting/hub.py` lists them:

| Hub dataset | Repo | Sources | Card | Private |
|---|---|---|---|---|
| `lafan1` | [`Berkeley-Humanoids/Lite-LAFAN1-Dataset`][dataset] | `lafan1bvh` | `cards/lafan1.md` | Yes |
| `kimodo` | [`Berkeley-Humanoids/Lite-Kimodo-Dataset`][kimodo-hub] | `kimodo`, `kimodolh2`, `kimodolh3` | `cards/kimodo.md` | No |

Each repo holds the layers of its motions as `<layer>/<source>/<motion>.rrd`, because the
Hub takes at most 10,000 files in one directory. `web/` holds the same motions in the form
that the motion browser plays. `uv run publish` writes the repo into `dataset/hub/<name>/`
as hard links to the store, with the card and `web/`. Then it uploads that directory, in
several commits if it is large. Last, it uploads the page to the [Space][space], with a
collection for the `web/` of each Hub dataset. Thus the Space plays the motions of every
Hub dataset on one page.

```bash
uv run publish --dry-run          # write dataset/hub/<name>/, and upload nothing
uv run publish                    # every Hub dataset, then the page
uv run publish --datasets lafan1  # one Hub dataset, then the page
uv run publish --datasets         # only the page
```

`--clear` deletes each file in a repo that the upload does not replace. To write to the
Hub, run `hf auth login` or set `HF_TOKEN`.

To add a Hub dataset, add an entry to `HUB_DATASETS` and a card to `cards/`. A source goes
to one Hub dataset at most. Then retarget its sources again: `uv run retarget` writes the
robot geometry into each motion of a Hub dataset with `geometry=True`, so that the motion
opens alone in the Rerun web viewer, and the card can link each motion there. The geometry
takes 9 MB, more than a short motion. Thus a large Hub dataset, such as Kimodo, sets
`geometry=False`: its motions share `assets/lite_pro.rrd`. A Hub dataset with
`private=True` goes to a private repo, and `uv run publish` makes an existing repo private
too. In the Space, its motions show only to a visitor who signs in with an account that can
read the repo. The Rerun web viewer cannot read a private repo, so neither the card nor the
page links its motions there.

**Warning:** Check the license of a source before you add it to a Hub dataset. A private
repo still shares the motions with each account that can read it. The licenses of AMASS
and MotionDecode do not allow you to share their motions.

### Browse the motions

`site/` is the motion browser: a static page that lists every motion in `dataset/` as a
card. When you point at a card, the card plays its motion. When you open a card, the page
shows a stage with a timeline, frame stepping, the path of the pelvis, and the command
that opens the motion in Rerun. To open a motion directly, add `#<motion>` to the end of
the address.

```bash
uv run build-site                      # dataset/ -> site/data/
uv run serve                           # then open http://localhost:8000
```

`uv run build-site` reads the robot layer of each motion, which is the layer that training
reads. It writes the layer for the page as float32 rows, with the motions end to end in a
few packs per source. The page loads the robot meshes once. It reads a motion only when
the card of that motion comes into view, as a byte range of its pack. Serve the page with
`uv run serve`, or with another server that answers range requests, such as Caddy.
`python -m http.server` answers a range request with the whole file.

`--sources lafan1` builds a page with LAFAN1 only. For a page that other people can reach,
build with `--sources kimodo kimodolh2 kimodolh3`. The licenses of LAFAN1, AMASS and
MotionDecode do not allow you to share their motions with the public.

The page plays a motion in two views. **Animation** plays it on a character, and the page
shows this view first. **URDF** plays it on the robot of the URDF. The character is the
VRM 1.0 model at `site/character.vrm`, which git does not track. Without this file, the
page shows only the robot.

The character uses the joint angles unchanged. Each joint turns its bone about the axis
that the joint has in the URDF at rest, where the character and the robot both stand in a
T-pose. At the waist, neck, shoulders, wrists, hips and ankles, the robot stacks three
joints at one point. There, the bone is a ball joint and uses all three angles, in the
order that the URDF chains them. A finger uses its two knuckle joints the same way. The
pelvis of the robot moves the hips of the character, so that the hip joints of the two
line up and both stand with their soles on the floor.

`site/collections.json` lists the collections that the page reads: folders of motion data,
`data/` by default. A collection can also be on the Hub, because the Hub answers range
requests from any site. The [Space][space] works this way: its `collections.json` points
at `web/` of each Hub dataset, as of the last commit to that repo. A collection marked
`"private": true` loads only for a visitor who signs in with Hugging Face. The page reads
it with the token of that account, so a private repo shows only to the people who can read
it. The page merges the collections into one list, and draws the robot of the first.

### The package

The package, `src/lite_motion_retargeting/`, follows the layout of
[mjlab](https://github.com/mujocolab/mjlab). `scripts/` holds the steps, and the other
modules are the library that the steps share:

| Module | What it does |
|---|---|
| `scripts/` | A module per command: `download.py`, `retarget.py`, `reactor.py`, `compare.py`, `build_site.py`, `serve.py` and `publish.py`. `pyproject.toml` lists the commands. |
| `sources/` | Defines the sources and the rule that names a motion. `g1.py` reads the G1 sources. `human.py` defines the human skeleton that every human source maps onto, and reads LAFAN1 BVH files. |
| `retargeting/` | Retargets one clip, grounds it, and measures it. `g1.py` is the G1 path. `human.py` builds the goal of each Lite link from a human pose, plants the feet, and solves the IK that reaches the goals. |
| `reactor/` | The mjlab environment of ReActor: its terms in `mdp.py`, its configs in `env_cfg.py`, and the motions that it reads in `library.py`. See [ReActor](#reactor). |
| `robots.py` | Loads the Lite and G1 models, and names their joints and bodies. |
| `kinematics.py` | Resampling, forward kinematics, the soles, and foot contacts. |
| `store.py` | Defines the file format, and reads a robot layer back. |
| `browser.py` | Writes the motions and the robot meshes for the page. |
| `hub.py` | Lists the Hub datasets, and names the Space. |

`uv run retarget` takes `--source <name>`, `--clip <regex>`, `--no-ik`, `--ik-iters N`,
`--workers N`, `--out <dir>` and `--validate-only`. `--out` writes the store to another
directory. `--validate-only` prints the metrics of the first clip and writes nothing. The
regex of `--clip` matches the path of a clip below the clip directory of its source,
without the suffix, such as `CMU/07/07_01_poses_120_jpos`. The motion name comes from the
same path: `amass_CMU_07_07_01_poses_120_jpos`.

`uv run publish` takes `--datasets <name>...`, `--clear` and `--dry-run`, as
[Publish to the Hub](#publish-to-the-hub) shows.

### Cost

The IK of the G1 path solves about 500 frames per second on one core, and the IK of the
human path solves about 100. A robot layer holds 12 to 16 MB for each minute of motion. A
base layer keeps the source rate, so it adds 10% to 60% to that size. The measured totals
are:

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

One MotionDecode clip is empty upstream. On 128 cores, `uv run retarget` took 3 h 40 min
for 64,224 MotionDecode clips. `uv run build-site` takes about 45 min for all sources.

The store holds the 9 MB robot geometry once, in `dataset/assets/lite_pro.rrd`.
`rerun --asset` attaches the geometry to the motion that you open. Only the native viewer
loads assets. The card of a Hub dataset with `geometry=True` opens a motion in the web
viewer, so each of its motions carries its own copy of the geometry.

## How a clip is retargeted

A source runs at 30 to 250 Hz. `uv run retarget` writes every motion at 50 Hz, the control
rate of the training environment. It resamples the source first, so the IK solves every
frame that reaches the file. As a result, the body poses stay the exact kinematics of the
joints beside them. The loader of the source decides which of two paths a clip takes: the
human path or the G1 path.

### A human clip

1. **One skeleton.** `sources/human.py` reads the clip into the world poses of one human
   skeleton. The skeleton follows the keypoints of the Lite conventions: pelvis, hips,
   knees, feet, toes, waist, chest, neck, head, shoulders, elbows and hands. Each rotation
   is rest-normalized: it is identity when the body stands in a T-pose that faces +x.
   Thus, a new format needs only a reader.
2. **Goals.** `retargeting.human.goals` builds a goal skeleton with the proportions of
   Lite and the joint rotations of the human. Its pelvis follows the human pelvis, scaled
   by the ratio of the two pelvis heights. Every other joint hangs from the joint above it
   by the Lite segment between them, and that segment turns as the human segment turns.
   Each Lite link in `retargeting.human.LITE_PAIRS` gets a goal pose: the pelvis, thighs,
   shins, feet, chest, head, upper arms, forearms and hands.
3. **Foot planting.** A skeleton with other proportions does not step where the human
   steps. `kinematics.planted` finds the frames where a human foot stands on the floor,
   from the speed and height of its ankle and toe. Then `retargeting.human.plant` shifts
   the goal skeleton, frame by frame, so that a planted foot keeps its sole point in place
   and on the floor. It also lifts a foot whose sole would dip through the floor.
4. **IK.** Per-frame [`mink`](https://github.com/kevinzakka/mink) IK with a free base
   moves Lite onto the goals, in position and orientation. Each frame starts from the
   solution of the frame before. The thigh, shin, upper arm and forearm goals pin the
   twist of each limb. Without these goals, the IK drifted along a joint limit onto
   another branch of the hip, then snapped back in one frame. No joint turns faster than
   20 rad/s. This limit binds where an arm passes the T-pose, in which the shoulder pitch
   and yaw of Lite share an axis.

The goals are the interface that all backends share. A physics backend, such as an RL
policy that tracks the goals in simulation, can use them in place of the IK.
[ReActor](#reactor) does this.

### A G1 clip

1. **Direct remap.** `lite_q = sign * g1_q + offset`, from the table in
   `retargeting.g1.G1_TO_LITE`.
2. **IK refinement.** Per-frame IK moves the result toward the pelvis-local poses of both
   feet and both hands of the G1, in position and orientation. A per-DoF posture cost
   holds the solution near the result of step 1. Thus, this step adjusts the pose but does
   not rearrange it.

### Both paths

**Grounding.** A constant shift of the pelvis height per clip puts the lowest point of the
Lite soles on the floor in the median frame. Each source puts the G1 at its own height,
and AMASS puts the pelvis at the floor. Thus, `retargeting/` measures the shift on the
Lite result. The median ignores the few frames where a heel or a toe dips through the
floor.

**Driven joints.** A source drives only the Lite joints that it has data for, and every
other joint holds its default, 0. A G1 source drives the 29 joints of
`retargeting.g1.G1_TO_LITE`, and has no neck, ankle yaw or fingers. A human source drives
every joint between the pelvis and a link with a goal: 34 joints for LAFAN1, which has no
fingers. The robot layer keeps the driven joints as `active_joint_names`. Thus, a motion
with fewer driven joints can still train the robot that it covers, such as a G1 motion for
the legs of the Lite Biped.

The page in `site/` shows the driven joints of each motion, and how many joints of each
Lite variant the motion drives. The page also filters the motions by the joint groups that
they drive.

**Metrics.** `retargeting.motion_metrics` measures every motion. The robot layer keeps the
numbers as its `metrics` property, and a run prints their mean:

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

The lowest sole point is exact: `kinematics.sole` takes the collision capsules of each foot
as segments with a radius, and boxes as their corners. A metric over an empty set, such as
the slide of a motion that never touches the floor, is NaN.

The IK plants the feet where the source stands, and `foot_slide_cm_s` measures the slide
on the same frames. Thus `foot_slide_cm_s` measures what the IK aims for. A method that
does not plant the feet, such as ReActor, can still keep the feet still where the source
stands, so the metric applies to it too. `near_floor_slide_cm_s` uses the height of the
robot sole instead, so it does not depend on the mask of the source. But height alone
decides it, so it also counts a swing foot that passes close to the floor.

The human path and the G1 path aim at different goals, so you cannot compare their goal
errors. You can compare the floor metrics and the joint speed. This table gives their
means over the 40 LAFAN1 takes that both sources hold, with the current Lite Pro model:

| Path | Foot slide | Planted gap | Penetration | Fastest joint |
|---|---|---|---|---|
| `lafan1bvh`, human | 2.6 cm/s | 0.40 cm | 1.1 cm | 19 rad/s |
| `lafan1`, G1 | 9.2 cm/s | 1.53 cm | 4.1 cm | 21 rad/s |

The human path has less foot slide, a smaller planted gap, and less penetration on 39 or
more of the 40 takes.

The IK solves against the Lite Pro MJCF of `lite-description`, and the robot layer draws
the URDF of the same package. Thus, a new version of `lite-description` gives new motions.

## ReActor

`uv run reactor` retargets the motions of a store again, in physics, with the method of
Disney's [ReActor](https://arxiv.org/abs/2605.06593). The IK of `retargeting/` ignores
mass, contact and torque. Thus, its motions can float, sink, slide, or ask a motor for
more than the motor has. ReActor tracks the same goals with a policy in MuJoCo Warp,
through mjlab, and the simulated motion is the retargeted motion:

1. **Policy.** The policy sets a PD target per joint around the IK solution of the frame,
   and pushes the pelvis with a wrench. The reward follows table 1 of the paper and
   penalizes the wrench, so the policy learns to use little of it.
2. **Goals.** The goals learn a position offset and an orientation offset per link, and a
   height offset per motion. After each PPO iteration, one gradient step moves the goals
   toward the simulated links. The step uses the approximate bilevel gradient of the
   paper.
3. **Export.** The policy plays each motion from its first frame to its end. The simulated
   joints become a robot layer of a new store, with the learned goals, the metrics of
   `retargeting/`, and the peak and mean wrench. If the policy loses track of the goals,
   `export` cuts the motion into pieces, as described below.

The simulation is ideal: a PD position actuator per joint with the armature and effort
limit of the MJCF, every collision geom of the MJCF, and no noise, delay or domain
randomization. ReActor makes reference motions. The policy that runs on the robot trains
on these motions in
[Lite-Motion-Tracking](https://github.com/Berkeley-Humanoids/Lite-Motion-Tracking).

ReActor reads the robot layers of a store, so it runs on any source that `uv run retarget`
wrote, human or G1. ReActor welds every joint that no motion drives, such as the fingers.
The `reactor` extra adds mjlab, which pins MuJoCo to 3.11 for the whole project.

```bash
uv run --extra reactor reactor train --source lafan1bvh --clip '^walk' --iterations 3000
uv run --extra reactor reactor export --run logs/reactor/<run> --out dataset_reactor
```

`train` writes its run to `logs/reactor/`, with the motions that it read (`library.npz`),
the goal offsets (`goals.pt`), the PPO checkpoints, and TensorBoard logs. With 4,096
environments on one RTX A4500, an iteration takes 1.6 s.

A LAFAN1 take runs for minutes, and a policy that trained on 10 s episodes can stumble in
a take. If the policy stumbles, `export` cuts the motion 1 s before the stumble and starts
again from the IK solution. It writes each piece of 2 s or more as
`<motion>_f<first frame>`. A motion without a stumble keeps its name. The pieces after a
stumble track the goals less well, and their metrics show this.

Two runs on LAFAN1 give the first results:

| Run | Takes | Iterations | Time | Kept | Lost track | Played whole |
|---|---|---|---|---|---|---|
| Walking | 12 | 3,000 | 1.3 h | 99% of 48 min | 39 times | 6 takes |
| All | 77 | 10,000 | 4.4 h | 98% of 276 min | 354 times | 26 takes |

This table compares each run with the IK, on the takes that the run played whole, weighted
by length:

| Metric | Walking run | IK | All run | IK |
|---|---|---|---|---|
| Foot slide | 2.75 cm/s | 1.97 cm/s | 3.53 cm/s | 2.51 cm/s |
| Planted gap | 0.14 cm | 0.33 cm | 0.18 cm | 0.73 cm |
| Penetration | 0.50 cm | 0.92 cm | 0.61 cm | 1.46 cm |
| Fastest joint | | | 25 rad/s | 19 rad/s |
| Goal error, feet | 6.8 cm | 0.2 cm | 8.3 cm | 0.3 cm |
| Root wrench, mean | | none | 24 N | none |
| Root wrench, peak | 65 to 107 N | none | 265 N | none |

The simulated feet sit on the floor better than the IK feet. The All run has a smaller
planted gap on all 26 takes, and less penetration on 23. But the feet slide more, the
joints move faster, and the policy depends on the wrench.

Lite Pro weighs 327 N. The mean wrench of the All run is 7% of that weight, and its peak
is 80%. Over all 77 takes, the mean wrench goes from 17 N on the walking takes to 61 N on
the obstacle takes, and no motion has a mean below 10 N. The falls and the motions on the
ground are the worst, with a foot slide of 9 to 13 cm/s.

On the walking takes, the All run does better than the Walking run. It has less foot slide
on all 9 motions that both runs wrote, and less goal error on 7 of them. But its peak
wrench is higher.

The paper trains for 20,000 iterations, and names the wrench penalty its most sensitive
setting. The real test is downstream: a tracking policy without the wrench, trained on
either set of motions.

This implementation differs from the paper in four places. The paper has no public code to
compare against. The unofficial reproduction has no license, so this repo uses none of its
code.

| Paper | Here | Why |
|---|---|---|
| PD targets, and a reset at the nominal pose with a ramp-in | PD targets around the IK solution, and a reset at the IK state | The IK solution exists, so the policy starts from it |
| Offsets on the source body frames | Offsets on the goals of `retargeting/human.py`, in the frame of each link | The goals already fit Lite: its proportions, and planted feet |
| Offset bounds of 0.5 | 0.1 m, 0.3 rad, 0.05 m | Lite is about 1 m tall |
| Projected SGD on the offsets | Adam, then the projection | The paper gives no step size |

## The file format

`store.py` writes the store, `dataset/`, as one directory per layer: `base/` for the
source clip as published at its own rate, `lite_pro/` for the 50 Hz retargeted motion,
`assets/` for the robot geometry, and `blueprints/` for the viewer layout.
`cards/lafan1.md` documents the store, and `store.py` documents every entity path.

Training and deployment read the robot layer. The robot layer is self-contained: its
joints reset the robot, and its body poses are the goal. The goal uses the frame
convention of the Lite links, so a reward compares the goal with the robot directly. To
draw the robot, the viewer also needs the geometry, from the asset or from the motion
file. The robot layer stores no reached pose, because the reached pose is the forward
kinematics of the joints beside it.

`store.py` is the only module that writes Rerun data, and it uses only `rerun-sdk`. The
layout on disk matches the motion store of
[MikuMotionTools](https://github.com/T-K-233/MikuMotionTools), so its `MotionStore` reads
these files.

The training repo,
[Lite-Motion-Tracking](https://github.com/Berkeley-Humanoids/Lite-Motion-Tracking), still
reads the older single-layer `tracking/*.rrd`. It needs a reader for this store before it
can train on these motions.

[dataset]: https://huggingface.co/datasets/Berkeley-Humanoids/Lite-LAFAN1-Dataset
[space]: https://huggingface.co/spaces/Berkeley-Humanoids/Lite-Motions
[kimodo-hub]: https://huggingface.co/datasets/Berkeley-Humanoids/Lite-Kimodo-Dataset
[lafan1]: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
[lafan1bvh]: https://github.com/ubisoft/ubisoft-laforge-animation-dataset
[amass]: https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics
[motiondecode]: https://huggingface.co/datasets/CMRobot/MotionDecode
[kimodo]: https://huggingface.co/datasets/cagataydev/kimodo-g1-motion-v1
[kimodolh2]: https://huggingface.co/datasets/cagataydev/kimodo-g1-longhorizon-v2
[kimodolh3]: https://huggingface.co/datasets/cagataydev/kimodo-g1-longhorizon-v3
