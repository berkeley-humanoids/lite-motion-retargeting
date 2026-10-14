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
| `amassx`: [AMASS SMPL-X][amassx], human | Your copy | 60 to 250 Hz | 0.3 GB for 4.2 h at 30 Hz | AMASS: non-commercial research, no redistribution | No |
| `lafan1`: [LAFAN1][lafan1] | 40 | 30 Hz | 0.3 GB | CC BY-NC-ND 4.0 | No |
| `amass`: [AMASS][amass], all subsets | 17,717 | 60 to 250 Hz | 7 GB | AMASS: non-commercial research, no redistribution | No |
| `motiondecode`: [ChingMu MotionDecode][motiondecode] | 92,559 | 120 Hz | 148 GB | ChingMu: non-commercial research, not to be hosted as a competing dataset | No |
| `kimodo`: [Kimodo G1][kimodo] | 10,000 | 50 Hz | 1.2 GB | Apache-2.0 | Yes |
| `kimodolh2`: [Kimodo G1 long-horizon v2][kimodolh2] | 10,000 | 50 Hz | 5.8 GB | Apache-2.0 | Yes |
| `kimodolh3`: [Kimodo G1 long-horizon v3][kimodolh3] | 10,000 | 50 Hz | 12.8 GB | Apache-2.0 | Yes |

`src/lite_motion_retargeting/sources/` defines each source and its loader. The `amassx`,
`amass` and `motiondecode` motions stay in your local store, `dataset/`, for training.
`uv run publish` uploads `lafan1bvh` to a private repo, and the three Kimodo sets to a
public one.
See [Publish to the Hub](#publish-to-the-hub).

`lafan1bvh` is the human motion that the authors of `lafan1` retargeted: the BVH takes of
Ubisoft's LAFAN1. These takes come as a zip on GitHub, not on the Hub. `lafan1` holds 40
of them, with the same names and frames, so the two
[retargeting paths](#how-a-clip-is-retargeted) compare clip for clip.

`amassx` is the human motion behind `amass`: the AMASS clips in their SMPL-X form. Its
reader builds the body with the SMPL-X model of AMASS, which it reads from
`.cache/smplx/SMPLX_NEUTRAL.npz`. Both need an MPI account, so `uv run download` cannot
fetch them. Copy the clips to `.cache/amassx/smplx/` and the model to `.cache/smplx/`
yourself. [UMR on AMASS](#umr-on-amass) prepares a sample of them.

The other sources ship G1 motion only. MotionDecode gives its BVH files on request, and
Kimodo generated its clips on the G1 directly.

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
| `lafan1bvh`, human | 2.6 cm/s | 0.40 cm | 1.05 cm | 19 rad/s |
| `lafan1`, G1 | 9.2 cm/s | 1.52 cm | 4.07 cm | 21 rad/s |

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

Four runs give these results. Two ran on an older Lite Pro model, `lite-description`
`99cdef0`, and two on the current one:

| Run | Model | Motions | Iterations | GPU | Time | Kept | Lost track | Played whole |
|---|---|---|---|---|---|---|---|---|
| Walking | `99cdef0` | 12 | 3,000 | RTX A4500 | 1.3 h | 99% of 48 min | 39 times | 6 |
| All, 10k | `99cdef0` | 77 | 10,000 | RTX A4500 | 4.4 h | 98% of 276 min | 354 times | 26 |
| All, 20k | current | 77 | 20,000 | RTX 3090 | 6.7 h | 99% of 276 min | 104 times | 48 |
| Kimodo | current | 9,490 | 10,000 | RTX 3080 Ti | 6.6 h | 100% of 630 min | 71 times | 9,421 |

[Compare the methods](#compare-the-methods) measures the two runs on the current model
against the IK and UMR.

The policy depends on the wrench. On the 25 LAFAN1 takes that both All runs played whole,
the 20k run pushes the pelvis with a mean wrench of 20 N, against 24 N for the 10k run. But
its mean peak is 302 N, against 257 N. Lite Pro weighs 258 N. Over the 48 takes that the
20k run played whole, the mean is 33 N and the mean peak 394 N, because the takes that only
this run finishes are the hard ones. The Kimodo motions are short and calm: their mean
wrench is 2 N, and their median peak 9 N.

On the walking takes, the All, 10k run does better than the Walking run. It has less foot
slide on all 9 motions that both runs wrote, and less goal error on 7 of them. But its peak
wrench is higher.

The 20k run trains for the 20,000 iterations of the paper. The paper names the wrench
penalty its most sensitive setting. The real test is downstream: a tracking policy without
the wrench, trained on these motions.

This implementation differs from the paper in four places. The paper has no public code to
compare against. The unofficial reproduction has no license, so this repo uses none of its
code.

| Paper | Here | Why |
|---|---|---|
| PD targets, and a reset at the nominal pose with a ramp-in | PD targets around the IK solution, and a reset at the IK state | The IK solution exists, so the policy starts from it |
| Offsets on the source body frames | Offsets on the goals of `retargeting/human.py`, in the frame of each link | The goals already fit Lite: its proportions, and planted feet |
| Offset bounds of 0.5 | 0.1 m, 0.3 rad, 0.05 m | Lite is about 1 m tall |
| Projected SGD on the offsets | Adam, then the projection | The paper gives no step size |

## UMR

[UMR](https://hanyang9.github.io/UMR/) is a kinematic retargeter, like the IK of
`retargeting/`. It matches points on the surface of an SMPL-X body to points on the surface
of the robot. Thus it needs no table that pairs the human bodies with the links of the
robot. The IK and ReActor use the pairs of `retargeting/human.py`. UMR reads only SMPL-X
motion. It takes LAFAN1 after a conversion, and AMASS in its SMPL-X form, but not a G1
source such as Kimodo.

UMR pins its own versions of torch, MuJoCo and NumPy, so it runs in an environment of its
own, outside this package. `umr/` holds the files that run it on Lite Pro:

| File | What it does |
|---|---|
| `lite_pro.json` | The robot config of UMR for Lite Pro. Lite Pro stands in a T-pose at zero, so the T-pose moves no joint. The joint limits hold the fingers within 0.01 rad of 0, so UMR solves the 34 joints that the IK solves. A limit of zero width stops the QP solver of UMR. |
| `lafan1_to_smplx.py` | Converts LAFAN1 BVH takes to SMPL-X with [`lafan_to_smplx`](https://github.com/jaraujo98/lafan_to_smplx), with the 10 betas of the body that UMR builds. |
| `ground_smplx.py` | Puts the feet of each SMPL-X take on the floor, as the authors of UMR do. The conversion leaves the bodies 2 to 6 cm below the floor, and UMR does not move them. |
| `batch.json` | The batch config of UMR, which solves each take in order. The default of UMR solves it in both directions and then picks, which makes joints jump. |
| `manifest.py` | Writes `manifest.json` beside the motions of a run: the hash of the robot model, the commit of UMR, and the configs. |
| `amass_model.py` | Makes the SMPL-X model of AMASS loadable by UMR. |
| `prepare_amass.py` | Prepares AMASS SMPL-X clips for UMR and the IK: 30 Hz, 10 betas, and the standing feet on the floor. It skips the clips on stairs, beams or boxes. |
| `amass_clips.txt` | The AMASS clips of [the comparison](#compare-the-methods): 4.2 h from 47 subjects of 8 subsets. |

To retarget LAFAN1 with UMR, run these steps in an empty directory beside the clone of this
repo, such as `../umr`:

1. Clone UMR at commit `d1cdaed`, and clone `lafan_to_smplx`:

   ```bash
   git clone https://github.com/hanyang9/UMR && git -C UMR checkout d1cdaed
   git clone https://github.com/jaraujo98/lafan_to_smplx
   ```

2. Make the environment of UMR. GMR at commit `81ce42e` holds the BVH reader that
   `lafan_to_smplx` needs:

   ```bash
   uv venv --python 3.12
   uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
   uv pip install -r UMR/requirements-umr.txt
   uv pip install --no-deps "git+https://github.com/YanjieZe/GMR@81ce42e"
   ```

3. Download the SMPL-X v1.1 body models from the [SMPL-X site](https://smpl-x.is.tue.mpg.de/).
   Put the `.npz` files in `body_models/smplx/`, and link them where UMR reads them:

   ```bash
   ln -s ../../body_models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz UMR/smpl/
   ```

4. Copy Lite Pro and its config into UMR:

   ```bash
   lmr=$(realpath ../Lite-Motion-Retargeting)
   mjcf=$(cd $lmr && uv run python -c "from lite_description import get_mjcf_path; print(get_mjcf_path('lite_pro'))")
   robot=$(dirname "$(dirname "$mjcf")")
   mkdir -p UMR/assets/lite_pro && cp -r "$robot/mjcf" "$robot/meshes" UMR/assets/lite_pro/
   cp $lmr/umr/lite_pro.json UMR/robot_configs/
   cp $lmr/umr/batch.json UMR/lite_pro_batch.json
   ```

5. Convert the takes to SMPL-X on the CPU, then put them on the floor:

   ```bash
   ls $lmr/.cache/lafan1bvh/bvh | sed 's/\.bvh$//' > takes.txt
   xargs -P 16 -n 5 .venv/bin/python $lmr/umr/lafan1_to_smplx.py lafan_to_smplx/lafan_to_smplx.py \
       $lmr/.cache/lafan1bvh/bvh smplx body_models < takes.txt
   xargs .venv/bin/python $lmr/umr/ground_smplx.py body_models smplx smplx_grounded < takes.txt
   ```

6. Retarget the takes, and record what made them:

   ```bash
   cd UMR && ../.venv/bin/python scripts/humanoid_retarget_pipeline_batch.py \
       --config robot_configs/lite_pro.json --batch-config lite_pro_batch.json \
       --motion-folder ../smplx_grounded --workers 9 --retarget-gpus 0 --output-root ../output \
       --force-build --force-train
   ../.venv/bin/python $lmr/umr/manifest.py robot_configs/lite_pro.json \
       lite_pro_batch.json ../output/lite_pro
   ```

   UMR writes `output/lite_pro/<take>_lite_pro.npz`, at 30 Hz. `uv run compare`
   interpolates it to 50 Hz. A run that solved at 50 Hz did no better.

UMR resolves a relative path against the folder of its batch config, so the batch config
goes into the clone of UMR. With 9 workers, one RTX 3080 Ti retargets the 276 min of LAFAN1
in 33 min, and holds 7.3 GB. With 16 workers, the GPU runs out of memory.

UMR keeps the correspondence of the surfaces in `UMR/output/correspondence_lite_pro_*`. The
name of the robot and the betas of the body name this cache, and the geometry does not.
Thus step 6 trains the correspondence again with `--force-build --force-train`. Without
these flags, a run after a change to Lite Pro uses the old correspondence.
`uv run compare` stops if `manifest.json` names another model than the installed Lite Pro.

### UMR on AMASS

AMASS is SMPL-X motion already, so UMR and the IK read the same files. To retarget the
clips of `amass_clips.txt`, do steps 1 to 4 of [UMR](#umr), then these steps:

1. Download the SMPL-X N archives of ACCAD, CMU, EKUT, Eyes Japan, HDM05, KIT, SFU and
   WEIZMANN, and the SMPL-X model with the locked head, from the [AMASS site][amassx].
   Extract them into `amass/` and `amass_model/`.

2. Make the model of AMASS loadable by UMR. UMR reads it through a robot config of its
   own:

   ```bash
   .venv/bin/python $lmr/umr/amass_model.py amass_model/neutral/model.npz \
       body_models/smplx/SMPLX_NEUTRAL.npz body_models/amass
   .venv/bin/python -c "import json; c = json.load(open('UMR/robot_configs/lite_pro.json')); \
       c['smplx_model_dir'] = '../../body_models/amass'; \
       json.dump(c, open('UMR/robot_configs/lite_pro_amass.json', 'w'), indent=2)"
   ```

3. Prepare the clips into the cache of `amassx`, and give the IK the same model:

   ```bash
   .venv/bin/python $lmr/umr/prepare_amass.py body_models/amass/SMPLX_NEUTRAL.npz amass \
       $lmr/.cache/amassx/smplx $lmr/umr/amass_clips.txt
   mkdir -p $lmr/.cache/smplx && cp body_models/amass/SMPLX_NEUTRAL.npz $lmr/.cache/smplx/
   ```

4. Retarget the clips with UMR and with the IK:

   ```bash
   cd UMR && ../.venv/bin/python scripts/humanoid_retarget_pipeline_batch.py \
       --config robot_configs/lite_pro_amass.json --batch-config lite_pro_batch.json \
       --motion-folder $lmr/.cache/amassx/smplx --workers 9 --retarget-gpus 0 \
       --output-root ../output/amass --force-build --force-train
   ../.venv/bin/python $lmr/umr/manifest.py robot_configs/lite_pro_amass.json \
       lite_pro_batch.json ../output/amass/lite_pro
   cd $lmr && uv run retarget --source amassx --workers 40 --out dataset_amass
   ```

UMR trains a correspondence for each body, thus once for each of the 47 subjects.
`prepare_amass.py` moves each clip up or down, so that the median height of a standing foot
is `z = 0`. In most AMASS clips, a standing foot sinks 1 to 2 cm, and UMR anchors a foot to
the floor only within 5 mm of it. The authors' rule for LAFAN1 does not apply, because it
needs a T-pose at the ends of a take. UMR treats a folder named `motions` as its BONES-SEED
dataset, so the clips of `amassx` go in `smplx/`.

## Compare the methods

`uv run compare` measures the IK, UMR and ReActor on the same frames of one source:

```bash
uv run compare --source lafan1bvh --umr ../umr/output/lite_pro --reactor dataset_reactor
uv run compare --source amassx --store dataset_amass --umr ../umr/output/amass/lite_pro
uv run compare --source kimodo --reactor dataset_reactor
```

The metrics that each store keeps do not compare across methods. For example,
`reactor export` measures the slide where its learned goals plant a foot. `uv run compare`
measures every method in the same way:

- **One mask.** Every method uses the stance mask of the source: the human feet for a
  human source, and the goals of the IK for a G1 source.
- **The same frames.** With a ReActor store, every method is measured on the frames of
  each ReActor piece.
- **Pooled samples.** A mean, a median or a percentile is over every sample of every
  piece. Thus a long motion counts more than a short one, and the cut of a motion into
  pieces changes nothing.
- **Fidelity to the source.** The limb direction error, the root error over 1 s and the
  contact IoU compare each method with its source. The IK aims at the limb directions of
  the source, so this metric favors the IK, as the slide where the source stands does.
- **One world.** For LAFAN1, UMR uses a world turned 90° from that of the IK, and for
  AMASS the same world. `compare` turns a UMR motion by the fixed angle of its source, and
  shifts it by one offset per motion. UMR scales the human by its height, not by the
  height of its pelvis, so `compare` leaves out the goal errors of UMR.
- **One model.** Every robot layer and every UMR run records the hash of its Lite Pro
  model. `compare` stops if a hash differs from that of the installed model.

This table gives the results on LAFAN1, over the 138 pieces of the All, 20k run of
[ReActor](#reactor), 274 of the 276 min. UMR ran as [UMR](#umr) shows.

| Metric | IK | UMR | ReActor |
|---|---|---|---|
| Slide where the source stands, mean / median / p95 | **2.5** / **1.2** / **9.1** cm/s | 12.4 / 7.8 / 39.6 cm/s | 5.2 / 1.4 / 23.5 cm/s |
| Slide where the sole is within 1 cm of the floor, mean / median | 29.9 / **1.8** cm/s | 37.0 / 11.7 cm/s | **14.7** / 1.9 cm/s |
| Contact IoU with the source | 73.7% | 70.9% | **74.3%** |
| Gap where the source stands | 0.54 cm | 0.35 cm | **0.30 cm** |
| Penetration, p95 of frames | 1.42 cm | **0.09 cm** | 0.34 cm |
| Fastest joint, p99 / max | 15.4 / 20.0 rad/s | **8.4** / **11.4** rad/s | 13.8 / 37.9 rad/s |
| Limb direction error | **4.3°** | 19.9° | 9.1° |
| Root error over 1 s | **2.8 cm** | 6.3 cm | 6.6 cm |
| Goal error, feet / hands | 0.32 / 0.67 cm | | 10.0 / 9.7 cm |

Kimodo is a G1 source, so UMR cannot read it, and the IK copies the G1 joints without foot
planting. This table covers its 9,461 ReActor pieces, 627 of 630 min:

| Metric | IK | ReActor |
|---|---|---|
| Slide where the source stands, mean / median / p95 | 3.5 / 1.5 / 13.4 cm/s | **1.1** / **0.25** / **2.8** cm/s |
| Slide where the sole is within 1 cm of the floor, mean / median | 9.0 / 1.8 cm/s | **3.4** / **0.30** cm/s |
| Contact IoU with the source | 82.0% | **85.0%** |
| Gap where the source stands | 0.53 cm | **0.26 cm** |
| Penetration, p95 of frames | 1.05 cm | **0.19 cm** |
| Fastest joint, p99 / max | 9.8 / 56.3 rad/s | **8.1** / **37.3** rad/s |
| Goal error, feet / hands | **6.55** / 13.61 cm | 8.13 / **12.76** cm |

AMASS is SMPL-X motion, so UMR reads it without a conversion. This table covers the 1,136
clips of [UMR on AMASS](#umr-on-amass), 252 min. ReActor did not run on it.

| Metric | IK | UMR |
|---|---|---|
| Slide where the source stands, mean / median / p95 | **1.4** / **0.59** / **5.3** cm/s | 7.6 / 3.8 / 27.5 cm/s |
| Slide where the sole is within 1 cm of the floor, mean / median | **15.8** / **0.70** cm/s | 24.6 / 5.5 cm/s |
| Contact IoU with the source | 79.7% | **80.3%** |
| Gap where the source stands | 0.39 cm | **0.14 cm** |
| Penetration, p95 of frames | 0.58 cm | **0.10 cm** |
| Fastest joint, p99 / max | 10.5 / 20.0 rad/s | **7.4** / **11.7** rad/s |
| Limb direction error | **3.7°** | 12.0° |
| Root error over 1 s | 4.3 cm | **1.9 cm** |
| Goal error, feet / hands | 0.26 / 0.84 cm | |

No method is best on every metric:

- **The IK** keeps a foot still where the source stands, and copies the limbs of the
  source most closely. It plants its goals on exactly those frames, so both metrics measure
  what it aims for. Where a sole is near the floor, its median slide is low, but its mean is
  17 times its median: a few frames, such as a swing foot low over the floor, slide fast.
  Its feet sink the deepest.
- **ReActor** sits on the floor best, and its feet slide the least where a sole is near the
  floor. On Kimodo, it beats the IK on every floor metric. On LAFAN1, one of its joints
  peaks at 38 rad/s, its feet leave the goals of the IK by 10 cm, its limbs turn 9° from
  the source, and it needs the wrench. The motions on the ground are its worst: their feet
  slide 11.6 cm/s where the source stands.
- **UMR** sinks the least, and moves its joints the slowest. But where the source stands,
  its feet slide 5 times as fast as those of the IK on LAFAN1, and its limbs turn 20° from
  the source. On LAFAN1, UMR scales the human by 0.77 and the IK by 0.84, so a UMR root
  travels about 8% less, which adds to its root error. On AMASS, its root follows the
  source more closely than that of the IK. [Why UMR slides](#why-umr-slides) shows that
  part of its slide comes from its input, and the rest comes from UMR.

These metrics measure the motions, not the tracking policy that trains on them.

The three methods also differ in cost, on the 276 min of LAFAN1:

| Method | Runs on | Time |
|---|---|---|
| IK | CPU | 2.3 h on one core, or 5.4 min with 40 workers. On the 252 min of AMASS, 4.8 min with 40 workers |
| UMR | GPU | 33 min on one RTX 3080 Ti, with 9 workers. On the 252 min of AMASS, 34 min on an RTX 3090 and an RTX 3080 Ti with 18 workers, with the 47 correspondences |
| ReActor | GPU | 6.7 h of training on one RTX 3090 for 20,000 iterations, then 6 min of export |

### Why UMR slides

An outside audit of this comparison named possible causes of the UMR slide. Each row of
this table changes one thing, on 8 LAFAN1 takes of 8 kinds, 27 min. The IK slides 2.2 cm/s
on them, in the mean.

| UMR setup | Slide where the source stands, mean / median | Fastest joint |
|---|---|---|
| 1. The first run: 16 betas, the 5th-percentile grounding, solved in both directions | 11.5 / 7.6 cm/s | 27.7 rad/s |
| 2. 10 betas, and the grounding of the UMR authors | 11.5 / 7.5 cm/s | 27.7 rad/s |
| 3. Row 2, without the smoothing | 12.1 / 7.0 cm/s | 39.1 rad/s |
| 4. Row 2, with the fingers held at 0 | 11.5 / 7.5 cm/s | 27.8 rad/s |
| 5. Row 2, solved at 50 Hz | 11.7 / 7.9 cm/s | 23.3 rad/s |
| 6. Row 2, solved in order | 11.4 / 7.7 cm/s | 10.9 rad/s |
| 7. The setup of [UMR](#umr), with a ground anchor cost of 50, not 5 | 19.3 / 7.6 cm/s | 11.7 rad/s |
| 8. The same, with a ground anchor cost of 500 | 30.5 / 11.7 cm/s | 11.2 rad/s |

- **The input matches that of the authors.** `lafan1_to_smplx.py` gives the poses of the
  `dance1_subject2` file that UMR ships, and `ground_smplx.py` its height, to 1 mm.
- **No change to the input or the solver brings the slide near that of the IK.** Solving in
  order keeps the slide and stops the joint jumps, because the bidirectional selection of
  UMR ignores the root. [UMR](#umr) now runs this way.
- **A stronger ground anchor makes the slide worse.** The anchor holds a point of the foot
  while the source foot slides, then lets go.
- **UMR slides as much on the G1, the robot of the UMR paper.** On the same 8 takes at
  30 Hz, a G1 foot slides 11.7 / 8.0 cm/s, and a Lite Pro foot 12.1 / 7.9 cm/s. Thus the
  slide does not come from the Lite Pro config.
- **The input slides.** Where the human stands, the feet of the SMPL-X conversion slide
  7.0 cm/s, and those of the BVH 2.1 cm/s. UMR follows the surface of its input. Its ground
  anchors are active in 97% of the frames of `walk1_subject1`, but they do not stop the
  slide.

**On AMASS, the input does not slide, and UMR still does.** Where the human stands, the
lowest foot vertex of the AMASS clips slides 2.1 cm/s, as the LAFAN1 BVH does. UMR then
slides 7.6 cm/s, less than the 12.4 cm/s of LAFAN1, but 3.6 times its input and 5.6
times the IK. It slides more than the IK on all 1,136 clips, by a factor of 4 to 10 in
each of the 8 subsets. Thus on LAFAN1, part of the slide of UMR comes from its input, and
the rest comes from UMR itself.

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
[amassx]: https://amass.is.tue.mpg.de
[motiondecode]: https://huggingface.co/datasets/CMRobot/MotionDecode
[kimodo]: https://huggingface.co/datasets/cagataydev/kimodo-g1-motion-v1
[kimodolh2]: https://huggingface.co/datasets/cagataydev/kimodo-g1-longhorizon-v2
[kimodolh3]: https://huggingface.co/datasets/cagataydev/kimodo-g1-longhorizon-v3
