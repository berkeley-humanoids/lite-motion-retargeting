# Lite Motion Retargeting

Retargets motion capture sets from the Unitree G1 to the Berkeley Lite humanoid. Each
source is already retargeted to the 29-DoF G1 by its authors. One clip becomes one
layered [Rerun](https://rerun.io) motion. The LAFAN1 motions are published in
[`Berkeley-Humanoids/Lite-LAFAN1-Dataset`][dataset], which you can
[open in a browser][viewer] or browse in the [Lite Motion Dataset][space] Space.

## Sources

| Source | Clips | Rate | Size | License | On the Hub |
|---|---|---|---|---|---|
| `lafan1`: [LAFAN1][lafan1] | 40 | 30 Hz | 0.3 GB | CC BY-NC-ND 4.0 | Yes |
| `amass`: [AMASS][amass], all subsets | 17,717 | 60 to 250 Hz | 7 GB | AMASS: non-commercial research, no redistribution | No |
| `motiondecode`: [ChingMu MotionDecode][motiondecode] | 92,559 | 120 Hz | 148 GB | ChingMu: non-commercial research, not to be hosted as a competing dataset | No |
| `kimodo`: [Kimodo G1][kimodo] | 10,000 | 50 Hz | 1.2 GB | Apache-2.0 | No |
| `kimodolh2`: [Kimodo G1 long-horizon v2][kimodolh2] | 10,000 | 50 Hz | 5.8 GB | Apache-2.0 | No |
| `kimodolh3`: [Kimodo G1 long-horizon v3][kimodolh3] | 10,000 | 50 Hz | 12.8 GB | Apache-2.0 | No |

`scripts/sources.py` defines each source and its loader. The motions of `amass` and
`motiondecode` stay in your local `dataset/`, for training. `publish.py` uploads only
`lafan1`.

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
| `retarget.py` | Retargets each clip and writes its two layers under `dataset/`. |
| `sources.py` | The sources, the loader that reads a clip of each, and how a motion is named. |
| `store.py` | Defines the file format. |
| `publish.py` | Fills the motion index of `DATASET_CARD.md`, uploads it with the store and the browser data, then the page to the Space. |
| `build_site.py` | Writes the motions and the robot meshes that `site/` plays into `site/data/`. |
| `serve.py` | Serves `site/` on this machine, with range requests. |
| `common.py` | The joint table, the model loaders, resampling, and forward kinematics. |

`retarget.py` takes `--source <name>`, `--clip <regex>`, `--no-ik`, `--ik-iters N`,
`--workers N`, and `--validate-only`, which prints the end-effector error table and
writes nothing. The regex matches the path of a clip below the clip directory of its
source, without the suffix, such as `CMU/07/07_01_poses_120_jpos`. A motion is named
after the same path: `amass_CMU_07_07_01_poses_120_jpos`.
`publish.py` takes `--dry-run`, `--repo-id <id>`, `--space-id <id>`, and `--clear`,
which deletes every file in the Hub repo that this store does not replace.

### Cost

The IK solves about 500 frames per second on one core. A robot layer holds 12 to 16 MB
for each minute of motion. A base layer keeps the source rate, so it adds 10 to 60 percent
to that. These are the measured totals:

| Source | Motions | Hours | Store | `site/data` |
|---|---|---|---|---|
| `lafan1` | 40 | 2.5 | 2.6 GB | 0.07 GB |
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
the exact kinematics of the joints beside them.

1. **Direct remap.** `lite_q = sign * g1_q + offset`, from the table in
   `common.G1_TO_LITE`.
2. **IK refinement.** Per-frame [`mink`](https://github.com/kevinzakka/mink) IK moves
   the result toward the pelvis-local pose of both feet and both hands of the G1,
   position and orientation. A per-DOF posture cost holds the solution near step 1,
   so the step adjusts the pose and does not rearrange it.
3. **Grounding.** A constant shift of the pelvis height per clip puts the lowest point
   of the Lite soles on the floor in the median frame. Each source puts the G1 at its
   own height, and AMASS puts the pelvis at the floor, so the shift is measured on the
   Lite result. The median ignores the few frames where a heel or a toe dips through
   the floor, which the IK does not prevent.

The IK solves against the Lite Pro MJCF of `lite-description`, and the robot layer draws
the URDF of the same package. A new asset version therefore gives new motions.

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
[amass]: https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics
[motiondecode]: https://huggingface.co/datasets/CMRobot/MotionDecode
[kimodo]: https://huggingface.co/datasets/cagataydev/kimodo-g1-motion-v1
[kimodolh2]: https://huggingface.co/datasets/cagataydev/kimodo-g1-longhorizon-v2
[kimodolh3]: https://huggingface.co/datasets/cagataydev/kimodo-g1-longhorizon-v3
[viewer]: https://rerun.io/viewer?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FBerkeley-Humanoids%2FLite-LAFAN1-Dataset%2Fresolve%2Fmain%2Flite_pro%2Flafan1_dance1_subject1.rrd
