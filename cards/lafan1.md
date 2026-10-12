---
license: cc-by-nc-4.0
task_categories:
- robotics
pretty_name: Lite LAFAN1 Dataset
size_categories:
- n<1K
tags:
- rerun
- rrd
- humanoid
- motion-retargeting
- berkeley-humanoids
- lite
- lafan1
viewer: false
---

# Lite LAFAN1 Dataset

This dataset holds 40 [LAFAN1][lafan1] motions at 50 Hz.
[Lite-Motion-Retargeting][retarget] retargeted them from the Unitree G1 to the Berkeley
Lite Pro humanoid, and kept the 30 Hz source clip beside each motion. Each motion is one
[Rerun](https://rerun.io) recording. You can [open one in your browser][viewer], with no
install. The [table below](#motions) links every motion, and the
[Lite Motions][browser] Space plays all of them on one page.

```bash
hf download Berkeley-Humanoids/Lite-LAFAN1-Dataset --repo-type dataset \
    --include "*/lafan1_dance1_subject1.rrd" "blueprints/*" --local-dir .

rerun */lafan1_dance1_subject1.rrd blueprints/lite_pro.rbl
```

Pass the blueprint last. Blueprints do not merge, and the viewer uses the last one that it
loads.

## Layout

Each directory is a [layer](https://rerun.io/blog/data-layer-for-robot-learning), and each
file is a motion. This layout follows
[`rerun/arkitscenes-rrd`](https://huggingface.co/datasets/rerun/arkitscenes-rrd). All
layers of one motion share an `application_id` and a `recording_id`, so the viewer opens
any set of them as one recording.

| Path | What it holds |
|---|---|
| `base/<motion>.rrd` | The source clip as published, at 30 Hz: the 29 G1 joints, and the world poses that these joints put the G1 bodies at. It records provenance, so you can derive the motion again without the upstream dataset. |
| `lite_pro/<motion>.rrd` | Everything the robot needs, at 50 Hz: the retargeted joints, the goal that its bodies track, and the URDF geometry that draws it. |
| `blueprints/lite_pro.rbl` | The viewer layout. |
| `web/` | The robot layer in the form that the [motion browser][browser] plays: `index.json.gz` lists the motions, `robot.json` and `robot.bin` draw the robot, and each pack under `packs/` holds float32 rows of motions end to end. `src/lite_motion_retargeting/build_site.py` of [Lite-Motion-Retargeting][retarget] documents the format. |

The robot layer is self-contained, and its joints reset the robot. `/lite_pro/goal/` holds
the goal that the robot tracks, in the frame convention of the Lite links, so a reward
compares the goal with the robot directly. `source_body_names` records the G1 link that
each goal came from. The layer stores no reached pose, because the reached pose is the
forward kinematics of the joints beside it. The `/lite_pro/tf` chain already draws the
robot at that pose. The gap between the robot and its goal triads is the retargeting
error, frame by frame.

Both layers carry the pelvis, both feet, and both hands: `/lite_pro/goal/` under the Lite
link names, and `/base/body/` under the G1 link names.

Every array is in the world frame, in meters and radians, and each rotation is `xyzw`.
Each velocity is the backward difference from the frame before, so frame 0 holds zero.
Every column carries a `frame` index and a `time` in seconds. The two layers run at
different rates, so `time` lines them up. `src/lite_motion_retargeting/store.py` of
[Lite-Motion-Retargeting][retarget] lists every entity path and property.

## Motions

<!-- motions -->

## Source

The G1 subset of [`lvhaidong/LAFAN1_Retargeting_Dataset`][lafan1] is LAFAN1, already
retargeted to a 29-DoF Unitree G1, at 30 Hz. [Lite-Motion-Retargeting][retarget] resamples
it to 50 Hz and retargets it to Lite Pro in three steps: a per-joint remap, a per-frame
IK, and a per-clip shift that puts the soles on the floor. Lite-Motion-Retargeting
documents all three steps. The result depends on the Lite Pro model, so a new version of
[Lite-Description](https://github.com/Berkeley-Humanoids/Lite-Description) gives new
motions.

CC BY-NC 4.0, inherited from LAFAN1. Non-commercial use only.

[lafan1]: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
[retarget]: https://github.com/Berkeley-Humanoids/Lite-Motion-Retargeting
[browser]: https://huggingface.co/spaces/Berkeley-Humanoids/Lite-Motions
[viewer]: https://rerun.io/viewer?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FBerkeley-Humanoids%2FLite-LAFAN1-Dataset%2Fresolve%2Fmain%2Flite_pro%2Flafan1_dance1_subject1.rrd
