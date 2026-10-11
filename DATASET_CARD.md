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

40 [LAFAN1][lafan1] clips at 50 Hz, retargeted from the Unitree G1 to the Berkeley Lite
Pro humanoid by [Lite-Motion-Retargeting][retarget], with the 30 Hz source kept beside
them. One motion is one
[Rerun](https://rerun.io) recording. [Open one in your browser][viewer], with no
install. The [table below](#motions) links every motion, and the
[Lite Motion Dataset][browser] Space plays them all on one page.

```bash
hf download Berkeley-Humanoids/Lite-LAFAN1-Dataset --repo-type dataset \
    --include "*/lafan1_dance1_subject1.rrd" "blueprints/*" --local-dir .

rerun */lafan1_dance1_subject1.rrd blueprints/lite_pro.rbl
```

Pass the blueprint last. Blueprints do not merge, and the last one loaded wins.

## Layout

A directory is a [layer](https://rerun.io/blog/data-layer-for-robot-learning) and a file
is a motion, after [`rerun/arkitscenes-rrd`](https://huggingface.co/datasets/rerun/arkitscenes-rrd).
All layers of one motion share an `application_id` and a `recording_id`, so the viewer
opens any set of them as one recording.

| Path | What it holds |
|---|---|
| `base/<motion>.rrd` | The source clip as published, at 30 Hz: the 29 G1 joints, and the world poses they put its bodies at. Provenance, so a motion can be derived again without the upstream dataset. |
| `lite_pro/<motion>.rrd` | Everything the robot needs, at 50 Hz: the retargeted joints, the goal its bodies track, and the URDF geometry that draws it. |
| `blueprints/lite_pro.rbl` | The viewer layout. |
| `web/` | The robot layer as the [browser][browser] plays it: `index.json.gz` lists the motions, `robot.json` and `robot.bin` draw the robot, and each pack under `packs/` holds float32 rows of motions end to end. `scripts/build_site.py` documents the format. |

The robot layer is self-contained. Its joints reset the robot, and `/lite_pro/goal/`
holds what the robot tracks, in the frame convention of the Lite links, so a reward
compares it against the robot directly. `source_body_names` records which G1 link each
goal came from. The layer stores no reached pose, because that is the forward kinematics
of the joints beside it, and the `/lite_pro/tf` chain already draws the robot there. The
gap between the robot and its goal triads is the error of the retarget, frame by frame.

Both layers carry the pelvis, both feet, and both hands: `/lite_pro/goal/` under the Lite
link names, and `/base/body/` under the G1 ones.

Every array is world frame, in metres and radians, and a rotation is `xyzw`. A velocity
is the backward difference against the frame before it, so frame 0 holds zero. Every
column carries a `frame` index and a `time` in seconds. The two layers run at different
rates, so `time` is what lines them up. `scripts/store.py` of
[Lite-Motion-Retargeting][retarget] lists every entity path and property.

## Motions

<!-- motions -->

## Source

The G1 subset of [`lvhaidong/LAFAN1_Retargeting_Dataset`][lafan1] is LAFAN1 already
retargeted to a 29-DoF Unitree G1, at 30 Hz. [Lite-Motion-Retargeting][retarget]
resamples it to 50 Hz and takes it to Lite Pro with a per-joint remap, a per-frame IK, and
a per-clip shift that puts the soles on the floor, and documents all three. The result depends on the Lite Pro asset, so a new version of
[Lite-Description](https://github.com/Berkeley-Humanoids/Lite-Description) gives new
motions.

CC BY-NC 4.0, inherited from LAFAN1. Non-commercial use only.

[lafan1]: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
[retarget]: https://github.com/Berkeley-Humanoids/Lite-Motion-Retargeting
[browser]: https://huggingface.co/spaces/Berkeley-Humanoids/Lite-Motion-Dataset
[viewer]: https://rerun.io/viewer?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FBerkeley-Humanoids%2FLite-LAFAN1-Dataset%2Fresolve%2Fmain%2Flite_pro%2Flafan1_dance1_subject1.rrd
