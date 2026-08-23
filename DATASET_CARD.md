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
Pro humanoid by [Lite-Motion-Retargeting][retarget]. One motion is one
[Rerun](https://rerun.io) recording. [Open one in your browser][viewer], with no
install. The [table below](#motions) links every motion.

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
| `reference/<motion>.rrd` | The G1 bodies the retarget aims at: a world pose and velocity per body, and no joints. |
| `lite_pro/<motion>.rrd` | What the retarget reached: joint positions and velocities, the same bodies, and the URDF geometry that draws them. |
| `blueprints/lite_pro.rbl` | The viewer layout. |

Both layers carry the pelvis, both feet, and both hands. The robot layer names them after
the links of Lite, the reference layer after the links of the G1, and the
`reference_body_names` property records the pairing. The gap between the two is the error
of the retarget, frame by frame.

Every array is world frame, in metres and radians, and a rotation is `xyzw`. A velocity
is the backward difference against the frame before it, so frame 0 holds zero.
`scripts/store.py` of [Lite-Motion-Retargeting][retarget] lists every entity path and
property.

## Motions

<!-- motions -->

## Source

The G1 subset of [`lvhaidong/LAFAN1_Retargeting_Dataset`][lafan1] is LAFAN1 already
retargeted to a 29-DoF Unitree G1, at 30 Hz. [Lite-Motion-Retargeting][retarget]
resamples it to 50 Hz and takes it to Lite Pro with a per-joint remap and a per-frame IK,
and documents both. The result depends on the Lite Pro asset, so a new version of
[Lite-Description](https://github.com/Berkeley-Humanoids/Lite-Description) gives new
motions.

CC BY-NC 4.0, inherited from LAFAN1. Non-commercial use only.

[lafan1]: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
[retarget]: https://github.com/Berkeley-Humanoids/Lite-Motion-Retargeting
[viewer]: https://rerun.io/viewer?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FBerkeley-Humanoids%2FLite-LAFAN1-Dataset%2Fresolve%2Fmain%2Flite_pro%2Flafan1_dance1_subject1.rrd
