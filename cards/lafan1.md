---
license: cc-by-nc-nd-4.0
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

This dataset holds the 77 takes of Ubisoft's [LAFAN1][lafan1] motion capture, retargeted to
the Berkeley Lite Pro humanoid at 50 Hz. [Lite-Motion-Retargeting][retarget] retargeted
them from the human motion itself, and kept the 30 Hz take beside each motion. Each motion
is one [Rerun](https://rerun.io) recording. You can [open one in your browser][viewer], with
no install. The [table below](#motions) links every motion, and the [Lite Motions][browser]
Space plays all of them on one page.

```bash
hf download Berkeley-Humanoids/Lite-LAFAN1-Dataset --repo-type dataset \
    --include "*/lafan1bvh/lafan1bvh_dance1_subject1.rrd" "blueprints/*" --local-dir .

rerun */lafan1bvh/lafan1bvh_dance1_subject1.rrd blueprints/lite_pro.rbl
```

Pass the blueprint last. Blueprints do not merge, and the viewer uses the last one that it
loads.

## Layout

Each top directory is a [layer](https://rerun.io/blog/data-layer-for-robot-learning), with a
directory per source, and each file is a motion. This layout follows
[`rerun/arkitscenes-rrd`](https://huggingface.co/datasets/rerun/arkitscenes-rrd). All
layers of one motion share an `application_id` and a `recording_id`, so the viewer opens
any set of them as one recording.

| Path | What it holds |
|---|---|
| `base/lafan1bvh/<motion>.rrd` | The take as published, at 30 Hz: the world pose of each of the 22 joints of the LAFAN1 skeleton. It records provenance, so you can derive the motion again without the upstream dataset. |
| `lite_pro/lafan1bvh/<motion>.rrd` | Everything the robot needs, at 50 Hz: the retargeted joints, the goal that its links track, the joints that the take drives, and the URDF geometry that draws it. |
| `blueprints/lite_pro.rbl` | The viewer layout. |
| `web/` | The robot layer in the form that the [motion browser][browser] plays: `index.json.gz` lists the motions, `robot.json` and `robot.bin` draw the robot, and each pack under `packs/` holds float32 rows of motions end to end. `src/lite_motion_retargeting/browser.py` of [Lite-Motion-Retargeting][retarget] documents the format. |

The robot layer is self-contained, and its joints reset the robot. `/lite_pro/goal/` holds
the goal that the robot tracks, in the frame convention of the Lite links, so a reward
compares the goal with the robot directly. The goals are 15 Lite links: the pelvis, the
thighs, the shins, the feet, the chest, the head, the upper arms, the forearms and the
hands. `source_body_names` records the body of the human skeleton that each goal follows.
The layer stores no reached pose, because the reached pose is the forward kinematics of
the joints beside it. The `/lite_pro/tf` chain already draws the robot at that pose. The
gap between the robot and its goal triads is the retargeting error, frame by frame.

A take drives 34 joints of Lite Pro: every joint between the pelvis and a link with a goal.
`active_joint_names` lists them. LAFAN1 has no fingers, so the finger joints hold 0. The
`metrics` property records how well the retarget went: the goal errors, the foot slide,
the planted gap, the penetration and the fastest joint.

The base layer holds the 22 joints of the LAFAN1 skeleton under their LAFAN1 names, such
as `Hips` and `LeftUpLeg`, in `/base/body/`.

Every array is in the world frame, in meters and radians, and each rotation is `xyzw`.
Each velocity is the backward difference from the frame before, so frame 0 holds zero.
Every column carries a `frame` index and a `time` in seconds. The two layers run at
different rates, so `time` lines them up. `src/lite_motion_retargeting/store.py` of
[Lite-Motion-Retargeting][retarget] lists every entity path and property.

## Motions

<!-- motions -->

## Source

[LAFAN1][lafan1] is the animation dataset of Ubisoft La Forge: 77 takes of 5 subjects, as
BVH files at 30 Hz. [Lite-Motion-Retargeting][retarget] reads each take into one human
skeleton, and gives each Lite link a goal pose with the proportions of Lite and the joint
rotations of the human. It moves the goals so that a planted foot stays in place, and
solves a per-frame IK with a free base. Then it shifts the take so that the soles stand on
the floor. Lite-Motion-Retargeting documents every step. The result depends on the Lite
Pro model, so a new version of
[Lite-Description](https://github.com/Berkeley-Humanoids/Lite-Description) gives new
motions.

Until 2026-10-11, this repo held 40 of these takes as the Unitree G1 motion of
[`lvhaidong/LAFAN1_Retargeting_Dataset`][g1], copied to Lite. The retarget from the human
motion replaced them. Over those 40 takes, it lowers the mean foot slide from 9.2 to
2.6 cm/s, the planted gap from 1.53 to 0.40 cm, and the penetration from 4.1 to 1.1 cm. The
history of this repo keeps the earlier version.

CC BY-NC-ND 4.0, inherited from LAFAN1. Non-commercial use only. If you use these motions,
cite the LAFAN1 paper:

```bibtex
@article{harvey2020robust,
  author    = {Félix G. Harvey and Mike Yurick and Derek Nowrouzezahrai and Christopher Pal},
  title     = {Robust Motion In-Betweening},
  booktitle = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH)},
  publisher = {ACM},
  volume    = {39},
  number    = {4},
  year      = {2020}
}
```

[lafan1]: https://github.com/ubisoft/ubisoft-laforge-animation-dataset
[g1]: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
[retarget]: https://github.com/Berkeley-Humanoids/Lite-Motion-Retargeting
[browser]: https://huggingface.co/spaces/Berkeley-Humanoids/Lite-Motions
[viewer]: https://rerun.io/viewer?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FBerkeley-Humanoids%2FLite-LAFAN1-Dataset%2Fresolve%2Fmain%2Flite_pro%2Flafan1bvh%2Flafan1bvh_dance1_subject1.rrd
