---
license: apache-2.0
task_categories:
- robotics
pretty_name: Lite Kimodo Dataset
size_categories:
- 10K<n<100K
tags:
- rerun
- rrd
- humanoid
- motion-retargeting
- berkeley-humanoids
- lite
- kimodo
viewer: false
---

# Lite Kimodo Dataset

This dataset holds the motions of the three Kimodo G1 sets, retargeted to the Berkeley Lite
Pro humanoid at 50 Hz by [Lite-Motion-Retargeting][retarget]. Each motion is one
[Rerun](https://rerun.io) recording, with the G1 clip that it comes from beside it. The
[Lite Motions][browser] Space plays all of them on one page.

| Source | Set | Motions | Hours | A clip |
|---|---|---|---|---|
| `kimodo` | [Kimodo G1 v1][kimodo] | 9,490 | 10 | 4 s, one action |
| `kimodolh2` | [Kimodo G1 long-horizon v2][kimodolh2] | 9,947 | 54 | about 20 s, 3 to 8 actions |
| `kimodolh3` | [Kimodo G1 long-horizon v3][kimodolh3] | 9,941 | 118 | about 43 s, about 50 actions |

Kimodo is not motion capture. NVIDIA's [Kimodo][model] model generated each clip from a
text prompt, such as "a person throwing boxing punches". Then a tracking policy played the
clip on a simulated Unitree G1. A motion is named `<source>_<group>_episode_<index>`. The
group is the action that the prompt asks for, such as `bowing_forward`, or the family of a
long-horizon prompt, such as `combat_drill`.

```bash
hf download Berkeley-Humanoids/Lite-Kimodo-Dataset --repo-type dataset \
    --include "*/kimodo/kimodo_bowing_forward_episode_00001.rrd" "assets/*" "blueprints/*" \
    --local-dir .

rerun */kimodo/kimodo_bowing_forward_episode_00001.rrd --asset assets/lite_pro.rrd \
    blueprints/lite_pro.rbl
```

The motions share the geometry of the robot, in `assets/lite_pro.rrd`. The geometry takes
9 MB, and a 4 s motion takes 1 MB, so each motion does not carry a copy. Pass the asset to
`rerun` with `--asset`, and pass the blueprint last. The Rerun web viewer cannot load an
asset, so use the [Space][browser] to see a motion in a browser.

## Layout

Each top directory is a [layer](https://rerun.io/blog/data-layer-for-robot-learning), with a
directory per source, and each file is a motion. This layout follows
[`rerun/arkitscenes-rrd`](https://huggingface.co/datasets/rerun/arkitscenes-rrd). All
layers of one motion share an `application_id` and a `recording_id`, so the viewer opens
any set of them as one recording.

| Path | What it holds |
|---|---|
| `base/<source>/<motion>.rrd` | The source clip as published, at 50 Hz: the 29 G1 joints, and the world poses that these joints put the G1 bodies at. It records provenance, so you can derive the motion again without the upstream dataset. |
| `lite_pro/<source>/<motion>.rrd` | Everything the robot needs, at 50 Hz: the retargeted joints, the goal that its bodies track, and the joints that the source drives. |
| `assets/lite_pro.rrd` | The URDF geometry that draws the robot, for every motion. |
| `blueprints/lite_pro.rbl` | The viewer layout. |
| `web/` | The robot layer in the form that the [motion browser][browser] plays: `index.json.gz` lists the motions, `robot.json` and `robot.bin` draw the robot, and each pack under `packs/` holds float32 rows of motions end to end. `src/lite_motion_retargeting/browser.py` of [Lite-Motion-Retargeting][retarget] documents the format. |

The robot layer is self-contained, and its joints reset the robot. `/lite_pro/goal/` holds
the goal that the robot tracks, in the frame convention of the Lite links, so a reward
compares the goal with the robot directly. `source_body_names` records the G1 link that
each goal came from. The layer stores no reached pose, because the reached pose is the
forward kinematics of the joints beside it. The `/lite_pro/tf` chain draws the robot at
that pose.

Both layers carry the pelvis, both feet, and both hands: `/lite_pro/goal/` under the Lite
link names, and `/base/body/` under the G1 link names. A G1 clip drives 29 joints of Lite
Pro, and `active_joint_names` lists them. The neck, the ankle yaw and the fingers have no
G1 joint, so they hold 0. The `metrics` property records how well the retarget went.

Every array is in the world frame, in meters and radians, and each rotation is `xyzw`.
Each velocity is the backward difference from the frame before, so frame 0 holds zero.
Every column carries a `frame` index and a `time` in seconds.
`src/lite_motion_retargeting/store.py` of [Lite-Motion-Retargeting][retarget] lists every
entity path and property.

## Source

Each set holds 10,000 episodes on the Hub, as LeRobot shards. The last shard of each set
has no footer, and no reader can open it, so this dataset has fewer motions.
[Lite-Motion-Retargeting][retarget] copies each G1 joint to its Lite joint, and refines the
copy with a per-frame IK that matches the feet and the hands of the G1. Then it shifts the
motion so that the soles stand on the floor. Lite-Motion-Retargeting documents every step.
The result depends on the Lite Pro model, so a new version of
[Lite-Description](https://github.com/Berkeley-Humanoids/Lite-Description) gives new
motions.

Apache-2.0, inherited from the three sets. The sets come from
[cagataydev](https://huggingface.co/cagataydev). Their clips come from
[`nvidia/Kimodo-G1-RP-v1`][model], tracked on the G1 by
[`cagataydev/protomotions-gtp-unitree-g1`](https://huggingface.co/cagataydev/protomotions-gtp-unitree-g1)
with [strands-labs/robots](https://github.com/strands-labs/robots).

[kimodo]: https://huggingface.co/datasets/cagataydev/kimodo-g1-motion-v1
[kimodolh2]: https://huggingface.co/datasets/cagataydev/kimodo-g1-longhorizon-v2
[kimodolh3]: https://huggingface.co/datasets/cagataydev/kimodo-g1-longhorizon-v3
[model]: https://huggingface.co/nvidia/Kimodo-G1-RP-v1
[retarget]: https://github.com/Berkeley-Humanoids/Lite-Motion-Retargeting
[browser]: https://huggingface.co/spaces/Berkeley-Humanoids/Lite-Motions
