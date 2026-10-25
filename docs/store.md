# The file format

`store.py` writes the store, `dataset/`, as one directory per layer, and one [Rerun](https://rerun.io) file per motion in each layer:

| Path | What it holds |
|---|---|
| `base/<motion>.rrd` | The source clip as published, at its own rate. |
| `lite_pro/<motion>.rrd` | The retargeted motion at 50 Hz: the joints, the goal of each link, the metrics, and the joints that the source drives. |
| `assets/lite_pro.rrd` | The robot geometry, once for every motion. |
| `blueprints/lite_pro.rbl` | The viewer layout. |

`cards/lafan1.md` documents the store, and `store.py` documents every entity path.

Training and deployment read the robot layer. The robot layer is self-contained: its joints reset the robot, and its body poses are the goal. The goal uses the frame convention of the Lite links, so a reward compares the goal with the robot directly. To draw the robot, the viewer also needs the geometry, from the asset or from the motion file. The robot layer stores no reached pose, because the reached pose is the forward kinematics of the joints beside it.

## Open a motion

`rerun --asset` attaches the 9 MB robot geometry of the asset to the motion that you open. Put the blueprint last, because blueprints do not merge, and the viewer uses the last one that it loads:

```bash
uv run rerun dataset/*/amass_CMU_07_07_01_poses_120_jpos.rrd \
    --asset dataset/assets/lite_pro.rrd dataset/blueprints/lite_pro.rbl
```

Only the native viewer loads assets. The card of a Hub dataset with `geometry=True` opens a motion in the web viewer, so each of its motions carries its own copy of the geometry. See [Publish to the Hub](publish.md).

## Readers

`store.py` is the only module that writes Rerun data, and it uses only `rerun-sdk`. The layout on disk matches the motion store of [MikuMotionTools](https://github.com/T-K-233/MikuMotionTools), so its `MotionStore` reads these files.

The training repo, [Lite-Motion-Tracking](https://github.com/Berkeley-Humanoids/Lite-Motion-Tracking), reads the robot layer with `src/lite_motion_tracking/motion/rrd_motion.py`.
