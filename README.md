# Lite Motion Retargeting

Retargets the [LAFAN1](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
motion capture set from the Unitree G1 to the Berkeley Lite humanoid. One clip
becomes one layered [Rerun](https://rerun.io) motion in
[`Berkeley-Humanoids/Lite-LAFAN1-Dataset`][dataset], which you can
[open in a browser][viewer].

## Use

```bash
uv sync
uv run scripts/download_lafan.py         # G1 clips -> .cache/
uv run scripts/retarget.py --workers -1  # every clip, every core
uv run rerun dataset/*/lafan1_walk1_subject1.rrd dataset/blueprints/lite_pro.rbl
uv run scripts/publish.py --clear        # upload to the Hub
```

To write to the Hub, run `hf auth login` or set `HF_TOKEN`.

| Script | What it does |
|---|---|
| `download_lafan.py` | Downloads the G1 clips of LAFAN1 into `.cache/lafan1_g1/`. |
| `retarget.py` | Retargets each clip and writes its two layers under `dataset/`. |
| `store.py` | Defines the file format. |
| `publish.py` | Fills the motion index of `DATASET_CARD.md`, then uploads both to the Hub. |
| `common.py` | The joint table, the model loaders, the CSV reader, and forward kinematics. |

`retarget.py` takes `--clip <regex>`, `--no-ik`, `--ik-iters N`, `--workers N`, and
`--validate-only`, which prints the end-effector error table and writes nothing.
`publish.py` takes `--dry-run`, `--repo-id <id>`, and `--clear`, which deletes every
file in the Hub repo that this store does not replace.

## How a clip is retargeted

The source is 30 Hz and a motion is written at 50 Hz, the control rate of the
training environment. `retarget.py` resamples the source before it does anything
else, so the IK solves every frame that reaches the file, and the body poses stay
the exact kinematics of the joints beside them.

1. **Direct remap.** `lite_q = sign * g1_q + offset`, from the table in
   `common.G1_TO_LITE`. A constant shift of the pelvis height puts the feet of Lite
   on the ground.
2. **IK refinement.** Per-frame [`mink`](https://github.com/kevinzakka/mink) IK moves
   the result toward the pelvis-local pose of both feet and both hands of the G1,
   position and orientation. A per-DOF posture cost holds the solution near step 1,
   so the step adjusts the pose and does not rearrange it.

The IK solves against the Lite Pro MJCF of `robot-assets`, and the robot layer draws
the URDF of the same package. A new asset version therefore gives new motions, and
`retarget.py` prints the pelvis shift it measured so that such a change is visible.

## The file format

`store.py` writes `dataset/` as one directory per layer: `base/` for the source clip
as published at 30 Hz, `lite_pro/` for the 50 Hz retarget, and `blueprints/` for the
viewer layout. `DATASET_CARD.md` documents that store, and `store.py` documents every
entity path.

The robot layer is what training and deployment read, and it is self-contained: its
joints reset the robot, and its body poses are the goal, turned into the frame
convention of the Lite links so a reward compares them against the robot directly.
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
[viewer]: https://rerun.io/viewer?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FBerkeley-Humanoids%2FLite-LAFAN1-Dataset%2Fresolve%2Fmain%2Flite_pro%2Flafan1_dance1_subject1.rrd
