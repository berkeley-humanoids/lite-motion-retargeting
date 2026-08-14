# Lite Motion Retargeting

Retargets the [LAFAN1][lafan1] motion capture set from the Unitree G1 to the
Berkeley Lite humanoid. One clip becomes one Rerun `.rrd` motion in
[`Berkeley-Humanoids/Lite-Motion-Tracking-Dataset`][dataset], which
[Lite-Motion-Tracking][training] then trains against.

## Pipeline

Each clip goes through two steps:

1. **Direct remap.** `lite_q = sign * g1_q + offset`, from the table in
   `common.G1_TO_LITE`. A constant shift of the pelvis height puts the feet of
   Lite on the ground.
2. **IK refinement.** Per-frame [`mink`][mink] IK moves the result toward the
   pelvis-local end-effector poses of the G1: both feet and both hands, position
   and orientation. A per-DOF posture cost holds the solution near step 1, so the
   step adjusts the pose and does not rearrange it.

```bash
uv sync
uv run scripts/download_lafan.py              # G1 source clips -> .cache/
uv run scripts/retarget.py --workers -1       # every clip, every core
uv run scripts/visualize.py                   # inspect the result
uv run scripts/retarget.py --workers -1 --push  # retarget, then upload
```

## Scripts

| Script | What it does |
|---|---|
| `download_lafan.py` | Pulls the G1 subset of the LAFAN1 dataset into `.cache/lafan1_g1/`. |
| `retarget.py` | Runs both steps and writes `dataset/tracking/lite_pro_lafan1_<clip>.rrd`. `--push` uploads them. |
| `visualize.py` | A viser viewer. Draws Lite solid with the source G1 over it as a ghost. |
| `rrd.py` | Reads and writes the motion file. |

### `retarget.py` flags

| Flag | Default | Effect |
|---|---|---|
| `--clip <regex>` | none | Retarget the clips whose name matches. |
| `--validate-only` | off | Print the end-effector error table, and write nothing. |
| `--no-ik` | off | Keep step 1 only. |
| `--ik-iters N` | 15 | Newton steps per frame in step 2. |
| `--workers N` | 1 | Processes to run clips in parallel. `-1` uses every core. |
| `--push` | off | Upload the motions after the last clip. |
| `--repo-id <id>` | the dataset above | The Hub repo to upload to. |

`--push` needs a token with write access to the namespace:

```bash
hf auth login          # or: export HF_TOKEN=hf_...
```

## The motion file

A motion is a Rerun `.rrd` file that holds the base pose, the joint positions and
velocities, and the joint names, on a `frame` timeline. `scripts/rrd.py` reads and
writes it, and `rerun <file>.rrd` opens it.

The format belongs to the training repo, at
`lite_motion_tracking/tasks/tracking/reference_motion.py`. `rrd.py` repeats its
constants instead of importing them, because that package pulls mjlab and torch.
Keep the two equal.

## The robot model

`retarget.py` loads `robots/lite_pro/mjcf/lite_pro.xml` from `robot-assets`, the
same asset package and pin that the training repo uses, so both read one MJCF.

The result depends on that model: the pelvis shift comes from where the feet of
Lite are, and the IK solves against its kinematics. A new asset version therefore
gives a new motion.

The 40 clips published before 2026-08-13 came from an earlier model, and their
pelvis sat 40 mm **above** what this asset needs, so the feet of the reference
floated. The clips on the Hub now come from the model above. The regeneration
moved the pelvis down by 40.00 mm on every clip and left x and y untouched, and
it changed the joints that the IK solves for. The direct remap of step 1 did not
move: the median joint change is 0.

## Licence

The retargeted motion inherits **CC BY-NC 4.0** from LAFAN1. Non-commercial use
only.

[lafan1]: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
[dataset]: https://huggingface.co/datasets/Berkeley-Humanoids/Lite-Motion-Tracking-Dataset
[training]: https://github.com/Berkeley-Humanoids/Lite-Motion-Tracking
[mink]: https://github.com/kevinzakka/mink
