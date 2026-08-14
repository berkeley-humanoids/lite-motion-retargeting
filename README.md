# Lite Motion Retargeting

Retargets the [LAFAN1][lafan1] motion capture set from the Unitree G1 to the
Berkeley Lite humanoid.

These scripts produced the 40 LAFAN1 clips in
`Berkeley-Humanoids/Lite-Motion-Tracking-Dataset`. They lived in that dataset
repo until now. Code belongs in a code repo, so they moved here, and the dataset
repo keeps only motions.

## Pipeline

Four scripts under `scripts/`, run in order:

1. **`download_lafan.py`** — pulls the G1 subset of [`lvhaidong/LAFAN1_Retargeting_Dataset`][lafan1]
   into `.cache/lafan1_g1/`.
2. **`retarget.py`** — a direct G1 → Lite joint remap (the sign and offset table
   in `common.G1_TO_LITE`), then per-frame `mink` IK on both feet and both hands.
3. **`visualize.py`** — a viser viewer that draws Lite solid over the source G1 as
   a ghost.
4. **`push_to_hub.py`** — uploads the result.

```bash
uv sync
uv run scripts/download_lafan.py
uv run scripts/retarget.py                      # every clip
uv run scripts/retarget.py --clip 'walk1_subject1'   # one clip
```

## Output format

`retarget.py` writes a **LeRobot v3 dataset**, which is the format the training
side has since replaced with Rerun `.rrd`. The bridge is
`scripts/motions/import_lerobot_motion.py` in
[Lite-Motion-Tracking][training], which converts an episode to a `.rrd` motion.

Writing `.rrd` directly from `retarget.py` would drop that hop. It is the obvious
next change here, and nothing depends on the LeRobot output except that importer.

## Licence

The retargeted motion inherits **CC BY-NC 4.0** from LAFAN1. Non-commercial use
only.

[lafan1]: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
[training]: https://github.com/Berkeley-Humanoids/Lite-Motion-Tracking
