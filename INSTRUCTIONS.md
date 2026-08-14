# Instructions

How to reproduce this dataset locally and push it to the Hugging Face Hub.

## Setup

```bash
uv sync
```

## Generation pipeline

Four scripts under `scripts/`, run in order:

1. **`download_lafan.py`** — pulls the G1 subset of
   [`lvhaidong/LAFAN1_Retargeting_Dataset`][lafan1] into `.cache/lafan1_g1/`
   (gitignored).
2. **`retarget.py`** — step 1 direct G1 → Lite joint remap (sign + offset
   table baked into `common.G1_TO_LITE`) and step 2 per-frame `mink` IK
   refinement on both feet and both hands. Writes a HuggingFace
   LeRobotDataset at the repository root (`meta/` + `data/chunk-*/`).
3. **`visualize.py`** — viser web viewer; renders Lite (solid) overlaid with
   the source G1 (alpha-blended ghost). Episode dropdown lazy-loads each
   clip on selection.
4. **`push_to_hub.py`** — uploads the dataset to the Hugging Face Hub using
   `LeRobotDataset.push_to_hub` (auto-generates the dataset card).

End-to-end reproduction:

```bash
uv run scripts/download_lafan.py
uv run scripts/retarget.py --workers -1     # parallelise across all CPU cores
uv run scripts/visualize.py                 # browse and inspect
uv run scripts/push_to_hub.py               # publish
```

### `retarget.py` flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--clip <regex>` | none | Process only matching CSVs (e.g. `'walk1_subject1'`). |
| `--validate-only` | off | Print step-1 vs step-2 EE-error table without writing the dataset. |
| `--no-ik` | off | Output step 1 only (skips per-frame IK refinement). |
| `--ik-iters N` | 15 | `mink` Newton iterations per frame in step 2. |
| `--workers N` | 1 | Parallel processes across clips. `-1` = all CPU cores. |

### `visualize.py` flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--episode-index N` | 0 | Initial episode to load (switch later from the GUI dropdown). |
| `--port N` | 8080 | viser HTTP port. |
| `--no-ghost` | off | Hide the source G1 ghost; show Lite only. |

## Pushing to Hugging Face Hub

Authenticate once:

```bash
hf auth login                # interactive
# or, non-interactive:
export HF_TOKEN=hf_...
```

Your token needs **write** scope on the target namespace (your account or
an org you belong to).

Then upload:

```bash
uv run scripts/push_to_hub.py
uv run scripts/push_to_hub.py --repo-id your-username/your-dataset
uv run scripts/push_to_hub.py --private
uv run scripts/push_to_hub.py --branch v0.1
```

`push_to_hub.py` calls `LeRobotDataset.push_to_hub`, which:

* Creates the dataset repo if it doesn't exist.
* Uploads `meta/` + `data/` (and `videos/` if any — there are none here).
* Generates a LeRobot dataset card from `meta/info.json` (this overwrites
  whatever `README.md` was uploaded).
* Tags the revision with the LeRobot codebase version.

[lafan1]: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
