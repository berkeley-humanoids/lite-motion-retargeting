# Publish to the Hub

`uv run publish` uploads Hub datasets from the store. A Hub dataset is one Hub dataset repo with its own card, and it holds the motions of one or more sources. `HUB_DATASETS` in `src/lite_motion_retargeting/hub.py` lists them:

| Hub dataset | Repo | Sources | Card | Private |
|---|---|---|---|---|
| `lafan1` | [`Berkeley-Humanoids/Lite-LAFAN1-Dataset`](https://huggingface.co/datasets/Berkeley-Humanoids/Lite-LAFAN1-Dataset) | `lafan1bvh` | `cards/lafan1.md` | Yes |
| `kimodo` | [`Berkeley-Humanoids/Lite-Kimodo-Dataset`](https://huggingface.co/datasets/Berkeley-Humanoids/Lite-Kimodo-Dataset) | `kimodo`, `kimodolh2`, `kimodolh3` | `cards/kimodo.md` | No |

Each repo holds the layers of its motions as `<layer>/<source>/<motion>.rrd`, because the Hub takes at most 10,000 files in one directory. `web/` holds the same motions in the form that the [motion browser](browser.md) plays. `uv run publish` writes the repo into `dataset/hub/<name>/` as hard links to the store, with the card and `web/`. Then it uploads that directory, in several commits if it is large. Last, it uploads the page to the [Lite Motions](https://huggingface.co/spaces/Berkeley-Humanoids/Lite-Motions) Space, with a collection for the `web/` of each Hub dataset. Thus the Space plays the motions of every Hub dataset on one page.

```bash
uv run publish --dry-run          # write dataset/hub/<name>/, and upload nothing
uv run publish                    # every Hub dataset, then the page
uv run publish --datasets lafan1  # one Hub dataset, then the page
uv run publish --datasets         # only the page
```

`--clear` deletes each file in a repo that the upload does not replace. To write to the Hub, run `hf auth login` or set `HF_TOKEN`.

To add a Hub dataset, add an entry to `HUB_DATASETS` and a card to `cards/`. A source goes to one Hub dataset at most. Then retarget its sources again: `uv run retarget` writes the robot geometry into each motion of a Hub dataset with `geometry=True`, so that the motion opens alone in the Rerun web viewer, and the card can link each motion there. The geometry takes 9 MB, more than a short motion. Thus a large Hub dataset, such as Kimodo, sets `geometry=False`: its motions share `assets/lite_pro.rrd`. A Hub dataset with `private=True` goes to a private repo, and `uv run publish` makes an existing repo private too. In the Space, its motions show only to a visitor who signs in with an account that can read the repo. The Rerun web viewer cannot read a private repo, so neither the card nor the page links its motions there.

**Warning:** Check the license of a source before you add it to a Hub dataset. A private repo still shares the motions with each account that can read it. The licenses of AMASS, MotionDecode and BONES-SEED do not allow you to share their motions. See [Sources](sources.md).
