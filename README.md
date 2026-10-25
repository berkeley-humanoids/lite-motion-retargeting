# Lite Motion Retargeting

Lite Motion Retargeting retargets human and Unitree G1 motion datasets to the Berkeley Lite Pro humanoid, and writes each clip as a layered [Rerun](https://rerun.io) motion.

Browse the motions in the [Lite Motions](https://huggingface.co/spaces/Berkeley-Humanoids/Lite-Motions) Space. The Kimodo motions are in [`Berkeley-Humanoids/Lite-Kimodo-Dataset`](https://huggingface.co/datasets/Berkeley-Humanoids/Lite-Kimodo-Dataset), and the LAFAN1 motions in the private [`Berkeley-Humanoids/Lite-LAFAN1-Dataset`](https://huggingface.co/datasets/Berkeley-Humanoids/Lite-LAFAN1-Dataset). The LAFAN1 motions show in the Space after you sign in with a Hugging Face account that can read that repo.

## Use

```bash
uv sync
uv run download --source lafan1bvh
uv run retarget --source lafan1bvh --workers -1
uv run build-site
uv run serve                  # then open http://localhost:8000
```

| Command | What it does |
|---|---|
| `uv run download` | Downloads a source into `.cache/<source>/`. |
| `uv run retarget` | Retargets the clips of a source by IK into the store, `dataset/`. |
| `uv run --extra reactor reactor train`, `export` | Retargets the motions of a store again in physics, with ReActor. |
| `uv run compare` | Measures the IK, UMR and ReActor on the same frames. |
| `uv run build-site`, `serve` | Builds and serves the motion browser, `site/`. |
| `uv run publish` | Uploads the Hub datasets and the Space. |

`uv run <command> --help` lists the options of a command. The package runs from its clone, and keeps `.cache/`, `dataset/` and `logs/` beside the code. The sources are `lafan1bvh`, `seed`, `amassx`, `lafan1`, `amass`, `motiondecode`, `kimodo`, `kimodolh2` and `kimodolh3`.

**Warning:** Check the license of a source before you share its motions. Most sources do not allow you to share them. See [Sources](docs/sources.md).

## Docs

- [Sources](docs/sources.md): the datasets, their licenses, and how to download them.
- [Retargeting](docs/retargeting.md): the three methods, how the IK retargets a clip, the metrics, and the code layout.
- [ReActor](docs/reactor.md): retargeting in physics, and its runs and studies.
- [UMR](docs/umr.md): how to run UMR on Lite Pro.
- [Compare the methods](docs/comparison.md): the IK, UMR and ReActor on the same frames, and why UMR slides.
- [The file format](docs/store.md): the layers of a motion.
- [The motion browser](docs/browser.md) and [Publish to the Hub](docs/publish.md).
