# UMR

[UMR](https://hanyang9.github.io/UMR/) is a kinematic retargeter, like the IK of `retargeting/ik/`. It matches points on the surface of an SMPL-X body to points on the surface of the robot. Thus it needs no table that pairs the human bodies with the links of the robot. The IK and ReActor use the pairs of `retargeting/ik/human.py`. UMR reads only SMPL-X motion. It takes LAFAN1 after a conversion, and AMASS in its SMPL-X form, but not a G1 source such as Kimodo.

UMR pins its own versions of torch, MuJoCo and NumPy, so it runs in an environment of its own, outside this package. `src/lite_motion_retargeting/retargeting/umr/` holds the files that run it on Lite Pro:

| File | What it does |
|---|---|
| `lite_pro.json` | The robot config of UMR for Lite Pro. Lite Pro stands in a T-pose at zero, so the T-pose moves no joint. The joint limits hold the fingers within 0.01 rad of 0, so UMR solves the 34 joints that the IK solves. A limit of zero width stops the QP solver of UMR. |
| `lafan1_to_smplx.py` | Converts LAFAN1 BVH takes to SMPL-X with [`lafan_to_smplx`](https://github.com/jaraujo98/lafan_to_smplx), with the 10 betas of the body that UMR builds. |
| `ground_smplx.py` | Puts the feet of each SMPL-X take on the floor, as the authors of UMR do. The conversion leaves the bodies 2 to 6 cm below the floor, and UMR does not move them. |
| `batch.json` | The batch config of UMR, which solves each take in order. The default of UMR solves it in both directions and then picks, which makes joints jump. |
| `selection.patch` | Fixes two bugs in how UMR picks between the two directions. The solve in order does not need it. |
| `manifest.py` | Writes `manifest.json` beside the motions of a run: the hash of the robot model, the commit of UMR, and the configs. It runs in the environment of this package. |
| `amass_model.py` | Makes the SMPL-X model of AMASS loadable by UMR. |
| `prepare_amass.py` | Prepares AMASS SMPL-X clips for UMR and the IK: 30 Hz, 10 betas, and the standing feet on the floor. It skips the clips on stairs, beams or boxes. |
| `amass_clips.txt` | The AMASS clips of [the comparison](comparison.md): 4.2 h from 47 subjects of 8 subsets. |
| `__init__.py` | Reads a UMR motion into the world of the IK, for `uv run compare`. |

## UMR on LAFAN1

Run these steps in an empty directory beside the clone of this repo, such as `../umr`:

1. Clone UMR at commit `d1cdaed`, and clone `lafan_to_smplx`:

   ```bash
   git clone https://github.com/hanyang9/UMR && git -C UMR checkout d1cdaed
   git clone https://github.com/jaraujo98/lafan_to_smplx
   ```

2. Make the environment of UMR. GMR at commit `81ce42e` holds the BVH reader that `lafan_to_smplx` needs:

   ```bash
   uv venv --python 3.12
   uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
   uv pip install -r UMR/requirements-umr.txt
   uv pip install --no-deps "git+https://github.com/YanjieZe/GMR@81ce42e"
   ```

3. Download the SMPL-X v1.1 body models from the [SMPL-X site](https://smpl-x.is.tue.mpg.de/). Put the `.npz` files in `body_models/smplx/`, and link them where UMR reads them:

   ```bash
   ln -s ../../body_models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz UMR/smpl/
   ```

4. Copy Lite Pro and its config into UMR:

   ```bash
   lmr=$(realpath ../Lite-Motion-Retargeting)
   umr=$lmr/src/lite_motion_retargeting/retargeting/umr
   mjcf=$(cd $lmr && uv run python -c "from lite_description import get_mjcf_path; print(get_mjcf_path('lite_pro'))")
   robot=$(dirname "$(dirname "$mjcf")")
   mkdir -p UMR/assets/lite_pro && cp -r "$robot/mjcf" "$robot/meshes" UMR/assets/lite_pro/
   cp $umr/lite_pro.json UMR/robot_configs/
   cp $umr/batch.json UMR/lite_pro_batch.json
   ```

5. Convert the takes to SMPL-X on the CPU, then put them on the floor:

   ```bash
   ls $lmr/.cache/lafan1bvh/bvh | sed 's/\.bvh$//' > takes.txt
   xargs -P 16 -n 5 .venv/bin/python $umr/lafan1_to_smplx.py lafan_to_smplx/lafan_to_smplx.py \
       $lmr/.cache/lafan1bvh/bvh smplx body_models < takes.txt
   xargs .venv/bin/python $umr/ground_smplx.py body_models smplx smplx_grounded < takes.txt
   ```

6. Retarget the takes, and record what made them:

   ```bash
   cd UMR && ../.venv/bin/python scripts/humanoid_retarget_pipeline_batch.py \
       --config robot_configs/lite_pro.json --batch-config lite_pro_batch.json \
       --motion-folder ../smplx_grounded --workers 9 --retarget-gpus 0 --output-root ../output \
       --force-build --force-train
   uv run --project $lmr python -m lite_motion_retargeting.retargeting.umr.manifest \
       robot_configs/lite_pro.json lite_pro_batch.json ../output/lite_pro
   ```

   UMR writes `output/lite_pro/<take>_lite_pro.npz`, at 30 Hz. `uv run compare` interpolates it to 50 Hz. A run that solved at 50 Hz did no better.

UMR resolves a relative path against the folder of its batch config, so the batch config goes into the clone of UMR. With 9 workers, one RTX 3080 Ti retargets the 276 min of LAFAN1 in 33 min, and holds 7.3 GB. With 16 workers, the GPU runs out of memory.

UMR keeps the correspondence of the surfaces in `UMR/output/correspondence_lite_pro_*`. The name of the robot and the betas of the body name this cache, and the geometry does not. Thus step 6 trains the correspondence again with `--force-build --force-train`. Without these flags, a run after a change to Lite Pro uses the old correspondence. `uv run compare` stops if `manifest.json` names another model than the installed Lite Pro.

## UMR on AMASS

AMASS is SMPL-X motion already, so UMR and the IK read the same files. To retarget the clips of `amass_clips.txt`, do steps 1 to 4 of [UMR on LAFAN1](#umr-on-lafan1), then these steps:

1. Download the SMPL-X N archives of ACCAD, CMU, EKUT, Eyes Japan, HDM05, KIT, SFU and WEIZMANN, and the SMPL-X model with the locked head, from the [AMASS site](https://amass.is.tue.mpg.de). Extract them into `amass/` and `amass_model/`.

2. Make the model of AMASS loadable by UMR. UMR reads it through a robot config of its own:

   ```bash
   .venv/bin/python $umr/amass_model.py amass_model/neutral/model.npz \
       body_models/smplx/SMPLX_NEUTRAL.npz body_models/amass
   .venv/bin/python -c "import json; c = json.load(open('UMR/robot_configs/lite_pro.json')); \
       c['smplx_model_dir'] = '../../body_models/amass'; \
       json.dump(c, open('UMR/robot_configs/lite_pro_amass.json', 'w'), indent=2)"
   ```

3. Prepare the clips into the cache of `amassx`, and give the IK the same model:

   ```bash
   .venv/bin/python $umr/prepare_amass.py body_models/amass/SMPLX_NEUTRAL.npz amass \
       $lmr/.cache/amassx/smplx $umr/amass_clips.txt
   mkdir -p $lmr/.cache/smplx && cp body_models/amass/SMPLX_NEUTRAL.npz $lmr/.cache/smplx/
   ```

4. Retarget the clips with UMR and with the IK:

   ```bash
   cd UMR && ../.venv/bin/python scripts/humanoid_retarget_pipeline_batch.py \
       --config robot_configs/lite_pro_amass.json --batch-config lite_pro_batch.json \
       --motion-folder $lmr/.cache/amassx/smplx --workers 9 --retarget-gpus 0 \
       --output-root ../output/amass --force-build --force-train
   uv run --project $lmr python -m lite_motion_retargeting.retargeting.umr.manifest \
       robot_configs/lite_pro_amass.json lite_pro_batch.json ../output/amass/lite_pro
   cd $lmr && uv run retarget --source amassx --workers 40 --out dataset_amass
   ```

UMR trains a correspondence for each body, thus once for each of the 47 subjects. `prepare_amass.py` moves each clip up or down, so that the median height of a standing foot is `z = 0`. In most AMASS clips, a standing foot sinks 1 to 2 cm, and UMR anchors a foot to the floor only within 5 mm of it. The authors' rule for LAFAN1 does not apply, because it needs a T-pose at the ends of a take. UMR treats a folder named `motions` as its BONES-SEED dataset, so the clips of `amassx` go in `smplx/`.
