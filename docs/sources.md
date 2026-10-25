# Sources

A source is one motion dataset. It holds either human motion, or motion that its authors already retargeted to the 29-DoF Unitree G1. `src/lite_motion_retargeting/sources/` defines each source and its loader.

| Source | Clips | Rate | Size | License | On the Hub |
|---|---|---|---|---|---|
| `lafan1bvh`: [LAFAN1 BVH][lafan1bvh], human | 77 | 30 Hz | 0.35 GB | CC BY-NC-ND 4.0 | Private |
| `amassx`: [AMASS SMPL-X][amassx], human | Your copy | 60 to 250 Hz | 0.3 GB for 4.2 h at 30 Hz | AMASS: non-commercial research, no redistribution | No |
| `seed`: [BONES-SEED][seed], human | 142,220 | 120 Hz | 45 GB, 276 GB unpacked | BONES-SEED: academic or startup (revenue under $1M) use only, no redistribution | No |
| `lafan1`: [LAFAN1][lafan1] | 40 | 30 Hz | 0.3 GB | CC BY-NC-ND 4.0 | No |
| `amass`: [AMASS][amass], all subsets | 17,717 | 60 to 250 Hz | 7 GB | AMASS: non-commercial research, no redistribution | No |
| `motiondecode`: [ChingMu MotionDecode][motiondecode] | 92,559 | 120 Hz | 148 GB | ChingMu: non-commercial research, not to be hosted as a competing dataset | No |
| `kimodo`: [Kimodo G1][kimodo] | 10,000 | 50 Hz | 1.2 GB | Apache-2.0 | Yes |
| `kimodolh2`: [Kimodo G1 long-horizon v2][kimodolh2] | 10,000 | 50 Hz | 5.8 GB | Apache-2.0 | Yes |
| `kimodolh3`: [Kimodo G1 long-horizon v3][kimodolh3] | 10,000 | 50 Hz | 12.8 GB | Apache-2.0 | Yes |

The `seed`, `amassx`, `amass` and `motiondecode` motions stay in your local store, `dataset/`, for training. `uv run publish` uploads `lafan1bvh` to a private repo, and the three Kimodo sets to a public one. See [Publish to the Hub](publish.md).

## Human sources

`lafan1bvh` is the human motion that the authors of `lafan1` retargeted: the BVH takes of Ubisoft's LAFAN1. These takes come as a zip on GitHub, not on the Hub. `lafan1` holds 40 of them, with the same names and frames, so the two [IK paths](retargeting.md#how-the-ik-retargets-a-clip) compare clip for clip.

`amassx` is the human motion behind `amass`: the AMASS clips in their SMPL-X form. Its reader builds the body with the SMPL-X model of AMASS, which it reads from `.cache/smplx/SMPLX_NEUTRAL.npz`. Both need an MPI account, so `uv run download` cannot fetch them. Copy the clips to `.cache/amassx/smplx/` and the model to `.cache/smplx/` yourself. [UMR on AMASS](umr.md#umr-on-amass) prepares a sample of them.

`seed` is [BONES-SEED][seed] by Bones Studio: 288 h of motion capture by 522 actors. Half of its takes are mirrors of the other half, and their names end in `_M`. This source reads the SOMA Proportional form: BVH skeletons that keep the proportions of each actor, and thus the foot contacts of the capture. The reader takes the rest pose of each joint from the T-pose of the SOMA rig in the download. The hands have finger joints, but no finger moves in 60 random takes.

The Hub repo of BONES-SEED is gated, so accept its license on the Hub first. Then `uv run download --source seed` fetches the 45 GB archive, and files each take under `clips/<category>/`, after the 20 categories of the metadata. With 120 workers, the IK retargets the 288 h in 3 h, and the store takes 1.3 TB, most of it the base layer: 78 joints at 120 Hz. Where an actor stands, a Lite Pro foot slides 1.8 cm/s in the mean take.

In 476 takes, some channels are NaN between valid frames, or a bone length is NaN, and the reader skips the take. In 206 other takes, NaN frames end the take, and the reader drops these frames.

**Warning:** Keep the `seed` clips and their retargets in a directory that only you can read, and out of the Hub and the motion browser. The license of BONES-SEED admits academic users and startups with a revenue under $1M a year only. It forbids any redistribution, and it binds you to keep the data confidential. A model or a page that uses the motions must credit "Motion Data by Bones Studio", with a link to <https://bones.studio/>.

## G1 sources

The other sources ship G1 motion only. MotionDecode gives its BVH files on request, and Kimodo generated its clips on the G1 directly.

`kimodo` is not motion capture. NVIDIA's Kimodo model generated each clip from a text prompt, such as "a person throwing boxing punches". Then a tracking policy played the clip on a simulated G1. Each clip is 4 s long, and its category is its prompt. `kimodo` ships as LeRobot shards, and `uv run download` splits them into one file per clip.

`kimodolh2` and `kimodolh3` come from the same model and policy, but each prompt chains several actions, as in "combat drill: kick → walk → sidestep". A v2 clip chains 3 to 8 actions over about 20 s, and a v3 clip chains about 50 actions over about 43 s. The category of these clips is the family that the prompt names.

AMASS bundles a license per subset, in `.cache/amass/g1/<subset>/license.txt`. Some subsets, such as CMU, are less restrictive than AMASS as a whole.

## Download part of a source

To download only part of a large source, pass Hub glob patterns to `--include`. They replace the default patterns of the source. Then retarget the clips that you downloaded:

```bash
uv run download --source amass --include 'g1/CMU/**'
uv run retarget --source amass --clip '^CMU/' --workers -1

uv run download --source motiondecode \
    --include 'samples/1.3.Basic_Gait_Category/**' 'samples/5.Dance/**'
uv run retarget --source motiondecode --workers -1
```

The regex of `--clip` matches the path of a clip below the clip directory of its source, without the suffix, such as `CMU/07/07_01_poses_120_jpos`. The motion name comes from the same path: `amass_CMU_07_07_01_poses_120_jpos`.

## Size and time

The IK of the G1 path solves about 500 frames per second on one core, and the IK of the human path solves about 100. A robot layer holds 12 to 16 MB for each minute of motion. A base layer keeps the source rate, so it adds 10% to 60% to that size. The measured totals are:

| Source | Motions | Hours | Store | `site/data` |
|---|---|---|---|---|
| `lafan1` | 40 | 2.5 | 2.6 GB | 0.07 GB |
| `lafan1bvh` | 77 | 4.6 | 6.0 GB | |
| `amass`, all subsets | 17,717 | 59 | 66 GB | 2.0 GB |
| `motiondecode` | 92,558 | 991 | 1,146 GB | 29.9 GB |
| `kimodo` | 9,490 | 10 | 11 GB | 0.3 GB |
| `kimodolh2` | 9,947 | 54 | 44 GB | 1.8 GB |
| `kimodolh3` | 9,941 | 118 | 104 GB | 3.5 GB |
| All | 139,693 | 1,234 | 1.37 TB | 37.6 GB |

One MotionDecode clip is empty upstream. On 128 cores, `uv run retarget` took 3 h 40 min for 64,224 MotionDecode clips. `uv run build-site` takes about 45 min for all sources.

[lafan1]: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
[lafan1bvh]: https://github.com/ubisoft/ubisoft-laforge-animation-dataset
[amass]: https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics
[amassx]: https://amass.is.tue.mpg.de
[seed]: https://huggingface.co/datasets/bones-studio/seed
[motiondecode]: https://huggingface.co/datasets/CMRobot/MotionDecode
[kimodo]: https://huggingface.co/datasets/cagataydev/kimodo-g1-motion-v1
[kimodolh2]: https://huggingface.co/datasets/cagataydev/kimodo-g1-longhorizon-v2
[kimodolh3]: https://huggingface.co/datasets/cagataydev/kimodo-g1-longhorizon-v3
