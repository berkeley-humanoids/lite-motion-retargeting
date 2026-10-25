# Compare the methods

`uv run compare` measures the IK, UMR and ReActor on the same frames of one source:

```bash
uv run compare --source lafan1bvh --umr ../umr/output/lite_pro --reactor dataset_reactor
uv run compare --source amassx --store dataset_amass --umr ../umr/output/amass/lite_pro
uv run compare --source kimodo --reactor dataset_reactor
```

The metrics that each store keeps do not compare across methods. For example, `reactor export` measures the slide where its learned goals plant a foot. `uv run compare` measures every method in the same way:

- **One mask.** Every method uses the stance mask of the source: the human feet for a human source, and the goals of the IK for a G1 source.
- **The same frames.** With a ReActor store, every method is measured on the frames of each ReActor piece.
- **Pooled samples.** A mean, a median or a percentile is over every sample of every piece. Thus a long motion counts more than a short one, and the cut of a motion into pieces changes nothing.
- **Fidelity to the source.** The limb direction error, the root error over 1 s and the contact IoU compare each method with its source. The IK aims at the limb directions of the source, so this metric favors the IK, as the slide where the source stands does.
- **One world.** For LAFAN1, UMR uses a world turned 90° from that of the IK, and for AMASS the same world. `compare` turns a UMR motion by the fixed angle of its source, and shifts it by one offset per motion. UMR scales the human by its height, not by the height of its pelvis, so `compare` leaves out the goal errors of UMR.
- **One model.** Every robot layer and every UMR run records the hash of its Lite Pro model. `compare` stops if a hash differs from that of the installed model.

## Results

This table gives the results on LAFAN1, over the 138 pieces of the All, 20k run of [ReActor](reactor.md#runs), 274 of the 276 min. UMR ran as [UMR on LAFAN1](umr.md#umr-on-lafan1) shows.

| Metric | IK | UMR | ReActor |
|---|---|---|---|
| Slide where the source stands, mean / median / p95 | **2.5** / **1.2** / **9.1** cm/s | 12.4 / 7.8 / 39.6 cm/s | 5.2 / 1.4 / 23.5 cm/s |
| Slide where the sole is within 1 cm of the floor, mean / median | 29.9 / **1.8** cm/s | 37.0 / 11.7 cm/s | **14.7** / 1.9 cm/s |
| Contact IoU with the source | 73.7% | 70.9% | **74.3%** |
| Gap where the source stands | 0.54 cm | 0.35 cm | **0.30 cm** |
| Penetration, p95 of frames | 1.42 cm | **0.09 cm** | 0.34 cm |
| Fastest joint, p99 / max | 15.4 / 20.0 rad/s | **8.4** / **11.4** rad/s | 13.8 / 37.9 rad/s |
| Limb direction error | **4.3°** | 19.9° | 9.1° |
| Root error over 1 s | **2.8 cm** | 6.3 cm | 6.6 cm |
| Goal error, feet / hands | 0.32 / 0.67 cm | | 10.0 / 9.7 cm |

Kimodo is a G1 source, so UMR cannot read it, and the IK copies the G1 joints without foot planting. This table covers its 9,461 ReActor pieces, 627 of 630 min:

| Metric | IK | ReActor |
|---|---|---|
| Slide where the source stands, mean / median / p95 | 3.5 / 1.5 / 13.4 cm/s | **1.1** / **0.25** / **2.8** cm/s |
| Slide where the sole is within 1 cm of the floor, mean / median | 9.0 / 1.8 cm/s | **3.4** / **0.30** cm/s |
| Contact IoU with the source | 82.0% | **85.0%** |
| Gap where the source stands | 0.53 cm | **0.26 cm** |
| Penetration, p95 of frames | 1.05 cm | **0.19 cm** |
| Fastest joint, p99 / max | 9.8 / 56.3 rad/s | **8.1** / **37.3** rad/s |
| Goal error, feet / hands | **6.55** / 13.61 cm | 8.13 / **12.76** cm |

AMASS is SMPL-X motion, so UMR reads it without a conversion. This table covers the 1,136 clips of [UMR on AMASS](umr.md#umr-on-amass), 252 min. ReActor did not run on it.

| Metric | IK | UMR |
|---|---|---|
| Slide where the source stands, mean / median / p95 | **1.4** / **0.59** / **5.3** cm/s | 7.6 / 3.8 / 27.5 cm/s |
| Slide where the sole is within 1 cm of the floor, mean / median | **15.8** / **0.70** cm/s | 24.6 / 5.5 cm/s |
| Contact IoU with the source | 79.7% | **80.3%** |
| Gap where the source stands | 0.39 cm | **0.14 cm** |
| Penetration, p95 of frames | 0.58 cm | **0.10 cm** |
| Fastest joint, p99 / max | 10.5 / 20.0 rad/s | **7.4** / **11.7** rad/s |
| Limb direction error | **3.7°** | 12.0° |
| Root error over 1 s | 4.3 cm | **1.9 cm** |
| Goal error, feet / hands | 0.26 / 0.84 cm | |

No method is best on every metric:

- **The IK** keeps a foot still where the source stands, and copies the limbs of the source most closely. It plants its goals on exactly those frames, so both metrics measure what it aims for. Where a sole is near the floor, its median slide is low, but its mean is 17 times its median: a few frames, such as a swing foot low over the floor, slide fast. Its feet sink the deepest.
- **ReActor** sits on the floor best, and its feet slide the least where a sole is near the floor. On Kimodo, it beats the IK on every floor metric. On LAFAN1, one of its joints peaks at 38 rad/s, its feet leave the goals of the IK by 10 cm, its limbs turn 9° from the source, and it needs the wrench. The motions on the ground are its worst: their feet slide 11.6 cm/s where the source stands.
- **UMR** sinks the least, and moves its joints the slowest. But where the source stands, its feet slide 5 times as fast as those of the IK on LAFAN1, and its limbs turn 20° from the source. On LAFAN1, UMR scales the human by 0.77 and the IK by 0.84, so a UMR root travels about 8% less, which adds to its root error. On AMASS, its root follows the source more closely than that of the IK. [Why UMR slides](#why-umr-slides) shows that part of its slide comes from its input, and the rest comes from UMR.

These metrics measure the motions, not the tracking policy that trains on them.

The three methods also differ in cost, on the 276 min of LAFAN1:

| Method | Runs on | Time |
|---|---|---|
| IK | CPU | 2.3 h on one core, or 5.4 min with 40 workers. On the 252 min of AMASS, 4.8 min with 40 workers |
| UMR | GPU | 33 min on one RTX 3080 Ti, with 9 workers. On the 252 min of AMASS, 34 min on an RTX 3090 and an RTX 3080 Ti with 18 workers, with the 47 correspondences |
| ReActor | GPU | 6.7 h of training on one RTX 3090 for 20,000 iterations, then 6 min of export |

## Why UMR slides

An outside audit of this comparison named possible causes of the UMR slide. Each row of this table changes one thing, on 8 LAFAN1 takes of 8 kinds, 27 min. The IK slides 2.2 cm/s on them, in the mean.

| UMR setup | Slide where the source stands, mean / median | Fastest joint |
|---|---|---|
| 1. The first run: 16 betas, the 5th-percentile grounding, solved in both directions | 11.5 / 7.6 cm/s | 27.7 rad/s |
| 2. 10 betas, and the grounding of the UMR authors | 11.5 / 7.5 cm/s | 27.7 rad/s |
| 3. Row 2, without the smoothing | 12.1 / 7.0 cm/s | 39.1 rad/s |
| 4. Row 2, with the fingers held at 0 | 11.5 / 7.5 cm/s | 27.8 rad/s |
| 5. Row 2, solved at 50 Hz | 11.7 / 7.9 cm/s | 23.3 rad/s |
| 6. Row 2, solved in order | 11.4 / 7.7 cm/s | 10.9 rad/s |
| 7. The setup of [UMR on LAFAN1](umr.md#umr-on-lafan1), with a ground anchor cost of 50, not 5 | 19.3 / 7.6 cm/s | 11.7 rad/s |
| 8. The same, with a ground anchor cost of 500 | 30.5 / 11.7 cm/s | 11.2 rad/s |

- **The input matches that of the authors.** `lafan1_to_smplx.py` gives the poses of the `dance1_subject2` file that UMR ships, and `ground_smplx.py` its height, to 1 mm.
- **No change to the input or the solver brings the slide near that of the IK.** Solving in order keeps the slide and stops the joint jumps. [UMR on LAFAN1](umr.md#umr-on-lafan1) now runs this way.
- **A stronger ground anchor makes the slide worse.** The anchor holds a point of the foot while the source foot slides, then lets go.
- **UMR slides as much on the G1, the robot of the UMR paper.** On the same 8 takes at 30 Hz, a G1 foot slides 11.7 / 8.0 cm/s, and a Lite Pro foot 12.1 / 7.9 cm/s. Thus the slide does not come from the Lite Pro config.
- **The input slides.** Where the human stands, the feet of the SMPL-X conversion slide 7.0 cm/s, and those of the BVH 2.1 cm/s. UMR follows the surface of its input. Its ground anchors are active in 97% of the frames of `walk1_subject1`, but they do not stop the slide.

**Picking between the two directions makes the joints jump, even after a fix.** By default, UMR solves a take forward and backward, then picks a solve for each frame. `selection.patch` fixes two bugs in that pick:

- **The cost before the step.** UMR compares each frame by its residual before the last solver step, thus by its start pose. At the solution, the other direction is cheaper on 20% of the frames. This fix changes the pick on 18% of the frames of the 8 takes, and lowers the fastest joint from 27.8 to 9.2 rad/s.
- **The root.** The cost of a switch between the two solves counts the joints, not the root. The two solves rarely differ at the root, by 0.2 to 0.8 cm in the median take, so this fix changes the pick on only 0.7% of the frames.

On all of LAFAN1, the fixed pick still switches 2,303 times, and joints jump at some switches. In `aiming2_subject3`, the two solves hold the right arm in two ways, 2.33 rad apart, and the switch costs less than what the cheaper solve saves over 50 frames:

| UMR on LAFAN1 | Slide where the source stands, mean | Slide where the sole is near the floor, mean | Fastest joint, p99 / max | Time |
|---|---|---|---|---|
| In order | 12.4 cm/s | **37.0 cm/s** | **8.4** / **11.4** rad/s | **1×** |
| Both directions, both fixes | **12.0 cm/s** | 38.5 cm/s | 8.7 / 47.7 rad/s | About 2× |

Thus [UMR on LAFAN1](umr.md#umr-on-lafan1) keeps solving in order.

**On AMASS, the input does not slide, and UMR still does.** Where the human stands, the lowest foot vertex of the AMASS clips slides 2.1 cm/s, as the LAFAN1 BVH does. UMR then slides 7.6 cm/s, less than the 12.4 cm/s of LAFAN1, but 3.6 times its input and 5.6 times the IK. It slides more than the IK on all 1,136 clips, by a factor of 4 to 10 in each of the 8 subsets. Thus on LAFAN1, part of the slide of UMR comes from its input, and the rest comes from UMR itself.
