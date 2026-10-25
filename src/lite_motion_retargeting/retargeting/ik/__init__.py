"""The IK: retarget a clip by inverse kinematics, frame by frame.

A source is either human motion, such as the LAFAN1 BVH takes, or motion that its authors
already retargeted to the 29-DoF Unitree G1. The loader of the source decides which path a
clip takes, after the pipeline resamples the source to the output rate:

* **Human** (``human.py``): ``goals`` poses a skeleton with the proportions of Lite and the
  joint rotations of the human, ``plant`` keeps a planted foot still, and ``solve_ik``
  reaches the skeleton frame by frame with a floating base.
* **G1** (``g1.py``): a direct joint copy, ``lite_q = sign * g1_q + offset``, from the
  static :data:`g1.G1_TO_LITE` table. Then a per-frame ``mink`` IK refines the copy to
  match the feet and hands of the G1 relative to the pelvis, in position and orientation.
  A posture cost per DoF pulls the solution toward the copy, so the IK changes it only a
  little.
"""
