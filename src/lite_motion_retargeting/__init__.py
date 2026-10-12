"""Retarget motion datasets to the Berkeley Lite humanoid, and publish them.

Each step of the pipeline is a module with a command of its own, such as
``uv run retarget`` for ``retarget.py``. The other modules hold the parts that the steps
share:

    sources.py   the motion sets, and a loader for each
    human.py     human motion in one skeleton
    mapping.py   human motion -> goals for the Lite links, and the IK that reaches them
    common.py    the robot models, the paths of the clone, resampling and kinematics
    store.py     the layers of a motion, as Rerun files
"""
