---
title: Lite Motions
emoji: 🤖
colorFrom: blue
colorTo: yellow
sdk: static
pinned: false
short_description: Motions retargeted to the Berkeley Lite Pro humanoid
datasets:
- Berkeley-Humanoids/Lite-LAFAN1-Dataset
hf_oauth: true
hf_oauth_scopes:
- read-repos
- gated-repos
---

# Lite Motions

This Space is the motion browser of [Lite-Motion-Retargeting][retarget]. It plays human
motion, retargeted from the Unitree G1 to the Berkeley Lite Pro, in your browser.

The page is static. `collections.json` lists the dataset folders, or collections, that the
page shows. The page reads each motion directly from the Hub, as a byte range of a pack.
To also see the private collections that your account can read, sign in with Hugging Face.

`scripts/publish.py` of [Lite-Motion-Retargeting][retarget] writes this Space. Change the
files there, not here.

[retarget]: https://github.com/Berkeley-Humanoids/Lite-Motion-Retargeting
