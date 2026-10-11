---
title: Lite Motion Dataset
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

# Lite Motion Dataset

The motion browser of [Lite-Motion-Retargeting][retarget]: human motion retargeted from
the Unitree G1 to the Berkeley Lite Pro, played in your browser.

The page is static. `collections.json` lists the dataset folders it shows, and the page
reads each motion straight from the Hub, as a byte range of a pack. Sign in with Hugging
Face to also see the private collections your account can read.

`scripts/publish.py` of [Lite-Motion-Retargeting][retarget] writes this Space. Change the
files there, not here.

[retarget]: https://github.com/Berkeley-Humanoids/Lite-Motion-Retargeting
