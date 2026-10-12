---
title: Lite Motions
emoji: 🤖
colorFrom: blue
colorTo: yellow
sdk: static
pinned: false
short_description: Motions retargeted to the Berkeley Lite Pro humanoid
# uv run publish replaces this list with the public Hub datasets.
datasets:
- Berkeley-Humanoids/Lite-Kimodo-Dataset
hf_oauth: true
hf_oauth_scopes:
- read-repos
- gated-repos
---

# Lite Motions

This Space is the motion browser of [Lite-Motion-Retargeting][retarget]. It plays the
motions that Lite-Motion-Retargeting retargeted to the Berkeley Lite Pro, in your browser.

The page is static. `collections.json` lists the collections that the page shows: the
`web/` folder of each Hub dataset of Lite-Motion-Retargeting. The page reads each motion
directly from the Hub, as a byte range of a pack. To also see the private collections that
your account can read, sign in with Hugging Face. The LAFAN1 motions are private, because
the license of LAFAN1 does not allow anyone to share a retarget of it with the public.

`uv run publish` of [Lite-Motion-Retargeting][retarget] writes this Space. Change the
files there, not here.

[retarget]: https://github.com/Berkeley-Humanoids/Lite-Motion-Retargeting
