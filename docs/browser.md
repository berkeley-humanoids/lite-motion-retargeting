# The motion browser

`site/` is the motion browser: a static page that lists every motion in `dataset/` as a card. When you point at a card, the card plays its motion. When you open a card, the page shows a stage with a timeline, frame stepping, the path of the pelvis, and the command that opens the motion in Rerun. To open a motion directly, add `#<motion>` to the end of the address.

```bash
uv run build-site                      # dataset/ -> site/data/
uv run serve                           # then open http://localhost:8000
```

`uv run build-site` reads the robot layer of each motion, which is the layer that training reads. It writes the layer for the page as float32 rows, with the motions end to end in a few packs per source. `browser.py` documents the format. The page loads the robot meshes once. It reads a motion only when the card of that motion comes into view, as a byte range of its pack. Serve the page with `uv run serve`, or with another server that answers range requests, such as Caddy. `python -m http.server` answers a range request with the whole file.

`--sources lafan1` builds a page with LAFAN1 only. For a page that other people can reach, build with `--sources kimodo kimodolh2 kimodolh3`. The licenses of LAFAN1, AMASS and MotionDecode do not allow you to share their motions with the public.

The page shows the joints that each motion drives, and how many joints of each Lite variant the motion drives. It also filters the motions by the joint groups that they drive.

## Two views

The page plays a motion in two views. **Animation** plays it on a character, and the page shows this view first. **URDF** plays it on the robot of the URDF. The character is the VRM 1.0 model at `site/character.vrm`, which git does not track. Without this file, the page shows only the robot. If `site/` holds the file, `uv run publish` uploads it to the Space.

The character uses the joint angles unchanged. Each joint turns its bone about the axis that the joint has in the URDF at rest, where the character and the robot both stand in a T-pose. At the waist, neck, shoulders, wrists, hips and ankles, the robot stacks three joints at one point. There, the bone is a ball joint and uses all three angles, in the order that the URDF chains them. A finger uses its two knuckle joints the same way. The pelvis of the robot moves the hips of the character, so that the hip joints of the two line up and both stand with their soles on the floor.

## Collections

`site/collections.json` lists the collections that the page reads: folders of motion data, `data/` by default. A collection can also be on the Hub, because the Hub answers range requests from any site. The [Lite Motions](https://huggingface.co/spaces/Berkeley-Humanoids/Lite-Motions) Space works this way: its `collections.json` points at `web/` of each Hub dataset, as of the last commit to that repo. A collection marked `"private": true` loads only for a visitor who signs in with Hugging Face. The page reads it with the token of that account, so a private repo shows only to the people who can read it. The page merges the collections into one list, and draws the robot of the first.
