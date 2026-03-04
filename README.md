# Robust Template Matching (30% Occlusion Tolerant)

This repository provides a pragmatic template matching pipeline for scenes with partial occlusion.

Key design choices:
- Candidate generation is **not** based on a single global hard constraint.
- It combines:
  - `ORB + RANSAC` for coarse similarity/affine pose.
  - `part-based voting` by splitting template edges into blocks.
- Final score is robust:
  - truncated Chamfer distance
  - trimmed mean over best points (default keep 70%)
  - inlier ratio (`distance < threshold`)
- Lightweight refinement only adjusts `x, y` (fast and enough for low precision targets).

## Why it fits your requirement

- Allowing 30% occlusion means score should ignore a bad subset of points.
- The matcher uses `--occlusion 0.30` by default, which keeps only the best `70%` transformed template edge points for the main distance term.
- Part voting still works when some blocks are occluded, as long as enough blocks agree on pose.

## Files

- `robust_template_match.py`: main implementation and CLI.
- `line2dup_make_template.py`: build reusable line2Dup-like template model.
- `line2dup_find_with_template.py`: find objects in scene using built model.
- `line2dup_edit_template_points.py`: interactive add/delete editor for template feature points.
- `line2dup_roi_template_workbench.py`: select ROI, auto-generate points from parameters, then edit and save.
- `line2dup_template_workbench.py`: unified entry for template create/edit/find.

## Install

```bash
py -3 -m pip install -r requirements.txt
```

## Usage

```bash
py -3 robust_template_match.py ^
  --template path\\to\\template.png ^
  --scene path\\to\\scene.png ^
  --out match_result.png ^
  --occlusion 0.30
```

Two-stage line2Dup-like workflow:

1) Build template model (with visualization):

```bash
py -3 line2dup_make_template.py ^
  --template path\\to\\template.png ^
  --out-model path\\to\\template_model.json ^
  --class-id object ^
  --num-features 192 ^
  --weak-thresh 20 ^
  --strong-thresh 35 ^
  --levels 4,8 ^
  --angle-start 0 --angle-end 360 --angle-step 10
```

2) Find with model:

```bash
py -3 line2dup_find_with_template.py ^
  --model path\\to\\template_model.json ^
  --scene path\\to\\scene.png ^
  --out find_result.png ^
  --threshold 55 ^
  --nms-iou 0.3 ^
  --topk 20
```

Case-like robust CLI usage (threshold fallback + stride crop):

```bash
py -3 line2dup_find_with_template.py ^
  --model path\\to\\template_model.json ^
  --scene path\\to\\scene.png ^
  --class-ids object ^
  --threshold 90 ^
  --auto-sweep --sweep-min 20 --sweep-step 5 ^
  --crop-stride 32 ^
  --nms-iou 0.5 ^
  --topk 20 ^
  --out find_result.png
```

3) Interactive edit points (optional):

```bash
py -3 line2dup_edit_template_points.py ^
  --model path\\to\\template_model.json ^
  --template-image path\\to\\template.png ^
  --zoom 2.0
```

Mouse and hotkeys:
- Left drag: add point and infer direction label from drag angle
- Right click: delete nearest point
- Mouse wheel: zoom in/out
- `s`: save model
- `q` / `Esc`: quit
- `c` / `x`: next / prev class
- `n` / `p`: next / prev template id
- `l` / `k`: next / prev pyramid level
- `]` / `[` : label + / -
- `+` / `-`: zoom in / out, `0`: reset zoom to 1x
- `u`: undo
- `h`: toggle help text
- `i`: toggle status text

4) ROI workbench (recommended template creation flow):

```bash
py -3 line2dup_roi_template_workbench.py ^
  --image path\\to\\image.png ^
  --out-model path\\to\\template_model.json ^
  --class-id object ^
  --levels 4,8 ^
  --num-features 128 ^
  --weak-thresh 30 ^
  --strong-thresh 60 ^
  --zoom 2.0
```

Inside workbench:
- First select ROI on the image.
- Trackbars auto-regenerate feature points when parameters change.
- Left drag adds points with direction, right click removes points.
- `s` saves model, `r` forces regenerate, `h`/`i` toggles overlays, mouse wheel or `+/-` zooms.

5) Unified entry (all modes in one command, mouse/menu first):

```bash
py -3 line2dup_template_workbench.py
```

The GUI includes two tabs:
- `Create Template`: select image, drag ROI, extract points, mouse-edit points, save model.
- `Edit Model`: open model, optionally load template image, mouse-edit points, save model.
- `Find`: open model + scene, tune threshold/NMS/verify parameters, run match, save overlay.

In `Create Template`, angle/scale range is also configurable in the panel:
- `Angle Start/End/Step`
- `Scale Start/End/Step`
`Save Model` in GUI now follows `line2dup_make_template.py` flow:
- per angle/scale pose: transform ROI image + mask
- re-extract features via `add_template` (not point-only geometric warp)
- feature count scales with `scale`

Mouse operations (no keyboard required):
- Left drag: ROI select (Create tab, ROI mode) or directional point add (point mode).
- Right click: delete nearest point.
- Mouse wheel: zoom in/out.

Optional preload arguments:

```bash
py -3 line2dup_template_workbench.py ^
  --image path\\to\\image.png ^
  --model path\\to\\template_model.json ^
  --scene path\\to\\scene.png ^
  --template-image path\\to\\template.png ^
  --out-model path\\to\\output_model.json
```

Main tunable arguments:
- `--occlusion`: allowed occlusion ratio. `0.30` means keep best 70%.
- `--grid-rows`, `--grid-cols`: number of template blocks for part voting.
- `--part-topk`: per-block top local candidates.
- `--part-score-thresh`: minimum local block match score.
- `--part-vote-max`: max translation vote hypotheses before robust scoring.
- `--vote-bin`: translation vote quantization.
- `--rotation-step`: rotation sweep step in degrees for candidate expansion.
- `--trunc-dist`: distance cap for truncated Chamfer.
- `--inlier-dist`: threshold for inlier ratio.
- `--spatial-nms-radius`, `--angle-nms-deg`: post-score local deduplication.
- `--cell-size`, `--max-per-cell`: diversity quota across scene regions.
- `--quality-min-score`, `--quality-min-inlier`, `--quality-max-trimmed`: post-score quality filters.
- `--cluster-radius`: spatial radius for cluster round-robin ordering (improves multi-instance coverage).
- `--draw-top`, `--legend-top`: visualization density controls.
- `--draw-nms-iou`, `--draw-nms-center`: visualization-only dedup controls.
- `--no-refine`: disable XY refinement for extra speed.

## Output

- Console prints top-N match hypotheses with:
  - score
  - source (`orb_ransac` or `part_vote`)
  - trimmed mean distance
  - inlier ratio
  - orientation error
  - affine matrix
- Image output shows the best transformed template quadrilateral.

## Notes

- Current implementation focuses on practical robustness with moderate compute.
- If rotation/scale variation is large and ORB is weak, increase texture or add a coarse angle/scale sweep before scoring.
