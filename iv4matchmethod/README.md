# iv4matchmethod

This project implements the MVP described in
`模板条件化轻量定位器_方案文档.md`:

- template-conditioned localization
- depthwise cross-correlation
- pose decoding (`center + angle + scale`)
- ROI follow
- optional prototype-based OK/NG judgement

## Quick start

Create an isolated virtual environment:

```powershell
.\scripts\bootstrap_venv.ps1
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Generate synthetic data:

```powershell
python -m iv4matchmethod synthesize --output demo_data --train-samples 48 --val-samples 12
```

Train the locator:

```powershell
python -m iv4matchmethod train --manifest demo_data/train.jsonl --output runs/smoke --epochs 2 --batch-size 4
```

The default training path uses the official `torchvision` `MobileNet V3 Small`
ImageNet weights for the backbone. To fall back to the original scratch
backbone:

```powershell
python -m iv4matchmethod train --manifest demo_data/train.jsonl --output runs/legacy --no-pretrained-backbone --backbone-variant legacy
```

Build a prototype bank for ROI-level OK/NG:

```powershell
python -m iv4matchmethod build-prototypes --manifest demo_data/train.jsonl --output runs/smoke/prototypes.npz
```

Open the local annotation tool:

```powershell
python -m iv4matchmethod annotate-template --image demo_data/user_images/template_full.png
```

Label a search image and optionally append it to a training manifest:

```powershell
python -m iv4matchmethod annotate-search `
  --template-annotation demo_data/user_images/template_full_annotation.json `
  --image demo_data/user_images/search_0001.png `
  --append-manifest demo_data/train_real.jsonl
```

Run XFeat matching (default: LighterGlue):

```powershell
python -m iv4matchmethod match-xfeat `
  --template-annotation demo_data/user_images/template_annotation.json `
  --search-image demo_data/user_images/test.png `
  --output-dir runs/xfeat_match
```

Run the lighter MNN matcher:

```powershell
python -m iv4matchmethod match-xfeat `
  --template-annotation demo_data/user_images/template_annotation.json `
  --search-image demo_data/user_images/test.png `
  --output-dir runs/xfeat_match_mnn `
  --matcher mnn `
  --max-dim 320 `
  --top-k 512
```

Run inference:

```powershell
python -m iv4matchmethod infer `
  --checkpoint runs/smoke/last.pt `
  --template-image demo_data/images/template_0000.png `
  --template-bbox "[80,88,96,80]" `
  --search-image demo_data/images/search_0000.png `
  --roi-ref-polygon "[[-20,-12],[20,-12],[20,12],[-20,12]]" `
  --prototype-bank runs/smoke/prototypes.npz `
  --debug-image runs/smoke/debug.png
```

## Manifest format

The training and prototype tools accept `.json` or `.jsonl` manifests with
records shaped like:

```json
{
  "template_image": "images/template_0000.png",
  "template_bbox": [80, 88, 96, 80],
  "search_image": "images/search_0000.png",
  "center": [191.5, 144.0],
  "angle_deg": 12.0,
  "scale": [1.05, 0.97],
  "roi_ref_polygon": [[-20, -12], [20, -12], [20, 12], [-20, 12]],
  "ok_ng": "OK"
}
```

`roi_ref_polygon` is expected in template-object coordinates with the template
`template_bbox` center as origin.

## Template annotation tool

Use the built-in GUI to mark a full-object template image:

```powershell
python -m iv4matchmethod annotate-template `
  --image demo_data/user_images/template_full.png `
  --output demo_data/user_images/template_full_annotation.json
```

Controls:

- `BBox mode`: drag a rectangle around the full target object.
- `ROI mode`: left-click to add polygon points for the inspection region.
- `Right click` or `Backspace`: remove the last ROI point.
- `Ctrl+S`: save JSON and preview image.
- `Enter`: save and exit.

The tool writes:

- `template_bbox`: `[x, y, w, h]` in template image coordinates
- `roi_image_polygon`: ROI polygon in image coordinates
- `roi_ref_polygon`: the same ROI shifted into `template_bbox`-center coordinates
- `*_preview.png`: overlay preview for quick validation

## Search image label tool

Use the search label GUI after the template annotation is done:

```powershell
python -m iv4matchmethod annotate-search `
  --template-annotation demo_data/user_images/template_full_annotation.json `
  --image demo_data/user_images/search_0001.png `
  --output demo_data/user_images/search_0001_label.json `
  --append-manifest demo_data/train_real.jsonl
```

Workflow:

- click the object center
- click the center of the same tooth / ROI target
- the tool computes `center`, `angle_deg`, and isotropic `scale`
- save a per-image label JSON
- optionally append a ready-to-train record to a `.jsonl` manifest

Controls:

- `Center mode`: click the object center
- `Tooth mode`: click the same ROI target in the search image
- `Set OK / Set NG`: choose label
- `Right click` or `Backspace`: undo the last point
- `Ctrl+S`: save JSON and preview image
- `Enter`: save and exit

## XFeat Matchers

This project also ships a registration path using the official XFeat model with
either the XFeat-specific LighterGlue weights or a simpler MNN/cosine matcher.

```powershell
python -m iv4matchmethod match-xfeat `
  --template-annotation demo_data/user_images/template_annotation.json `
  --search-image demo_data/user_images/test.png `
  --output-dir runs/xfeat_match `
  --matcher lighterglue `
  --max-dim 1024 `
  --top-k 4096
```

To use the simpler matcher:

```powershell
python -m iv4matchmethod match-xfeat `
  --template-annotation demo_data/user_images/template_annotation.json `
  --search-image demo_data/user_images/test.png `
  --output-dir runs/xfeat_match `
  --matcher mnn `
  --min-confidence 0.82
```

Common tradeoff:

- `--matcher lighterglue`: slower, usually more stable on hard images
- `--matcher mnn`: lighter and faster on CPU, but easier to drift on repeated or symmetric structures
- a practical CPU preset for `mnn` is `--max-dim 320 --top-k 512`

Outputs written to `--output-dir`:

- `xfeat_lighterglue_*.png/.json` or `xfeat_mnn_*.png/.json`
- the JSON result includes the selected `matcher`, keypoint counts, match counts, inliers, homography, and ROI polygon

## Notes

- The locator training path optimizes the pose heads.
- The OK/NG stage is implemented as a lightweight prototype bank built from
  aligned ROI patches.
- The default model is CPU-friendly and does not depend on `torchvision`.
