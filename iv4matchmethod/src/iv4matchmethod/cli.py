from __future__ import annotations

import argparse
import json
from pathlib import Path

from iv4matchmethod.data import load_manifest
from iv4matchmethod.annotate import run_annotation_tool
from iv4matchmethod.infer import run_inference
from iv4matchmethod.prototype import build_bank_from_manifest
from iv4matchmethod.search_label import run_search_label_tool
from iv4matchmethod.synthetic import synthesize_dataset
from iv4matchmethod.train import train_locator
from iv4matchmethod.xfeat_lighterglue import run_xfeat_match


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Template-conditioned lightweight locator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train the locator network")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--log-interval", type=int, default=5)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", default=None)
    train.add_argument("--backbone-variant", choices=["legacy", "torchvision"], default="torchvision")
    train.set_defaults(pretrained_backbone=True)
    train.add_argument("--pretrained-backbone", dest="pretrained_backbone", action="store_true")
    train.add_argument("--no-pretrained-backbone", dest="pretrained_backbone", action="store_false")
    train.add_argument("--template-size", type=int, default=128)
    train.add_argument("--search-size", type=int, default=384)
    train.add_argument("--feature-stride", type=int, default=8)
    train.add_argument("--fuse-channels", type=int, default=96)
    train.add_argument("--head-channels", type=int, default=64)
    train.add_argument("--heatmap-sigma", type=float, default=1.75)

    infer = subparsers.add_parser("infer", help="Run locator inference")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--template-image", required=True)
    infer.add_argument("--template-bbox", required=True)
    infer.add_argument("--search-image", required=True)
    infer.add_argument("--roi-ref-polygon", default=None)
    infer.add_argument("--prototype-bank", default=None)
    infer.add_argument("--debug-image", default=None)
    infer.add_argument("--roi-patch-output", default=None)
    infer.add_argument("--roi-size", type=int, default=128)
    infer.add_argument("--device", default=None)

    synth = subparsers.add_parser("synthesize", help="Generate a synthetic dataset")
    synth.add_argument("--output", required=True)
    synth.add_argument("--train-samples", type=int, default=48)
    synth.add_argument("--val-samples", type=int, default=12)
    synth.add_argument("--seed", type=int, default=13)

    proto = subparsers.add_parser("build-prototypes", help="Build prototype bank from aligned ROI patches")
    proto.add_argument("--manifest", required=True)
    proto.add_argument("--output", required=True)
    proto.add_argument("--roi-size", type=int, default=128)

    annotate = subparsers.add_parser("annotate-template", help="Open a local GUI to mark template_bbox and ROI")
    annotate.add_argument("--image", required=True)
    annotate.add_argument("--output", default=None)
    annotate.add_argument("--load", default=None)
    annotate.add_argument("--fit-size", type=int, default=1200)

    search_annotate = subparsers.add_parser("annotate-search", help="Label search images with center/angle/scale")
    search_annotate.add_argument("--template-annotation", required=True)
    search_annotate.add_argument("--image", required=True)
    search_annotate.add_argument("--output", default=None)
    search_annotate.add_argument("--load", default=None)
    search_annotate.add_argument("--append-manifest", default=None)
    search_annotate.add_argument("--ok-ng", default="OK")
    search_annotate.add_argument("--fit-size", type=int, default=1200)

    xfeat = subparsers.add_parser("match-xfeat", help="Match template and search images with XFeat-based matchers")
    xfeat.add_argument("--template-annotation", required=True)
    xfeat.add_argument("--template-image", default=None)
    xfeat.add_argument("--search-image", required=True)
    xfeat.add_argument("--output-dir", default=None)
    xfeat.add_argument("--matcher", choices=["lighterglue", "mnn"], default="lighterglue")
    xfeat.add_argument("--top-k", type=int, default=4096)
    xfeat.add_argument("--detection-threshold", type=float, default=0.05)
    xfeat.add_argument("--min-confidence", type=float, default=None)
    xfeat.add_argument("--max-dim", type=int, default=1024)
    xfeat.add_argument("--ransac-reproj-threshold", type=float, default=4.0)
    xfeat.add_argument("--max-draw-matches", type=int, default=80)
    xfeat.add_argument("--no-write-visuals", action="store_true")
    xfeat.add_argument("--no-write-json", action="store_true")
    xfeat.add_argument("--quiet", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        checkpoint = train_locator(args)
        print(json.dumps({"checkpoint": str(checkpoint.resolve())}, indent=2))
        return 0
    if args.command == "infer":
        run_inference(args)
        return 0
    if args.command == "synthesize":
        synthesize_dataset(args)
        return 0
    if args.command == "build-prototypes":
        manifest_path = Path(args.manifest)
        records = load_manifest(manifest_path)
        stats = build_bank_from_manifest(records, args.output, manifest_path.parent, roi_size=args.roi_size)
        print(json.dumps({"output": str(Path(args.output).resolve()), **stats}, indent=2))
        return 0
    if args.command == "annotate-template":
        return run_annotation_tool(args)
    if args.command == "annotate-search":
        return run_search_label_tool(args)
    if args.command == "match-xfeat":
        run_xfeat_match(args)
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
