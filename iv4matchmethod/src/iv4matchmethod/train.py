from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from iv4matchmethod.config import ModelConfig
from iv4matchmethod.data import EpisodeDataset
from iv4matchmethod.losses import locator_loss
from iv4matchmethod.models.network import TemplateConditionedLocator


def choose_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_locator(args) -> Path:
    config = ModelConfig(
        backbone_variant=args.backbone_variant,
        pretrained_backbone=args.pretrained_backbone,
        template_size=args.template_size,
        search_size=args.search_size,
        feature_stride=args.feature_stride,
        fuse_channels=args.fuse_channels,
        head_channels=args.head_channels,
    )
    config.validate()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    dataset = EpisodeDataset(args.manifest, config=config, heatmap_sigma=args.heatmap_sigma)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TemplateConditionedLocator(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_total = 0.0
        for step, batch in enumerate(loader, start=1):
            template = batch["template"].to(device)
            search = batch["search"].to(device)
            targets = {key: value.to(device) for key, value in batch["target"].items()}

            optimizer.zero_grad(set_to_none=True)
            predictions = model(template, search)
            loss_dict = locator_loss(predictions, targets)
            loss = loss_dict["total"]
            loss.backward()
            optimizer.step()

            running_total += float(loss.item())
            if step % args.log_interval == 0 or step == len(loader):
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "loss_total": round(float(loss.item()), 6),
                            "loss_heatmap": round(float(loss_dict["heatmap"].item()), 6),
                            "loss_offset": round(float(loss_dict["offset"].item()), 6),
                            "loss_scale": round(float(loss_dict["scale"].item()), 6),
                            "loss_angle": round(float(loss_dict["angle"].item()), 6),
                            "loss_quality": round(float(loss_dict["quality"].item()), 6),
                        }
                    )
                )

        epoch_summary = {
            "epoch": epoch,
            "mean_loss": running_total / max(len(loader), 1),
        }
        history.append(epoch_summary)
        checkpoint = {
            "model": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "history": history,
        }
        torch.save(checkpoint, output_dir / "last.pt")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output_dir / "last.pt"
