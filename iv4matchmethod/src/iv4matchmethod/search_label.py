from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from iv4matchmethod.annotate import bbox_center
from iv4matchmethod.image_ops import draw_prediction_overlay, load_rgb


def polygon_centroid(points: Iterable[Iterable[float]]) -> tuple[float, float]:
    pts = np.asarray(list(points), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        raise ValueError("polygon must be shaped [N, 2]")
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def compute_pose_from_clicks(
    template_bbox: Iterable[float],
    template_roi_polygon_image: Iterable[Iterable[float]],
    center_point: Iterable[float],
    roi_point: Iterable[float],
) -> dict[str, object]:
    template_center = np.asarray(bbox_center(template_bbox), dtype=np.float32)
    template_roi_center = np.asarray(polygon_centroid(template_roi_polygon_image), dtype=np.float32)
    search_center = np.asarray(list(center_point), dtype=np.float32)
    search_roi_center = np.asarray(list(roi_point), dtype=np.float32)

    template_vector = template_roi_center - template_center
    search_vector = search_roi_center - search_center

    template_radius = float(np.linalg.norm(template_vector))
    search_radius = float(np.linalg.norm(search_vector))
    if template_radius < 1e-6:
        raise ValueError("template ROI center must not coincide with template bbox center")
    if search_radius < 1e-6:
        raise ValueError("search ROI click must not coincide with search center click")

    template_angle = math.atan2(float(template_vector[1]), float(template_vector[0]))
    search_angle = math.atan2(float(search_vector[1]), float(search_vector[0]))
    theta = search_angle - template_angle
    scale = search_radius / template_radius

    return {
        "center": [float(search_center[0]), float(search_center[1])],
        "angle_deg": float(math.degrees(theta)),
        "scale": [float(scale), float(scale)],
        "template_center": [float(template_center[0]), float(template_center[1])],
        "template_roi_center": [float(template_roi_center[0]), float(template_roi_center[1])],
        "search_roi_center": [float(search_roi_center[0]), float(search_roi_center[1])],
    }


def build_manifest_record(
    template_annotation: dict[str, object],
    search_image: str | Path,
    center: Iterable[float],
    angle_deg: float,
    scale: Iterable[float],
    ok_ng: str = "OK",
) -> dict[str, object]:
    return {
        "template_image": str(template_annotation["template_image"]),
        "template_bbox": template_annotation["template_bbox"],
        "search_image": str(search_image),
        "center": [float(center[0]), float(center[1])],
        "angle_deg": float(angle_deg),
        "scale": [float(scale[0]), float(scale[1])],
        "roi_ref_polygon": template_annotation["roi_ref_polygon"],
        "ok_ng": str(ok_ng).upper(),
    }


def append_jsonl_record(path: str | Path, record: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def draw_search_annotation(
    image: Image.Image,
    center: Iterable[float] | None = None,
    roi_point: Iterable[float] | None = None,
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    if center is not None:
        cx, cy = float(center[0]), float(center[1])
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(0, 220, 0))
    if roi_point is not None:
        rx, ry = float(roi_point[0]), float(roi_point[1])
        draw.ellipse((rx - 6, ry - 6, rx + 6, ry + 6), fill=(255, 80, 0))
    if center is not None and roi_point is not None:
        draw.line((float(center[0]), float(center[1]), float(roi_point[0]), float(roi_point[1])), fill=(255, 80, 0), width=3)
    return result


@dataclass(slots=True)
class SearchLabelState:
    center_point: list[float] | None = None
    roi_point: list[float] | None = None
    ok_ng: str = "OK"


class SearchLabelApp:
    def __init__(
        self,
        template_annotation_path: str | Path,
        image_path: str | Path,
        output_path: str | Path,
        fit_size: int = 1200,
        existing_label: str | Path | None = None,
        append_manifest: str | Path | None = None,
        default_label: str = "OK",
    ) -> None:
        import tkinter as tk
        from tkinter import messagebox

        self.tk = tk
        self.messagebox = messagebox
        self.template_annotation_path = Path(template_annotation_path)
        self.template_annotation = json.loads(self.template_annotation_path.read_text(encoding="utf-8"))
        self.image_path = Path(image_path)
        self.output_path = Path(output_path)
        self.preview_path = self.output_path.with_name(f"{self.output_path.stem}_preview.png")
        self.append_manifest = Path(append_manifest) if append_manifest else None
        self.base_image = load_rgb(self.image_path)
        self.fit_size = max(400, int(fit_size))
        self.scale = min(
            self.fit_size / self.base_image.width,
            self.fit_size / self.base_image.height,
            1.0,
        )
        self.display_size = (
            max(1, int(round(self.base_image.width * self.scale))),
            max(1, int(round(self.base_image.height * self.scale))),
        )
        self.display_image = self.base_image.resize(self.display_size, resample=Image.Resampling.BILINEAR)
        self.state = SearchLabelState(ok_ng=default_label.upper())
        if existing_label:
            self._load_existing(existing_label)

        self.active_target = "center"

        self.root = tk.Tk()
        self.root.title(f"iv4matchmethod search label - {self.image_path.name}")
        self.main = tk.Frame(self.root)
        self.main.pack(fill=tk.BOTH, expand=True)
        self.sidebar = tk.Frame(self.main, width=360, padx=10, pady=10)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas = tk.Canvas(
            self.main,
            width=self.display_size[0],
            height=self.display_size[1],
            background="#202020",
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        self.photo = ImageTk.PhotoImage(self.display_image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.status_var = tk.StringVar(value="")
        self.mode_var = tk.StringVar(value="")
        self.pose_var = tk.StringVar(value="")
        self.label_var = tk.StringVar(value="")

        self._build_sidebar()
        self._bind_events()
        self._render()

    def _load_existing(self, label_path: str | Path) -> None:
        payload = json.loads(Path(label_path).read_text(encoding="utf-8"))
        if payload.get("center"):
            self.state.center_point = [float(v) for v in payload["center"]]
        search_roi_center = payload.get("search_roi_center")
        if search_roi_center:
            self.state.roi_point = [float(v) for v in search_roi_center]
        if payload.get("ok_ng"):
            self.state.ok_ng = str(payload["ok_ng"]).upper()

    def _build_sidebar(self) -> None:
        tk = self.tk
        template_preview = draw_prediction_overlay(
            load_rgb(self.template_annotation["template_image"]),
            center=bbox_center(self.template_annotation["template_bbox"]),
            polygon=self.template_annotation.get("roi_image_polygon"),
        )
        preview = template_preview.resize((240, 240), resample=Image.Resampling.BILINEAR)
        self.template_preview_photo = ImageTk.PhotoImage(preview)

        tk.Label(self.sidebar, text="Search Label Tool", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        tk.Label(
            self.sidebar,
            text=(
                "1. 先点目标中心。\n"
                "2. 再点同一颗齿的中心位置。\n"
                "3. 工具会自动算 center / angle / scale。\n"
                "4. 可选追加到 manifest。"
            ),
            justify=tk.LEFT,
            wraplength=320,
        ).pack(anchor=tk.W, pady=(8, 12))
        tk.Label(self.sidebar, image=self.template_preview_photo).pack(anchor=tk.W, pady=(0, 12))

        tk.Button(self.sidebar, text="Center Mode (1)", command=lambda: self._set_target("center")).pack(fill=tk.X)
        tk.Button(self.sidebar, text="Tooth Mode (2)", command=lambda: self._set_target("roi")).pack(fill=tk.X, pady=(6, 0))
        tk.Button(self.sidebar, text="Set OK", command=lambda: self._set_ok_ng("OK")).pack(fill=tk.X, pady=(18, 0))
        tk.Button(self.sidebar, text="Set NG", command=lambda: self._set_ok_ng("NG")).pack(fill=tk.X, pady=(6, 0))
        tk.Button(self.sidebar, text="Clear Points", command=self._clear_points).pack(fill=tk.X, pady=(18, 0))
        tk.Button(self.sidebar, text="Save (Ctrl+S)", command=self._save).pack(fill=tk.X, pady=(18, 0))
        tk.Button(self.sidebar, text="Save And Exit", command=self._save_and_exit).pack(fill=tk.X, pady=(6, 0))

        tk.Label(self.sidebar, textvariable=self.mode_var, justify=tk.LEFT, wraplength=320).pack(anchor=tk.W, pady=(18, 0))
        tk.Label(self.sidebar, textvariable=self.label_var, justify=tk.LEFT, wraplength=320).pack(anchor=tk.W, pady=(6, 0))
        tk.Label(self.sidebar, textvariable=self.pose_var, justify=tk.LEFT, wraplength=320).pack(anchor=tk.W, pady=(6, 0))
        tk.Label(self.sidebar, textvariable=self.status_var, justify=tk.LEFT, wraplength=320, fg="#0a6").pack(anchor=tk.W, pady=(18, 0))

    def _bind_events(self) -> None:
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.root.bind("1", lambda _event: self._set_target("center"))
        self.root.bind("2", lambda _event: self._set_target("roi"))
        self.root.bind("<BackSpace>", lambda _event: self._undo_last())
        self.root.bind("<Control-s>", lambda _event: self._save())
        self.root.bind("<Return>", lambda _event: self._save_and_exit())
        self.root.bind("<Escape>", lambda _event: self.root.destroy())

    def _set_target(self, target: str) -> None:
        self.active_target = target
        self._render()

    def _set_ok_ng(self, label: str) -> None:
        self.state.ok_ng = label
        self._render()

    def _clear_points(self) -> None:
        self.state.center_point = None
        self.state.roi_point = None
        self._render()

    def _undo_last(self) -> None:
        if self.state.roi_point is not None:
            self.state.roi_point = None
        elif self.state.center_point is not None:
            self.state.center_point = None
        self._render()

    def _clamp_canvas(self, x: float, y: float) -> tuple[float, float]:
        x = min(max(x, 0.0), float(self.display_size[0]))
        y = min(max(y, 0.0), float(self.display_size[1]))
        return x, y

    def _canvas_to_image(self, x: float, y: float) -> tuple[float, float]:
        x, y = self._clamp_canvas(x, y)
        return x / self.scale, y / self.scale

    def _image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale, y * self.scale

    def _on_left_click(self, event) -> None:
        x, y = self._canvas_to_image(event.x, event.y)
        if self.active_target == "center":
            self.state.center_point = [x, y]
            if self.state.roi_point is None:
                self.active_target = "roi"
        else:
            self.state.roi_point = [x, y]
        self._render()

    def _on_right_click(self, _event) -> None:
        self._undo_last()

    def _compute_pose(self) -> dict[str, object] | None:
        if self.state.center_point is None or self.state.roi_point is None:
            return None
        return compute_pose_from_clicks(
            self.template_annotation["template_bbox"],
            self.template_annotation["roi_image_polygon"],
            self.state.center_point,
            self.state.roi_point,
        )

    def _render(self) -> None:
        self.canvas.delete("overlay")
        if self.state.center_point is not None:
            cx, cy = self._image_to_canvas(*self.state.center_point)
            self.canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="#00dc00", outline="", tags="overlay")
        if self.state.roi_point is not None:
            rx, ry = self._image_to_canvas(*self.state.roi_point)
            self.canvas.create_oval(rx - 5, ry - 5, rx + 5, ry + 5, fill="#ff6b00", outline="", tags="overlay")
        if self.state.center_point is not None and self.state.roi_point is not None:
            cx, cy = self._image_to_canvas(*self.state.center_point)
            rx, ry = self._image_to_canvas(*self.state.roi_point)
            self.canvas.create_line(cx, cy, rx, ry, fill="#ff6b00", width=2, tags="overlay")

        self.mode_var.set(f"Click target: {'Center' if self.active_target == 'center' else 'Tooth'}")
        self.label_var.set(f"ok_ng: {self.state.ok_ng}")
        pose = self._compute_pose()
        if pose is None:
            self.pose_var.set("pose: waiting for center and tooth clicks")
        else:
            center = pose["center"]
            scale = pose["scale"]
            self.pose_var.set(
                f"center: [{center[0]:.1f}, {center[1]:.1f}]\n"
                f"angle_deg: {pose['angle_deg']:.2f}\n"
                f"scale: [{scale[0]:.4f}, {scale[1]:.4f}]"
            )
        self.root.update_idletasks()

    def _save(self) -> bool:
        pose = self._compute_pose()
        if pose is None:
            self.messagebox.showerror("Save failed", "center and tooth point are required")
            return False

        payload = {
            "template_annotation": str(self.template_annotation_path),
            "search_image": str(self.image_path),
            "center": pose["center"],
            "angle_deg": pose["angle_deg"],
            "scale": pose["scale"],
            "search_roi_center": pose["search_roi_center"],
            "ok_ng": self.state.ok_ng,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        preview = draw_search_annotation(
            self.base_image,
            center=self.state.center_point,
            roi_point=self.state.roi_point,
        )
        preview.save(self.preview_path)

        manifest_path = None
        if self.append_manifest is not None:
            record = build_manifest_record(
                self.template_annotation,
                self.image_path,
                pose["center"],
                pose["angle_deg"],
                pose["scale"],
                ok_ng=self.state.ok_ng,
            )
            append_jsonl_record(self.append_manifest, record)
            manifest_path = self.append_manifest

        status = [f"Saved: {self.output_path.name}", f"Preview: {self.preview_path.name}"]
        if manifest_path is not None:
            status.append(f"Appended manifest: {manifest_path.name}")
        self.status_var.set("\n".join(status))
        return True

    def _save_and_exit(self) -> None:
        if self._save():
            self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def run_search_label_tool(args) -> int:
    image_path = Path(args.image)
    output = Path(args.output) if args.output else image_path.with_name(f"{image_path.stem}_label.json")
    app = SearchLabelApp(
        template_annotation_path=args.template_annotation,
        image_path=image_path,
        output_path=output,
        fit_size=args.fit_size,
        existing_label=args.load,
        append_manifest=args.append_manifest,
        default_label=args.ok_ng,
    )
    return app.run()
