from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageTk

from iv4matchmethod.image_ops import draw_prediction_overlay, load_rgb, parse_bbox


def bbox_center(bbox: Iterable[float]) -> tuple[float, float]:
    x, y, w, h = parse_bbox(bbox)
    return x + w / 2.0, y + h / 2.0


def polygon_to_relative(
    polygon_image: Iterable[Iterable[float]],
    bbox: Iterable[float],
) -> list[list[float]]:
    cx, cy = bbox_center(bbox)
    points: list[list[float]] = []
    for x, y in polygon_image:
        points.append([float(x) - cx, float(y) - cy])
    return points


def polygon_to_image(
    polygon_relative: Iterable[Iterable[float]],
    bbox: Iterable[float],
) -> list[list[float]]:
    cx, cy = bbox_center(bbox)
    points: list[list[float]] = []
    for x, y in polygon_relative:
        points.append([float(x) + cx, float(y) + cy])
    return points


def annotation_to_json(
    image_path: str | Path,
    image_size: tuple[int, int],
    bbox: Iterable[float],
    polygon_image: Iterable[Iterable[float]],
) -> dict[str, object]:
    bbox_list = [round(float(v), 3) for v in bbox]
    polygon_image_list = [[round(float(x), 3), round(float(y), 3)] for x, y in polygon_image]
    polygon_relative = polygon_to_relative(polygon_image_list, bbox_list)
    return {
        "template_image": str(image_path),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "template_bbox": bbox_list,
        "roi_image_polygon": polygon_image_list,
        "roi_ref_polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in polygon_relative],
        "roi_origin": "template_bbox_center",
    }


def draw_template_annotation(
    image: Image.Image,
    bbox: Iterable[float] | None = None,
    polygon_image: Iterable[Iterable[float]] | None = None,
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    if bbox is not None:
        x, y, w, h = parse_bbox(bbox)
        draw.rectangle((x, y, x + w, y + h), outline=(0, 200, 255), width=3)
    if polygon_image is not None:
        result = draw_prediction_overlay(result, polygon=polygon_image)
    return result


@dataclass(slots=True)
class TemplateAnnotationState:
    bbox: list[float] | None = None
    polygon_image: list[list[float]] | None = None

    def __post_init__(self) -> None:
        if self.polygon_image is None:
            self.polygon_image = []


class TemplateAnnotationApp:
    def __init__(
        self,
        image_path: str | Path,
        output_path: str | Path,
        fit_size: int = 1200,
        existing_annotation: str | Path | None = None,
    ) -> None:
        import tkinter as tk
        from tkinter import messagebox

        self.tk = tk
        self.messagebox = messagebox
        self.image_path = Path(image_path)
        self.output_path = Path(output_path)
        self.preview_path = self.output_path.with_name(f"{self.output_path.stem}_preview.png")
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
        self.state = TemplateAnnotationState()
        if existing_annotation:
            self._load_existing(existing_annotation)

        self.mode = "bbox"
        self.drag_origin: tuple[float, float] | None = None
        self.temp_bbox: list[float] | None = None

        self.root = tk.Tk()
        self.root.title(f"iv4matchmethod annotate - {self.image_path.name}")

        self.main = tk.Frame(self.root)
        self.main.pack(fill=tk.BOTH, expand=True)

        self.sidebar = tk.Frame(self.main, width=320, padx=10, pady=10)
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
        self.bbox_var = tk.StringVar(value="")
        self.roi_var = tk.StringVar(value="")

        self._build_sidebar()
        self._bind_events()
        self._render()

    def _load_existing(self, annotation_path: str | Path) -> None:
        payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
        bbox = payload.get("template_bbox")
        if bbox:
            self.state.bbox = [float(v) for v in bbox]
        polygon_relative = payload.get("roi_ref_polygon")
        polygon_image = payload.get("roi_image_polygon")
        if polygon_image:
            self.state.polygon_image = [[float(x), float(y)] for x, y in polygon_image]
        elif polygon_relative and self.state.bbox:
            self.state.polygon_image = polygon_to_image(polygon_relative, self.state.bbox)

    def _build_sidebar(self) -> None:
        tk = self.tk
        tk.Label(
            self.sidebar,
            text="Template Annotation",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor=tk.W)
        instructions = (
            "1. 先切到 BBox 模式，拖框圈住完整目标。\n"
            "2. 再切到 ROI 模式，左键逐点画 ROI。\n"
            "3. 右键或 Backspace 撤销最后一个 ROI 点。\n"
            "4. 保存后会输出 JSON 和预览图。"
        )
        tk.Label(self.sidebar, text=instructions, justify=tk.LEFT, wraplength=280).pack(anchor=tk.W, pady=(8, 12))

        tk.Button(self.sidebar, text="BBox Mode (1)", command=lambda: self._set_mode("bbox")).pack(fill=tk.X)
        tk.Button(self.sidebar, text="ROI Mode (2)", command=lambda: self._set_mode("roi")).pack(fill=tk.X, pady=(6, 0))
        tk.Button(self.sidebar, text="Clear BBox", command=self._clear_bbox).pack(fill=tk.X, pady=(18, 0))
        tk.Button(self.sidebar, text="Clear ROI", command=self._clear_roi).pack(fill=tk.X, pady=(6, 0))
        tk.Button(self.sidebar, text="Save (Ctrl+S)", command=self._save).pack(fill=tk.X, pady=(18, 0))
        tk.Button(self.sidebar, text="Save And Exit", command=self._save_and_exit).pack(fill=tk.X, pady=(6, 0))

        tk.Label(self.sidebar, textvariable=self.mode_var, justify=tk.LEFT, wraplength=280).pack(anchor=tk.W, pady=(18, 0))
        tk.Label(self.sidebar, textvariable=self.bbox_var, justify=tk.LEFT, wraplength=280).pack(anchor=tk.W, pady=(6, 0))
        tk.Label(self.sidebar, textvariable=self.roi_var, justify=tk.LEFT, wraplength=280).pack(anchor=tk.W, pady=(6, 0))
        tk.Label(self.sidebar, textvariable=self.status_var, justify=tk.LEFT, wraplength=280, fg="#0a6").pack(anchor=tk.W, pady=(18, 0))

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<Button-3>", self._on_right_click)

        self.root.bind("1", lambda _event: self._set_mode("bbox"))
        self.root.bind("2", lambda _event: self._set_mode("roi"))
        self.root.bind("<BackSpace>", lambda _event: self._undo_roi_point())
        self.root.bind("<Control-s>", lambda _event: self._save())
        self.root.bind("<Return>", lambda _event: self._save_and_exit())
        self.root.bind("<Escape>", lambda _event: self.root.destroy())

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
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

    def _normalize_bbox(self, x0: float, y0: float, x1: float, y1: float) -> list[float]:
        left = max(0.0, min(x0, x1))
        top = max(0.0, min(y0, y1))
        right = min(float(self.base_image.width), max(x0, x1))
        bottom = min(float(self.base_image.height), max(y0, y1))
        width = max(1.0, right - left)
        height = max(1.0, bottom - top)
        return [left, top, width, height]

    def _on_left_press(self, event) -> None:
        x, y = self._canvas_to_image(event.x, event.y)
        if self.mode == "bbox":
            self.drag_origin = (x, y)
            self.temp_bbox = [x, y, 1.0, 1.0]
            self._render()
            return
        self.state.polygon_image.append([x, y])
        self._render()

    def _on_left_drag(self, event) -> None:
        if self.mode != "bbox" or self.drag_origin is None:
            return
        x0, y0 = self.drag_origin
        x1, y1 = self._canvas_to_image(event.x, event.y)
        self.temp_bbox = self._normalize_bbox(x0, y0, x1, y1)
        self._render()

    def _on_left_release(self, event) -> None:
        if self.mode != "bbox" or self.drag_origin is None:
            return
        x0, y0 = self.drag_origin
        x1, y1 = self._canvas_to_image(event.x, event.y)
        self.state.bbox = self._normalize_bbox(x0, y0, x1, y1)
        self.drag_origin = None
        self.temp_bbox = None
        self._render()

    def _on_right_click(self, _event) -> None:
        self._undo_roi_point()

    def _undo_roi_point(self) -> None:
        if self.state.polygon_image:
            self.state.polygon_image.pop()
            self._render()

    def _clear_bbox(self) -> None:
        self.state.bbox = None
        self.temp_bbox = None
        self._render()

    def _clear_roi(self) -> None:
        self.state.polygon_image = []
        self._render()

    def _render(self) -> None:
        tk = self.tk
        self.canvas.delete("overlay")
        bbox = self.temp_bbox if self.temp_bbox is not None else self.state.bbox
        if bbox is not None:
            x, y, w, h = bbox
            x0, y0 = self._image_to_canvas(x, y)
            x1, y1 = self._image_to_canvas(x + w, y + h)
            self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline="#00c8ff",
                width=2,
                tags="overlay",
            )
            cx, cy = bbox_center(bbox)
            cx_c, cy_c = self._image_to_canvas(cx, cy)
            self.canvas.create_line(cx_c - 8, cy_c, cx_c + 8, cy_c, fill="#00c8ff", width=2, tags="overlay")
            self.canvas.create_line(cx_c, cy_c - 8, cx_c, cy_c + 8, fill="#00c8ff", width=2, tags="overlay")
        if self.state.polygon_image:
            points_canvas: list[float] = []
            for px, py in self.state.polygon_image:
                cx, cy = self._image_to_canvas(px, py)
                points_canvas.extend([cx, cy])
                self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="#ff6b00", outline="", tags="overlay")
            if len(points_canvas) >= 4:
                self.canvas.create_line(*points_canvas, fill="#ff6b00", width=2, tags="overlay")
            if len(points_canvas) >= 6:
                self.canvas.create_line(
                    points_canvas[-2],
                    points_canvas[-1],
                    points_canvas[0],
                    points_canvas[1],
                    fill="#ff6b00",
                    width=1,
                    dash=(4, 2),
                    tags="overlay",
                )

        self.mode_var.set(f"Mode: {'BBox' if self.mode == 'bbox' else 'ROI'}")
        if self.state.bbox is None:
            self.bbox_var.set("template_bbox: not set")
        else:
            x, y, w, h = self.state.bbox
            self.bbox_var.set(f"template_bbox: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")
        if not self.state.polygon_image:
            self.roi_var.set("roi_ref_polygon: no points")
        elif self.state.bbox is None:
            self.roi_var.set(f"roi points: {len(self.state.polygon_image)} (set bbox first before saving)")
        else:
            relative = polygon_to_relative(self.state.polygon_image, self.state.bbox)
            self.roi_var.set(
                "roi_ref_polygon: "
                + json.dumps([[round(x, 1), round(y, 1)] for x, y in relative], ensure_ascii=False)
            )
        self.root.update_idletasks()

    def _validate(self) -> None:
        if self.state.bbox is None:
            raise ValueError("template_bbox is required")
        if len(self.state.polygon_image) < 3:
            raise ValueError("roi_ref_polygon needs at least 3 points")

    def _save(self) -> bool:
        try:
            self._validate()
            payload = annotation_to_json(
                self.image_path,
                self.base_image.size,
                self.state.bbox,
                self.state.polygon_image,
            )
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

            preview = draw_template_annotation(
                self.base_image,
                bbox=self.state.bbox,
                polygon_image=self.state.polygon_image,
            )
            preview.save(self.preview_path)
            self.status_var.set(f"Saved: {self.output_path.name}\nPreview: {self.preview_path.name}")
            return True
        except Exception as exc:  # pragma: no cover - GUI path
            self.messagebox.showerror("Save failed", str(exc))
            return False

    def _save_and_exit(self) -> None:
        if self._save():
            self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def run_annotation_tool(args) -> int:
    image_path = Path(args.image)
    output = Path(args.output) if args.output else image_path.with_name(f"{image_path.stem}_annotation.json")
    app = TemplateAnnotationApp(
        image_path=image_path,
        output_path=output,
        fit_size=args.fit_size,
        existing_annotation=args.load,
    )
    return app.run()
