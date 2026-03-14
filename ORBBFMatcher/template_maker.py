from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys

import cv2
import numpy as np


@dataclass
class TemplateSelection:
    selection_type: str
    bbox: tuple[int, int, int, int]
    polygon_points: list[tuple[int, int]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a template image and mask from a rectangle or polygon selection."
    )
    parser.add_argument("--image", required=True, help="Source image path.")
    parser.add_argument(
        "--output-template",
        required=True,
        help="Output template path. PNG can store transparent alpha.",
    )
    parser.add_argument(
        "--output-mask",
        help="Optional output mask path. Default: <template_stem>_mask.png",
    )
    parser.add_argument(
        "--output-preview",
        help="Optional output preview path. Default: <template_stem>_preview.png",
    )
    parser.add_argument(
        "--output-meta",
        help="Optional output metadata path. Default: <template_stem>.json",
    )
    parser.add_argument(
        "--mode",
        choices=("polygon", "rect"),
        default="polygon",
        help="Initial interactive selection mode. Default: polygon",
    )
    parser.add_argument(
        "--rect",
        help="Create without GUI using x,y,w,h in source-image coordinates.",
    )
    parser.add_argument(
        "--polygon",
        help="Create without GUI using x1,y1,x2,y2,... in source-image coordinates.",
    )
    parser.add_argument(
        "--display-max-width",
        type=int,
        default=1600,
        help="Max preview width for the interactive window. Default: 1600",
    )
    parser.add_argument(
        "--display-max-height",
        type=int,
        default=1000,
        help="Max preview height for the interactive window. Default: 1000",
    )
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def clamp_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    return max(0, min(x, width - 1)), max(0, min(y, height - 1))


def parse_rect(rect_text: str, image_shape: tuple[int, int, int]) -> TemplateSelection:
    parts = [part.strip() for part in rect_text.split(",")]
    if len(parts) != 4:
        raise ValueError("--rect expects x,y,w,h")

    x, y, w, h = (int(part) for part in parts)
    if w <= 0 or h <= 0:
        raise ValueError("--rect width and height must be positive")

    image_height, image_width = image_shape[:2]
    x, y = clamp_point(x, y, image_width, image_height)
    x2, y2 = clamp_point(x + w - 1, y + h - 1, image_width, image_height)
    bbox = normalize_bbox(x, y, x2, y2)
    return TemplateSelection(selection_type="rect", bbox=bbox)


def parse_polygon(
    polygon_text: str, image_shape: tuple[int, int, int]
) -> TemplateSelection:
    parts = [part.strip() for part in polygon_text.split(",")]
    if len(parts) < 6 or len(parts) % 2 != 0:
        raise ValueError("--polygon expects x1,y1,x2,y2,... with at least 3 points")

    image_height, image_width = image_shape[:2]
    points: list[tuple[int, int]] = []
    for index in range(0, len(parts), 2):
        x = int(parts[index])
        y = int(parts[index + 1])
        points.append(clamp_point(x, y, image_width, image_height))

    bbox = bounding_box_from_points(points)
    return TemplateSelection(
        selection_type="polygon",
        bbox=bbox,
        polygon_points=points,
    )


def normalize_bbox(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    return left, top, right - left + 1, bottom - top + 1


def bounding_box_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def create_mask(
    image_shape: tuple[int, int, int], selection: TemplateSelection
) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if selection.selection_type == "rect":
        x, y, w, h = selection.bbox
        cv2.rectangle(mask, (x, y), (x + w - 1, y + h - 1), 255, thickness=-1)
        return mask

    polygon = np.array(selection.polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def crop_to_mask(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    non_zero = cv2.findNonZero(mask)
    if non_zero is None:
        raise ValueError("The selection mask is empty")

    x, y, w, h = cv2.boundingRect(non_zero)
    return image[y : y + h, x : x + w].copy(), mask[y : y + h, x : x + w].copy(), (x, y, w, h)


def create_preview_image(template_image: np.ndarray, template_mask: np.ndarray) -> np.ndarray:
    preview = cv2.cvtColor(template_image, cv2.COLOR_BGR2BGRA)
    preview[:, :, 3] = template_mask
    return preview


def derive_output_path(template_path: Path, suffix: str) -> Path:
    return template_path.with_name(f"{template_path.stem}{suffix}")


def write_outputs(
    source_image: np.ndarray,
    source_path: Path,
    selection: TemplateSelection,
    output_template: Path,
    output_mask: Path,
    output_preview: Path,
    output_meta: Path,
) -> None:
    full_mask = create_mask(source_image.shape, selection)
    template_image, template_mask, crop_bbox = crop_to_mask(source_image, full_mask)
    preview_image = create_preview_image(template_image, template_mask)

    output_template.parent.mkdir(parents=True, exist_ok=True)
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    output_preview.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)

    if output_template.suffix.lower() == ".png":
        cv2.imwrite(str(output_template), preview_image)
    else:
        cv2.imwrite(str(output_template), template_image)

    cv2.imwrite(str(output_mask), template_mask)
    cv2.imwrite(str(output_preview), preview_image)

    crop_x, crop_y, crop_w, crop_h = crop_bbox
    polygon_points_local = [
        [point[0] - crop_x, point[1] - crop_y] for point in selection.polygon_points
    ]
    metadata = {
        "source_image": str(source_path.resolve()),
        "selection_type": selection.selection_type,
        "crop_bbox": {
            "x": crop_x,
            "y": crop_y,
            "width": crop_w,
            "height": crop_h,
        },
        "output_template": str(output_template.resolve()),
        "output_mask": str(output_mask.resolve()),
        "output_preview": str(output_preview.resolve()),
        "polygon_points_source": [list(point) for point in selection.polygon_points],
        "polygon_points_local": polygon_points_local,
    }
    output_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"template saved to: {output_template}")
    print(f"mask saved to: {output_mask}")
    print(f"preview saved to: {output_preview}")
    print(f"metadata saved to: {output_meta}")


class InteractiveTemplateMaker:
    def __init__(
        self,
        image: np.ndarray,
        initial_mode: str,
        display_max_width: int,
        display_max_height: int,
    ) -> None:
        self.image = image
        image_height, image_width = image.shape[:2]
        self.scale = min(
            display_max_width / image_width,
            display_max_height / image_height,
            1.0,
        )
        display_width = max(1, int(round(image_width * self.scale)))
        display_height = max(1, int(round(image_height * self.scale)))
        interpolation = cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LINEAR
        self.display_image = cv2.resize(
            image,
            (display_width, display_height),
            interpolation=interpolation,
        )
        self.window_name = "Template Maker"
        self.mode = initial_mode
        self.show_help = True
        self.rect_start: tuple[int, int] | None = None
        self.rect_end: tuple[int, int] | None = None
        self.dragging_rect = False
        self.polygon_points: list[tuple[int, int]] = []
        self.polygon_closed = False
        self.cursor_point: tuple[int, int] | None = None

    def display_to_image(self, x: int, y: int) -> tuple[int, int]:
        if self.scale == 0:
            return 0, 0
        image_height, image_width = self.image.shape[:2]
        image_x = int(round(x / self.scale))
        image_y = int(round(y / self.scale))
        return clamp_point(image_x, image_y, image_width, image_height)

    def image_to_display(self, point: tuple[int, int]) -> tuple[int, int]:
        return int(round(point[0] * self.scale)), int(round(point[1] * self.scale))

    def clear_selection(self) -> None:
        self.rect_start = None
        self.rect_end = None
        self.dragging_rect = False
        self.polygon_points = []
        self.polygon_closed = False

    def build_selection(self) -> TemplateSelection | None:
        if self.mode == "rect" and self.rect_start and self.rect_end:
            return TemplateSelection(
                selection_type="rect",
                bbox=normalize_bbox(
                    self.rect_start[0],
                    self.rect_start[1],
                    self.rect_end[0],
                    self.rect_end[1],
                ),
            )

        if self.mode == "polygon" and self.polygon_closed and len(self.polygon_points) >= 3:
            return TemplateSelection(
                selection_type="polygon",
                bbox=bounding_box_from_points(self.polygon_points),
                polygon_points=list(self.polygon_points),
            )
        return None

    def close_polygon(self) -> None:
        if len(self.polygon_points) >= 3:
            self.polygon_closed = True

    def mouse_callback(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        point = self.display_to_image(x, y)
        self.cursor_point = point

        if self.mode == "rect":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.rect_start = point
                self.rect_end = point
                self.dragging_rect = True
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging_rect:
                self.rect_end = point
            elif event == cv2.EVENT_LBUTTONUP and self.dragging_rect:
                self.rect_end = point
                self.dragging_rect = False
            return

        if event == cv2.EVENT_LBUTTONDOWN and not self.polygon_closed:
            self.polygon_points.append(point)
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            self.close_polygon()

    def render(self) -> np.ndarray:
        canvas = self.display_image.copy()

        if self.rect_start and self.rect_end:
            start = self.image_to_display(self.rect_start)
            end = self.image_to_display(self.rect_end)
            cv2.rectangle(canvas, start, end, (0, 255, 0), 2, cv2.LINE_AA)

        if self.polygon_points:
            polygon_display = np.array(
                [self.image_to_display(point) for point in self.polygon_points],
                dtype=np.int32,
            )
            cv2.polylines(
                canvas,
                [polygon_display],
                self.polygon_closed,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            for point in polygon_display:
                cv2.circle(canvas, tuple(point), 4, (0, 0, 255), thickness=-1)

            if not self.polygon_closed and self.cursor_point is not None:
                last_point = self.image_to_display(self.polygon_points[-1])
                cursor_display = self.image_to_display(self.cursor_point)
                cv2.line(
                    canvas,
                    last_point,
                    cursor_display,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        if self.show_help:
            lines = [
                f"mode: {self.mode}",
                "r: rect mode",
                "p: polygon mode",
                "left drag: draw rect",
                "left click: add polygon point",
                "right click / Enter: close polygon",
                "u: undo polygon point",
                "c: clear selection",
                "s: save",
                "h: toggle help",
                "q or Esc: quit",
            ]
            for index, line in enumerate(lines):
                y = 24 + index * 22
                cv2.putText(
                    canvas,
                    line,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    line,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        return canvas

    def run(self) -> TemplateSelection | None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            cv2.imshow(self.window_name, self.render())
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue
            if key in (27, ord("q")):
                cv2.destroyAllWindows()
                return None
            if key == ord("h"):
                self.show_help = not self.show_help
            elif key == ord("r"):
                self.mode = "rect"
                self.clear_selection()
            elif key == ord("p"):
                self.mode = "polygon"
                self.clear_selection()
            elif key == ord("c"):
                self.clear_selection()
            elif key == ord("u") and self.mode == "polygon" and not self.polygon_closed:
                if self.polygon_points:
                    self.polygon_points.pop()
            elif key in (13, 10, 32) and self.mode == "polygon":
                self.close_polygon()
            elif key == ord("s"):
                selection = self.build_selection()
                if selection is not None:
                    cv2.destroyAllWindows()
                    return selection


def resolve_selection(
    args: argparse.Namespace, image: np.ndarray
) -> TemplateSelection | None:
    if args.rect and args.polygon:
        raise ValueError("Use either --rect or --polygon, not both")

    if args.rect:
        return parse_rect(args.rect, image.shape)
    if args.polygon:
        return parse_polygon(args.polygon, image.shape)

    selector = InteractiveTemplateMaker(
        image=image,
        initial_mode=args.mode,
        display_max_width=args.display_max_width,
        display_max_height=args.display_max_height,
    )
    return selector.run()


def main() -> int:
    args = parse_args()
    try:
        image_path = Path(args.image)
        image = load_image(args.image)

        output_template = Path(args.output_template)
        output_mask = Path(args.output_mask) if args.output_mask else derive_output_path(
            output_template, "_mask.png"
        )
        output_preview = (
            Path(args.output_preview)
            if args.output_preview
            else derive_output_path(output_template, "_preview.png")
        )
        output_meta = Path(args.output_meta) if args.output_meta else derive_output_path(
            output_template, ".json"
        )

        selection = resolve_selection(args, image)
        if selection is None:
            print("Template creation cancelled.", file=sys.stderr)
            return 1

        write_outputs(
            source_image=image,
            source_path=image_path,
            selection=selection,
            output_template=output_template,
            output_mask=output_mask,
            output_preview=output_preview,
            output_meta=output_meta,
        )
    except (FileNotFoundError, ValueError, cv2.error) as exc:
        print(exc, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
