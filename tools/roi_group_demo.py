"""Isolated template editor for testing named local ROI localization groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def default_demo_product(workdir: Path) -> Path:
    """Prefer the organized workspace; reuse an unmigrated legacy copy in place."""
    destination = workdir / "experiments/localization_groups/product_1125"
    legacy = workdir / "reports/localization_groups_demo/product_1125"
    return legacy if not destination.exists() and legacy.exists() else destination


def prepare_demo_product(source: Path | None, destination: Path) -> Path:
    """Copy once. Reopening preserves the user's experimental configuration."""
    recipe_path = destination / "shape" / "cam1" / "recipe.json"
    if recipe_path.is_file():
        return destination
    if destination.exists():
        raise RuntimeError(f"测试目录已存在但没有配方，请另选空目录：{destination}")
    if source is None:
        raise RuntimeError("首次准备测试副本需要指定 --source-product 或选择源产品目录。")
    original_recipe = source / "shape" / "cam1" / "recipe.json"
    if not original_recipe.is_file():
        raise RuntimeError(f"找不到源产品配方：{original_recipe}")
    from shape.core.recipe import load_recipe
    recipe = load_recipe(str(original_recipe))
    reference = Path(recipe.reference_image)
    if not reference.is_file():
        raise RuntimeError(f"找不到参考图：{reference}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="roi-group-prepare-", dir=destination.parent) as temp:
        stage = Path(temp) / "product"
        role_dir = stage / "shape" / "cam1"
        shutil.copytree(source / "shape" / "cam1", role_dir)
        train = stage / "samples" / "train"
        if (source / "samples" / "train").is_dir():
            shutil.copytree(source / "samples" / "train", train)
        else:
            train.mkdir(parents=True)
        copied_reference = train / reference.name
        if not copied_reference.exists():
            shutil.copy2(reference, copied_reference)
        payload = recipe.to_dict()
        payload["model_path"] = "model.json"
        payload["reference_image"] = "../../samples/train/" + reference.name
        (role_dir / "recipe.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        # Model source geometry stays unchanged; only point its source image at
        # the independent reference, so the editor cannot switch to production.
        model_file = role_dir / "model.json"
        model = json.loads(model_file.read_text(encoding="utf-8"))
        for class_data in model.get("classes", {}).values():
            if isinstance(class_data, dict) and isinstance(class_data.get("source"), dict):
                class_data["source"]["image_path"] = payload["reference_image"]
        model_file.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
        (stage / "DEMO_COPY.json").write_text(json.dumps({"source": str(source.resolve()), "purpose": "独立定位组测试副本"}, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.rename(destination)
    return destination


def _run_demo(args) -> int:
    product = (args.product_dir or default_demo_product(args.workdir)).resolve()
    source = args.source_product.resolve() if args.source_product is not None else None
    if not product.exists() and source is None:
        from PySide6 import QtWidgets
        from ui.i18n import tr
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            None, tr("permission.product.select"), str(args.workdir / ".qr_session"),
        )
        if not selected:
            return 0
        source = Path(selected).resolve()
    product = prepare_demo_product(source, product)
    if args.prepare_only:
        print(product)
        return 0
    from PySide6 import QtCore, QtWidgets
    from shape.core.recipe import load_recipe
    from ui.i18n import tr
    from ui.shape_template.template_page_pyside6 import ShapeTemplateDialog
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    recipe = load_recipe(str(product / "shape/cam1/recipe.json"))
    dialog = ShapeTemplateDialog(product_name=tr("template.groups.demo_product", product=product.name), product_dir=str(product), initial_image_path=recipe.reference_image)
    dialog.setWindowTitle(tr("template.groups.demo_title"))
    for path in sorted((product / "samples/train").glob("*")):
        if path.suffix.lower() not in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        if "undistorted" in path.stem.lower() and "undistorted" not in Path(recipe.reference_image).stem.lower():
            continue
        item = QtWidgets.QListWidgetItem(path.name)
        item.setData(QtCore.Qt.UserRole, str(path))
        item.setToolTip(str(path))
        dialog.list_find_images.addItem(item)
    if dialog.list_find_images.count():
        dialog.list_find_images.setCurrentRow(0)
    dialog.tabs.setCurrentWidget(dialog.tab_groups)
    dialog.show()
    return app.exec()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=ROOT, help="Workspace root for demo data and logs")
    parser.add_argument("--source-product", type=Path, help="Source product directory for the first copy only")
    parser.add_argument("--product-dir", type=Path, help="Open or prepare this independent product copy")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    args.workdir = args.workdir.resolve()
    destination = (args.product_dir or default_demo_product(args.workdir)).resolve()
    if args.prepare_only and not destination.exists() and args.source_product is None:
        parser.error("--source-product is required for the first --prepare-only run")
    try:
        return _run_demo(args)
    except Exception:
        log = args.workdir / "artifacts/reports/localization_groups_demo/startup_error.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc(), encoding="utf-8")
        if not args.prepare_only:
            try:
                from PySide6 import QtWidgets
                from ui.i18n import tr
                app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
                QtWidgets.QMessageBox.critical(None, tr("template.groups.demo_failed"), tr("template.groups.demo_error_log", path=log, error=traceback.format_exc()))
            except Exception:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
