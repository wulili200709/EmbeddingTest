"""Auto ROI workflow helpers for ToolPage."""

from __future__ import annotations

from . import page as page_module

List = page_module.List
Optional = page_module.Optional
os = page_module.os
QtCore = page_module.QtCore
QtWidgets = page_module.QtWidgets
qr_core = page_module.qr_core
line2dup_locator = page_module.line2dup_locator
ncc_locator = page_module.ncc_locator
_filter_paths_for_camera = page_module._filter_paths_for_camera


def _set_reference(self, path: str) -> None:
    camera_role = self.current_camera_role()
    self._clear_training_roi_review_state(camera_role)
    self.ref_image = path
    if self.lbl_ref is not None:
        self.lbl_ref.setText(f"参考图：{os.path.basename(path)}")
        self.lbl_ref.setToolTip(path)
    try:
        recipe = self.line2dup_recipe_for_role(camera_role) or line2dup_locator.load_recipe_for_product(
            self.session.product_dir,
            camera_role,
        )
        recipe.reference_image = path
        recipe.model_path = self.line2dup_model_path_for_role(camera_role)
        line2dup_locator.save_recipe_for_product(self.session.product_dir, recipe, camera_role)
        self._line2dup_recipes_by_role[camera_role] = recipe
        self.line2dup_recipe = recipe
    except Exception:
        pass
    self._save_session()

def _set_ref_from_current(self) -> None:
    p = self.canvas.image_path()
    if not p:
        QtWidgets.QMessageBox.warning(self, "提示", "请先在右侧打开一张图片")
        return
    self._set_reference(p)

def _pick_ref_image(self) -> None:
    p, _ = QtWidgets.QFileDialog.getOpenFileName(
        self, "选择参考图", "",
        "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
    )
    if not p:
        return
    self._set_reference(p)

def _open_line2dup_template_page(self) -> None:
    from ui.debug import Line2DupTemplateDialog

    if self._template_editor_dialog is not None and self._template_editor_dialog.isVisible():
        self._template_editor_dialog.raise_()
        self._template_editor_dialog.activateWindow()
        return
    _ensure_line2dup_reference_regions_sync_timer(self)
    initial = self.ref_image or self.canvas.image_path() or ""
    dlg = Line2DupTemplateDialog(
        product_name=self.session.current_product,
        product_dir=self.session.product_dir,
        camera_role=self.current_camera_role(),
        initial_image_path=initial,
        parent=self.window(),
    )
    dlg.setModal(False)
    dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
    dlg.modelSaved.connect(self._on_line2dup_model_saved)
    dlg.referenceRegionsChanged.connect(self._on_line2dup_reference_regions_changed)
    dlg.destroyed.connect(self._on_template_editor_dialog_destroyed)
    self._template_editor_dialog = dlg
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()

def _ensure_line2dup_reference_regions_sync_timer(self):
    timer = getattr(self, "_line2dup_reference_regions_sync_timer", None)
    if timer is not None:
        return timer
    timer = QtCore.QTimer(self)
    timer.setSingleShot(True)
    timer.timeout.connect(self._flush_line2dup_reference_regions_sync)
    self._line2dup_reference_regions_sync_timer = timer
    self._line2dup_reference_regions_sync_pending = False
    return timer

def _schedule_line2dup_reference_regions_sync(self) -> None:
    timer = _ensure_line2dup_reference_regions_sync_timer(self)
    self._line2dup_reference_regions_sync_pending = True
    timer.start(120)

def _flush_line2dup_reference_regions_sync(self) -> None:
    timer = getattr(self, "_line2dup_reference_regions_sync_timer", None)
    if timer is not None and timer.isActive():
        timer.stop()
    if not bool(getattr(self, "_line2dup_reference_regions_sync_pending", False)):
        return
    self._line2dup_reference_regions_sync_pending = False
    self._clear_training_roi_review_state(self.current_camera_role())
    self._sync_line2dup_recipe_and_items()
    current_path = self.canvas.image_path()
    if current_path and os.path.exists(current_path):
        self._load_canvas_image(current_path)
    self.lbl_status.setText("状态：参考ROI已同步到运行界面")

def _on_template_editor_dialog_destroyed(self, *_args) -> None:
    self._flush_line2dup_reference_regions_sync()
    self._template_editor_dialog = None

def _on_line2dup_model_saved(self, model_path: str, recipe_path: str) -> None:
    camera_role = self.current_camera_role()
    self._clear_training_roi_review_state(camera_role)
    try:
        self.line2dup_recipe = self.line2dup_recipe_for_role(camera_role, force_reload=True)
    except Exception:
        self.line2dup_recipe = None
    self._reload_inspection_items()
    self._apply_current_role_recipe_state()
    self.lbl_status.setText(f"状态：模板模型已保存 {os.path.basename(model_path)}")

def _on_line2dup_reference_regions_changed(self) -> None:
    self._schedule_line2dup_reference_regions_sync()
    self.lbl_status.setText("状态：参考ROI已更新，正在同步运行界面")

def _sync_line2dup_recipe_and_items(self) -> None:
    try:
        self.line2dup_recipe = self.line2dup_recipe_for_role(self.current_camera_role(), force_reload=True)
    except Exception:
        self.line2dup_recipe = None
    self._reload_inspection_items()

def _update_loc_ui(self) -> None:
    return None

def _on_loc_method_changed(self, method: str) -> None:
    self.loc_method = method
    self._clear_training_roi_review_state()
    self._update_loc_ui()
    self._reload_inspection_items()
    current_path = self.canvas.image_path()
    if current_path and os.path.exists(current_path):
        self._load_canvas_image(current_path)
    self._save_session()

def _resolve_autogen_targets(
    self,
    paths: List[str],
    *,
    only_missing: bool,
    silent: bool,
    camera_role=None,
) -> List[str]:
    self._skip_empty_autogen_message = False
    if not paths:
        return []
    missing = self._missing_roi_files(paths, camera_role=camera_role)
    if not missing and not only_missing:
        if silent:
            return list(paths)
        reply = QtWidgets.QMessageBox.question(
            self,
            "覆盖已存在 ROI？",
            f"当前列表中 {len(paths)} 张图片都已有 ROI。\n是否覆盖并重新创建 ROI？",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            return list(paths)
        self._skip_empty_autogen_message = True
        return []
    if not missing:
        if not silent:
            QtWidgets.QMessageBox.information(self, "提示", "这些图片已经存在 ROI。")
            self._skip_empty_autogen_message = True
        return []

    missing_set = set(missing)
    existing = [p for p in paths if p not in missing_set]
    if not existing or silent:
        return list(missing) if only_missing else list(paths)

    default_button = (
        QtWidgets.QMessageBox.StandardButton.No
        if only_missing
        else QtWidgets.QMessageBox.StandardButton.Yes
    )
    reply = QtWidgets.QMessageBox.question(
        self, "覆盖已存在ROI？",
        (
            f"当前列表中已有 ROI 的图片有 {len(existing)} 张。\n"
            "是否覆盖并重新创建这些 ROI？\n\n"
            '选择"是"将重建整个列表；选择"否"只创建缺失 ROI；选择"取消"终止。'
        ),
        QtWidgets.QMessageBox.StandardButton.Yes
        | QtWidgets.QMessageBox.StandardButton.No
        | QtWidgets.QMessageBox.StandardButton.Cancel,
        default_button,
    )
    if reply == QtWidgets.QMessageBox.StandardButton.Cancel:
        self._skip_empty_autogen_message = True
        return []
    if reply == QtWidgets.QMessageBox.StandardButton.No:
        return list(missing)
    return list(paths)

def _autogen_roi_for_images(
    self,
    paths: List[str],
    only_missing: bool,
    silent: bool = False,
    *,
    camera_role=None,
) -> None:
    if not paths:
        if not silent:
            QtWidgets.QMessageBox.information(self, "提示", "没有可处理的图片")
        return
    ref_image = self.ref_image
    method = self.loc_method
    role = self.current_camera_role() if camera_role is None else str(camera_role)
    if method == "line2dup":
        try:
            recipe = self.line2dup_recipe_for_role(role, force_reload=True)
            if role == self.current_camera_role():
                self.line2dup_recipe = recipe
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "提示", f"无法加载模板 recipe：{exc}")
            return
        if recipe.reference_image and os.path.exists(recipe.reference_image):
            ref_image = recipe.reference_image
            if self.ref_image != ref_image:
                if role == self.current_camera_role():
                    self._set_reference(ref_image)
                else:
                    self.ref_image = ref_image
                    if getattr(self, "lbl_ref", None) is not None:
                        self.lbl_ref.setText(f"参考图: {os.path.basename(ref_image)}")
                        self.lbl_ref.setToolTip(ref_image)
        if not os.path.exists(self.line2dup_model_path_for_role(role)):
            QtWidgets.QMessageBox.warning(self, "提示", "当前产品还没有模板模型，请先创建模板。")
            return
        recipe_region_labels = {
            str(region.get("output_label") or region.get("reference_label") or "").strip()
            for region in (recipe.reference_regions or [])
            if isinstance(region, dict)
        }
        recipe_region_labels.discard("")
        reference_labels = list(recipe_region_labels)
        if (not ref_image or not os.path.exists(ref_image)) and not recipe_region_labels:
            QtWidgets.QMessageBox.warning(self, "提示", "模板定位需要参考图或已保存的参考 ROI。")
            return
        if reference_labels:
            missing_labels = [label for label in reference_labels if label not in recipe_region_labels]
            if missing_labels:
                ref_json = qr_core.labelme_json_of_image(ref_image) if ref_image else ""
                if not ref_json or not os.path.exists(ref_json):
                    QtWidgets.QMessageBox.warning(
                        self, "提示",
                        f"参考图缺少 labelme json，且 recipe 中也没有这些参考ROI：{', '.join(missing_labels)}",
                    )
                    return
                missing_labels = [
                    label for label in missing_labels
                    if qr_core.read_shape_from_labelme(ref_json, label) is None
                ]
                if missing_labels:
                    QtWidgets.QMessageBox.warning(
                        self, "提示",
                        f"参考图缺少参考ROI标注：{', '.join(missing_labels)}",
                    )
                    return
    elif method == "ncc":
        model_path = ncc_locator.resolved_model_path_for_product(self.session.product_dir, role)
        if not os.path.exists(model_path):
            QtWidgets.QMessageBox.warning(self, "提示", "当前产品还没有 NCC 模型，请先在 NCC 工具里制作模板。")
            return
    else:
        if not ref_image or not os.path.exists(ref_image):
            QtWidgets.QMessageBox.warning(self, "提示", "请先设置参考图")
            return
        ref_json = qr_core.labelme_json_of_image(ref_image)
        if not os.path.exists(ref_json):
            QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少标注 json（需要 anchor + roi）")
            return
        if qr_core.try_read_xywh_from_labelme(ref_json, "anchor") is None:
            QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少 anchor 标注")
            return
        if qr_core.try_read_xywh_from_labelme(ref_json, "roi") is None:
            QtWidgets.QMessageBox.warning(self, "提示", "参考图缺少 roi 标注")
            return

    todo = self._resolve_autogen_targets(paths, only_missing=only_missing, silent=silent, camera_role=role)
    if not todo:
        if getattr(self, "_skip_empty_autogen_message", False):
            self._skip_empty_autogen_message = False
            return
        if not silent:
            QtWidgets.QMessageBox.information(self, "提示", "这些图片已存在 ROI")
        return

    line2dup_detector = None
    ncc_model_path = ""
    ncc_model = None
    ncc_compiled = None
    if method == "line2dup":
        try:
            from line2dup.like_matcher import load_detector_model

            line2dup_detector = load_detector_model(self.line2dup_model_path_for_role(role))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Info", f"Failed to load line2dup model: {exc}")
            return
    elif method == "ncc":
        try:
            from ncc.runtime_service import NccCompiledModel

            ncc_model_path = ncc_locator.resolved_model_path_for_product(self.session.product_dir, role)
            ncc_model = ncc_locator.load_model(ncc_model_path).normalized()
            ncc_compiled = NccCompiledModel(ncc_model_path, ncc_model)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Info", f"Failed to load NCC model: {exc}")
            return

    ok = 0
    errs: List[str] = []
    try:
        for index, p in enumerate(todo, start=1):
            try:
                if method == "line2dup":
                    run = line2dup_locator.autogen_roi_json_from_line2dup_timed(
                        tgt_img_path=p, ref_img_path=ref_image,
                        product_dir=self.session.product_dir,
                        camera_role=role,
                        recipe=recipe,
                        detector=line2dup_detector,
                    )
                    self._line2dup_match_ms_by_image[p] = float(run.total_ms)
                    self._line2dup_autogen_ms_by_image[p] = float(run.total_ms)
                elif method == "ncc":
                    run = ncc_locator.autogen_roi_json_from_ncc_timed(
                        tgt_img_path=p,
                        product_dir=self.session.product_dir,
                        camera_role=role,
                        model_path=ncc_model_path,
                        model=ncc_model,
                        compiled_model=ncc_compiled,
                    )
                    self._line2dup_match_ms_by_image[p] = float(run.locate_ms)
                    self._line2dup_autogen_ms_by_image[p] = float(run.total_ms)
                else:
                    qr_core.autogen_roi_json_from_reference(
                        tgt_img_path=p, ref_img_path=ref_image,
                        method=method, anchor_label="anchor", roi_label="roi",
                    )
                ok += 1
            except Exception as e:
                errs.append(f"{os.path.basename(p)}: {e}")
                if method == "ncc":
                    self._line2dup_match_ms_by_image.pop(p, None)
                    self._line2dup_autogen_ms_by_image.pop(p, None)
            if not silent:
                self.lbl_status.setText(f"状态：自动 ROI {index}/{len(todo)}，成功 {ok}，失败 {len(errs)}")
                QtWidgets.QApplication.processEvents()
    finally:
        if ncc_compiled is not None:
            try:
                ncc_compiled.close()
            except Exception:
                pass

    if not silent:
        msg = f"自动 ROI 完成：成功 {ok} / 失败 {len(errs)}"
        if errs:
            msg += "\n\n失败示例（前10）：\n" + "\n".join(errs[:10])
        QtWidgets.QMessageBox.information(self, "完成", msg)
        if ok:
            self.lbl_status.setText(f"状态：当前列表已生成ROI，成功 {ok} 张，失败 {len(errs)} 张")

    if ok or (errs and method == "ncc"):
        invalidate_shape_cache = getattr(self, "_invalidate_shape_lookup_cache", None)
        if callable(invalidate_shape_cache):
            for image_path in todo:
                invalidate_shape_cache(image_path)
    if ok and not silent:
        self._reload_inspection_items()
    if ok or (errs and method == "ncc"):
        self.roiGeometryChanged.emit()

    cur = self.canvas.image_path()
    if not silent and cur and cur in todo:
        self._load_canvas_image(cur)
        self._set_status_for_current_image(cur)

    if silent and errs and ok == 0:
        raise RuntimeError("\n".join(errs[:10]))

def _autogen_roi_current_tab(self) -> None:
    tab = self.tabs.currentIndex()
    if tab == 0:
        paths = self._sample_paths_for_kind("train", self.current_camera_role())
    else:
        paths = _filter_paths_for_camera(self, self.test_files, self.current_camera_role())
    self._autogen_roi_for_images(paths, only_missing=self.chk_only_missing.isChecked())

def _autogen_roi_all(self) -> None:
    train_files = list(getattr(self, "train_files", []) or [])
    if not train_files:
        train_files = list(getattr(self, "ok_files", []) or []) + list(getattr(self, "ng_files", []) or [])
    paths = list(dict.fromkeys(train_files + list(self.test_files)))
    self._autogen_roi_for_images(paths, only_missing=self.chk_only_missing.isChecked())

def _clear_roi_for_images(
    self,
    paths: List[str],
    *,
    labels: Optional[List[str]] = None,
    silent: bool = False,
    camera_role=None,
) -> None:
    if not paths:
        if not silent:
            QtWidgets.QMessageBox.information(self, "提示", "没有可处理的图片")
        return
    role = self.current_camera_role() if camera_role is None else str(camera_role)
    self._clear_training_roi_review_state(role)
    if labels is None:
        labels, _clear_mode = self._clear_roi_labels_for_paths(paths, camera_role=role)
    labels = [str(label).strip() for label in (labels or []) if str(label).strip()] or ["roi"]

    removed = 0
    touched = 0
    for path in paths:
        try:
            path_removed = int(qr_core.delete_labelme_shapes(path, labels))
        except Exception:
            path_removed = 0
        if path_removed > 0:
            removed += path_removed
            touched += 1
            self._line2dup_match_ms_by_image.pop(path, None)
            self._line2dup_autogen_ms_by_image.pop(path, None)

    cur = self.canvas.image_path()
    if cur and cur in paths:
        self._load_canvas_image(cur)
        self._set_status_for_current_image(cur)

    if touched:
        self.roiGeometryChanged.emit()

    if not silent:
        QtWidgets.QMessageBox.information(
            self, "完成",
            f"已清空 ROI：{touched} 张图片，删除 {removed} 个标签。\n标签: {', '.join(labels)}",
        )
        self.lbl_status.setText(f"状态：已清空 ROI，图片 {touched} 张，标签 {removed} 个")

def _clear_roi_current_tab(self) -> None:
    tab = self.tabs.currentIndex()
    if tab == 0:
        paths = self._sample_paths_for_kind("train", self.current_camera_role())
        tab_name = "训练样本"
    else:
        paths = _filter_paths_for_camera(self, self.test_files, self.current_camera_role())
        tab_name = "测试样本"

    if not paths:
        QtWidgets.QMessageBox.information(self, "提示", "当前列表没有图片")
        return

    labels, clear_mode = self._clear_roi_labels_for_paths(paths)
    if clear_mode == "stale_only":
        action_text = "将删除当前列表中已失效的 ROI 标签: "
    elif clear_mode == "all_existing":
        action_text = "将删除当前列表中的全部相关 ROI 标签: "
    else:
        action_text = "将删除标签: "
    reply = QtWidgets.QMessageBox.question(
        self, "清空ROI",
        f"确定清空当前 {tab_name} 列表中的 ROI 吗？\n{action_text}{', '.join(labels)}",
        QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
        QtWidgets.QMessageBox.StandardButton.Cancel,
    )
    if reply != QtWidgets.QMessageBox.StandardButton.Yes:
        return
    self._clear_roi_for_images(paths, labels=labels, silent=False)
