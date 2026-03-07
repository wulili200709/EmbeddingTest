from iv4matchmethod.search_label import build_manifest_record, compute_pose_from_clicks


def test_compute_pose_from_clicks_identity():
    result = compute_pose_from_clicks(
        template_bbox=[0, 0, 100, 100],
        template_roi_polygon_image=[[70, 50], [80, 50], [80, 60], [70, 60]],
        center_point=[50, 50],
        roi_point=[75, 55],
    )

    assert result["center"] == [50.0, 50.0]
    assert abs(result["angle_deg"]) < 1e-6
    assert result["scale"][0] == result["scale"][1] == 1.0


def test_build_manifest_record_uses_template_annotation():
    template_annotation = {
        "template_image": "template.png",
        "template_bbox": [10, 20, 30, 40],
        "roi_ref_polygon": [[1, 2], [3, 4], [5, 6]],
    }

    record = build_manifest_record(
        template_annotation=template_annotation,
        search_image="search.png",
        center=[11, 22],
        angle_deg=12.5,
        scale=[0.9, 0.9],
        ok_ng="NG",
    )

    assert record["template_image"] == "template.png"
    assert record["template_bbox"] == [10, 20, 30, 40]
    assert record["search_image"] == "search.png"
    assert record["center"] == [11.0, 22.0]
    assert record["angle_deg"] == 12.5
    assert record["scale"] == [0.9, 0.9]
    assert record["roi_ref_polygon"] == [[1, 2], [3, 4], [5, 6]]
    assert record["ok_ng"] == "NG"

