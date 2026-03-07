from iv4matchmethod.annotate import annotation_to_json, polygon_to_image, polygon_to_relative


def test_polygon_round_trip_uses_template_bbox_center():
    bbox = [100, 50, 80, 40]
    polygon_image = [[140, 70], [150, 70], [150, 80], [140, 80]]
    relative = polygon_to_relative(polygon_image, bbox)

    assert relative == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    assert polygon_to_image(relative, bbox) == polygon_image


def test_annotation_json_contains_relative_and_absolute_roi():
    payload = annotation_to_json(
        image_path="template_full.png",
        image_size=(512, 512),
        bbox=[0, 0, 512, 512],
        polygon_image=[[300, 180], [360, 180], [360, 220], [300, 220]],
    )

    assert payload["template_bbox"] == [0.0, 0.0, 512.0, 512.0]
    assert payload["roi_image_polygon"] == [[300.0, 180.0], [360.0, 180.0], [360.0, 220.0], [300.0, 220.0]]
    assert payload["roi_ref_polygon"] == [[44.0, -76.0], [104.0, -76.0], [104.0, -36.0], [44.0, -36.0]]
    assert payload["roi_origin"] == "template_bbox_center"

