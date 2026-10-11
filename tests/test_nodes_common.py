import numpy as np

from cloth_sort.nodes.common import arrow_to_image, arrow_to_json, image_to_arrow, json_to_arrow


def test_image_roundtrip():
    img = (np.arange(2 * 3 * 3) % 255).astype(np.uint8).reshape(2, 3, 3)
    arr, meta = image_to_arrow(img)
    back = arrow_to_image(arr, meta)
    assert back.shape == img.shape and (back == img).all()


def test_json_roundtrip_accepts_numpy_types():
    obj = {"bbox": [np.int64(1), np.int64(2)], "conf": np.float32(0.5), "xy": np.array([3.0, 4.0]), "s": "番号"}
    back = arrow_to_json(json_to_arrow(obj))
    assert back == {"bbox": [1, 2], "conf": 0.5, "xy": [3.0, 4.0], "s": "番号"}
