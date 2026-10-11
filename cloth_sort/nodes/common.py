"""dora ノード間で画像・JSON を Arrow 配列として受け渡すための小さな補助。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pyarrow as pa


def image_to_arrow(img: np.ndarray) -> tuple[pa.Array, dict]:
    """(H, W, 3) uint8 を平坦な UInt8 配列と形状メタデータにする。"""
    return pa.array(img.reshape(-1)), {"shape": list(img.shape)}


def arrow_to_image(value: pa.Array, metadata: dict) -> np.ndarray:
    return value.to_numpy(zero_copy_only=False).reshape(metadata["shape"])


def _jsonable(o: Any) -> Any:
    """numpy のスカラー・配列を JSON にできる型へ落とす。"""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"JSON にできない型: {type(o).__name__}")


def json_to_arrow(obj: Any) -> pa.Array:
    return pa.array([json.dumps(obj, ensure_ascii=False, default=_jsonable)])


def arrow_to_json(value: pa.Array) -> Any:
    return json.loads(value.to_pylist()[0])
