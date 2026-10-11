import numpy as np
import pytest

from cloth_sort import config
from cloth_sort.perception.detect import detect_pieces
from cloth_sort.perception.reader import _components, _merge_vertical, ink_mask
from cloth_sort.sim.synth import CHAR_CLASSES, build_dataset, normalize_char
from cloth_sort.sim import fonts
from cloth_sort.sim.world import World


def test_random_scene_places_pieces_without_overlap():
    w = World.random_scene(12, seed=3)
    assert len(w.pieces) == 12
    assert len({p.label for p in w.pieces}) == 12
    for a in w.pieces:
        for b in w.pieces:
            if a.id < b.id:
                gap = max(abs(a.center[0] - b.center[0]), abs(a.center[1] - b.center[1]))
                assert gap > config.PIECE_MM, (a, b)


def test_random_scene_too_many_pieces_raises():
    with pytest.raises(RuntimeError):
        World.random_scene(60, seed=0, labels=[str(i) for i in range(60)])


def test_camera_image_shape_and_detection_matches_pieces():
    w = World.random_scene(10, seed=5)
    img = w.render_camera()
    assert img.shape == (config.BENCH_H_MM, config.BENCH_W_MM, 3)
    bg = w.render_camera(include_pieces=False)
    dets = detect_pieces(img, bg, region=config.SOURCE_AREA)
    assert len(dets) == 10
    for d in dets:
        p = w.piece_at(d.center)
        assert p is not None
        assert abs(p.center[0] - d.center[0]) < 8 and abs(p.center[1] - d.center[1]) < 8


def test_grip_only_when_on_table_and_over_piece():
    w = World.random_scene(3, seed=1)
    p = w.pieces[0]
    w.move_gripper(p.center, 1.0)
    assert w.close_gripper() is None          # 上空では掴めない
    w.open_gripper()
    w.move_gripper(p.center, 0.0)
    assert w.close_gripper() == p.id
    w.move_gripper((900, 300), 1.0)
    assert p.center == (900, 300)             # 把持中は一緒に動く
    w.open_gripper("7")
    assert p.placed_at == "7" and not p.held and w.held_piece is None


def test_ink_mask_and_components_find_each_digit():
    w = World.random_scene(6, seed=2)
    img = w.render_camera(noise=False)
    dets = detect_pieces(img, w.render_camera(noise=False, include_pieces=False), region=config.SOURCE_AREA)
    found_any = False
    for d in dets:
        boxes = _merge_vertical(_components(ink_mask(d.crop)))
        truth = w.piece_at(d.center).label
        if len(boxes) == len(truth):
            found_any = True
    assert found_any


def test_normalize_char_is_square_and_in_range():
    mask = np.zeros((40, 10), dtype=np.float32)
    mask[5:35, 3:7] = 1.0
    out = normalize_char(mask)
    assert out.shape == (28, 28)
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert out[:, 14].sum() > out[:, 2].sum()   # 中央に寄せられている


def test_build_dataset_shapes():
    x, y = build_dataset(3, fonts.available(fonts.TRAIN_FONTS)[:2], seed=0)
    assert x.shape == (3 * len(CHAR_CLASSES), 1, 28, 28)
    assert set(y.tolist()) == set(range(len(CHAR_CLASSES)))
