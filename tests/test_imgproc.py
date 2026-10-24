import numpy as np

from opencymple_linux import imgproc
from opencymple_linux.eye import LANDMARK_SWAPS, ROI_SWAPS, neutral_rescale, unmirror_points


def test_tensor_recipe():
    img = np.array([[0, 255], [127, 128]], np.uint8)
    t = imgproc.to_tensor(np.kron(img, np.ones((48, 48), np.uint8)), 96)
    assert t.shape == (1, 3, 96, 96) and t.dtype == np.float32
    assert np.array_equal(t[0, 0], t[0, 2])
    assert t[0, 0, 0, 0] == np.float32(-127.5) * np.float32(0.00784314)
    assert abs(t[0, 0, 0, 95] - 1.0) < 1e-4


def test_quad_valid():
    good = np.float32([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert imgproc.quad_valid(good)
    assert not imgproc.quad_valid(np.float32([[0, 0], [0, 0.01], [1, 0], [1, 1]]))
    assert not imgproc.quad_valid(np.float32([[0, 0], [0, 1], [np.nan, 0], [1, 1]]))


def test_warp_identity_quad():
    img = np.random.default_rng(1).integers(0, 255, (96, 96), dtype=np.uint8)
    out = imgproc.warp_quad(img, np.float32([[0, 0], [0, 1], [1, 0], [1, 1]]))
    assert np.array_equal(out, img)


def test_manual_mouth_rect_matches_hand_calculation():
    # [Cymple_Mouth] values from the shipped user INI, 240x176 frame
    assert imgproc.manual_mouth_rect(240, 176, 0.327485, 0.828460, 0.600877, 0.5) == (47, 57, 146, 89)
    # bottom above upper is pushed to upper + 1 px; width > 1 clamps to the full frame
    assert imgproc.manual_mouth_rect(240, 176, 0.9, 0.1, 2.0, 0.0) == (0, 158, 240, 2)


def test_rotate_180_maps_about_integer_centre():
    img = np.zeros((176, 240), np.uint8)
    img[10, 20] = 255
    out = imgproc.rotate_frame(img, -180.0)
    assert out[176 - 10, 240 - 20] == 255          # x -> 2*120 - x, y -> 2*88 - y
    assert imgproc.rotate_frame(img, 0.00005) is img


def test_unmirror():
    pts = np.float32([[0.1 * i, 0.05 * i] for i in range(8)])
    lm = unmirror_points(pts, LANDMARK_SWAPS)
    assert np.allclose(lm[0], [1 - 0.1, 0.05]) and np.allclose(lm[1], [1.0, 0.0])
    assert np.allclose(lm[6], [1 - 0.6, 0.3]) and np.allclose(lm[7], [1 - 0.7, 0.35])
    q = unmirror_points(pts[:4], ROI_SWAPS)
    assert np.allclose(q[0], [1 - 0.2, 0.1]) and np.allclose(q[3], [1 - 0.1, 0.05])


def test_neutral_rescale():
    assert neutral_rescale(0.5, 0.0) == 0.5
    assert neutral_rescale(0.75, 0.5) == 0.5
    assert neutral_rescale(0.2, 0.5) == 0.0
    assert neutral_rescale(1.5, 0.0) == 1.0
    assert neutral_rescale(0.9, 1.0) == 0.0
