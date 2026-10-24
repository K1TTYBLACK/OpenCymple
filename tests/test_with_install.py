"""Tests against the local Cymple install, capture and dataset; skipped when they are absent."""
import glob
import os

import cv2
import numpy as np
import pytest

from cymple_linux import calibration as C
from cymple_linux.receiver import Ingest, ReplaySource

HERE = os.path.dirname(__file__)
CAPTURE = os.path.join(HERE, "..", "captures", "live-test-1.cap")
INI = C.find_ini()
MODELS = C.find_model_dir(INI) if INI else None
need_install = pytest.mark.skipif(not (INI and MODELS), reason="no Cymple install found")


@pytest.mark.skipif(not os.path.exists(CAPTURE), reason="no live capture")
def test_replay_capture_frame_counts():
    frames = []
    ing = Ingest(frames.append)
    src = ReplaySource(ing, CAPTURE, speed=0)
    src.run()
    counts = {s: sum(f.stream == s for f in frames) for s in (0, 1, 2)}
    assert counts == {0: 992, 1: 989, 2: 987}         # same as cymple_udp_probe.py on this file
    assert not ing.errors
    assert {d.name for d in ing.devices.values()} == {"EYE_V3", "MOUTH_SINGLE_V1"}


ORACLE = os.path.join(HERE, "..", "captures", "corpus1.oracle.jsonl")
CORPUS = os.path.join(HERE, "..", "captures", "corpus1.cap")


@need_install
@pytest.mark.skipif(not (os.environ.get("CYMPLE_ORACLE_TESTS") and os.path.exists(ORACLE)),
                    reason="set CYMPLE_ORACLE_TESTS=1 (takes ~3 min) and record corpus1 + its oracle")
def test_fields_match_oracle():
    from cymple_linux.models import load_models
    from cymple_linux.oracle import compare
    from cymple_linux.pipeline import Pipeline
    pipeline = Pipeline(load_models(MODELS, ["eye_landmark", "mouth_roi", "mouth_v1"]), C.load(INI))
    stats = compare(CORPUS, ORACLE, pipeline, lambda p, t: p.build_packet(t))["fields"]
    assert len(stats) == 70
    for name, s in stats.items():
        if s["oracle_range"][1] - s["oracle_range"][0] < 0.1:
            assert s["max_abs"] < 0.02, (name, s)        # pupils, ShrugUpper/Lower, tongue 66..68
            continue
        assert s["mean_abs"] < 0.015, (name, s)
        assert s["corr"] > 0.97, (name, s)


@need_install
def test_models_load_from_memory():
    from cymple_linux.models import SPECS, load_models
    models = load_models(MODELS, list(SPECS))
    shapes = {n: [tuple(o.shape) for o in m.session.get_outputs()] for n, m in models.items()}
    assert shapes == {"eye_roi": [(1, 8)], "eye_landmark": [(1, 16), (1, 3)], "mouth_roi": [(1, 8)],
                      "mouth_v1": [(1, 30)], "mouth_v2": [(1, 30)]}


def _dataset_frame(side):
    root = os.path.join(os.path.dirname(INI), "dataset", "eye", "0_eye_normal_%s" % side)
    files = sorted(glob.glob(os.path.join(root, "*.jpg")))
    if not files:
        pytest.skip("no eye dataset")
    # dataset frames are the 240x176 camera image squeezed to 200x200
    return cv2.resize(cv2.imread(files[len(files) // 2], cv2.IMREAD_GRAYSCALE), (240, 176),
                      interpolation=cv2.INTER_AREA)


@need_install
@pytest.mark.parametrize("eye,side", [(0, "left"), (1, "right")])
def test_eye_roi_and_landmarks_on_dataset(eye, side):
    from cymple_linux.eye import EyeTracker, calibrate_roi, landmark_sanity
    from cymple_linux.models import load_models
    cal = C.load(INI)
    m = load_models(MODELS, ["eye_roi", "eye_landmark"])
    frame = _dataset_frame(side)
    quad = calibrate_roi(m["eye_roi"], frame, eye)
    assert quad is not None
    assert np.max(np.abs(quad - cal.eyes[eye].roi_quad)) < 0.08   # model0 reproduces the saved quad
    obs = EyeTracker(m["eye_landmark"], eye, cal.eyes[eye].roi_quad).process(frame)
    s = landmark_sanity(obs)
    assert obs.roi_used and s["lids_ordered"] and s["in_crop"] and s["corner_span"] > 0.3


ORACLE_OSC = os.path.join(HERE, "..", "captures", "corpus1.oracle-osc.jsonl")


@pytest.mark.skipif(not (os.environ.get("CYMPLE_ORACLE_TESTS") and os.path.exists(ORACLE_OSC)),
                    reason="set CYMPLE_ORACLE_TESTS=1 and record corpus1's --mode osc oracle")
def test_osc_formulas_match_cymple():
    from cymple_linux.oracle import osc_formula_check

    stats, _ = osc_formula_check(ORACLE_OSC)
    assert len(stats) == 400                       # every address Cymple sent on corpus1
    for a, s in stats.items():
        if s["type"] == "bool":
            assert s["mean"] * s["n"] <= max(2, 0.01 * s["n"]), a   # bits flip at the same 1/(2^n-1) steps
        elif "Squeeze" in a:
            assert s["mean"] < 1e-3, a             # pow(openness, 0.15) magnifies the deadband
        else:
            assert s["max"] < 2e-4, a              # exact up to Cymple's 1e-4 send deadband
