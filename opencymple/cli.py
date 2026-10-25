"""opencymple-linux command line."""
import argparse
import json
import logging
import os
import signal
import sys
import threading
import time

from . import __version__
from . import calibration as C
from . import logs
from . import protocol as P

log = logging.getLogger("opencymple")


def _paths(args, own=True):
    """(INI, model dir). own=True: our settings copy (imported from Cymple's INI on first use);
    own=False: Cymple's own iniFile.ini (the oracle runs the Windows app with it)."""
    from . import settings as S
    ini = S.resolve_ini(args.ini, own)
    if not ini or not os.path.isfile(ini):
        sys.exit("Cymple iniFile.ini not found; pass --ini or set CYMPLE_INI")
    models = args.models or C.find_model_dir(ini) or C.find_model_dir(C.find_ini())
    if not models or not os.path.isdir(models):
        sys.exit("Cymple model directory not found; pass --models or set CYMPLE_MODELS")
    return ini, models


def _add_common(p):
    p.add_argument("--ini", help="settings INI in Cymple's format; default: ours in ~/.config/opencymple-linux "
                                 "(oracle commands: Cymple's own iniFile.ini)")
    p.add_argument("--models", help="directory with model0.bin ... model6.bin; default: next to the INI")
    p.add_argument("--provider", default="auto",
                   help="ONNX Runtime execution provider: auto/cpu, cuda, rocm, migraphx, openvino, ...")
    p.add_argument("--threads", type=int, default=1, help="intra-op threads per model (default 1)")


def _add_source(p):
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=P.INGEST_PORT)
    p.add_argument("--no-beacon", action="store_true", help="do not announce this PC on UDP 23302")
    p.add_argument("--beacon-to", action="append", default=[], metavar="IP",
                   help="extra beacon target, e.g. the headset IP (repeatable)")
    p.add_argument("--record", metavar="FILE", help="also write every datagram to FILE (CYMPCAP1)")
    p.add_argument("--replay", metavar="FILE", help="read a CYMPCAP1 capture instead of the network")
    p.add_argument("--speed", type=float, default=1.0, help="replay speed factor (0 = unpaced)")
    p.add_argument("--loop", action="store_true", help="loop the replay")


def _source_kwargs(args):
    return dict(replay=args.replay, speed=args.speed, loop=args.loop, bind=args.bind, port=args.port,
                beacon=not args.no_beacon, beacon_to=args.beacon_to, record=args.record)


def _start_source(args, ingest):
    from .receiver import ReplaySource, UdpSource
    if args.replay:
        src = ReplaySource(ingest, args.replay, speed=args.speed, loop=args.loop)
    else:
        src = UdpSource(ingest, bind=args.bind, port=args.port, beacon=not args.no_beacon,
                        beacon_to=args.beacon_to, record_path=args.record)
    src.start()
    return src


def _wait(stop, src, args, tick):
    deadline = time.monotonic() + args.duration if args.duration else None
    while not stop.is_set():
        stop.wait(1.0)
        tick()
        if deadline and time.monotonic() >= deadline:
            break
        if args.replay and src.finished.is_set():
            time.sleep(0.3)                  # let the workers drain the last frames
            tick()
            break


def _install_sigint(stop):
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())


def cmd_run(args):
    from .runtime import Runtime

    ini, model_dir = _paths(args)
    rt = Runtime(ini, model_dir, args.provider, args.threads, mouth=not args.no_mouth)
    if args.osc:
        rt.start_osc(args.osc_target, args.osc_avatar, args.osc_listen, args.osc_config_dir)
    rt.start()
    stop = threading.Event()
    _install_sigint(stop)
    src = rt.start_source(**_source_kwargs(args))

    def tick():
        if args.json:
            snap = rt.pipeline.snapshot()
            print(json.dumps({"rx": rt.ingest.snapshot(), "workers": snap["workers"], "gaze": snap["gaze"],
                              "mouth": None if snap["mouth"] is None else
                              [round(float(v), 4) for v in snap["mouth"].values]}), flush=True)
        else:
            print(rt.status_line(), flush=True)
        if args.debug_image:
            import cv2
            from .debugview import render
            cv2.imwrite(args.debug_image, render(rt.pipeline))

    try:
        _wait(stop, src, args, tick)
    finally:
        rt.stop()
    print(json.dumps({"rx": rt.ingest.snapshot(), "workers": rt.pipeline.snapshot()["workers"],
                      "model_ms": {n: round(m.total_ms / m.calls, 3) for n, m in rt.models.items() if m.calls}},
                     indent=2, default=str))


def cmd_gui(args):
    try:
        from .ui.app import run_gui
    except ImportError as e:
        sys.exit("the window needs PySide6 (%s): uv pip install --python .venv/bin/python -e '.[gui]'" % e)
    ini, model_dir = _paths(args)
    return run_gui(args, ini, model_dir, _source_kwargs(args))


def cmd_config(args):
    from . import settings as S
    from .oracle import cymple_pids

    if args.action == "import":
        src = args.source or C.find_ini()
        if not src:
            sys.exit("Cymple iniFile.ini not found; pass --from")
        print("imported %s -> %s" % (src, S.import_from_cymple(src, force=args.force)))
        return 0
    own = S.own_ini()
    if args.action == "export":
        target = args.to or C.find_ini()
        if not os.path.isfile(own) or not target:
            sys.exit("nothing to export: %s" % ("no Cymple INI found; pass --to" if os.path.isfile(own)
                                                 else "%s does not exist yet" % own))
        if args.dry_run:
            keys, files = S.export_plan(own, target)
        elif cymple_pids():
            sys.exit("Windows Cymple is running; close it first (it writes its INI again when it exits)")
        else:
            keys, files, backup = S.export_to_cymple(own, target)
        for section, values in keys.items():
            for k, v in values.items():
                print("[%s] %s = %s" % (section, k, v))
        for name in files:
            print("%s/%s" % (S.AVATAR_DIR, name))
        if not keys and not files:
            print("Cymple's settings already match ours")
        elif args.dry_run:
            print("(dry run: %s not changed)" % target)
        else:
            print("written to %s%s; close Windows Cymple first next time, it saves its INI on exit"
                  % (target, " (backup %s)" % backup if backup else ""))
        return 0
    opts = S.load_options()
    print("config directory  %s" % S.config_dir())
    print("settings INI      %s%s" % (own, "" if os.path.isfile(own) else " (not imported yet)"))
    print("imported from     %s" % opts.get("imported_from", "-"))
    print("Cymple's INI      %s" % (C.find_ini() or "not found"))
    return 0


def cmd_models(args):
    import numpy as np
    from .models import SPECS, load_models

    _, model_dir = _paths(args)
    models = load_models(model_dir, list(SPECS), args.provider, args.threads)
    for m in models.values():
        print(m.describe())
    if args.bench:
        for m in models.values():
            x = np.random.default_rng(0).uniform(-1, 1, (1, 3, m.spec.input_size, m.spec.input_size)).astype(np.float32)
            for _ in range(5):
                m.run(x)
            t = time.perf_counter()
            for _ in range(args.bench):
                m.run(x)
            print("%-12s %.3f ms/inference" % (m.name, (time.perf_counter() - t) * 1000 / args.bench))


def cmd_calibrate_roi(args):
    from .eye import calibrate_roi, EYE_NAMES
    from .imgproc import decode_gray
    from .models import load_models
    from .receiver import Ingest

    ini, model_dir = _paths(args)
    eyes = [0, 1] if args.eye == "both" else [EYE_NAMES.index(args.eye)]
    roi = load_models(model_dir, ["eye_roi"], args.provider, args.threads)["eye_roi"]
    frames = {}
    if args.image:
        import cv2
        if len(eyes) != 1:
            sys.exit("--image needs --eye left or --eye right")
        img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            sys.exit("cannot read %s" % args.image)
        frames[eyes[0]] = img
    else:
        counts = {e: 0 for e in eyes}
        done = threading.Event()

        def on_frame(f):
            eye = P.EYE_STREAMS.get(f.stream)
            if eye in counts:
                frames[eye] = f.jpeg
                counts[eye] += 1
                if all(c >= args.frames for c in counts.values()):
                    done.set()

        ingest = Ingest(on_frame)
        src = _start_source(args, ingest)
        print("collecting %d frames per eye (look straight ahead)..." % args.frames, flush=True)
        ok = done.wait(args.timeout)
        src.stop()
        if not ok:
            sys.exit("timed out; frames so far: %s" % {EYE_NAMES[e]: c for e, c in counts.items()})
        frames = {e: decode_gray(j) for e, j in frames.items()}
    status = 0
    for eye in eyes:
        quad = calibrate_roi(roi, frames[eye], eye) if frames.get(eye) is not None else None
        if quad is None:
            print("%s eye: ROI model returned an invalid quadrilateral" % EYE_NAMES[eye])
            status = 1
            continue
        print("%s eye ROI: %s" % (EYE_NAMES[eye], " ".join("%s=%.6f" % kv for kv in zip(C.ROI_KEYS, quad.reshape(-1)))))
        if args.write_ini:
            C.write_roi_quad(ini, eye, quad)
            print("  written to %s (backup kept next to it)" % ini)
    return status


def cmd_record(args):
    from .receiver import Ingest
    from .record import Beeper, run_script, wait_for_streams

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.name or time.strftime("corpus-%Y%m%d-%H%M%S"))
    if not args.replay:
        args.record = base + ".cap"
    ingest = Ingest(lambda f: None)
    stop = threading.Event()
    _install_sigint(stop)
    src = _start_source(args, ingest)
    beeper = Beeper(enabled=not args.no_sound)
    try:
        streams = [0, 1] + ([] if args.no_mouth else [2])
        print("waiting for camera streams %s at >= %.0f fps..." % (streams, args.min_fps), flush=True)
        missing = wait_for_streams(ingest, streams, args.min_fps, args.wait, stop)
        if missing and not args.allow_missing:
            print("streams not running: %s (%s); is the headset relay on? Use --allow-missing to record anyway."
                  % (missing, ingest.snapshot()["streams"]), file=sys.stderr)
            return 1
        labels = run_script(base + ".labels.json", beeper, stop, time_scale=args.time_scale,
                            capture=None if args.replay else os.path.basename(args.record))
    finally:
        src.stop()
        src.join(timeout=2.0)
        beeper.close()
    rx = ingest.snapshot()
    print(json.dumps({"labels": base + ".labels.json", "capture": None if args.replay else args.record,
                      "segments": len(labels["segments"]), "rx": rx}, indent=2))
    return 0


def cmd_oracle_record(args):
    from .oracle import record

    ini, _ = _paths(args, own=False)
    suffix = ".oracle.jsonl" if args.mode == "vrcft" else ".oracle-%s.jsonl" % args.mode
    out = args.out or os.path.splitext(args.capture)[0] + suffix
    n = record(args.capture, out, ini, mode=args.mode, settle_s=args.settle, cymple_log=out + ".cymple.log")
    print("%d oracle datagrams -> %s" % (n, out))
    return 0 if n else 1


def _oracle_compare_osc(golden):
    from .oracle import osc_formula_check

    stats, ticks = osc_formula_check(golden)
    print("%d Cymple OSC ticks; our formulas recomputed from its own eye state:" % ticks)
    print("%-48s %5s %6s %9s %9s" % ("address", "type", "n", "mean", "max"))
    for a, s in sorted(stats.items()):
        print("%-48s %5s %6d %9.6f %9.6f" % (a.replace("/avatar/parameters/", ""), s["type"], s["n"], s["mean"],
                                             s["max"]))
    return stats


def cmd_oracle_compare(args):
    from .models import load_models
    from .oracle import compare
    from .pipeline import Pipeline

    golden = args.golden or os.path.splitext(args.capture)[0] + ".oracle.jsonl"
    with open(golden) as f:
        mode = json.loads(next(f)).get("mode")
    if mode == "osc":
        stats = _oracle_compare_osc(golden)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(stats, f, indent=1)
        return 0

    ini, model_dir = _paths(args, own=False)
    calib = C.load(ini)
    models = load_models(model_dir, ["eye_landmark", "mouth_roi", "mouth_v1"], args.provider, args.threads)
    pipeline = Pipeline(models, calib)

    def produce(p, t):
        return p.build_packet(t)

    result = compare(args.capture, golden, pipeline, produce)
    print("%d oracle packets" % result["packets"])
    print("%-22s %6s %9s %9s %9s %7s  oracle range" % ("field", "n", "mean|e|", "p95|e|", "max|e|", "corr"))
    for name, s in result["fields"].items():
        print("%-22s %6d %9.4f %9.4f %9.4f %7.3f  [%+.2f, %+.2f]" % (
            name, s["n"], s["mean_abs"], s["p95_abs"], s["max_abs"], s["corr"], *s["oracle_range"]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0


def cmd_selftest(args):
    from .gaze import selftest
    ini = args.ini or C.find_ini()
    if not ini:
        sys.exit("Cymple iniFile.ini not found; pass --ini")
    return 0 if selftest(ini) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="opencymple-linux", description="Native Linux runtime for Cymple eye/face trackers.")
    ap.add_argument("--version", action="version", version="%(prog)s " + __version__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="receive camera streams and run eye/mouth inference")
    _add_common(p)
    _add_source(p)
    p.add_argument("--no-mouth", action="store_true", help="eyes only")
    p.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = until Ctrl+C)")
    p.add_argument("--json", action="store_true", help="print one JSON status object per second")
    p.add_argument("--debug-image", metavar="PNG", help="rewrite a debug image (frames, ROI, landmarks) every second")
    p.add_argument("--osc", action="store_true", help="send VRChat avatar parameters over OSC (Mode A)")
    p.add_argument("--osc-target", default="127.0.0.1:9000", metavar="HOST:PORT")
    p.add_argument("--osc-avatar", default="auto", metavar="auto|none|FILE",
                   help="where the avatar's parameter list comes from: auto = VRChat OSCQuery, else its OSC "
                        "config file on /avatar/change; none = default addresses + VRChat native eye tracking; "
                        "FILE = an avatar OSC config JSON")
    p.add_argument("--osc-config-dir", default="", metavar="DIR",
                   help="VRChat's OSC config folder (the one holding usr_*/Avatars); default: autodetect "
                        "in the Steam Proton prefix")
    p.add_argument("--osc-listen", type=int, default=9001, metavar="PORT",
                   help="VRChat OSC output port to watch for /avatar/change (0 = off)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("gui", help="the same as run, in a window: previews, values, sliders, VRChat output")
    _add_common(p)
    _add_source(p)
    p.add_argument("--no-mouth", action="store_true", help="eyes only")
    p.add_argument("--no-osc", action="store_true", help="start with the VRChat output off")
    p.add_argument("--duration", type=float, default=0, help="close after N seconds (0 = when closed)")
    p.add_argument("--screenshot", metavar="PNG", help="with --duration: save each tab as <PNG stem>-<tab>.png "
                                                       "before closing")
    p.add_argument("--osc-listen", type=int, default=9001, metavar="PORT",
                   help="VRChat OSC output port to watch for /avatar/change (0 = off)")
    p.add_argument("--calibrate", action="store_true",
                   help="start the 13-point gaze calibration in the headset (OpenXR overlay) right away")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("config", help="our settings copy: show, import from Cymple, export back to Cymple")
    p.add_argument("action", nargs="?", choices=("show", "import", "export"), default="show")
    p.add_argument("--from", dest="source", metavar="INI", help="import: Cymple INI to copy (default: auto-detect)")
    p.add_argument("--force", action="store_true", help="import: replace our copy (a backup is kept)")
    p.add_argument("--to", metavar="INI", help="export: Cymple INI to update (default: auto-detect)")
    p.add_argument("--dry-run", action="store_true", help="export: only list what would change")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("models", help="load every model from memory and print its I/O")
    _add_common(p)
    p.add_argument("--bench", type=int, default=0, metavar="N", help="time N inferences per model")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("calibrate-roi", help="run the eye ROI model (model0) to find each eye's crop quad")
    _add_common(p)
    _add_source(p)
    p.add_argument("--eye", choices=("left", "right", "both"), default="both")
    p.add_argument("--frames", type=int, default=50, help="frames to wait for per eye before sampling the last one")
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--image", help="use this raw camera frame instead of the stream (needs --eye)")
    p.add_argument("--write-ini", action="store_true", help="store the quads in Cymple's INI (backup first)")
    p.set_defaults(func=cmd_calibrate_roi)

    p = sub.add_parser("record", help="guided recording of a labelled test corpus (wear the headset)")
    _add_source(p)
    p.add_argument("--name", help="file name stem (default corpus-<date>-<time>)")
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                     "captures"))
    p.add_argument("--no-sound", action="store_true", help="no beep cues")
    p.add_argument("--no-mouth", action="store_true", help="do not wait for the mouth camera")
    p.add_argument("--min-fps", type=float, default=40.0)
    p.add_argument("--wait", type=float, default=30.0, help="seconds to wait for the streams before giving up")
    p.add_argument("--allow-missing", action="store_true", help="record even if a stream is missing")
    p.add_argument("--time-scale", type=float, default=1.0, help="shorten (<1) every step, for testing")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("oracle-record", help="replay a capture into Windows Cymple (Bottles) and log its output")
    _add_common(p)
    p.add_argument("capture")
    p.add_argument("--out", help="default: <capture>.oracle.jsonl")
    p.add_argument("--settle", type=float, default=25.0, help="seconds to let Cymple start before replaying")
    p.add_argument("--mode", choices=("vrcft", "osc"), default="vrcft",
                   help="vrcft: log the 70-value packet on 22999; osc: log direct OSC on 9000 (ForceRelevant)")
    p.set_defaults(func=cmd_oracle_record)

    p = sub.add_parser("oracle-compare", help="compare our output with an oracle log on the same capture")
    _add_common(p)
    p.add_argument("capture")
    p.add_argument("--golden", help="default: <capture>.oracle.jsonl (an --mode osc log checks the OSC formulas)")
    p.add_argument("--json", metavar="FILE", help="also write the statistics as JSON")
    p.set_defaults(func=cmd_oracle_compare)

    p = sub.add_parser("selftest", help="check the gaze mesh math against the saved calibration")
    p.add_argument("--ini")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logs.attach()                                # the window's Logs tab, for `gui`
    try:
        return args.func(args) or 0
    except (OSError, RuntimeError) as e:
        log.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
