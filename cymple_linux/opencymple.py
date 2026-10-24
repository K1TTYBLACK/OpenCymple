"""`opencymple`: the app for everyday use (the developer and oracle commands stay in `cymple-linux`).

    opencymple                      window + tray icon (a running OpenCymple just opens its window)
    opencymple --headless           tray icon only
    opencymple calibrate precise    13-point calibration in the headset
    opencymple calibrate coarse     5 s look-around range calibration
    opencymple recenter             recenter both eyes (look straight ahead)
    opencymple status | quit
    opencymple install | uninstall  desktop entry and icon (menu, WiVRn's application list)

Commands go to the running OpenCymple; when none runs, a headless one is started in the background
and keeps running afterwards."""
import argparse
import logging
import logging.handlers
import os
import sys

from . import instance

COMMANDS = ("calibrate", "recenter", "status", "quit")


def _parser():
    p = argparse.ArgumentParser(prog="opencymple", description="OpenCymple: Cymple eye and face tracking "
                                "for VRChat on Linux.")
    p.add_argument("--headless", action="store_true", help="start without the window (tray icon only)")
    p.add_argument("--no-osc", action="store_true", help="start with the VRChat output off")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    # for tests and debugging; passed on to a background instance started by a command
    p.add_argument("--replay", help=argparse.SUPPRESS)
    p.add_argument("--loop", action="store_true", help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    c = sub.add_parser("calibrate", help="start a gaze calibration")
    c.add_argument("mode", choices=("precise", "coarse"),
                   help="precise: 13 points in the headset; coarse: look around for 5 seconds")
    sub.add_parser("recenter", help="recenter both eyes: look straight ahead with relaxed eyes")
    sub.add_parser("status", help="is OpenCymple running, are the eyes tracked")
    sub.add_parser("quit", help="stop the running OpenCymple")
    sub.add_parser("install", help="add the desktop entry and icon (menu, WiVRn's application list)")
    sub.add_parser("uninstall", help="remove the desktop entry and icon")
    return p


def _setup_logging(verbose):
    os.makedirs(instance.state_dir(), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname).1s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    to_file = logging.handlers.RotatingFileHandler(os.path.join(instance.state_dir(), "opencymple.log"),
                                                   maxBytes=2_000_000, backupCount=2)
    for h in (logging.StreamHandler(), to_file):
        h.setFormatter(fmt)
        root.addHandler(h)


def _request(args):
    if args.command == "calibrate":
        return {"cmd": "calibrate", "mode": args.mode}
    return {"cmd": args.command}


def _print(reply):
    print(reply.get("message", ""))
    return 0 if reply.get("ok") else 1


def _command(args):
    """A command for the running app, starting a headless one if needed."""
    req = _request(args)
    timeout = 15.0 if args.command == "recenter" else 5.0
    reply = instance.send(req, timeout=timeout)
    if reply is not None:
        return _print(reply)
    if args.command in ("status", "quit"):
        print("OpenCymple is not running")
        return 1
    print("starting OpenCymple in the background...", flush=True)
    extra = ["--replay", args.replay] + (["--loop"] if args.loop else []) if args.replay else []
    if args.no_osc:
        extra.append("--no-osc")
    if not instance.spawn(extra):
        print("OpenCymple did not start; see %s" % os.path.join(instance.state_dir(), "opencymple.log"))
        return 1
    if args.command == "recenter" and not instance.wait_tracking():
        print("no eye is tracked yet: is the headset relay on?")
        return 1
    return _print(instance.send(req, timeout=timeout) or {"ok": False, "message": "OpenCymple went away"})


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command in ("install", "uninstall"):
        from . import desktop
        if args.command == "install":
            for path in desktop.install():
                print("wrote", path)
            print("OpenCymple is in the application menu and in WiVRn's list of applications to start "
                  "(not selected there).")
            bin_dir = os.path.dirname(desktop.link_path())
            if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
                print("note: %s is not on your PATH" % bin_dir)
        else:
            for path in desktop.uninstall() or ["nothing to remove"]:
                print("removed", path)
        return 0
    if args.command in COMMANDS:
        return _command(args)
    # the app itself; an instance that already runs just shows its window
    reply = instance.send({"cmd": "ping" if args.headless else "show"}, timeout=3.0)
    if reply is not None:
        print("OpenCymple is already running%s" % ("" if args.headless else "; window opened"))
        return 0
    _setup_logging(args.verbose)
    try:
        from .ui.shell import run_app
    except ImportError as e:
        sys.exit("OpenCymple needs PySide6 (%s): uv pip install --python .venv/bin/python -e '.[gui,xr]'" % e)
    from .cli import _paths
    from .protocol import INGEST_PORT
    args.ini = args.models = None
    args.provider, args.threads, args.osc_listen = "auto", 1, 9001
    ini, model_dir = _paths(args)
    source = dict(replay=args.replay, speed=1.0, loop=args.loop, bind="0.0.0.0", port=INGEST_PORT,
                  beacon=True, beacon_to=[], record=None)
    return run_app(args, ini, model_dir, source)


if __name__ == "__main__":
    sys.exit(main())
