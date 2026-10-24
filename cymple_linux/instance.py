"""One OpenCymple per user: the running app listens on a Unix socket, and `opencymple <command>`
sends it one JSON line and prints the JSON reply. Only one process can receive the camera streams
(UDP 23303), so commands always go to the running app; when none runs, a headless one is started
in the background first.

Requests: {"cmd": "ping" | "show" | "status" | "recenter" | "quit"} or
{"cmd": "calibrate", "mode": "precise" | "coarse"}. Replies: {"ok": bool, "message": str, ...}."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

APP_ID = "opencymple"
START_TIMEOUT_S = 40.0                   # model loading takes a few seconds
TRACKING_TIMEOUT_S = 15.0


def socket_path():
    base = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return os.path.join(base, "%s.sock" % APP_ID)


def state_dir():
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return os.path.join(base, APP_ID)


def send(request, timeout=5.0, path=None):
    """The running app's reply as a dict, or None when no app listens."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(path or socket_path())
    except (FileNotFoundError, ConnectionRefusedError):
        s.close()
        return None
    try:
        s.sendall(json.dumps(request).encode() + b"\n")
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    finally:
        s.close()
    if not data.strip():
        return {"ok": False, "message": "no reply from OpenCymple"}
    return json.loads(data)


def running():
    return send({"cmd": "ping"}, timeout=2.0) is not None


def spawn(extra_args=()):
    """Start `opencymple --headless` detached from this terminal; wait until it answers."""
    cmd = [sys.executable, "-m", "cymple_linux.opencymple", "--headless", *extra_args]
    subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True, close_fds=True)
    end = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < end:
        if running():
            return True
        time.sleep(0.25)
    return False


def wait_tracking(timeout=TRACKING_TIMEOUT_S):
    """After a fresh start: until at least one eye is tracked (or the timeout)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        rep = send({"cmd": "status"}, timeout=2.0)
        if rep and any(rep.get("eyes", ())):
            return True
        time.sleep(0.3)
    return False
