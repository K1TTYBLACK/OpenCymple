"""OpenCymple as a desktop app: the runtime, VRChat output and headset calibration live here, with a
tray icon (Open Window, Precise Calibration, Quit) and the single-instance socket that
`opencymple <command>` talks to (instance.py). The window is created when asked for and closing it
leaves OpenCymple running in the tray."""
import json
import logging
import os
import signal
import sys
import threading

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication, QIcon
from PySide6.QtNetwork import QLocalServer
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .. import instance
from .. import settings as S
from ..desktop import icon_path, installed
from ..runtime import Runtime
from .xr_overlay import HeadsetLauncher

log = logging.getLogger(__name__)
APP_NAME = "OpenCymple"


def app_icon():
    return QIcon(icon_path())


class Shell(QObject):
    reply_ready = Signal(object, object)          # (socket, reply) from worker threads to the GUI thread

    def __init__(self, rt, args, options):
        super().__init__()
        self.rt, self.args, self.opts = rt, args, options
        self.headset = HeadsetLauncher(rt)
        self.window = None
        self.quitting = False
        self._buffers = {}
        self.reply_ready.connect(self._write_reply)
        self.rt.pipeline.afk.enabled = self.opts["afk"]["enabled"]
        self._start_osc()
        self.tray = QSystemTrayIcon(app_icon(), self)
        self.tray.setToolTip(APP_NAME)
        menu = QMenu()
        for text, slot in (("Open Window", self.show_window), ("Precise Calibration", self.precise_calibration),
                           ("Quit", self.quit)):
            act = QAction(text, menu)
            act.triggered.connect(slot)
            menu.addAction(act)
        self.menu = menu
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()
        if not QSystemTrayIcon.isSystemTrayAvailable():
            log.warning("no system tray: control OpenCymple with `opencymple <command>`")
        self.server = QLocalServer(self)
        self._listen()

    # ---- pieces ---------------------------------------------------------------------------
    def _start_osc(self):
        o = self.opts["osc"]
        if not o["enabled"] or self.args.no_osc:
            return
        try:
            self.rt.start_osc(o["target"], o["avatar"], self.args.osc_listen, o["config_dir"])
        except (OSError, ValueError) as e:
            log.error("cannot start the VRChat output: %s", e)

    def _listen(self):
        path = instance.socket_path()
        QLocalServer.removeServer(path)          # a stale socket from a crashed run (main() checked it is dead)
        if not self.server.listen(path):
            log.error("cannot listen on %s: %s", path, self.server.errorString())
            return
        self.server.newConnection.connect(self._connection)
        log.info("listening for commands on %s", path)

    def _tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.show_window()

    # ---- actions --------------------------------------------------------------------------
    def show_window(self):
        if self.window is None:
            from .app import MainWindow
            self.window = MainWindow(self.rt, self.args, self.opts, shell=self)
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def window_closed(self, win):
        if self.window is win:
            self.window = None
            win.deleteLater()

    def precise_calibration(self):
        why = self.headset.start()
        if why:
            self.tray.showMessage(APP_NAME, why, app_icon(), 6000)
        return why

    def quit(self):
        if self.quitting:
            return
        self.quitting = True
        self.headset.stop(wait=5.0)
        if self.window is not None:
            self.window.close()                  # saves its geometry
        self.server.close()
        self.tray.hide()
        self.rt.stop()
        QApplication.quit()

    # ---- commands from `opencymple <command>` ---------------------------------------------
    def _connection(self):
        while self.server.hasPendingConnections():
            sock = self.server.nextPendingConnection()
            self._buffers[id(sock)] = b""
            sock.readyRead.connect(lambda s=sock: self._read(s))
            sock.disconnected.connect(lambda s=sock: (self._buffers.pop(id(s), None), s.deleteLater()))

    def _read(self, sock):
        key = id(sock)
        if key not in self._buffers:             # request already taken
            return
        buf = self._buffers[key] + bytes(sock.readAll())
        if b"\n" not in buf:
            self._buffers[key] = buf
            return
        del self._buffers[key]
        try:
            req = json.loads(buf.split(b"\n", 1)[0])
            self.handle(req, lambda reply: self.reply_ready.emit(sock, reply))
        except Exception as e:                   # noqa: BLE001
            log.exception("bad command")
            self._write_reply(sock, {"ok": False, "message": "bad command: %s" % e})

    def handle(self, req, reply):
        """Run one request; `reply(dict)` may be called from another thread."""
        cmd = req.get("cmd")
        if cmd == "ping":
            reply({"ok": True, "message": "%s is running" % APP_NAME})
        elif cmd == "show":
            self.show_window()
            reply({"ok": True, "message": "window opened"})
        elif cmd == "status":
            reply({"ok": True, "eyes": list(self.rt.eyes_tracked()),
                   "message": self.headset.status() or self.rt.status_line()})
        elif cmd == "calibrate":
            mode = req.get("mode", "precise")
            if mode == "precise":
                why = self.precise_calibration()
                reply({"ok": why is None, "message": why or "headset calibration started"})
            elif mode == "coarse":
                if self.rt.calib.precise_gaze:
                    self.rt.set_precise(False)
                ok = self.rt.start_coarse()
                reply({"ok": bool(ok), "message": "coarse calibration: look around for 5 seconds" if ok
                       else "coarse calibration did not start"})
            else:
                reply({"ok": False, "message": "unknown calibration mode %r" % mode})
        elif cmd == "recenter":
            def work():
                try:
                    rep = self.rt.recenter(wait=True)
                    ok = isinstance(rep, dict) and any(isinstance(r, dict) for r in rep.values())
                    reply({"ok": ok, "message": self.rt.calib_status})
                except Exception as e:           # noqa: BLE001
                    log.exception("recenter failed")
                    reply({"ok": False, "message": "recenter failed: %s" % e})
            threading.Thread(target=work, name="recenter", daemon=True).start()
        elif cmd == "quit":
            reply({"ok": True, "message": "%s stopped" % APP_NAME})
            QTimer.singleShot(100, self.quit)
        else:
            reply({"ok": False, "message": "unknown command %r" % cmd})

    def _write_reply(self, sock, reply):
        try:
            sock.write(json.dumps(reply).encode() + b"\n")
            sock.flush()
            sock.disconnectFromServer()
        except RuntimeError:                     # the client went away
            pass


def run_app(args, ini, model_dir, source_kwargs):
    """`opencymple`: window (unless --headless) + tray until Quit."""
    from .app import _apply_theme
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland.textinput=false")
    # file dialogs go through xdg-desktop-portal: the desktop's own file manager, not Qt's built-in one
    os.environ["QT_QPA_PLATFORMTHEME"] = "xdgdesktopportal"
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    if installed():                              # Wayland app id = desktop entry: icon and name in docks
        QGuiApplication.setDesktopFileName(instance.APP_ID)
    app.setWindowIcon(app_icon())
    app.setQuitOnLastWindowClosed(False)
    _apply_theme(app)
    rt = Runtime(ini, model_dir, args.provider, args.threads, mouth=True)
    rt.start()
    rt.start_source(**source_kwargs)
    shell = Shell(rt, args, S.load_options())
    if not args.headless:
        shell.show_window()
    signal.signal(signal.SIGINT, lambda *_: shell.quit())
    signal.signal(signal.SIGTERM, lambda *_: shell.quit())    # WiVRn stops its apps this way
    wake = QTimer(interval=200)                  # lets Python run its signal handlers
    wake.timeout.connect(lambda: None)
    wake.start()
    try:
        return app.exec()
    finally:
        if not shell.quitting:
            shell.headset.stop(wait=5.0)
            rt.stop()
