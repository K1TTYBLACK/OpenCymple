"""`opencymple install`: the desktop entry and icon in ~/.local/share. The `X-WiVRn-VR` category
puts OpenCymple in WiVRn's list of applications to start with the headset (wivrn-server and
wivrn-dashboard scan desktop entries for it; the flatpak reads xdg-data/applications). Choosing it
there is left to the user, so WiVRn's configuration is not touched."""
import os
import shutil
import subprocess
import sys

APP_ID = "opencymple"
HERE = os.path.dirname(os.path.abspath(__file__))


def icon_path():
    return os.path.join(HERE, "resources", "%s.svg" % APP_ID)


def data_home():
    return os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")


def entry_path():
    return os.path.join(data_home(), "applications", "%s.desktop" % APP_ID)


def installed_icon_path():
    return os.path.join(data_home(), "icons", "hicolor", "scalable", "apps", "%s.svg" % APP_ID)


def link_path():
    """`opencymple` on the PATH: ~/.local/bin (XDG's place for user commands)."""
    return os.path.join(os.path.expanduser("~/.local/bin"), APP_ID)


def installed():
    return os.path.isfile(entry_path())


def executable():
    """The command that starts OpenCymple: the console script next to this Python, else -m."""
    script = os.path.join(os.path.dirname(sys.executable), APP_ID)
    if os.path.isfile(script) and os.access(script, os.X_OK):
        return [script]
    return [sys.executable, "-m", "cymple_linux.opencymple"]


def _quote(arg):
    """Desktop Entry Exec quoting."""
    if arg and not any(c in arg for c in ' \t\n"\'\\><~|&;$*?#()`'):
        return arg
    return '"%s"' % arg.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")


def entry_text(cmd=None):
    cmd = cmd or executable()
    exe = " ".join(_quote(a) for a in cmd)
    lines = ["[Desktop Entry]",
             "Type=Application",
             "Name=OpenCymple",
             "GenericName=Eye and face tracking",
             "Comment=Cymple eye and face tracking for VRChat",
             "Exec=%s" % exe,                 # no TryExec: WiVRn's flatpak cannot see the path
             "Icon=%s" % APP_ID,
             "Terminal=false",
             "Categories=Utility;X-WiVRn-VR;",
             "Keywords=VR;eye tracking;face tracking;Cymple;VRChat;",
             "StartupWMClass=%s" % APP_ID,
             "Actions=calibrate;recenter;quit;",
             ""]
    for action, name, args in (("calibrate", "Precise Calibration", "calibrate precise"),
                               ("recenter", "Recenter Eyes", "recenter"), ("quit", "Quit", "quit")):
        lines += ["[Desktop Action %s]" % action, "Name=%s" % name, "Exec=%s %s" % (exe, args), ""]
    return "\n".join(lines)


def install():
    """Write the entry and the icon and link the command; returns the paths written."""
    entry, icon = entry_path(), installed_icon_path()
    for path in (entry, icon):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.copyfile(icon_path(), icon)
    with open(entry, "w", encoding="utf-8") as f:
        f.write(entry_text())
    _refresh_icon_cache()
    written = [entry, icon]
    script, link = executable()[0], link_path()
    if os.path.basename(script) == APP_ID:               # the console script (not `python -m`)
        if os.path.islink(link):
            os.remove(link)
        if not os.path.exists(link):                     # never replace a real file of someone else's
            os.makedirs(os.path.dirname(link), exist_ok=True)
            os.symlink(script, link)
            written.append(link)
    return written


def uninstall():
    removed = []
    for path in (entry_path(), installed_icon_path()):
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    link = link_path()
    if os.path.islink(link) and os.path.basename(os.readlink(link)) == APP_ID:
        os.remove(link)
        removed.append(link)
    _refresh_icon_cache()
    return removed


def _refresh_icon_cache():
    """Only an icon cache that already exists is refreshed; creating one could hide icons later."""
    theme = os.path.join(data_home(), "icons", "hicolor")
    if os.path.isfile(os.path.join(theme, "icon-theme.cache")) and shutil.which("gtk-update-icon-cache"):
        subprocess.run(["gtk-update-icon-cache", "-q", "-t", "-f", theme], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
