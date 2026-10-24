"""Mode A: VRChat avatar parameters over OSC (what Cymple's direct-OSC mode sends).

Runs on the 10 ms packet timer: the eye fields become VRCFT's parameters for the current avatar
(vrcft_params.resolve), and only values that moved are sent. Cymple's sender skips float
changes smaller than 1e-4 (the smallest step in its OSC output on corpus1); we do the same.
Everything is re-sent every `refresh_s` so a VRChat restart or avatar reload picks it up.
"""
import logging

from . import osc
from .vrcft_params import resolve, unified_from_fields

log = logging.getLogger(__name__)
DEADBAND = 1e-4


def _moved(old, new):
    if isinstance(new, tuple):
        return any(abs(a - b) >= DEADBAND * 0.999 for a, b in zip(old, new))
    if isinstance(new, bool):
        return old != new
    return abs(old - new) >= DEADBAND * 0.999


class VRChatOutput:
    def __init__(self, sender=None, avatars=None, refresh_s=5.0):
        """sender: osc.OscSender (None = only compute); avatars: AvatarWatcher or anything
        with `current` and `version` (None = no avatar information, default addresses)."""
        self.sender, self.avatars, self.refresh_s = sender, avatars, refresh_s
        self.bindings, self.avatar = [], None
        self._version = None
        self._last = {}
        self._refresh_at = None
        self._fields = {}                    # latest value of every field seen (eyes may drop out)
        self.sent = 0

    def _rebind(self):
        version = self.avatars.version if self.avatars is not None else 0
        if version == self._version:
            return
        self._version = version
        info = self.avatars.current if self.avatars is not None else None
        self.avatar = info
        self.bindings = resolve(info.params if info else None, info.full_face_tracking if info else False)
        self._last = {}
        native = [b.address for b in self.bindings if b.address.startswith("/tracking/")]
        log.info("VRChat OSC: %d parameters for %s%s", len(self.bindings),
                 "avatar " + (info.name or info.id) if info else "unknown avatar (default addresses)",
                 " incl. " + ", ".join(native) if native else "")

    def messages(self, fields, now, eye_active=True):
        """[(address, args)] to send for this tick."""
        self._rebind()
        if self._refresh_at is None or now >= self._refresh_at:
            self._last = {}
            self._refresh_at = now + self.refresh_s
        u = unified_from_fields(fields, eye_active)
        out = []
        for b in self.bindings:
            v = b.value(u)
            old = self._last.get(b.address)
            if old is not None and not _moved(old, v):
                continue
            self._last[b.address] = v
            out.append((b.address, v if isinstance(v, tuple) else (v,)))
        return out

    def update(self, fields, now):
        self._fields.update(fields)
        msgs = self.messages(self._fields, now, eye_active="EyeYaw_L" in fields)
        if msgs and self.sender is not None:
            self.sender.send([osc.message(a, *args) for a, args in msgs])
        self.sent += len(msgs)
        return msgs

    def status(self):
        a = self.avatar
        return {"avatar": (a.name or a.id) if a else None, "source": a.source if a else "default",
                "parameters": len(self.bindings), "sent": self.sent}
