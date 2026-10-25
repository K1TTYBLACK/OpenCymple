"""AFK detection (not part of the original): neutral values while the headset is off.

The eye cameras light the eye with their own IR LEDs, so with nobody in the headset the eye crop
turns almost black. On h3-vrchat (95 min) the crop's mean never fell below 46 while worn,
including blinks, closed eyes and pushing the headset up, and sat at 7..13 once it was off; on
corpus1 the lowest worn value was 46 as well. Meanwhile the models keep producing values from the
dark frames (and the mouth camera sees whatever is in front of it), so those creep onto the avatar.
While away the pipeline sends every field at its default (0: eyes open and centred, face relaxed).
"""
import logging

import numpy as np

log = logging.getLogger(__name__)

DARK_LEVEL = 30.0        # eye crop mean below this = no eye in front of the camera
EYE_FRESH_S = 0.2        # an eye counts as present for this long after its last lit frame
ENTER_S = 1.0            # no lit eye for this long (or no eye frames at all) -> away
LEAVE_S = 0.5            # a lit eye for this long -> back


class AfkDetector:
    """Fed every eye observation (under the pipeline lock), asked once per packet tick."""

    def __init__(self):
        self.enabled = False
        self.away = False
        self.seen = False                    # any eye frame so far (before that there is nothing to reset)
        self.lit = [None, None]              # per eye: time of the last lit frame
        self.back_since = None

    def on_eye(self, obs):
        self.seen = True
        if float(np.mean(obs.crop)) >= DARK_LEVEL:
            self.lit[obs.eye] = obs.t

    def update(self, now):
        """True while the values should be reset."""
        if not (self.enabled and self.seen):
            self.away, self.back_since = False, None
            return False
        last = max((t for t in self.lit if t is not None), default=None)
        was = self.away
        if self.away:
            if last is not None and now - last <= EYE_FRESH_S:
                self.back_since = now if self.back_since is None else self.back_since
                if now - self.back_since >= LEAVE_S:
                    self.away, self.back_since = False, None
            else:
                self.back_since = None
        elif last is None or now - last >= ENTER_S:
            self.away = True
        if self.away != was:
            log.info("AFK: %s", "headset off, sending default values" if self.away else "headset back on")
        return self.away
