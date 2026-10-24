"""Geometry of Cymple's full-view gaze calibration overlay (cympleface.dll), so the 13 dots sit where
Cymple's SteamVR overlay puts them for the same stored height and coverage.

The overlay is a head-locked square 2 m in front of the head (sub_1801811b0): an 896 px texture,
1.65 m x coverage wide, centred `y_offset` m above eye level. Dot i sits at pixel
(448 + int(330 tx), 448 - int(330 ty)) for the canonical target (tx, ty) (sub_18018ee70), so the
outer ring radius is 1.65 * coverage * 330 / 896 m. The thumbstick moves height and coverage at a
fixed rate (sub_180183cf0, sub_180183e00); coverage is capped so every dot stays inside 6..94 % of
each eye's view (sub_180182600).

The saved mesh targets stay the canonical ones: sub_180183060 divides each half axis by the visible
extent and sub_180182b60 normalises it back to the outer dot, which gives the canonical layout
whenever all dots are in view."""
import math

from .calib import MESH_TARGETS

DISTANCE_M = 2.0                       # sub_1801811b0: z = -2
TEXTURE_PX = 896
RING_PX = 330                          # sub_1801811a0
WIDTH_PER_SCALE = 1.65                 # sub_180181130
Y_RANGE = (-0.75, 0.75)                # sub_180183cf0
SCALE_RANGE = (0.45, 6.0)              # sub_1801810f0, sub_1801829b0
STICK_DEADZONE, STICK_SPAN = 0.2, 0.8  # sub_180184c30
STEP_DT_MAX_S = 0.05
HEIGHT_RATE = 0.7                      # m/s at full deflection
COVERAGE_RATE = 0.85                   # scale/s at full deflection
FIT = (0.06, 0.94)                     # sub_180182820: dots inside this part of each eye's view
CHANGE_EPS = 1e-4


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def overlay_width(scale):
    return WIDTH_PER_SCALE * clamp(scale, *SCALE_RANGE)


def ring_radius(scale):
    """Outer ring radius in metres at DISTANCE_M."""
    return overlay_width(scale) * RING_PX / TEXTURE_PX


def dot_position(i, y_offset, scale):
    """Head-space position (x right, y up, z forward negative) of dot i."""
    tx, ty = MESH_TARGETS[i]
    m = overlay_width(scale) / TEXTURE_PX
    return int(RING_PX * tx) * m, y_offset + int(RING_PX * ty) * m, -DISTANCE_M


def stick_step(axis, rate, dt):
    """Change for one poll with the stick at `axis` (-1..1) for `dt` seconds."""
    a = abs(axis)
    if not a >= STICK_DEADZONE:
        return 0.0
    t = clamp((a - STICK_DEADZONE) / STICK_SPAN, 0.0, 1.0)
    return math.copysign(t, axis) * rate * min(max(dt, 0.0), STEP_DT_MAX_S)


class EyeView:
    """One eye relative to the head: position, orientation quaternion (x, y, z, w) and the tangent
    extents of its view (left < 0 < right, down < 0 < up)."""

    def __init__(self, position, orientation, tan_left, tan_right, tan_up, tan_down):
        self.p = tuple(position)
        self.q = tuple(orientation)
        self.left, self.right, self.up, self.down = tan_left, tan_right, tan_up, tan_down

    @classmethod
    def from_angles(cls, position, orientation, left, right, up, down):
        return cls(position, orientation, math.tan(left), math.tan(right), math.tan(up), math.tan(down))

    def project(self, point):
        """(u, v) of a head-space point across this eye's view (0..1 left to right, bottom to top),
        or None when it is behind the eye."""
        d = [point[k] - self.p[k] for k in range(3)]
        x, y, z = _rotate_inverse(self.q, d)
        if z >= -1e-6:
            return None
        tx, ty = x / -z, y / -z
        w, h = self.right - self.left, self.up - self.down
        if w <= 1e-6 or h <= 1e-6:
            return None
        return (tx - self.left) / w, (ty - self.down) / h


def _rotate_inverse(q, v):
    """Rotate v by the inverse of unit quaternion q = (x, y, z, w)."""
    qx, qy, qz, qw = -q[0], -q[1], -q[2], q[3]
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v); v' = v + w t + cross(q.xyz, t)
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    return (vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx))


def fits(y_offset, scale, eyes):
    lo, hi = FIT
    for eye in eyes:
        for i in range(len(MESH_TARGETS)):
            uv = eye.project(dot_position(i, y_offset, scale))
            if uv is None or not (lo <= uv[0] <= hi and lo <= uv[1] <= hi):
                return False
    return True


def max_scale(y_offset, eyes):
    """Largest coverage that keeps every dot in view of every eye (sub_180182600: 24 bisections)."""
    lo, hi = SCALE_RANGE
    if not eyes or fits(y_offset, hi, eyes):
        return hi
    if not fits(y_offset, lo, eyes):
        return lo
    for _ in range(24):
        mid = (lo + hi) * 0.5
        if fits(y_offset, mid, eyes):
            lo = mid
        else:
            hi = mid
    return lo


def fov_targets(y_offset, scale, extents):
    """Mesh targets Cymple derives from one eye's view (sub_180183060 + sub_180182b60), for a scale
    already within limits. extents: tangents (right, left, up, down), all positive. None when Cymple
    falls back to the canonical layout."""
    eps = 1e-5
    right, left, up, down = (e * DISTANCE_M for e in extents)
    up, down = up - y_offset, down + y_offset
    if min(right, left, up, down) <= eps:
        return None
    m = overlay_width(scale) / TEXTURE_PX
    raw = []
    for tx, ty in MESH_TARGETS:
        if abs(tx) <= eps and abs(ty) <= eps:
            raw.append((0.0, 0.0))
            continue
        px, py = tx * RING_PX * m, ty * RING_PX * m
        raw.append((clamp(px / (right if px >= 0 else left), -1.0, 1.0),
                    clamp(py / (up if py >= 0 else down), -1.0, 1.0)))

    def normalise(values):                               # sub_180182dd0
        pos = max([0.0] + [v for v in values if v > 0])
        neg = max([0.0] + [-v for v in values if v < 0])
        return [clamp(v / pos, -1.0, 1.0) if v > eps and pos > eps else
                clamp(v / neg, -1.0, 1.0) if v < -eps and neg > eps else 0.0 for v in values]
    xs, ys = normalise([t[0] for t in raw]), normalise([t[1] for t in raw])
    return list(zip(xs, ys))


class OverlayGeometry:
    """Height and coverage while the user adjusts them with the thumbsticks."""

    def __init__(self, y_offset=0.0, scale=1.0, eyes=()):
        self.eyes = list(eyes)
        self.y = clamp(float(y_offset), *Y_RANGE)
        self._limit = None
        self.scale = self.clamp_scale(float(scale))

    def set_eyes(self, eyes):
        self.eyes = list(eyes)
        self._limit = None
        self.scale = self.clamp_scale(self.scale)

    def limit(self):
        if self._limit is None or self._limit[0] != self.y:
            self._limit = (self.y, max_scale(self.y, self.eyes))
        return self._limit[1]

    def clamp_scale(self, scale):                       # sub_1801829b0
        return clamp(scale, SCALE_RANGE[0], max(SCALE_RANGE[0], self.limit()))

    def adjust_height(self, axis, dt):                  # sub_180183cf0
        d = stick_step(axis, HEIGHT_RATE, dt)
        if abs(d) < 1e-6:
            return False
        y = clamp(self.y + d, *Y_RANGE)
        if abs(y - self.y) < CHANGE_EPS:
            return False
        self.y = y
        self.scale = self.clamp_scale(self.scale)
        return True

    def adjust_coverage(self, axis, dt):                # sub_180183e00
        d = stick_step(axis, COVERAGE_RATE, dt)
        if abs(d) < 1e-6:
            return False
        s = self.clamp_scale(self.scale + d)
        if abs(s - self.scale) < CHANGE_EPS:
            return False
        self.scale = s
        return True

    def ring_degrees(self):
        """Outer ring radius as a visual angle, for display."""
        return math.degrees(math.atan2(ring_radius(self.scale), DISTANCE_M))

    def dots(self):
        return [dot_position(i, self.y, self.scale) for i in range(len(MESH_TARGETS))]
