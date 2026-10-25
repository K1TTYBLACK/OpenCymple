"""Cymple 4.18.0 eye-gaze math, re-implemented from cympleface.dll.

Function names in comments are the cympleface.dll addresses this code mirrors.
See Cymple-gaze-pipeline.md for the evidence behind each step.

Landmark input: model2 `landmark_output[16]` as 8 (x, y) points, normalized 0..1 in the
96x96 eye crop, after Cymple's un-mirroring (image-ordered pairs):
  pt0, pt1  eye corners            (image-left, image-right)
  pt2, pt3  upper-lid points       (image-left, image-right)
  pt4, pt5  lower-lid points       (image-left, image-right)
  pt6       pupil centre           (moves with gaze)
  pt7       eye-centre reference   (smoothed, nearly static)
Eye index: 0 = left, 1 = right.  Output yaw > 0 = looking right, pitch > 0 = looking up.

Self-test against a saved calibration:  opencymple-linux selftest [--ini path/to/iniFile.ini]
"""
import math
import time

TWO_PI = 6.2831855          # float literal used by the binary (0x1801dc988)
PI = 3.1415927


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def finite(*xs):
    return all(math.isfinite(x) for x in xs)


def lerp(a, b, t):                       # sub_1800c77d0
    t = clamp(t, 0.0, 1.0)
    return a * (1.0 - t) + b * t


# ---------------------------------------------------------------- landmark features
def corners_valid(lm):                   # sub_1800c6110 (separation threshold 0.01)
    (x0, y0), (x1, y1) = lm[0], lm[1]
    return finite(x0, y0, x1, y1) and max(abs(x0 - x1), abs(y0 - y1)) > 0.01


def ref_valid(lm):                       # sub_1800c6250: pt7 finite and inside the crop
    x, y = lm[7]
    return finite(x, y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def pupil_valid(lm):                     # sub_1800c5f60
    return finite(*lm[6])


def blend_reference(lm, widen):          # sub_1800c64f0 + sub_1800c6430
    """Pull pt7 toward the corner midpoint when the (neutral-rescaled) widen channel > 0.5."""
    w = clamp((clamp(widen, 0.0, 1.0) - 0.5) / 0.5, 0.0, 1.0) if finite(widen) else 0.0
    if w <= 0.0 or not (ref_valid(lm) and corners_valid(lm)):
        return lm
    mx = 0.5 * (lm[0][0] + lm[1][0])
    my = 0.5 * (lm[0][1] + lm[1][1])
    out = list(lm)
    out[7] = ((1 - w) * lm[7][0] + w * mx, (1 - w) * lm[7][1] + w * my)
    return out


def feature_mode2(lm):                   # sub_1800c7370 + sub_1800c6e60
    """Eye-local frame: u along the corners, v toward the lower lid; scales L and 0.42 L."""
    if not (pupil_valid(lm) and corners_valid(lm) and ref_valid(lm)):
        return None
    ex, ey = lm[1][0] - lm[0][0], lm[1][1] - lm[0][1]
    L = math.hypot(ex, ey)
    if L <= 1e-5:
        return None
    ux, uy = ex / L, ey / L
    vx, vy = -uy, ux
    upx, upy = 0.5 * (lm[2][0] + lm[3][0]), 0.5 * (lm[2][1] + lm[3][1])
    lox, loy = 0.5 * (lm[4][0] + lm[5][0]), 0.5 * (lm[4][1] + lm[5][1])
    if vx * (lox - upx) + vy * (loy - upy) < 0.0:
        vx, vy = -vx, -vy
    dx, dy = lm[7][0] - lm[6][0], lm[7][1] - lm[6][1]
    raw_yaw = (dx * ux + dy * uy) / L
    raw_pitch = (dx * vx + dy * vy) / (0.42 * L)
    return (raw_yaw, raw_pitch) if finite(raw_yaw, raw_pitch) else None


def feature_mode0(lm):                   # sub_1800c6350
    if not (ref_valid(lm) and pupil_valid(lm)):
        return None
    return lm[7][0] - lm[6][0], lm[7][1] - lm[6][1]


# ---------------------------------------------------------------- 13-point gaze mesh
def wrap(a):                             # sub_1800c2fc0
    while a < 0.0:
        a += TWO_PI
    while a >= TWO_PI:
        a -= TWO_PI
    return a


def unwrap_near(a, ref):                 # sub_1800c3020
    a, ref = wrap(a), wrap(ref)
    while a - ref > PI:
        a -= TWO_PI
    while ref - a > PI:
        a += TWO_PI
    return a


def slope(x0, y0, x1, y1):               # sub_1800c30c0
    dx = x1 - x0
    return 0.0 if dx <= 1e-4 else (y1 - y0) / dx


def hermite(x, x0, x1, y0, y1, m0, m1):  # sub_1800c3120
    h = x1 - x0
    if h <= 1e-4:
        return y0
    t = clamp((x - x0) / h, 0.0, 1.0)
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * h * m0 +
            (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * h * m1)


def build_ring(mesh, start, n):          # sub_1800cda80 (inner, n=4) / sub_1800cde40 (outer, n=8)
    cx, cy = mesh[0][0], mesh[0][1]
    ring = []
    for e in mesh[start:start + n]:
        if not finite(*e):
            return None
        dx, dy = e[0] - cx, e[1] - cy
        r = math.hypot(dx, dy)
        if r <= 1e-5:
            return None
        ring.append((wrap(math.atan2(dy, dx)), r, e[2], e[3]))
    ring.sort(key=lambda el: el[0])
    for k in range(n):
        a, b = ring[k][0], ring[(k + 1) % n][0]
        if a >= b:
            b += TWO_PI
        if b - a <= 1e-4:
            return None
    return ring


def ring_eval(ring, theta):              # sub_1800ce670 (n=4) / sub_1800cec60 (n=8)
    """Periodic cubic-Hermite (Catmull-Rom style tangents) of (radius, target_yaw, target_pitch)."""
    n = len(ring)
    q0 = wrap(theta)
    seg = None
    for k in range(n):
        a, b = ring[k][0], ring[(k + 1) % n][0]
        if a >= b:
            b += TWO_PI
        q = q0 + TWO_PI if a > q0 else q0
        if q <= b + 1e-4:
            seg, qe = k, q
            break
    if seg is None:
        return None
    P, Pp, Pn, Pnn = ring[seg], ring[(seg - 1) % n], ring[(seg + 1) % n], ring[(seg + 2) % n]
    thp = unwrap_near(Pp[0], P[0])
    if thp >= P[0]:
        thp -= TWO_PI
    thn = unwrap_near(Pn[0], P[0])
    if P[0] >= thn:
        thn += TWO_PI
    thnn = unwrap_near(Pnn[0], thn)
    if thn >= thnn:
        thnn += TWO_PI
    out = []
    for ch in (1, 2, 3):
        m0 = slope(thp, Pp[ch], thn, Pn[ch])
        m1 = slope(P[0], P[ch], thnn, Pnn[ch])
        out.append(hermite(qe, P[0], thn, P[ch], Pn[ch], m0, m1))
    return tuple(out) if finite(*out) else None


def extrapolation_cap(c, o):             # sub_1800c32d0 + sub_1800c3370
    cap = math.inf
    for ci, oi in zip(c, o):
        d = oi - ci
        if abs(d) <= 1e-5:
            continue
        s = ((1.0 if d > 0 else -1.0) - ci) / d
        if s >= 1.0:
            cap = min(cap, s)
        elif s > 0.0:
            cap = min(cap, 1.0)
    return cap if math.isfinite(cap) and cap >= 1.0 else 1.0


def mesh_map(x, y, mesh):                # sub_1800c34c0
    """mesh: 13 x (raw_yaw, raw_pitch, target_yaw, target_pitch); 0 = centre, 1-4 inner, 5-12 outer.
    Returns (yaw, pitch) in [-1, 1] or None when the mesh is unusable (caller falls back)."""
    if not finite(x, y):
        return None
    for e in mesh:                                       # exact calibration hit
        if (x - e[0]) ** 2 + (y - e[1]) ** 2 <= 1e-10:
            return clamp(e[2], -1, 1), clamp(e[3], -1, 1)
    c = mesh[0]
    dx, dy = x - c[0], y - c[1]
    r = math.hypot(dx, dy)
    if r <= 1e-5:
        return clamp(c[2], -1, 1), clamp(c[3], -1, 1)
    inner, outer = build_ring(mesh, 1, 4), build_ring(mesh, 5, 8)
    if inner is None or outer is None:
        return None
    th = math.atan2(dy, dx)
    I, O = ring_eval(inner, th), ring_eval(outer, th)
    if I is None or O is None:
        return None
    ri, ix, iy = I
    ro, ox, oy = O
    if ri <= 1e-5 or ri + 1e-5 >= ro:
        return None
    if r <= ri:                                          # centre -> inner ring
        t = clamp(r / ri, 0.0, 1.0)
        res = (c[2] + t * (ix - c[2]), c[3] + t * (iy - c[3]))
    elif r <= ro:                                        # inner -> outer ring
        span = ro - ri
        t = 1.0 if span <= 1e-5 else clamp((r - ri) / span, 0.0, 1.0)
        res = (ix + t * (ox - ix), iy + t * (oy - iy))
    else:                                                # beyond outer ring: capped extrapolation
        oc = (clamp(ox, -1, 1), clamp(oy, -1, 1))
        s = clamp(r / ro, 1.0, extrapolation_cap((c[2], c[3]), oc))
        res = (c[2] + s * (oc[0] - c[2]), c[3] + s * (oc[1] - c[3]))
    q = clamp(r / ri, 0.0, 1.0)                          # centre dead-zone (smoothstep 8%..18%)
    if q <= 0.08:
        w = 0.0
    elif q < 0.18:
        u = clamp((q - 0.08) / 0.1, 0.0, 1.0)
        w = u * u * (3.0 - 2.0 * u)
    else:
        w = 1.0
    yaw = clamp(c[2] + w * (res[0] - c[2]), -1, 1)
    pitch = clamp(c[3] + w * (res[1] - c[3]), -1, 1)
    return yaw, pitch


def range_map(x, rng):                   # sub_1800c5530 (legacy/coarse fallback)
    mn, mid, mx = sorted(rng)
    x = clamp(x, -1.0, 1.0)
    if x >= mid:
        d = mx - mid
        return 0.0 if d <= 1e-5 else clamp((x - mid) / d, -1.0, 1.0)
    d = mid - mn
    return 0.0 if d <= 1e-5 else clamp((x - mid) / d, -1.0, 1.0)


# ---------------------------------------------------------------- smoothing
class OneEuro:                           # sub_1800cff00 / sub_1800cf820
    def __init__(self, min_cutoff=1.0, beta=0.2, d_cutoff=1.0):
        self.d_cutoff = d_cutoff
        self.set_params(min_cutoff, beta)
        self.x = self.dx = self.t = None

    def set_params(self, min_cutoff, beta):
        """New slider value: the filter keeps its state."""
        self.min_cutoff, self.beta = max(min_cutoff, 1e-4), max(beta, 0.0)

    @staticmethod
    def _alpha(te, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, now=None):
        now = time.monotonic() if now is None else now
        if not math.isfinite(x):
            return self.x if self.x is not None else x
        if self.x is None:
            self.x, self.dx, self.t = x, 0.0, now
            return x
        te = now - self.t
        self.t = now
        te = 0.02 if (not math.isfinite(te) or te <= 1e-4) else clamp(te, 1.0 / 240.0, 0.2)
        dx = (x - self.x) / te
        ad = self._alpha(te, self.d_cutoff)
        self.dx = ad * dx + (1 - ad) * self.dx
        a = self._alpha(te, self.min_cutoff + self.beta * abs(self.dx))
        self.x = a * x + (1 - a) * self.x
        return self.x


def gaze_filter_params(eye_gaze_filter):  # sub_1800bbd60 / sub_1800c5c60, filters 0-4
    t = clamp((clamp(eye_gaze_filter, 0.2, 1.0) - 0.2) / 0.8, 0.0, 1.0)
    return 4.0 + (1.8 - 4.0) * t, 0.06 + (0.02 - 0.06) * t   # (min_cutoff, beta)


# ---------------------------------------------------------------- per-frame gaze solve
def lead_weight(v):                      # sub_1800c7860 (v = -yawL for left, +yawR for right)
    if not math.isfinite(v) or v <= 0.2:
        return 0.0
    return 1.0 if v >= 0.5 else (v - 0.2) / 0.3


class GazeSolver:
    """Mirrors sub_1800c0950 + sub_1800ba5f0 + filters 0-2 (sub_1800bc250) + sub_1800bcb00."""

    def __init__(self, meshes, mesh_valid=(True, True), feature_mode=(2, 2),
                 ranges=None, eye_gaze_filter=0.8, amplitude=1.0, squint_comp=0.0, precise=True):
        self.meshes, self.mesh_valid, self.mode = meshes, mesh_valid, feature_mode
        # Precise mode off: sub_1800c7590 -> sub_1800c6940, which ends in mode 0 for these landmarks
        self.precise = precise
        self.features = [None, None]              # last (yaw, pitch) feature per eye
        # legacy (min, mid, max) per eye for yaw and pitch; used only when the mesh fails
        self.ranges = ranges or {e: {"yaw": (-1.0, 0.0, 1.0), "pitch": (-1.0, 0.0, 1.0)} for e in (0, 1)}
        self.prev = [0.0, 0.0, 0.0, 0.0]          # pitchL, pitchR, yawL, yawR  (this+0x584)
        self.filters = [OneEuro() for _ in range(3)]
        self.configure(eye_gaze_filter, amplitude, squint_comp)

    def configure(self, eye_gaze_filter, amplitude, squint_comp):
        for f in self.filters:
            f.set_params(*gaze_filter_params(eye_gaze_filter))
        self.amplitude = clamp(amplitude, 0.5, 1.0)
        self.squint_comp = clamp(squint_comp, 0.0, 1.0)

    def eye_gaze(self, eye, lm, widen=0.0):
        lm = blend_reference(lm, widen)
        feat = feature_mode2(lm) if self.precise and self.mode[eye] == 2 else feature_mode0(lm)
        self.features[eye] = feat
        if feat is None:
            return None
        out = mesh_map(feat[0], feat[1], self.meshes[eye]) if self.mesh_valid[eye] else None
        if out is None:
            out = (range_map(feat[0], self.ranges[eye]["yaw"]), range_map(feat[1], self.ranges[eye]["pitch"]))
        return out                                  # (yaw, pitch)

    def step(self, lm_left=None, lm_right=None, widen=(0.0, 0.0), lid_close=(0.0, 0.0),
             squint=(0.0, 0.0), now=None):
        """lid_close: max(filtered, raw) lid-close per eye; widen: neutral-rescaled ch0;
        squint: raw squint per eye (only used with squint_comp > 0)."""
        present = (lm_left is not None, lm_right is not None)
        self.features = [None, None]
        g = list(self.prev)                         # pitchL, pitchR, yawL, yawR
        valid = [False, False]
        for eye, lm in ((0, lm_left), (1, lm_right)):
            if lm is None or lid_close[eye] >= 0.95:
                continue
            r = self.eye_gaze(eye, lm, widen[eye])
            if r is not None:
                g[2 + eye], g[eye] = r[0], r[1]
                valid[eye] = True
        p = self.prev
        if g[2] > 0.7:                              # nasal extreme damping (left eye)
            g[2], g[0] = 0.9 * p[2] + 0.1 * g[2], 0.9 * p[0] + 0.1 * g[0]
        if g[3] < -0.7:                             # nasal extreme damping (right eye)
            g[3], g[1] = 0.9 * p[3] + 0.1 * g[3], 0.9 * p[1] + 0.1 * g[1]
        if present[0] and not present[1]:
            g[1], g[3] = g[0], g[2]
        elif present[1] and not present[0]:
            g[0], g[2] = g[1], g[3]
        wl = lead_weight(-g[2]) if valid[0] else 0.0
        wr = lead_weight(g[3]) if valid[1] else 0.0
        if wl > 0.0 or wr > 0.0:                    # temporal-side eye leads, the other follows
            eq = abs(wl - wr) <= 1e-5
            if wl > wr or (eq and wl > 0 and abs(g[2]) > abs(g[3])):
                g[3], g[1] = lerp(g[3], g[2], wl), lerp(g[1], g[0], wl)
            elif wr > wl or (eq and wr > 0 and abs(g[3]) > abs(g[2])):
                g[2], g[0] = lerp(g[2], g[3], wr), lerp(g[0], g[1], wr)
        self.prev = list(g)
        k = self.squint_comp
        pitch = 0.5 * ((g[0] - clamp(squint[0], 0, 1) * k) + (g[1] - clamp(squint[1], 0, 1) * k))
        yl, yr = g[2], g[3]
        if yr > yl:                                 # divergence is removed
            yl = yr = 0.5 * (yl + yr)
        else:                                       # convergence is limited
            t1 = clamp(0.5 * yr + 0.5, 0.0, 1.0)
            t2 = clamp(-0.5 * yl + 0.5, 0.0, 1.0)
            yl, yr = yl * (1 - t1) + t1 * yr, yr * (1 - t2) + t2 * yl
        pitch, yl, yr = (f(v, now) for f, v in zip(self.filters, (pitch, yl, yr)))
        self.filtered = (pitch, yl, yr)             # this+0x1138 / 0x113C / 0x1140
        a = self.amplitude
        return {"EyePitch_L": clamp(pitch * a, -1, 1), "EyePitch_R": clamp(pitch * a, -1, 1),
                "EyeYaw_L": clamp(yl * a, -1, 1), "EyeYaw_R": clamp(yr * a, -1, 1)}


# ---------------------------------------------------------------- INI loading + self-test
def load_meshes(ini_path):
    vals = {}
    for line in open(ini_path, encoding="utf-8", errors="replace"):
        if "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip()
    meshes, valid, modes = [], [], []
    for side in ("Left", "Right"):
        mesh = []
        for i in range(13):
            key = "%s_Eyegaze_mesh_%d_" % (side, i)
            mesh.append(tuple(float(vals[key + f]) for f in ("raw_yaw", "raw_pitch", "target_yaw", "target_pitch")))
        meshes.append(mesh)
        valid.append(vals.get("%s_Eyegaze_mesh_valid" % side, "false").lower() == "true")
        modes.append(int(vals.get("%s_Eyegaze_feature_mode" % side, "2")))
    return meshes, valid, modes


def selftest(ini_path):
    meshes, valid, modes = load_meshes(ini_path)
    ok = True
    for eye, name in ((0, "left"), (1, "right")):
        mesh = meshes[eye]
        print("== %s eye: mesh_valid=%s feature_mode=%d" % (name, valid[eye], modes[eye]))
        worst = max(max(abs(a - b) for a, b in zip(mesh_map(e[0], e[1], mesh), (e[2], e[3]))) for e in mesh)
        print("   calibration points reproduced exactly: max error %.2e" % worst)
        ok &= worst < 1e-6
        inner, outer = build_ring(mesh, 1, 4), build_ring(mesh, 5, 8)
        print("   rings: inner=%s outer=%s" % (inner is not None, outer is not None))
        for ring, label in ((inner, "inner"), (outer, "outer")):
            if ring is None:
                ok = False
                continue
            err = max(max(abs(a - b) for a, b in zip(ring_eval(ring, el[0]), el[1:])) for el in ring)
            print("   %s ring passes through its points: max error %.2e" % (label, err))
            ok &= err < 1e-5
        c = mesh[0]
        xs = [e[0] for e in mesh]
        ys = [e[1] for e in mesh]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        step, n, fails, jumps = span / 120.0, 0, 0, 0.0
        prev_row = None
        for j in range(-80, 81):
            row = []
            for i in range(-80, 81):
                r = mesh_map(c[0] + i * step, c[1] + j * step, mesh)
                n += 1
                if r is None:
                    fails += 1
                    row.append(None)
                    continue
                ok &= -1 <= r[0] <= 1 and -1 <= r[1] <= 1
                if row and row[-1] is not None:
                    jumps = max(jumps, abs(r[0] - row[-1][0]), abs(r[1] - row[-1][1]))
                if prev_row and prev_row[len(row)] is not None:
                    jumps = max(jumps, abs(r[0] - prev_row[len(row)][0]), abs(r[1] - prev_row[len(row)][1]))
                row.append(r)
            prev_row = row
        print("   grid %d samples: %d unusable, largest neighbour jump %.3f" % (n, fails, jumps))
        for f in (0.5, 1.0, 1.5):
            e = mesh[5]
            x, y = c[0] + f * (e[0] - c[0]), c[1] + f * (e[1] - c[1])
            print("   %.1fx toward outer point 5 (target %+.2f,%+.2f): -> yaw %+.3f pitch %+.3f" %
                  ((f, e[2], e[3]) + mesh_map(x, y, mesh)))
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok
