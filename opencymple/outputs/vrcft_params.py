"""VRCFaceTracking's avatar parameters, computed from Cymple's 70-value array.

Port of VRCFaceTracking.Core: Params/Expressions/UnifiedExpressionsParameters.cs (v2/*),
Params/Expressions/Legacy/Eye/EyeTrackingParams.cs, Legacy/Lip (LipShapeMerger, UnifiedSRanMapper,
LipShapeConversion), NativeParameters.cs, UnifiedSimplifier and the OSC/DataTypes parameter classes.
Cymple's direct-OSC sender (sub_1800ec6a0) reproduces the same names and formulas; how it fills the
Unified shapes from its 70 fields (MOUTH_SHAPES) was read back from its OSC output on corpus1.
Head pose parameters are left out (no head data).

Relevance follows VRCFT's BaseParam.ResetParam: a parameter is sent only when the avatar has a
parameter of the same type whose address ends in "/<name>" (not preceded by "v<digits>", so
"EyeSquintLeft" does not catch ".../v2/EyeSquintLeft"), and it is sent to that address. Binary
parameters drive the avatar's "<name>1/2/4/..." bools and "<name>Negative".
"""
import math
import re
import struct

from ..params import FIELDS

PREFIX = "/avatar/parameters/"
SIDES = ("Left", "Right")
EYE_SHAPE_BASES = ("EyeSquint", "EyeWide", "BrowLowerer", "BrowInnerUp", "BrowOuterUp", "BrowPinch")
MOUTH_FIELDS = frozenset(FIELDS[20:])


def f32(v):
    """Round to float32, the type VRCFT compares and sends."""
    return struct.unpack("<f", struct.pack("<f", v))[0]


class Unified:
    """The part of VRCFT's UnifiedTrackingData the parameters read."""

    def __init__(self):
        self.gaze = [(0.0, 0.0), (0.0, 0.0)]        # per eye, Vector2(x = yaw, y = pitch)
        self.openness = [1.0, 1.0]
        self.pupil = [0.0, 0.0]
        self.shapes = {}
        self.eye_active = False
        self.face_active = False

    def s(self, name):
        return self.shapes.get(name, 0.0)

    def combined_gaze(self):
        (lx, ly), (rx, ry) = self.gaze
        return ((lx + rx) / 2.0, (ly + ry) / 2.0)

    def combined_openness(self):
        return (self.openness[0] + self.openness[1]) / 2.0

    def dilation(self):
        # VRCFT normalizes against a running min/max (and returns 1.0 before any change);
        # Cymple's pupil fields are always 0 and it sends 0, so keep the plain average.
        return (self.pupil[0] + self.pupil[1]) / 2.0


# UnifiedExpressions shape <- Cymple field (a name, or a pair to average), as Cymple's OSC output
# has them: CornerPull = CornerSlant = Smile, MouthUpperUp = MouthUpperDeepen = LipRaise,
# LipSuckCorner = mean of upper and lower, TongueFlat <- field 68 ("TongueWide"). Shapes not listed
# (nasal, JawBackward/Clench/MandibleRaise, Press, Tightener, TongueSquish/Twist, palate, throat,
# neck) stay 0.
MOUTH_SHAPES = {
    "CheekSquintRight": "CheekSquintRight", "CheekSquintLeft": "CheekSquintLeft",
    "CheekPuffRight": "CheekPuffRight", "CheekPuffLeft": "CheekPuffLeft",
    "CheekSuckRight": "CheekSuckRight", "CheekSuckLeft": "CheekSuckLeft",
    "JawOpen": "JawOpen", "JawRight": "JawRight", "JawLeft": "JawLeft", "JawForward": "JawFwd",
    "MouthClosed": "MouthClosed",
    "LipSuckUpperRight": "LipSuckUR", "LipSuckUpperLeft": "LipSuckUL",
    "LipSuckLowerRight": "LipSuckDR", "LipSuckLowerLeft": "LipSuckDL",
    "LipSuckCornerRight": ("LipSuckUR", "LipSuckDR"), "LipSuckCornerLeft": ("LipSuckUL", "LipSuckDL"),
    "LipFunnelUpperRight": "MouthFunnelUR", "LipFunnelUpperLeft": "MouthFunnelUL",
    "LipFunnelLowerRight": "MouthFunnelDR", "LipFunnelLowerLeft": "MouthFunnelDL",
    "LipPuckerUpperRight": "MouthPuckerUpper", "LipPuckerUpperLeft": "MouthPuckerUpper",
    "LipPuckerLowerRight": "MouthPuckerLower", "LipPuckerLowerLeft": "MouthPuckerLower",
    "MouthUpperUpRight": "LipRaiseUR", "MouthUpperUpLeft": "LipRaiseUL",
    "MouthUpperDeepenRight": "LipRaiseUR", "MouthUpperDeepenLeft": "LipRaiseUL",
    "NoseSneerRight": "NoseSneerRight", "NoseSneerLeft": "NoseSneerLeft",
    "MouthLowerDownRight": "LipDepressDR", "MouthLowerDownLeft": "LipDepressDL",
    "MouthUpperRight": "MouthUpperShiftRight", "MouthUpperLeft": "MouthUpperShiftLeft",
    "MouthLowerRight": "MouthLowerShiftRight", "MouthLowerLeft": "MouthLowerShiftLeft",
    "MouthCornerPullRight": "MouthSmileRight", "MouthCornerPullLeft": "MouthSmileLeft",
    "MouthCornerSlantRight": "MouthSmileRight", "MouthCornerSlantLeft": "MouthSmileLeft",
    "MouthFrownRight": "MouthFrownRight", "MouthFrownLeft": "MouthFrownLeft",
    "MouthStretchRight": "MouthStretchRight", "MouthStretchLeft": "MouthStretchLeft",
    "MouthDimpleRight": "MouthDimpleRight", "MouthDimpleLeft": "MouthDimpleLeft",
    "MouthRaiserUpper": "MouthShrugUpper", "MouthRaiserLower": "MouthShrugLower",
    "TongueOut": "TongueOut", "TongueUp": "TongueUp", "TongueDown": "TongueDown",
    "TongueRight": "TongueRight", "TongueLeft": "TongueLeft", "TongueRoll": "TongueRoll",
    "TongueBendDown": "TongueBendDown", "TongueCurlUp": "TongueCurlUp", "TongueFlat": "TongueWide",
}


def unified_from_fields(fields, eye_active=True, face_active=None):
    """Fields by name (packet order of params.FIELDS; missing ones count as 0) -> Unified.
    Gaze = (yaw, pitch) and openness = 1 - lid close, as Cymple's OSC output shows
    (v2/EyeLeftX = EyeYaw_L). face_active defaults to "mouth fields present"."""
    g = lambda k: fields.get(k, 0.0)
    u = Unified()
    for i, side in enumerate(SIDES):
        s = side[0]
        u.gaze[i] = (g("EyeYaw_" + s), g("EyePitch_" + s))
        u.openness[i] = 1.0 - g("EyeLidClose" + side)
        u.pupil[i] = g("Eye_Pupil_" + side)
        for base in EYE_SHAPE_BASES:
            u.shapes[base + side] = g(base + side)
    for shape, src in MOUTH_SHAPES.items():
        u.shapes[shape] = (g(src[0]) + g(src[1])) / 2.0 if isinstance(src, tuple) else g(src)
    u.eye_active = eye_active
    u.face_active = (not MOUTH_FIELDS.isdisjoint(fields)) if face_active is None else face_active
    return u


# ---- parameter kinds ------------------------------------------------------------------------

class Param:
    """VRCFT BaseParam<T>: one value of type bool or float under `name`."""
    __slots__ = ("name", "type", "fn")

    def __init__(self, name, typ, fn):
        self.name, self.type, self.fn = name, typ, fn


class Binary:
    """VRCFT BinaryBaseParameter: the magnitude spread over the avatar's <name>2^k bools."""
    __slots__ = ("name", "fn")

    def __init__(self, name, fn):
        self.name, self.fn = name, fn


class Native:
    """VRCFT NativeParameter: VRChat's built-in /tracking/eye/* endpoints, used only when the
    avatar has no matching eye parameters of its own (`kind` "gaze" or "lid")."""
    __slots__ = ("address", "kind", "fn")

    def __init__(self, address, kind, fn):
        self.address, self.kind, self.fn = address, kind, fn


def eparam(name, fn, threshold=0.5, binary=True):
    """VRCFT EParam: bool (value < threshold), float and (optionally) binary variants."""
    out = [Param(name, bool, lambda u: fn(u) < threshold), Param(name, float, fn)]
    if binary:
        out.append(Binary(name, fn))
    return out


def eparam2(name, fn):
    """VRCFT EParam over a Vector2: <name>X and <name>Y."""
    return eparam(name + "X", lambda u: fn(u)[0]) + eparam(name + "Y", lambda u: fn(u)[1])


def _max(a, b):
    return a if a > b else b                    # VRCFT's "a > b ? a : b"


def brow_up(u, side):                           # UnifiedSimplifier
    return u.s("BrowOuterUp" + side) * 0.6 + u.s("BrowInnerUp" + side) * 0.4


def brow_down(u, side):
    return u.s("BrowLowerer" + side) * 0.75 + u.s("BrowPinch" + side) * 0.25


def brow_expression(u, side):
    return min(1.0, u.s("BrowInnerUp" + side) * 0.5 + u.s("BrowOuterUp" + side) * 0.5) - brow_down(u, side)


def lid_expanded(u, i):
    return u.s("EyeWide" + SIDES[i]) * 0.2 + u.openness[i] * 0.8


def squeeze(u, i):
    return (1.0 - u.openness[i] ** 0.15) * u.s("EyeSquint" + SIDES[i])


def _lid_expanded_binary(u, i):
    if lid_expanded(u, i) > 0.8:
        return u.s("EyeWide" + SIDES[i])
    return u.openness[i]


def _lid_expanded_squeeze_binary(u, i):
    lid = lid_expanded(u, i) - squeeze(u, i)
    if lid > 0.8:
        return u.s("EyeWide" + SIDES[i])
    if lid >= 0:
        return u.openness[i]
    return squeeze(u, i)


def _combined_lid_expanded_squeeze_binary(u):
    lid = ((u.s("EyeWideLeft") + u.s("EyeWideRight")) * 0.2 + (u.openness[0] + u.openness[1]) * 0.8
           - squeeze(u, 0) - squeeze(u, 1)) * 0.5
    if lid > 0.8:
        return (u.s("EyeWideRight") + u.s("EyeWideLeft")) * 0.5
    if lid >= 0:
        return u.combined_openness()
    return (squeeze(u, 0) + squeeze(u, 1)) * 0.5


UNIFIED_EXPRESSIONS = (                     # enum UnifiedExpressions, in order (without Max)
    "EyeSquintRight", "EyeSquintLeft", "EyeWideRight", "EyeWideLeft", "BrowPinchRight", "BrowPinchLeft",
    "BrowLowererRight", "BrowLowererLeft", "BrowInnerUpRight", "BrowInnerUpLeft", "BrowOuterUpRight",
    "BrowOuterUpLeft", "NasalDilationRight", "NasalDilationLeft", "NasalConstrictRight", "NasalConstrictLeft",
    "CheekSquintRight", "CheekSquintLeft", "CheekPuffRight", "CheekPuffLeft", "CheekSuckRight", "CheekSuckLeft",
    "JawOpen", "JawRight", "JawLeft", "JawForward", "JawBackward", "JawClench", "JawMandibleRaise", "MouthClosed",
    "LipSuckUpperRight", "LipSuckUpperLeft", "LipSuckLowerRight", "LipSuckLowerLeft", "LipSuckCornerRight",
    "LipSuckCornerLeft", "LipFunnelUpperRight", "LipFunnelUpperLeft", "LipFunnelLowerRight", "LipFunnelLowerLeft",
    "LipPuckerUpperRight", "LipPuckerUpperLeft", "LipPuckerLowerRight", "LipPuckerLowerLeft", "MouthUpperUpRight",
    "MouthUpperUpLeft", "MouthUpperDeepenRight", "MouthUpperDeepenLeft", "NoseSneerRight", "NoseSneerLeft",
    "MouthLowerDownRight", "MouthLowerDownLeft", "MouthUpperRight", "MouthUpperLeft", "MouthLowerRight",
    "MouthLowerLeft", "MouthCornerPullRight", "MouthCornerPullLeft", "MouthCornerSlantRight", "MouthCornerSlantLeft",
    "MouthFrownRight", "MouthFrownLeft", "MouthStretchRight", "MouthStretchLeft", "MouthDimpleRight",
    "MouthDimpleLeft", "MouthRaiserUpper", "MouthRaiserLower", "MouthPressRight", "MouthPressLeft",
    "MouthTightenerRight", "MouthTightenerLeft", "TongueOut", "TongueUp", "TongueDown", "TongueRight",
    "TongueLeft", "TongueRoll", "TongueBendDown", "TongueCurlUp", "TongueSquish", "TongueFlat", "TongueTwistRight",
    "TongueTwistLeft", "SoftPalateClose", "ThroatSwallow", "NeckFlexRight", "NeckFlexLeft",
)


def smile(u, side):
    return u.s("MouthCornerPull" + side) * 0.8 + u.s("MouthCornerSlant" + side) * 0.2


def sad(u, side):
    return _max(u.s("MouthFrown" + side), u.s("MouthStretch" + side))


SIMPLE = (                                  # enum UnifiedSimpleExpressions -> UnifiedSimplifier
    ("BrowUpRight", lambda u: brow_up(u, "Right")), ("BrowUpLeft", lambda u: brow_up(u, "Left")),
    ("BrowDownRight", lambda u: brow_down(u, "Right")), ("BrowDownLeft", lambda u: brow_down(u, "Left")),
    ("MouthSmileRight", lambda u: smile(u, "Right")), ("MouthSmileLeft", lambda u: smile(u, "Left")),
    ("MouthSadRight", lambda u: sad(u, "Right")), ("MouthSadLeft", lambda u: sad(u, "Left")),
)


def _avg(*names):
    return lambda u: sum(u.s(n) for n in names) / len(names)


def _diff(a, b):
    return lambda u: u.s(a) - u.s(b)


def _v2():
    """UnifiedExpressionsParameters.ExpressionParameters: every v2/<UnifiedExpression> and
    v2/<UnifiedSimpleExpression> (bool threshold 0), then UnifiedCombinedShapes."""
    L, R = 0, 1
    p = []
    for n in UNIFIED_EXPRESSIONS:
        p += eparam("v2/" + n, lambda u, n=n: u.s(n), 0.0)
    for n, fn in SIMPLE:
        p += eparam("v2/" + n, fn, 0.0)
    p += eparam2("v2/Eye", lambda u: u.combined_gaze())
    p += eparam2("v2/EyeLeft", lambda u: u.gaze[L])
    p += eparam2("v2/EyeRight", lambda u: u.gaze[R])
    p += eparam("v2/PupilDilation", lambda u: u.dilation())
    p += eparam("v2/PupilDiameterLeft", lambda u: u.pupil[L] * 0.1)
    p += eparam("v2/PupilDiameterRight", lambda u: u.pupil[R] * 0.1)
    p += eparam("v2/PupilDiameter", lambda u: (u.pupil[L] + u.pupil[R]) * 0.05)
    p += eparam("v2/EyeOpenLeft", lambda u: u.openness[L])
    p += eparam("v2/EyeOpenRight", lambda u: u.openness[R])
    p += eparam("v2/EyeOpen", lambda u: u.combined_openness())
    p += eparam("v2/EyeClosedLeft", lambda u: 1 - u.openness[L])
    p += eparam("v2/EyeClosedRight", lambda u: 1 - u.openness[R])
    p += eparam("v2/EyeClosed", lambda u: 1 - u.combined_openness())
    p += eparam("v2/EyeWide", lambda u: _max(u.s("EyeWideLeft"), u.s("EyeWideRight")))
    p += eparam("v2/EyeLidLeft", lambda u: u.openness[L] * 0.75 + u.s("EyeWideLeft") * 0.25)
    p += eparam("v2/EyeLidRight", lambda u: u.openness[R] * 0.75 + u.s("EyeWideRight") * 0.25)
    p += eparam("v2/EyeLid", lambda u: u.combined_openness() * 0.75
                + (u.s("EyeWideRight") + u.s("EyeWideLeft")) / 2.0 * 0.25)
    p += eparam("v2/EyeSquint", lambda u: _max(u.s("EyeSquintLeft"), u.s("EyeSquintRight")))
    p += eparam("v2/EyesSquint", lambda u: _max(u.s("EyeSquintLeft"), u.s("EyeSquintRight")))
    p += eparam("v2/BrowUp", lambda u: (brow_up(u, "Right") + brow_up(u, "Left")) * 0.5)
    p += eparam("v2/BrowDown", lambda u: (brow_down(u, "Right") + brow_down(u, "Left")) * 0.5)
    p += eparam("v2/BrowInnerUp", _avg("BrowInnerUpLeft", "BrowInnerUpRight"))
    p += eparam("v2/BrowOuterUp", _avg("BrowOuterUpLeft", "BrowOuterUpRight"))
    p += eparam("v2/BrowExpressionRight", lambda u: brow_expression(u, "Right"))
    p += eparam("v2/BrowExpressionLeft", lambda u: brow_expression(u, "Left"))
    p += eparam("v2/BrowExpression", lambda u: (brow_expression(u, "Right") + brow_expression(u, "Left")) * 0.5)
    p += eparam("v2/JawX", _diff("JawRight", "JawLeft"))
    p += eparam("v2/JawZ", _diff("JawForward", "JawBackward"))
    p += eparam("v2/CheekSquint", _avg("CheekSquintLeft", "CheekSquintRight"))
    p += eparam("v2/CheekPuffSuckLeft", _diff("CheekPuffLeft", "CheekSuckLeft"))
    p += eparam("v2/CheekPuffSuckRight", _diff("CheekPuffRight", "CheekSuckRight"))
    p += eparam("v2/CheekPuffSuck", lambda u: (u.s("CheekPuffRight") + u.s("CheekPuffLeft")) / 2.0
                - (u.s("CheekSuckRight") + u.s("CheekSuckLeft")) / 2.0)
    p += eparam("v2/CheekSuck", _avg("CheekSuckLeft", "CheekSuckRight"))
    p += eparam("v2/MouthUpperX", _diff("MouthUpperRight", "MouthUpperLeft"))
    p += eparam("v2/MouthLowerX", _diff("MouthLowerRight", "MouthLowerLeft"))
    p += eparam("v2/MouthX", lambda u: (u.s("MouthUpperRight") + u.s("MouthLowerRight")) / 2.0
                - (u.s("MouthUpperLeft") + u.s("MouthLowerLeft")) / 2.0)
    p += eparam("v2/LipSuckUpper", _avg("LipSuckUpperRight", "LipSuckUpperLeft"))
    p += eparam("v2/LipSuckLower", _avg("LipSuckLowerRight", "LipSuckLowerLeft"))
    p += eparam("v2/LipSuck", _avg("LipSuckUpperRight", "LipSuckUpperLeft", "LipSuckLowerRight", "LipSuckLowerLeft"))
    p += eparam("v2/LipFunnelUpper", _avg("LipFunnelUpperRight", "LipFunnelUpperLeft"))
    p += eparam("v2/LipFunnelLower", _avg("LipFunnelLowerRight", "LipFunnelLowerLeft"))
    p += eparam("v2/LipFunnel", _avg("LipFunnelUpperRight", "LipFunnelUpperLeft", "LipFunnelLowerRight",
                                     "LipFunnelLowerLeft"))
    p += eparam("v2/LipPuckerUpper", _avg("LipPuckerUpperRight", "LipPuckerUpperLeft"))
    p += eparam("v2/LipPuckerLower", _avg("LipPuckerLowerRight", "LipPuckerLowerLeft"))
    p += eparam("v2/LipPuckerRight", _avg("LipPuckerUpperRight", "LipPuckerLowerRight"))
    p += eparam("v2/LipPuckerLeft", _avg("LipPuckerUpperLeft", "LipPuckerLowerLeft"))
    p += eparam("v2/LipPucker", _avg("LipPuckerUpperRight", "LipPuckerUpperLeft", "LipPuckerLowerRight",
                                     "LipPuckerLowerLeft"))
    for v in ("Upper", "Lower"):
        p += eparam("v2/LipSuckFunnel" + v, lambda u, v=v: (u.s("LipSuck%sRight" % v) + u.s("LipSuck%sLeft" % v)) / 2.0
                    - (u.s("LipFunnel%sRight" % v) + u.s("LipFunnel%sLeft" % v)) / 2.0)
    for v in ("LowerLeft", "LowerRight", "UpperLeft", "UpperRight"):
        p += eparam("v2/LipSuckFunnel" + v, _diff("LipSuck" + v, "LipFunnel" + v))
    p += eparam("v2/MouthUpperUp", _avg("MouthUpperUpRight", "MouthUpperUpLeft"))
    p += eparam("v2/MouthLowerDown", _avg("MouthLowerDownRight", "MouthLowerDownLeft"))
    p += eparam("v2/MouthOpen", _avg("MouthUpperUpRight", "MouthUpperUpLeft", "MouthLowerDownRight",
                                     "MouthLowerDownLeft"))
    p += eparam("v2/MouthStretch", _avg("MouthStretchRight", "MouthStretchLeft"))
    p += eparam("v2/MouthTightener", _avg("MouthTightenerRight", "MouthTightenerLeft"))
    p += eparam("v2/MouthPress", _avg("MouthPressRight", "MouthPressLeft"))
    p += eparam("v2/MouthDimple", _avg("MouthDimpleRight", "MouthDimpleLeft"))
    p += eparam("v2/NoseSneer", _avg("NoseSneerRight", "NoseSneerLeft"))
    p += eparam("v2/MouthTightenerStretch", lambda u: (u.s("MouthTightenerRight") + u.s("MouthTightenerLeft")) / 2.0
                - (u.s("MouthStretchRight") + u.s("MouthStretchLeft")) / 2.0)
    p += eparam("v2/MouthTightenerStretchLeft", _diff("MouthTightenerLeft", "MouthStretchLeft"))
    p += eparam("v2/MouthTightenerStretchRight", _diff("MouthTightenerRight", "MouthStretchRight"))
    p += eparam("v2/MouthCornerYLeft", _diff("MouthCornerSlantLeft", "MouthFrownLeft"))
    p += eparam("v2/MouthCornerYRight", _diff("MouthCornerSlantRight", "MouthFrownRight"))
    p += eparam("v2/MouthCornerY", lambda u: (u.s("MouthCornerSlantLeft") - u.s("MouthFrownLeft")
                                              + u.s("MouthCornerSlantRight") - u.s("MouthFrownRight")) * 0.5)
    p += eparam("v2/SmileFrownRight", lambda u: smile(u, "Right") - u.s("MouthFrownRight"))
    p += eparam("v2/SmileFrownLeft", lambda u: smile(u, "Left") - u.s("MouthFrownLeft"))
    p += eparam("v2/SmileFrown", lambda u: smile(u, "Right") * 0.5 + smile(u, "Left") * 0.5
                - u.s("MouthFrownRight") * 0.5 - u.s("MouthFrownLeft") * 0.5)
    p += eparam("v2/SmileSadRight", lambda u: smile(u, "Right") - sad(u, "Right"))
    p += eparam("v2/SmileSadLeft", lambda u: smile(u, "Left") - sad(u, "Left"))
    p += eparam("v2/SmileSad", lambda u: (smile(u, "Left") + smile(u, "Right")) / 2.0
                - (sad(u, "Left") + sad(u, "Right")) / 2.0)
    p += eparam("v2/TongueX", _diff("TongueRight", "TongueLeft"))
    p += eparam("v2/TongueY", _diff("TongueUp", "TongueDown"))
    p += eparam("v2/TongueArchY", _diff("TongueCurlUp", "TongueBendDown"))
    p += eparam("v2/TongueShape", _diff("TongueFlat", "TongueSquish"))
    p.append(Param("EyeTrackingActive", bool, lambda u: u.eye_active))
    p.append(Param("ExpressionTrackingActive", bool, lambda u: u.face_active))
    p.append(Param("LipTrackingActive", bool, lambda u: u.face_active))
    return p


# ---- legacy SRanipal lip shapes (Legacy/Lip) ------------------------------------------------------
# Cymple follows an older VRCFT here; its OSC output on corpus1 shows (and we do the same):
# SRanipal JawOpen = JawOpen (no "- MouthClosed"), MouthUpperUp/MouthLowerDown without the funnel
# term, MouthSad = max(MouthFrown, MouthStretch) of that side without subtracting the smile, every
# averaged merge averages (no useMax), and TongueSteps = step1 - step2.

def _sran_smile(u, side):
    return _max(smile(u, side), u.s("MouthDimple" + side))


def _sran_sad(u, side):
    return _max(u.s("MouthFrown" + side), u.s("MouthStretch" + side))


SRANIPAL = (                                # enum SRanipal_LipShape_v2 -> UnifiedSRanMapper (without Max)
    ("JawRight", lambda u: u.s("JawRight")), ("JawLeft", lambda u: u.s("JawLeft")),
    ("JawForward", lambda u: u.s("JawForward")),
    ("JawOpen", lambda u: u.s("JawOpen")),
    ("MouthApeShape", lambda u: u.s("MouthClosed")),
    ("MouthUpperRight", lambda u: u.s("MouthUpperRight")), ("MouthUpperLeft", lambda u: u.s("MouthUpperLeft")),
    ("MouthLowerRight", lambda u: u.s("MouthLowerRight")), ("MouthLowerLeft", lambda u: u.s("MouthLowerLeft")),
    ("MouthUpperOverturn", _avg("LipFunnelUpperLeft", "LipFunnelUpperRight")),
    ("MouthLowerOverturn", _avg("LipFunnelLowerLeft", "LipFunnelLowerRight")),
    ("MouthPout", _avg("LipPuckerUpperLeft", "LipPuckerUpperRight", "LipPuckerLowerLeft", "LipPuckerLowerRight")),
    ("MouthSmileRight", lambda u: _sran_smile(u, "Right")), ("MouthSmileLeft", lambda u: _sran_smile(u, "Left")),
    ("MouthSadRight", lambda u: _sran_sad(u, "Right")), ("MouthSadLeft", lambda u: _sran_sad(u, "Left")),
    ("CheekPuffRight", lambda u: u.s("CheekPuffRight")), ("CheekPuffLeft", lambda u: u.s("CheekPuffLeft")),
    ("CheekSuck", _avg("CheekSuckLeft", "CheekSuckRight")),
    ("MouthUpperUpRight", lambda u: u.s("MouthUpperUpRight")), ("MouthUpperUpLeft", lambda u: u.s("MouthUpperUpLeft")),
    ("MouthLowerDownRight", lambda u: u.s("MouthLowerDownRight")),
    ("MouthLowerDownLeft", lambda u: u.s("MouthLowerDownLeft")),
    ("MouthUpperInside", lambda u: max(0.0, (u.s("LipSuckUpperLeft") + u.s("LipSuckUpperRight")) / 2.0)),
    ("MouthLowerInside", lambda u: max(0.0, (u.s("LipSuckLowerLeft") + u.s("LipSuckLowerRight")) / 2.0)),
    ("MouthLowerOverlay", lambda u: u.s("MouthRaiserLower")),
    ("TongueLongStep1", lambda u: min(1.0, u.s("TongueOut") * 2.0)),
    ("TongueLongStep2", lambda u: min(1.0, max(0.0, u.s("TongueOut") * 2.0 - 1.0))),
    ("TongueDown", lambda u: u.s("TongueDown")), ("TongueUp", lambda u: u.s("TongueUp")),
    ("TongueRight", lambda u: u.s("TongueRight")), ("TongueLeft", lambda u: u.s("TongueLeft")),
    ("TongueRoll", lambda u: u.s("TongueRoll")),
    ("TongueUpLeftMorph", lambda u: 0.0), ("TongueUpRightMorph", lambda u: 0.0),
    ("TongueDownLeftMorph", lambda u: 0.0), ("TongueDownRightMorph", lambda u: 0.0),
)
_SR = dict(SRANIPAL)


def _pn(pos, neg):                          # PositiveNegativeShape
    return lambda u: _SR[pos](u) - _SR[neg](u)


def _pna(pos, neg):                         # PositiveNegativeAveragedShape (always averaged, see above)
    return lambda u: sum(_SR[n](u) for n in pos) / len(pos) - sum(_SR[n](u) for n in neg) / len(neg)


def _merged():
    """LipShapeMerger.MergedShapes, in order."""
    UP, LO, LR = ("MouthUpperUpLeft", "MouthUpperUpRight"), ("MouthLowerDownLeft", "MouthLowerDownRight"), \
        ("MouthSmileLeft", "MouthSmileRight")
    PUFF, OVER = ("CheekPuffRight", "CheekPuffLeft"), ("MouthUpperOverturn", "MouthLowerOverturn")
    m = [("JawX", _pn("JawRight", "JawLeft")), ("MouthUpper", _pn("MouthUpperRight", "MouthUpperLeft")),
         ("MouthLower", _pn("MouthLowerRight", "MouthLowerLeft")),
         ("MouthX", _pna(("MouthUpperRight", "MouthLowerRight"), ("MouthUpperLeft", "MouthLowerLeft"))),
         ("SmileSadRight", _pn("MouthSmileRight", "MouthSadRight")),
         ("SmileSadLeft", _pn("MouthSmileLeft", "MouthSadLeft")),
         ("SmileSad", _pna(("MouthSmileLeft", "MouthSmileRight"), ("MouthSadLeft", "MouthSadRight"))),
         ("TongueY", _pn("TongueUp", "TongueDown")), ("TongueX", _pn("TongueRight", "TongueLeft")),
         ("PuffSuckRight", _pn("CheekPuffRight", "CheekSuck")), ("PuffSuckLeft", _pn("CheekPuffLeft", "CheekSuck")),
         ("PuffSuck", _pna(("CheekPuffLeft", "CheekPuffRight"), ("CheekSuck",))),
         ("JawOpenApe", _pn("JawOpen", "MouthApeShape")),
         ("JawOpenPuff", _pna(("JawOpen",), ("CheekPuffLeft", "CheekPuffRight"))),
         ("JawOpenPuffRight", _pn("JawOpen", "CheekPuffRight")), ("JawOpenPuffLeft", _pn("JawOpen", "CheekPuffLeft")),
         ("JawOpenSuck", _pn("JawOpen", "CheekSuck")), ("JawOpenForward", _pn("JawOpen", "JawForward")),
         ("JawOpenOverlay", _pn("JawOpen", "MouthLowerOverlay"))]
    for lip, name, inside in (("MouthUpperUp", "UpperInside", "MouthUpperInside"),
                              ("MouthLowerDown", "LowerInside", "MouthLowerInside")):
        for side in ("Right", "Left"):
            s = lip + side
            m += [(s + name, _pn(s, inside)), (s + "Puff" + side, _pn(s, "CheekPuff" + side)),
                  (s + "Ape", _pn(s, "MouthApeShape")), (s + "Pout", _pn(s, "MouthPout")),
                  (s + "Overlay", _pn(s, "MouthLowerOverlay")), (s + "Suck", _pn(s, "CheekSuck"))]
        both = UP if lip == "MouthUpperUp" else LO
        m += [(lip + name, _pna(both, (inside,))),
              (lip + "Inside", _pna(both, ("MouthUpperInside", "MouthLowerInside"))),
              (lip + "Puff", _pna(both, ("CheekPuffLeft", "CheekPuffRight"))),
              (lip + "PuffLeft", _pna(both, ("CheekPuffLeft",))), (lip + "PuffRight", _pna(both, ("CheekPuffRight",))),
              (lip + "Ape", _pna(both, ("MouthApeShape",))), (lip + "Pout", _pna(both, ("MouthPout",))),
              (lip + "Overlay", _pna(both, ("MouthLowerOverlay",))), (lip + "Suck", _pna(both, ("CheekSuck",)))]
    m += [("MouthUpperInsideOverturn", _pn("MouthUpperInside", "MouthUpperOverturn")),
          ("MouthLowerInsideOverturn", _pn("MouthLowerInside", "MouthLowerOverturn"))]
    for side in ("Right", "Left"):
        s = "MouthSmile" + side
        m += [("Smile%sUpperOverturn" % side, _pn(s, "MouthUpperOverturn")),
              ("Smile%sLowerOverturn" % side, _pn(s, "MouthLowerOverturn")),
              ("Smile%sOverturn" % side, _pna((s,), OVER)), ("Smile%sApe" % side, _pn(s, "MouthApeShape")),
              ("Smile%sOverlay" % side, _pn(s, "MouthLowerOverlay")), ("Smile%sPout" % side, _pn(s, "MouthPout"))]
    m += [("SmileUpperOverturn", _pna(LR, ("MouthUpperOverturn",))),
          ("SmileLowerOverturn", _pna(LR, ("MouthLowerOverturn",))), ("SmileOverturn", _pna(LR, OVER)),
          ("SmileApe", _pna(LR, ("MouthApeShape",))), ("SmileOverlay", _pna(LR, ("MouthLowerOverlay",))),
          ("SmilePout", _pna(LR, ("MouthPout",)))]
    for side in ("Right", "Left"):
        s = "CheekPuff" + side
        m += [("Puff%sUpperOverturn" % side, _pn(s, "MouthUpperOverturn")),
              ("Puff%sLowerOverturn" % side, _pn(s, "MouthLowerOverturn")),
              ("Puff%sOverturn" % side, _pna((s,), OVER))]
    m += [("PuffUpperOverturn", _pna(PUFF, ("MouthUpperOverturn",))),
          ("PuffLowerOverturn", _pna(PUFF, ("MouthLowerOverturn",))), ("PuffOverturn", _pna(PUFF, OVER)),
          ("TongueSteps", _pn("TongueLongStep1", "TongueLongStep2"))]
    return m


def _legacy_lip():
    """LipShapeMerger.AllLipParameters: every SRanipal shape, then the merged shapes (threshold 0)."""
    p = []
    for name, fn in SRANIPAL + tuple(_merged()):
        p += eparam(name, fn, 0.0)
    return p


def _legacy_eye():
    """EyeTrackingParams.ParameterList (with its Quest Pro cheek/mouth entries)."""
    L, R = 0, 1
    p = []
    p += eparam2("Eyes", lambda u: u.combined_gaze())
    p += eparam2("LeftEye", lambda u: u.gaze[L])
    p += eparam2("RightEye", lambda u: u.gaze[R])
    p += eparam("LeftEyeWiden", lambda u: u.s("EyeWideLeft"))
    p += eparam("RightEyeWiden", lambda u: u.s("EyeWideRight"))
    p += eparam("EyeWiden", lambda u: (u.s("EyeWideLeft") + u.s("EyeWideRight")) / 2.0)
    p += eparam("LeftEyeSqueeze", lambda u: u.s("EyeSquintLeft"))
    p += eparam("RightEyeSqueeze", lambda u: u.s("EyeSquintRight"))
    p += eparam("EyesSqueeze", lambda u: (u.s("EyeSquintLeft") + u.s("EyeSquintRight")) / 2.0)
    p += eparam("EyesDilation", lambda u: u.dilation())
    p += eparam("EyesPupilDiameter", lambda u: (u.pupil[L] + u.pupil[R]) * 0.5)
    p += eparam("LeftEyeLid", lambda u: u.openness[L])
    p += eparam("RightEyeLid", lambda u: u.openness[R])
    p += eparam("CombinedEyeLid", lambda u: u.combined_openness())
    p += eparam("LeftEyeLidExpanded", lambda u: lid_expanded(u, L), binary=False)
    p += eparam("RightEyeLidExpanded", lambda u: lid_expanded(u, R), binary=False)
    p += eparam("EyeLidExpanded", lambda u: (u.s("EyeWideLeft") + u.s("EyeWideRight")) * 0.1
                + (u.openness[L] + u.openness[R]) * 0.4, binary=False)
    p += eparam("LeftEyeLidExpandedSqueeze", lambda u: lid_expanded(u, L) - squeeze(u, L), binary=False)
    p += eparam("RightEyeLidExpandedSqueeze", lambda u: lid_expanded(u, R) - squeeze(u, R), binary=False)
    p += eparam("EyeLidExpandedSqueeze", lambda u: ((u.s("EyeWideLeft") + u.s("EyeWideRight")) * 0.2
                                                     + (u.openness[L] + u.openness[R]) * 0.8
                                                     - squeeze(u, L) - squeeze(u, R)) * 0.5, binary=False)
    p.append(Binary("LeftEyeLidExpanded", lambda u: _lid_expanded_binary(u, L)))
    p.append(Binary("RightEyeLidExpanded", lambda u: _lid_expanded_binary(u, R)))
    p.append(Binary("CombinedEyeLidExpanded", lambda u: (
        (u.s("EyeWideLeft") + u.s("EyeWideRight")) / 2.0 if (u.s("EyeWideLeft") + u.s("EyeWideRight")) / 2.0 > 0
        else u.combined_openness())))
    p.append(Binary("LeftEyeLidExpandedSqueeze", lambda u: _lid_expanded_squeeze_binary(u, L)))
    p.append(Binary("RightEyeLidExpandedSqueeze", lambda u: _lid_expanded_squeeze_binary(u, R)))
    p.append(Binary("CombinedEyeLidExpandedSqueeze", _combined_lid_expanded_squeeze_binary))
    p.append(Param("LeftEyeWidenToggle", bool, lambda u: lid_expanded(u, L) > 0.8))
    p.append(Param("RightEyeWidenToggle", bool, lambda u: lid_expanded(u, R) > 0.8))
    p.append(Param("EyesWidenToggle", bool, lambda u: (lid_expanded(u, R) + lid_expanded(u, L)) / 2.0 > 0.8))
    p.append(Param("LeftEyeSqueezeToggle", bool, lambda u: lid_expanded(u, L) - squeeze(u, L) < 0))
    p.append(Param("RightEyeSqueezeToggle", bool, lambda u: lid_expanded(u, R) - squeeze(u, R) < 0))
    p.append(Param("EyesSqueezeToggle", bool, lambda u: (lid_expanded(u, R) - squeeze(u, R)
                                                         + lid_expanded(u, L) - squeeze(u, L)) / 2.0 < 0))
    # Quest Pro legacy
    p += eparam("BrowsInnerUp", lambda u: _max(u.s("BrowInnerUpLeft"), u.s("BrowInnerUpRight")))
    p += eparam("BrowInnerUpLeft", lambda u: u.s("BrowInnerUpLeft"))
    p += eparam("BrowInnerUpRight", lambda u: u.s("BrowInnerUpRight"))
    p += eparam("BrowsOuterUp", lambda u: _max(u.s("BrowOuterUpLeft"), u.s("BrowOuterUpRight")))
    p += eparam("BrowOuterUpLeft", lambda u: u.s("BrowOuterUpLeft"))
    p += eparam("BrowOuterUpRight", lambda u: u.s("BrowOuterUpRight"))
    p += eparam("BrowsDown", lambda u: _max((u.s("BrowPinchLeft") + u.s("BrowLowererLeft")) / 2.0,
                                            (u.s("BrowPinchRight") + u.s("BrowLowererRight")) / 2.0))
    p += eparam("BrowDownLeft", lambda u: (u.s("BrowPinchLeft") + u.s("BrowLowererLeft")) / 2.0)
    p += eparam("BrowDownRight", lambda u: (u.s("BrowPinchRight") + u.s("BrowLowererRight")) / 2.0)
    p += eparam("MouthRaiserLower", lambda u: u.s("MouthRaiserLower"))
    p += eparam("MouthRaiserUpper", lambda u: u.s("MouthRaiserUpper"))
    p += eparam("EyesSquint", lambda u: (u.s("EyeSquintLeft") + u.s("EyeSquintRight")) / 2.0)
    p += eparam("EyeSquintLeft", lambda u: u.s("EyeSquintLeft"))
    p += eparam("EyeSquintRight", lambda u: u.s("EyeSquintRight"))
    for name, base in (("Cheeks", "CheekSquint"), ("Mouth", "MouthDimple"), ("Mouth", "MouthPress"),
                       ("Mouth", "MouthStretch"), ("Mouth", "MouthTightener"), ("Nose", "NoseSneer")):
        combined = "CheeksSquint" if name == "Cheeks" else base
        p += eparam(combined, _avg(base + "Left", base + "Right"))
        p += eparam(base + "Left", lambda u, n=base + "Left": u.s(n))
        p += eparam(base + "Right", lambda u, n=base + "Right": u.s(n))
    # Cymple's own status flags (not VRCFT): sent once with the tracking state
    p.append(Param("EyeTrackingEnabled", bool, lambda u: u.eye_active))
    p.append(Param("FaceTrackingEnabled", bool, lambda u: u.face_active))
    return p


def _deg(a):
    return math.degrees(math.atan(a))


NATIVE = (
    Native("/tracking/eye/LeftRightPitchYaw", "gaze", lambda u: (
        -_deg(u.gaze[0][1]), _deg(u.gaze[0][0]), -_deg(u.gaze[1][1]), _deg(u.gaze[1][0]))),
    Native("/tracking/eye/EyesClosedAmount", "lid", lambda u: 1 - u.combined_openness()),
)

_V2, _LEGACY_EYE = _v2(), _legacy_eye()
CATALOG = tuple(_V2 + _legacy_lip() + _legacy_eye())      # VRCFT AllParameters order
EYE_PARAM_NAMES = frozenset(n for q in _V2 + _LEGACY_EYE for n in (q.name, q.name + "Negative"))


# ---- relevance --------------------------------------------------------------------------------

_V_SUFFIX = re.compile(r"v\d+$")
_DIGITS = re.compile(r"(\d+)$")


def name_matches(address, name):
    """VRCFT's (?<!(v\\d+))(/NAME)$|^(NAME)$ without the variable-width lookbehind."""
    if address == name:
        return True
    tail = "/" + name
    return address.endswith(tail) and not _V_SUFFIX.search(address[:-len(tail)])


def binary_index(address, name):
    """Bit index k of an avatar bool "<...>/<name><2^k>", or None (GetBinarySteps)."""
    m = _DIGITS.search(address)
    if not m or not name_matches(address[:m.start()], name):
        return None
    n = int(m.group(1))
    return n.bit_length() - 1 if n > 0 and n & (n - 1) == 0 else None


def binary_bit(value, index, max_int, negative_relevant):
    """BinaryBaseParameter.ProcessBinary."""
    if value < 0 and not negative_relevant:
        return False
    value = abs(value)
    if value > 0.99999:
        return True
    return ((int(f32(value) * max_int) >> index) & 1) == 1


class Binding:
    """One resolved output: OSC address, type and value function."""
    __slots__ = ("address", "type", "fn")

    def __init__(self, address, typ, fn):
        self.address, self.type, self.fn = address, typ, fn

    def value(self, u):
        v = self.fn(u)
        if self.type is bool:
            return bool(v)
        if self.type is tuple:
            return tuple(f32(x) for x in v)
        return f32(v)


def resolve(avatar_params, full_face_tracking=False, catalog=CATALOG, native=NATIVE):
    """Bindings for an avatar's parameters [(address, type)], type in {bool, int, float}.
    avatar_params None = no avatar information: every float parameter goes to its default
    address (VRCFT's force-relevant mode, minus its bool variants, which would fight the floats
    on the same address) and VRChat's native eye endpoints are used."""
    out = []
    if avatar_params is None:
        out = [Binding(PREFIX + q.name, float, q.fn) for q in catalog if isinstance(q, Param) and q.type is float]
        out += [Binding(n.address, tuple if n.kind == "gaze" else float, n.fn) for n in native]
        return out
    if full_face_tracking:
        return out
    taken = set()
    for q in catalog:
        if isinstance(q, Param):
            for addr, typ in avatar_params:
                if typ is q.type and addr not in taken and name_matches(addr, q.name):
                    out.append(Binding(addr, q.type, q.fn))
                    taken.add(addr)
                    break
        else:
            neg = next((a for a, t in avatar_params if t is bool and name_matches(a, q.name + "Negative")), None)
            if neg is not None and neg not in taken:
                out.append(Binding(neg, bool, lambda u, fn=q.fn: fn(u) < 0))
                taken.add(neg)
            bits = []
            for addr, typ in avatar_params:
                if typ is bool and addr not in taken:
                    k = binary_index(addr, q.name)
                    if k is not None:
                        bits.append((addr, k))
            max_int = 2 ** len(bits) - 1        # Cymple: steps of 1/(2^n - 1); VRCFT uses 2^n
            for addr, k in bits:
                out.append(Binding(addr, bool, lambda u, fn=q.fn, k=k, m=max_int, n=neg is not None:
                                   binary_bit(fn(u), k, m, n)))
                taken.add(addr)
    for n in native:                            # VRCNativeParameters conditions
        words = ("X", "Y") if n.kind == "gaze" else ("Open", "Lid")
        own = [a for a, _ in avatar_params if "Eye" in a and any(w in a[len(PREFIX):] for w in words)]
        if not any(a.endswith(e) for a in own for e in EYE_PARAM_NAMES):
            out.append(Binding(n.address, tuple if n.kind == "gaze" else float, n.fn))
    return out
