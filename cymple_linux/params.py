"""The 70-value expression array shared by both outputs (VRCFT UDP 22999 and direct OSC).

Order from the CympleFaceTracking module's Constants.blendShapeNames (.cctor IL), identical to
the native sender's packet layout (sub_1800c2820). Index 68's string is "TongueWide" but the
module stores it into TongueFlat.
"""
import struct

FIELDS = (
    "EyePitch_L", "EyePitch_R", "EyeYaw_L", "EyeYaw_R", "Eye_Pupil_Left", "Eye_Pupil_Right",
    "EyeLidCloseLeft", "EyeLidCloseRight", "EyeSquintLeft", "EyeSquintRight", "EyeWideLeft", "EyeWideRight",
    "BrowLowererLeft", "BrowLowererRight", "BrowInnerUpLeft", "BrowInnerUpRight", "BrowOuterUpLeft",
    "BrowOuterUpRight", "BrowPinchLeft", "BrowPinchRight", "NoseSneerLeft", "NoseSneerRight",
    "CheekSquintLeft", "CheekSquintRight", "CheekPuffLeft", "CheekPuffRight", "CheekSuckLeft", "CheekSuckRight",
    "JawLeft", "JawRight", "JawFwd", "JawOpen", "MouthClosed", "MouthUpperShiftLeft", "MouthUpperShiftRight",
    "MouthLowerShiftLeft", "MouthLowerShiftRight", "LipSuckUL", "LipSuckUR", "LipSuckDL", "LipSuckDR",
    "MouthFunnelUL", "MouthFunnelUR", "MouthFunnelDL", "MouthFunnelDR", "MouthPuckerUpper", "MouthPuckerLower",
    "LipRaiseUL", "LipRaiseUR", "LipDepressDL", "LipDepressDR", "MouthSmileLeft", "MouthSmileRight",
    "MouthDimpleLeft", "MouthDimpleRight", "MouthFrownLeft", "MouthFrownRight", "MouthStretchLeft",
    "MouthStretchRight", "MouthShrugUpper", "MouthShrugLower", "TongueOut", "TongueLeft", "TongueRight",
    "TongueUp", "TongueDown", "TongueCurlUp", "TongueBendDown", "TongueWide", "TongueRoll",
)
assert len(FIELDS) == 70
INDEX = {name: i for i, name in enumerate(FIELDS)}

FLAG_MOUTH, FLAG_EYE = 1, 2
VRCFT_MAGIC, VRCFT_TYPE, VRCFT_LEN = -3, 0, 280
VRCFT_SIZE = 12 + VRCFT_LEN                       # int32 LE magic, u32 BE flags, u16 type, u16 len, 70 f32 LE


def decode_vrcft(data):
    """(flags, [70 floats]) from a 292-byte packet, or None if it is not one."""
    if len(data) < VRCFT_SIZE:
        return None
    (magic,) = struct.unpack_from("<i", data, 0)
    (flags,) = struct.unpack_from(">I", data, 4)
    ptype, _length = struct.unpack_from("<HH", data, 8)
    if magic != VRCFT_MAGIC or ptype != VRCFT_TYPE:
        return None
    return flags, list(struct.unpack_from("<70f", data, 12))


def encode_vrcft(flags, values):
    return (struct.pack("<i", VRCFT_MAGIC) + struct.pack(">I", flags)
            + struct.pack("<HH70f", VRCFT_TYPE, VRCFT_LEN, *values))
