"""VR outputs (next phase).

Planned modules, wire formats already verified (Cymple-technical-QA.md section 4):
  osc.py    Mode A: VRChat OSC on 127.0.0.1:9000, /avatar/parameters/v2/*, bundles <= 4096 B,
            native /tracking/eye/LeftRightPitchYaw; OSCQuery avatar discovery.
  vrcft.py  Mode B: 292-byte packet to 127.0.0.1:22999 (magic -3, BE flags, type 0, len 280,
            70 float32 LE in Appendix A order), at most every 10 ms.
Still unknown: the lid / squint math and the mapping of the 30 mouth outputs to those fields.
"""
