# OpenCymple (opencymple-linux)

Native Linux runtime for Cymple eye/face trackers. It receives the ESP32 camera streams, runs
Cymple's own ONNX models (decrypted in memory from the installed `model/*.bin`), computes Cymple's
70-value array (eyes: gaze, lids, squint, wide, brows; mouth: jaw, lips, cheeks, tongue) and sends
it to VRChat over OSC as VRCFaceTracking-style avatar parameters. No Windows app, Wine or SteamVR
involved.

Status: receiver, inference, the whole 70-value array (eyes M3, mouth M4) and VRChat OSC for eyes
and mouth (M2, M5) are done, checked against Windows Cymple and verified in VRChat. The preview
window with Cymple's sliders and switches (M6, `opencymple-linux gui`) and Cymple's calibration flows
(M7: recenter with expression neutral, coarse range, 13-point gaze mesh) are done, and the 13-point
calibration runs in the headset as an OpenXR overlay (M8). Everyday use is the `opencymple` app: tray
icon, headless mode, single-instance commands, desktop entry and WiVRn's application list (M9).
VRCFT UDP 22999 is set aside for now.

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev,gui,xr]"
```

`xr` (pyopenxr, PyOpenGL) is only needed for the headset calibration overlay. Then
`.venv/bin/opencymple install` adds the desktop entry and icon and links `opencymple` into
`~/.local/bin` (see OpenCymple below).

The Cymple install is found automatically in the Bottles prefix (`iniFile.ini` for calibration,
`Program Files/Cymple/model` for the models). Override with `--ini` / `--models` or
`CYMPLE_INI` / `CYMPLE_MODELS`. Close the Windows Cymple app first: only one program can own UDP 23303.

Settings: the first `run`, `gui` or `calibrate-roi` copies Cymple's `iniFile.ini` and
`avatarParaments/` into `~/.config/opencymple-linux/` (`OPENCYMPLE_LINUX_CONFIG` overrides) and works on
that copy from then on, in Cymple's key format. Nothing here changes the Windows app's files
except `config export`, which backs them up first and refuses while Windows Cymple runs (it
rewrites its INI on exit). The oracle commands keep using Cymple's own INI, since the Windows app
runs with it. Our own options (VRChat target, window size) are in `opencymple-linux.json` there.

## Commands

```bash
.venv/bin/opencymple                                         # the app: window + tray icon
.venv/bin/opencymple --headless                              # tray icon only
.venv/bin/opencymple calibrate precise                       # 13-point calibration in the headset
.venv/bin/opencymple recenter                                # recenter both eyes (also: calibrate coarse, status, quit)
.venv/bin/opencymple install                                 # desktop entry, icon, ~/.local/bin link (uninstall removes them)
.venv/bin/opencymple-linux gui                                    # live, in a window (VRChat output as last set there)
.venv/bin/opencymple-linux gui --calibrate                        # ... and open the 13-point calibration in the headset
.venv/bin/opencymple-linux gui --replay captures/corpus1.cap --loop  # the window on a capture
.venv/bin/opencymple-linux config                                 # where our settings copy is, where it came from
.venv/bin/opencymple-linux config export --dry-run                # what export would write into Cymple's INI
.venv/bin/opencymple-linux config export                          # write our changed keys + amplitude files back
.venv/bin/opencymple-linux config import --force                  # re-copy Cymple's settings (ours kept as backup)
.venv/bin/opencymple-linux run                                    # live: beacon, receive, infer, status line each second
.venv/bin/opencymple-linux run --osc                              # ... and drive the VRChat avatar (OSC to 127.0.0.1:9000)
.venv/bin/opencymple-linux run --osc --osc-avatar none            # default addresses + VRChat native eye tracking only
.venv/bin/opencymple-linux run --debug-image /tmp/opencymple.png      # also render frames + ROI + landmarks every second
.venv/bin/opencymple-linux run --record session.cap               # save raw datagrams for later replay
.venv/bin/opencymple-linux run --replay captures/live-test-1.cap  # same pipeline from a capture
.venv/bin/opencymple-linux run --json                             # machine-readable status
.venv/bin/opencymple-linux record --name corpus1                  # guided, labelled recording (beep cues, ~3.5 min)
.venv/bin/opencymple-linux oracle-record captures/corpus1.cap     # replay into Windows Cymple (Bottles), log its 70 values
.venv/bin/opencymple-linux oracle-compare captures/corpus1.cap    # our deterministic replay vs that log, per field
.venv/bin/opencymple-linux oracle-record captures/corpus1.cap --mode osc   # same, Cymple's direct OSC instead
.venv/bin/opencymple-linux oracle-compare captures/corpus1.cap --golden captures/corpus1.oracle-osc.jsonl  # OSC formulas
.venv/bin/opencymple-linux models --bench 200                     # load all 5 models, show I/O, time them
.venv/bin/opencymple-linux calibrate-roi --write-ini              # eye ROI quads via model0 into our settings copy (backup kept)
.venv/bin/opencymple-linux selftest                               # gaze mesh math vs saved calibration
.venv/bin/python -m pytest
```

`--provider cuda|rocm|migraphx|openvino` picks another ONNX Runtime execution provider. That needs
the matching onnxruntime build. CPU is the default because it is fast enough. Measured on this machine,
single thread: eye landmark 1.0 ms, eye ROI 0.45 ms, mouth ROI 0.85 ms, mouth V1/V2 1.25 ms. Three
streams at 50 fps, including JPEG decode and preprocessing, use about 27% of one core. The open
window adds about 18% of one core (replay of corpus1, Values tab showing); minimized, it paints
nothing.

## Layout

| Module | Role |
|---|---|
| `protocol.py` | Datagram types 0/1/5/6, validation rules, stream → device map |
| `receiver.py` | `Ingest` (decode, reassembly, registry, stats), `UdpSource` (socket + beacon), `ReplaySource`, CYMPCAP1 captures |
| `models.py` | In-memory decryption, ORT sessions, provider selection |
| `imgproc.py` | JPEG → gray, the shared tensor recipe, quad warp, rotation, fixed mouth rect |
| `eye.py` | Per-eye landmark pipeline, ROI calibration with model0 |
| `mouth.py` | Mouth V1: rotation, auto-ROI scheduler with smoothing, model4 |
| `gaze.py` | Feature, 13-point mesh, binocular post-processing, One-Euro (moved from `opencymple_gaze.py`) |
| `calibration.py` | Reads Cymple's `iniFile.ini` format; writes keys in place (`update_ini`) |
| `settings.py` | Our copy in `~/.config/opencymple-linux/`: import, batched writes, amplitude sets, export |
| `runtime.py` | What `run` and `gui` share: models, pipeline, source, VRChat output, live setting changes, calibration runs |
| `calib.py` | Recenter + expression neutral, coarse range, 13-point gaze mesh (M7), writing Cymple's INI keys |
| `face.py` | Eye core (M3): lid close, squint, wide/brow fields, closed-eye hold, blink state machine; mouth core (M4): filters, mapping, amplitudes |
| `params.py` | The 70 field names, VRCFT 22999 packet encode/decode |
| `pipeline.py` | One worker thread per stream with a newest-frame mailbox; 10 ms packet timer feeding the outputs |
| `outputs/osc.py` | OSC encoder (bundles ≤ 4096 B like Cymple), sender |
| `outputs/vrcft_params.py` | VRCFaceTracking's avatar parameters (v2, legacy, binary, native) from the fields |
| `outputs/avatar.py` | Current avatar's parameter list: OSCQuery, else VRChat's OSC config files |
| `outputs/vrchat.py` | Mode A sender: relevance, 1e-4 deadband, refresh every 5 s |
| `oracle.py` | Runs Windows Cymple in Bottles on a capture and compares |
| `ui/app.py`, `ui/widgets.py` | The window (PySide6): previews, values, eye/mouth/amplitude pages, output |
| `ui/calibration.py` | Desktop target display for the 13-point calibration |
| `overlay.py` | Cymple's calibration overlay geometry: dot positions, stick steps, FOV limit (M8) |
| `ui/xr_overlay.py` | The 13-point calibration as an OpenXR overlay session (EGL, quad layers, controller actions) |
| `opencymple.py` | The `opencymple` app and its commands |
| `instance.py` | Single-instance socket client, background start |
| `desktop.py` | Desktop entry and icon (`opencymple install`) |
| `ui/shell.py` | Tray icon, command server, VRChat output and headset calibration without a window |
| `debugview.py` | Debug image for headless runs (`--debug-image`) |
| `cli.py` | Commands above |

## Pipeline as implemented

Facts from the earlier reports, plus these new ones from `cympleface.dll` (REA/Hopper, this session):

| Step | Behavior | Evidence |
|---|---|---|
| Stream → device | 0 = left eye, 1 = right eye, 2 = mouth V1 ("face tracker"), 3/4 = mouth V2 | `sub_1800b14e0`, names in `sub_1800b1e30`; eye 0 = "Eye left" decoder in `sub_18002c330` |
| Eye ROI at runtime | **Stored INI quad**, not per-frame model0. model0 runs only for ROI calibration ("saved and will be reused at runtime") | `sub_180032370`, `sub_18003fb60` |
| Eye crop | Valid quad → warpPerspective to 96×96 (INTER_LINEAR, border 0); else resize to 96×96 **INTER_CUBIC** | `sub_1800412a0`, `sub_18003fa60` |
| Right eye | Flip crop horizontally before model2, un-mirror landmarks after; for ROI calibration flip the whole frame first | `sub_18003f390`, `sub_18004cfe0` |
| Quad validity | All finite; edges 0-1, 2-3, 0-2, 1-3 longer than 0.02 | `sub_18003d570`, `sub_180037ab0` |
| Mouth rotation | warpAffine about (cols/2, rows/2), scale 1, skipped if \|angle\| ≤ 1e-4 | `sub_180035f80` |
| Mouth auto-ROI | model3 on the rotated frame at most every 1000 ms; quad += 0.25·(new − quad) | `sub_1800368a0`, constant at `0x1801d1130` |
| Mouth fixed ROI | Rect from `upper/bottom/width/x_offset` (floor/ceil, clamps) when no quad | `sub_1800370c0` |
| Source ownership | Each device kind sticks to one source IP; another IP is ignored until the owner has been silent 5 s | `sub_180090f80` |
| 70-value array | One array feeds both outputs; mouth fields = filtered model outputs m0–m29 (mapping in `sub_1800bee00`, confirmed per expression on `corpus1`), × `Amp_*` | `sub_1800c2820`, `sub_1800bee00`, `sub_1800bd5e0` |
| Mouth values | On each mouth frame the 30 outputs are clamped to [0, 1] (NaN → 0) and a "mouth active" flag is set for good; every 10 ms tick runs 30 One-Euro filters on the latest values (min cutoff 9 − 7t, β 0.12 − 0.09t, t = `filer`/0.5), clamps, maps; LipRaise/LipDepress × \|1 − m23\| | `sub_1800b7c30`, `sub_1800c5ea0`, `sub_1800bed10`, `sub_1800c5ba0`, `sub_180035960` |
| Amplitudes | 28 slots: `Amp_<name>Amp` for 23 mouth keys, then 5 eye keys, each clamped to [0, 5]. The consumer reads slots 14–18 for Smile, Dimple, Frown, Stretch, ShrugLower, but the key table has ShrugLower, Smile, Dimple, Frown, Stretch there: Smile is scaled by `Amp_ShrugLowerAmp`, Dimple by `Amp_MouthSmileAmp`, Frown by `Amp_MouthDimpleAmp`, Stretch by `Amp_MouthFrownAmp`, ShrugLower by `Amp_MouthStretchAmp`. We do the same | cympleFacelib `sub_18002ce20`, `sub_18002c1e0`, tables `0x18003b4c0`/`0x18003b578`; cympleface `sub_1800bd5e0` |
| VRCFT vs direct OSC | 22999 packet only when `[VRCDIRECT] Enabled = false` | `sub_1800d18d0`, keys at `0x1801e1cc8` |
| Inference device | `[Face Inference Device] DeviceId = 0` is CPU | `sub_18005c5c0` |

Validation: replaying `captures/live-test-1.cap` gives the same frame counts as the probe
(992/989/987) and every worker keeps up at 50 fps. On the eye dataset frames, model0 reproduces
the saved ROI quads to within 0.05 for both eyes, and the landmarks land on corners, lids and
pupil (`tests/test_with_install.py`).

Oracle (Windows Cymple on CPU, fed `corpus1`): 11,694 VRCFT packets at ~59 Hz, 65 of 70 fields
move. Eye_Pupil_Left/Right and TongueCurlUp/BendDown/Wide are always 0. Our 20 eye fields against
it (`oracle-compare`): mean |error| 0.002–0.008, correlation ≥ 0.991 (pitch 0.996, yaw 0.998,
lid close 0.995–0.997, squint ≥ 0.991, wide ~0.995, brows ≥ 0.996), best at 0 ms lag. Mouth
fields: the 45 that move by more than 0.1 have correlation ≥ 0.976 (most ≥ 0.997) and mean
|error| ≤ 0.013 (LipSuck lower is the largest); the rest stay within 0.02.

## VRChat OSC (Mode A)

`run --osc` turns the 70 fields into VRCFaceTracking's parameters (Cymple's direct-OSC mode
sends the same names and formulas) and sends the ones the current avatar has, every 10 ms, only
when a value moved by ≥ 1e-4 (Cymple's deadband), plus a full refresh every 5 s. The catalog is
VRCFT's: every `v2/<UnifiedExpression>` and simple expression, the combined v2 shapes, the legacy
SRanipal lip set and the legacy eye set (head pose left out).

- Fields → Unified shapes as Cymple's OSC output has them: CornerPull = CornerSlant = Smile,
  MouthUpperUp = MouthUpperDeepen = LipRaise, MouthLowerDown = LipDepress, LipSuckCorner = mean of
  upper and lower, MouthRaiser = Shrug, TongueFlat ← field 68; nasal, press, tightener, clench,
  palate, throat and neck stay 0.
- Cymple's legacy lip formulas are an older VRCFT's and we follow them: SRanipal JawOpen = JawOpen
  (current VRCFT subtracts MouthClosed), MouthUpperUp/LowerDown without the funnel term, MouthSad =
  max(frown, stretch) of that side without subtracting the smile, averaged merges never take the
  max, TongueSteps = step1 − step2.

- Avatar parameters: VRChat's OSCQuery (mDNS `_oscjson._tcp`, polled each second); without it,
  `/avatar/change` on UDP 9001 and VRChat's config file in the Proton prefix
  (`compatdata/438100/.../LocalLow/VRChat/VRChat/OSC/usr_*/Avatars/<id>.json`). Until one of those
  is known, the floats go to their default addresses (`/avatar/parameters/v2/EyeLeftX`, ...).
- Relevance as in VRCFT: the avatar address must end in `/<name>` (not after `v<digits>`) with the
  same type; binary params use the avatar's `<name>1/2/4…` bools and `<name>Negative`.
- Avatars without eye parameters get VRChat's native `/tracking/eye/LeftRightPitchYaw`
  (degrees, `[-atan(pitchL), atan(yawL), -atan(pitchR), atan(yawR)]`) and
  `/tracking/eye/EyesClosedAmount`, as VRCFT does.
- Checked against Cymple's own OSC output (`--mode osc` oracle, 400 addresses, 19,752 ticks): our
  formulas, fed with the eye and mouth state Cymple sent, reproduce all 400 addresses to within its
  1e-4 deadband (bits: 2 boundary flips out of 29 on JawX1). Binary params step at multiples of 1/(2ⁿ−1) in Cymple (VRCFT's code uses 2ⁿ); we
  follow Cymple.
- Without avatar information VRCFT's force-relevant mode would also send each parameter's bool
  twin to the same address; we send only the floats.
- On samples where Cymple's values change, our eye fields differ from its OSC run about twice as
  much as two Cymple runs differ from each other (gaze 0.005–0.007, lids 0.015, wide 0.06 vs 0.026).

## Window (M6)

`opencymple-linux gui` runs the same pipeline as `run` with a window in the same process (Qt Widgets,
Fusion style, dark unless the desktop asks for light). The pages follow Windows Cymple's QML
(`uiApi.dll`): same controls, ranges and steps, and the same INI keys.

- Previews: per eye the camera frame with the stored ROI quad and the 96×96 model crop with the 8
  landmarks; the mouth frame after rotation with the auto-ROI quad or the manual rectangle, and its
  crop. Under each: device and firmware, received and tracked fps, inference time, latency (frame
  complete → result) and dropped frames. A gaze plot shows both eyes' yaw/pitch with lid bars.
- Values: all 70 fields as bars; the five Cymple never fills are greyed.
- Eyes: eyelid smoothing (`eyeExpressionFilter` 0–0.5), gaze smoothing (`eyeGazeFilter` 0.2–1),
  eye closing (`eyeBlinkCloseAmplitude` 0.1–0.3), gaze range (`eyeGazeAmplitude` 0.5–1), squint
  compensation (0–1), eye sync (`eyeGazeSyncRate` 0–100); switches flip vertical gaze (7), sync
  eyelids (9), silly mode (12); eye ROI calibration with model0 on the latest frames (both, left or
  right); status of the stored gaze mesh.
- Mouth: smoothing (`filer` 0–0.5), camera rotation (−180…180 in steps of 5), automatic mouth area,
  manual area by sliders or by dragging a rectangle on the mouth preview (Cymple's clamping rules:
  bottom below top, area inside the frame).
- Amplitudes: the 28 sliders (0–5, step 0.05) in Cymple's page order. Slider i sets slot i, saved
  under Cymple's key for that slot, so the shifted keys stay compatible (the Smile slider writes
  `Amp_ShrugLowerAmp`, as Cymple's does). The page edits the default set, or the current avatar's
  own set once one is created; a set can be created for and deleted from the current avatar.
- Output: VRChat OSC on/off, target, avatar-parameter source, live status; the camera source,
  devices and headset; export to Windows Cymple (lists the changes and asks first).

Every change applies at once and is saved to our INI 0.6 s after the last one. Behaviour that is
ours rather than Cymple's (unknown there): changing a smoothing slider keeps the filters' state
(only their parameters change); turning the automatic mouth area off, or changing the rotation,
drops the current auto-ROI quad, so detection starts over. The amplitude set follows VRChat's
current avatar (its own `avatarParaments/<id>.json`, else `default.json`), in `run --osc` too;
[Inferred] from Cymple's amplitude page ("only affects the current model").

Not in the window yet: mouth V2 pages, dataset recording, firmware.

## Calibration (M7)

Eyes tab → Gaze calibration: "Recenter both eyes", a precise/coarse switch
(`eye_gaze_use_precise_calibration`), "Calibrate in headset" (13 dots, see M8 below) and "Calibrate
on desktop" (13 dots in a window) or "Start coarse calibration (5 s)". Everything is written to our settings copy in Cymple's keys, so an export
gives Windows Cymple the same calibration. Evidence from `cympleface.dll` (Hopper via REA, objdump
for the float arguments):

| Flow | Behavior | Evidence |
|---|---|---|
| Before recenter / range calibration | Eye ROI calibration (model0) on the latest frames, as Cymple's eye class does | `sub_180033490` ("refreshed ROI calibration before recenter"), strings at `0x1801d0db0` |
| Recenter, per eye | Needs fresh landmarks, an open eye (mean lid opening > max(0.05 × corner distance, 1e-5)), a valid feature and finite lid points; uses the current landmarks only (no averaging) | `sub_1800c1220`, `sub_1800c8a00`, `sub_1800c7650` |
| Centre lid geometry | `<Side>_Eye{upper,lower}_{outer,inner}_y` = the four lid points; the left/centre/right `Eyelid_ref_*` move to the new centre column by column (each ref's outer and inner columns mapped from the old centre span onto the new one), yaw shifted by the centre's change; centre = current lids and mode-2 yaw | `sub_1800b9c10`, `sub_1800c7de0`, `sub_1800c7f60` |
| Lid-close refs | `vrcft<Side>Lidvalue0..2` = 0.8·o, k·o, o; o = lid opening at the pupil (≥ 1e-5), k = eye-closing slider clamped to 0.1–0.3 | `sub_1800c9930` (your INI: 0.16195 = 0.8 × 0.202438, 0.042512 = 0.21 × 0.202438) |
| Squint / wide refs | `squint<Side>value0` = ½(lower-lid mid + pupil y) − ref y, `value1` = lower-lid mid − ref y; `eyeWide<Side>value0` = total opening of both lid columns, `value1` = 1.2 × that; the third values stay | `sub_1800c1220`, `sub_1800c9020`, `0x1801dc940` |
| Lid state | Lid output, raw lid, lid filter and blink state machine reset | `sub_1800c1220`, `sub_1800c5810` |
| Expression neutral | Raw widen / brow-down / brow-inner-raise outputs that are finite and in [0, 0.9999) become the neutral; their filters reset | `sub_1800b9460`, `sub_1800cf890`, `sub_1800d0200` |
| Gaze centre | Precise mode with a valid mesh: all 13 raw points shift by (feature − point 0) when it moved by more than 1e-5, ranges = the mesh's bounding box with mid = point 0 (your INI matches this exactly). Otherwise each range keeps its spans (at least 0.08 yaw / 0.035 pitch) and moves its mid to the feature | `sub_1800c4980`, `sub_1800c46b0`, `sub_1800c5370` |
| Coarse range (precise off) | Start: ranges collapse to their mid. Every packet: open eyes with a feature widen min/max. After 5 s: old mid kept, a new min/max only if it reaches more than 0.08 past the mid, otherwise the old value | `sub_1800c01c0`, `sub_1800c0950`, `sub_1800c52a0`, `sub_1800c4f10`; 5 s from the eye page text |
| 13-point mesh | Points 0..12 in order, one capture per confirm; raw = each eye's mode-2 feature of the current frame after the widen blend; point 0 also runs the centre references (without the gaze part), points 9 / 5 the left / right lid references. An eye with all 13 points: mesh saved with feature mode 2, layout version 4 and the overlay geometry, ranges = bounding box | `sub_18017efc0`, `sub_1800ba3a0`, `sub_1800b8690`, `sub_1800ba100`, `sub_18017dd40` |
| Mesh targets | Centre; inner ring r 0.45 (up, left, down, right); outer ring r 1 every 45° from the right, counter-clockwise | `sub_180182ee0`; your saved targets are exactly these |
| Gaze feature, precise off | Mode 0 (pupil − reference). Cymple tries an iris-circle feature first, but the EYE_V3 landmark path never sets its circle flag [Inferred] | `sub_1800c7590` → `sub_1800c6940` → `sub_1800c6700` / `sub_1800c6350`; flag bits in `sub_180040xxx` |

Ours rather than Cymple's: after the ROI refresh the recenter waits until both eyes have landmarks from
the new crop (plus 0.6 s for the pt7 low-pass when a crop changed); Cymple recenters on whatever
landmarks it has, which can still come from the old crop. The desktop target window (Space captures)
keeps the stored overlay geometry. Feature mode 1 (`sub_1800c7210`) is not ported (your calibration
uses mode 2).

## Headset calibration overlay (M8)

"Calibrate in headset" (or `gui --calibrate`) opens an OpenXR overlay session (`XR_EXTX_overlay`,
EGL through `XR_MNDX_egl_enable`) on top of whatever runs in WiVRn, VRChat included. Step 1: left
stick up/down sets the size, right stick up/down the height, right trigger accepts (B / Y closes).
Step 2: look at the yellow dot and pull either trigger; a short buzz means captured, a long one
skipped; the panel shows whether each eye is tracked; B / Y goes back to step 1. After point 13 the
mesh is saved and the overlay closes 3 s later. Evidence from `cympleface.dll` (objdump):

| Piece | Behavior | Evidence |
|---|---|---|
| Placement | Head-locked square 2 m in front, 1.65 m × coverage wide, centred `y_offset` m above eye level; the texture is 896 px | `sub_1801811b0`, `sub_180181130`, constants −2.0, 1.65, 896 |
| Dots | Pixel (448 + int(330 tx), 448 − int(330 ty)) for the canonical target, so the outer ring is 1.65 × coverage × 330/896 m | `sub_18018ee70`, `sub_1801811a0`, table at `0x1801eb5d0` |
| Sticks | Dead zone 0.2, linear to full deflection at 1.0, at most 50 ms per poll; height 0.7 m/s within ±0.75 m, coverage 0.85/s from 0.45 up to the FOV limit (6 at most) | `sub_180184c30`, `sub_180183cf0`, `sub_180183e00` |
| FOV limit | Largest coverage (24 bisections) with every dot inside 6–94 % of each eye's view; a height change re-caps the coverage | `sub_180182600`, `sub_180182820`, `sub_18018de50` |
| Saved targets | Each half axis divided by the visible extent, then normalised back to the outermost dot: the canonical layout whenever the dots are in view (your INI has exactly that) | `sub_180183060`, `sub_180182b60`, `sub_180182dd0` |
| Stored geometry | `eye_gaze_calibration_overlay_{y_offset_meters,ring_scale}` on accept; the per-eye `*_mesh_overlay_*` keys with the mesh | `updateEyeGazeCalibrationOverlay*`, `updateEyeGazeMeshCalibrationWithOverlayGeometry` |

Ours rather than Cymple's: Cymple adjusts height, then coverage, with one stick, and confirms with
either trigger; here both sticks work at once and the right trigger accepts step 1. The overlay is
drawn as quad layers (dots, a pulsing ring on the current dot, a text panel) instead of one SteamVR
overlay. A point that catches only one eye still advances, as in Cymple, and that eye's mesh is then
not saved.

## OpenCymple (M9)

`opencymple` is the app for everyday use; `opencymple-linux` keeps the developer and oracle commands.

- One per user: the running app listens on `$XDG_RUNTIME_DIR/opencymple.sock` (one JSON line per
  request). Commands (`calibrate precise|coarse`, `recenter`, `status`, `quit`) go to it; when none
  runs, `calibrate` and `recenter` start a headless one in the background first, which keeps
  running. Plain `opencymple` with an app already running just opens its window.
- Tray icon: Open Window (also a left click), Precise Calibration (the headset overlay), Quit.
  Closing the window leaves OpenCymple running; Quit or `opencymple quit` stops it. SIGTERM (how
  WiVRn stops its applications) quits cleanly too. Logs: `~/.local/state/opencymple/opencymple.log`.
- `opencymple install` writes `~/.local/share/applications/opencymple.desktop` (Name OpenCymple,
  actions Precise Calibration / Recenter Eyes / Quit) and the icon in `hicolor/scalable/apps`. Its
  category `X-WiVRn-VR` is what wivrn-server and wivrn-dashboard look for, so OpenCymple appears in
  WiVRn's list of applications to start with the headset; WiVRn's configuration is not changed, so
  it is not selected there (WayVR stays the one WiVRn starts). There is no `TryExec`, since WiVRn's
  flatpak can't see the venv path. It also links `~/.local/bin/opencymple` to the venv's script
  (only a link of its own is ever replaced or removed), so `opencymple` works in any shell.
- The settings copy stays in `~/.config/opencymple-linux/` (`opencymple-linux.json` holds the window
  geometry and VRChat output options for both commands).

Checked on the desktop on 2026-10-24 (Hyprland): the tray item registers with the menu above, Open
Window and plain `opencymple` open the window (Wayland app id `opencymple`, matching the entry),
closing it keeps the app running, `status` answers, and the tray's Quit stops it and removes the
socket. Not yet tried with the headset: Precise Calibration from the tray or `opencymple calibrate
precise` (the same code as the window's button), and `recenter` with live cameras (tested on the
replayed capture).

## Not done yet / unknown

- Mouth V2 (streams 3/4, model6) uses a different preprocessing and is ignored with a warning.
- Squint with degenerate calibration (`squint*value0 == value1`) gives 0 instead of Cymple's NaN.
- We don't advertise our own OSCQuery service yet, so VRChat shows no "OSC app connected" popup;
  avatar changes are seen by polling OSCQuery or on UDP 9001.
- The legacy gaze fallback uses `Eye{pupil}_x/y_min/mid/max` as its ranges (inferred).
- Live path verified with the headset (3 × 50 fps, no drops, during a WiVRn session).
- VRChat (Proton + WiVRn) checked in the headset on 2026-10-23 with a face-tracking avatar: OSCQuery
  found VRChat and the avatar within a second; gaze, blinks/winks, squint, wide and brows work, and
  avatar switches are followed. On an avatar without eye parameters (FISH_Scrap Metal) the native
  `/tracking/eye/*` path moves the eyes and closes the lids.
- Mouth in VRChat checked in the headset on 2026-10-23 (H3, V's Obsidian - Temptation, 137 bound
  parameters): face tracking looks right.
- The window checked live on 2026-10-24 (relay on, VRChat running, same avatar): previews, rates and
  sliders look right, and changes made in the window show up on the avatar.
- The headset calibration overlay checked live on 2026-10-24 (WiVRn, Quest 3, VRChat running): all
  13 points captured for both eyes and saved with height −0.27 m and coverage 2.06.

The models stay Cymple's property. This code decrypts them in memory only; don't write or share
decrypted copies.
