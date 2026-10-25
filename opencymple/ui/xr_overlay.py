"""The 13-point gaze calibration in the headset: an OpenXR overlay session (XR_EXTX_overlay) on top of
whatever runs in WiVRn or Monado, laid out like Cymple's SteamVR overlay (geometry in overlay.py).

Step 1 places the dot pattern: left stick up/down sets the size (Cymple's coverage), right stick
up/down the height, right trigger accepts. Step 2 highlights the dots in Cymple's order; a trigger
pull captures the highlighted one while the user looks at it. B or Y goes back a step (in step 1 it
closes the overlay). Cymple has a height step and a coverage step, both on one stick; here both
sticks work at once.

Everything is drawn as head-locked quad layers whose textures are painted with QPainter and uploaded
through an EGL context (XR_MNDX_egl_enable), so no window or 3D renderer is involved."""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"        # PyOpenGL loads its functions through EGL; set before import

import ctypes                                    # noqa: E402
import logging                                   # noqa: E402
import threading                                 # noqa: E402
import time                                      # noqa: E402

from PySide6.QtCore import QPointF, QRectF, Qt   # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen   # noqa: E402

from .. import overlay as G                      # noqa: E402
from ..calib import MESH_TARGETS                 # noqa: E402

log = logging.getLogger(__name__)

ADJUST, PREPARING, CAPTURE, DONE = "adjust", "preparing", "capture", "done"
EXTENSIONS = ("XR_EXTX_overlay", "XR_MNDX_egl_enable", "XR_KHR_opengl_enable")
MAIN_PX = 1024                                   # Cymple's 896 px layout, drawn at 1024 px
TEXT_W, TEXT_H = 1024, 288
GLOW_PX = 128
TRIGGER_ON, TRIGGER_OFF = 0.7, 0.3
DONE_CLOSE_S = 3.0
TEXT_HZ = 15.0
GL_RGBA8, GL_SRGB8_ALPHA8 = 0x8058, 0x8C43

DOT_IDLE, DOT_PENDING = QColor(225, 230, 240), QColor(120, 126, 140)
DOT_CURRENT, DOT_DONE = QColor("#ffe000"), QColor("#3ccf8e")
TEXT, TEXT_DIM, TEXT_BAD, TEXT_GOOD = QColor("#f0f0f0"), QColor("#a8adb8"), QColor("#ff6b5b"), QColor("#3ccf8e")


def available():
    """None when the headset overlay can start, else why not."""
    try:
        import xr
    except Exception as e:                       # noqa: BLE001
        return "pyopenxr is not installed (%s)" % e
    try:
        names = set()
        for p in xr.enumerate_instance_extension_properties():
            n = p.extension_name
            names.add(n.decode() if isinstance(n, bytes) else n)
    except Exception as e:                       # noqa: BLE001
        return "no OpenXR runtime (%s)" % e
    missing = [n for n in EXTENSIONS if n not in names]
    return "the OpenXR runtime lacks %s" % ", ".join(missing) if missing else None


# ---- textures ----------------------------------------------------------------------------
def _image(w, h):
    img = QImage(w, h, QImage.Format.Format_RGBA8888_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    return img


def dot_pixel(i, size=MAIN_PX):
    """Dot i on the main texture: Cymple's pixel layout (sub_18018ee70) scaled to `size`."""
    tx, ty = MESH_TARGETS[i]
    k = size / G.TEXTURE_PX
    return size / 2 + int(G.RING_PX * tx) * k, size / 2 - int(G.RING_PX * ty) * k


def paint_main(current=None, captured=(), adjusting=False):
    img = _image(MAIN_PX, MAIN_PX)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QPointF(MAIN_PX / 2, MAIN_PX / 2)
    r = G.RING_PX * MAIN_PX / G.TEXTURE_PX
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(8, 9, 12, 150))
    p.drawEllipse(c, r + 52, r + 52)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(255, 255, 255, 70 if adjusting else 40), 3))
    p.drawEllipse(c, r, r)
    p.drawEllipse(c, 0.45 * r, 0.45 * r)
    for i in range(len(MESH_TARGETS)):
        pt = QPointF(*dot_pixel(i))
        color = (DOT_IDLE if adjusting else DOT_CURRENT if i == current else
                 DOT_DONE if i in captured else DOT_PENDING)
        rad = 12 if i == current else 9
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawEllipse(pt, rad, rad)
        p.setBrush(QColor(10, 10, 12))
        p.drawEllipse(pt, 2.5, 2.5)
    p.end()
    return img


def paint_glow():
    img = _image(GLOW_PX, GLOW_PX)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QPointF(GLOW_PX / 2, GLOW_PX / 2)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for width, alpha, rad in ((16, 50, 46), (6, 230, 44)):
        p.setPen(QPen(QColor(255, 224, 0, alpha), width))
        p.drawEllipse(c, rad, rad)
    p.end()
    return img


def paint_text(lines):
    """lines: [(text, QColor, point size)], centred on a dark rounded panel."""
    img = _image(TEXT_W, TEXT_H)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(14, 15, 20, 220))
    p.drawRoundedRect(QRectF(4, 4, TEXT_W - 8, TEXT_H - 8), 28, 28)
    heights = [size * 1.9 for _t, _c, size in lines]
    y = (TEXT_H - sum(heights)) / 2
    for (text, color, size), h in zip(lines, heights):
        f = QFont()
        f.setPixelSize(int(size))
        f.setBold(size >= 40)
        p.setFont(f)
        p.setPen(color)
        p.drawText(QRectF(24, y, TEXT_W - 48, h), Qt.AlignmentFlag.AlignCenter, text)
        y += h
    p.end()
    return img


# ---- OpenXR ------------------------------------------------------------------------------
class _Swapchain:
    def __init__(self, x, w, h):
        xr = x.xr
        self.x, self.w, self.h = x, w, h
        usage = xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT | xr.SwapchainUsageFlags.SAMPLED_BIT
        try:
            self.handle = xr.create_swapchain(x.session, self._info(usage | xr.SwapchainUsageFlags.TRANSFER_DST_BIT))
        except xr.XrException:
            self.handle = xr.create_swapchain(x.session, self._info(usage))
        self.images = xr.enumerate_swapchain_images(self.handle, xr.SwapchainImageOpenGLKHR)
        self.sub_image = xr.SwapchainSubImage(
            swapchain=self.handle,
            image_rect=xr.Rect2Di(offset=xr.Offset2Di(0, 0), extent=xr.Extent2Di(w, h)))

    def _info(self, usage):
        return self.x.xr.SwapchainCreateInfo(usage_flags=usage, format=self.x.format, sample_count=1,
                                             width=self.w, height=self.h, face_count=1, array_size=1,
                                             mip_count=1)

    def upload(self, qimage):
        xr, GL = self.x.xr, self.x.GL
        img = qimage.convertToFormat(QImage.Format.Format_RGBA8888_Premultiplied)
        # GL images start at the bottom row
        img = img.flipped(Qt.Orientation.Vertical) if hasattr(img, "flipped") else img.mirrored(False, True)
        data = bytes(img.constBits())[:self.w * self.h * 4]
        idx = xr.acquire_swapchain_image(self.handle, xr.SwapchainImageAcquireInfo())
        xr.wait_swapchain_image(self.handle, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.images[idx].image)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.w, self.h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glFinish()
        xr.release_swapchain_image(self.handle, xr.SwapchainImageReleaseInfo())

    def destroy(self):
        self.x.xr.destroy_swapchain(self.handle)


class XrOverlay:
    """Instance, overlay session, view space, swapchains and controller actions. Lives in one
    thread: the EGL context is current there."""

    def __init__(self):
        import xr
        from OpenGL import EGL, GL
        self.xr, self.EGL, self.GL = xr, EGL, GL
        self.session = self.instance = self.action_set = None
        self.swapchains = []
        self.running = self.exit_requested = False
        self.state = None
        self.instance = xr.create_instance(xr.InstanceCreateInfo(
            application_info=xr.ApplicationInfo(application_name="opencymple-linux calibration",
                                                application_version=1, engine_name="opencymple-linux",
                                                engine_version=1, api_version=xr.Version(1, 0, 0)),
            enabled_extension_names=list(EXTENSIONS)))
        self.system = xr.get_system(self.instance, xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY))
        xr.get_opengl_graphics_requirements_khr(self.instance, self.system)   # required before the session
        self._egl()
        self.overlay_info = xr.SessionCreateInfoOverlayEXTX(session_layers_placement=1)
        self.binding = xr.GraphicsBindingEGLMNDX(get_proc_address=self.get_proc_address, display=self.display,
                                                 config=self.config, context=self.context, next=self.overlay_info)
        self.session = xr.create_session(self.instance, xr.SessionCreateInfo(system_id=self.system, next=self.binding))
        self.space = xr.create_reference_space(self.session, xr.ReferenceSpaceCreateInfo(
            reference_space_type=xr.ReferenceSpaceType.VIEW, pose_in_reference_space=xr.Posef()))
        formats = list(xr.enumerate_swapchain_formats(self.session))
        self.format = next((f for f in (GL_SRGB8_ALPHA8, GL_RGBA8) if f in formats), formats[0])
        self._actions()

    def _egl(self):
        EGL = self.EGL
        self.display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        if not self.display or not EGL.eglInitialize(self.display, None, None):
            raise RuntimeError("EGL initialisation failed")
        configs = (EGL.EGLConfig * 1)()
        count = EGL.EGLint()
        attrs = [EGL.EGL_RED_SIZE, 8, EGL.EGL_GREEN_SIZE, 8, EGL.EGL_BLUE_SIZE, 8, EGL.EGL_ALPHA_SIZE, 8,
                 EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT, EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
                 EGL.EGL_NONE]
        if not EGL.eglChooseConfig(self.display, attrs, configs, 1, count) or count.value < 1:
            raise RuntimeError("no EGL config for OpenGL")
        self.config = configs[0]
        self.surface = EGL.eglCreatePbufferSurface(self.display, self.config,
                                                   [EGL.EGL_WIDTH, 1, EGL.EGL_HEIGHT, 1, EGL.EGL_NONE])
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)
        self.context = EGL.eglCreateContext(self.display, self.config, EGL.EGL_NO_CONTEXT, None)
        if not self.context or not EGL.eglMakeCurrent(self.display, self.surface, self.surface, self.context):
            raise RuntimeError("could not make an EGL OpenGL context current")
        self._libegl = ctypes.CDLL("libEGL.so.1")
        self.get_proc_address = ctypes.cast(self._libegl.eglGetProcAddress, self.xr.PFN_xrEglGetProcAddressMNDX)

    def _actions(self):
        xr = self.xr
        path = lambda s: xr.string_to_path(self.instance, s)         # noqa: E731
        self.hands = [path("/user/hand/left"), path("/user/hand/right")]
        self.action_set = xr.create_action_set(self.instance, xr.ActionSetCreateInfo(
            action_set_name="opencymple_gaze_calibration", localized_action_set_name="OpenCymple gaze calibration"))

        def action(name, kind, label, per_hand=False):
            return xr.create_action(self.action_set, xr.ActionCreateInfo(
                action_name=name, action_type=kind, localized_action_name=label,
                subaction_paths=self.hands if per_hand else None))
        T = xr.ActionType
        self.size_stick = action("size", T.VECTOR2F_INPUT, "Calibration area size")
        self.height_stick = action("height", T.VECTOR2F_INPUT, "Calibration area height")
        self.trigger = action("confirm", T.FLOAT_INPUT, "Accept / capture", per_hand=True)
        self.back = action("back", T.BOOLEAN_INPUT, "Back", per_hand=True)
        self.haptic = action("haptic", T.VIBRATION_OUTPUT, "Capture feedback", per_hand=True)
        touch = [(self.size_stick, "/user/hand/left/input/thumbstick"),
                 (self.height_stick, "/user/hand/right/input/thumbstick"),
                 (self.trigger, "/user/hand/left/input/trigger/value"),
                 (self.trigger, "/user/hand/right/input/trigger/value"),
                 (self.back, "/user/hand/left/input/y/click"),
                 (self.back, "/user/hand/right/input/b/click"),
                 (self.haptic, "/user/hand/left/output/haptic"),
                 (self.haptic, "/user/hand/right/output/haptic")]
        simple = [(self.trigger, "/user/hand/left/input/select/click"),
                  (self.trigger, "/user/hand/right/input/select/click"),
                  (self.back, "/user/hand/left/input/menu/click"),
                  (self.back, "/user/hand/right/input/menu/click"),
                  (self.haptic, "/user/hand/left/output/haptic"),
                  (self.haptic, "/user/hand/right/output/haptic")]
        for profile, binds in (("/interaction_profiles/oculus/touch_controller", touch),
                               ("/interaction_profiles/khr/simple_controller", simple)):
            try:
                xr.suggest_interaction_profile_bindings(self.instance, xr.InteractionProfileSuggestedBinding(
                    interaction_profile=path(profile),
                    suggested_bindings=[xr.ActionSuggestedBinding(a, path(p)) for a, p in binds]))
            except xr.XrException as e:
                log.warning("bindings for %s refused: %s", profile, e)
        xr.attach_session_action_sets(self.session, xr.SessionActionSetsAttachInfo(action_sets=[self.action_set]))

    # ---- per frame ------------------------------------------------------------------------
    def poll(self):
        """Handle pending events; False once the session is over."""
        xr = self.xr
        while True:
            try:
                ev = xr.poll_event(self.instance)
            except xr.EventUnavailable:
                return True
            kind = xr.StructureType(ev.type)
            if kind == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                return False
            if kind != xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                continue
            st = xr.SessionState(ctypes.cast(ctypes.byref(ev), ctypes.POINTER(xr.EventDataSessionStateChanged)).contents.state)
            self.state = st
            log.info("overlay session: %s", st.name)
            if st == xr.SessionState.READY:
                xr.begin_session(self.session, xr.SessionBeginInfo(xr.ViewConfigurationType.PRIMARY_STEREO))
                self.running = True
            elif st == xr.SessionState.STOPPING:
                xr.end_session(self.session)
                self.running = False
            elif st in (xr.SessionState.EXITING, xr.SessionState.LOSS_PENDING):
                return False

    @property
    def focused(self):
        return self.state == self.xr.SessionState.FOCUSED

    def request_exit(self):
        if self.running and not self.exit_requested:
            self.exit_requested = True
            self.xr.request_exit_session(self.session)

    def read_input(self):
        """(size stick y, height stick y, (left, right) trigger values, (left, right) back) or None."""
        xr = self.xr
        try:
            xr.sync_actions(self.session, xr.ActionsSyncInfo(active_action_sets=[xr.ActiveActionSet(self.action_set, 0)]))
        except xr.XrException:
            return None

        def get(fn, action, hand=0):
            return fn(self.session, xr.ActionStateGetInfo(action=action, subaction_path=hand))
        size = get(xr.get_action_state_vector2f, self.size_stick)
        height = get(xr.get_action_state_vector2f, self.height_stick)
        trig = tuple(get(xr.get_action_state_float, self.trigger, h).current_state for h in self.hands)
        back = tuple(bool(get(xr.get_action_state_boolean, self.back, h).current_state) for h in self.hands)
        return (size.current_state.y if size.is_active else 0.0,
                height.current_state.y if height.is_active else 0.0, trig, back)

    def pulse(self, hand, amplitude, seconds):
        xr = self.xr
        try:
            xr.apply_haptic_feedback(self.session, xr.HapticActionInfo(action=self.haptic, subaction_path=self.hands[hand]),
                                     xr.HapticVibration(duration=int(seconds * 1e9), frequency=0, amplitude=amplitude))
        except xr.XrException as e:
            log.debug("haptics: %s", e)

    def eye_views(self, display_time):
        xr = self.xr
        try:
            st, views = xr.locate_views(self.session, xr.ViewLocateInfo(
                view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO, display_time=display_time,
                space=self.space))
        except xr.XrException:
            return []
        need = xr.ViewStateFlags.ORIENTATION_VALID_BIT | xr.ViewStateFlags.POSITION_VALID_BIT
        if (st.view_state_flags & need) != need:
            return []
        out = []
        for v in views:
            p, q, f = v.pose.position, v.pose.orientation, v.fov
            out.append(G.EyeView.from_angles((p.x, p.y, p.z), (q.x, q.y, q.z, q.w),
                                             f.angle_left, f.angle_right, f.angle_up, f.angle_down))
        return out

    def swapchain(self, w, h):
        sc = _Swapchain(self, w, h)
        self.swapchains.append(sc)
        return sc

    def quad(self, sc, x, y, z, w, h):
        xr = self.xr
        return xr.CompositionLayerQuad(
            layer_flags=xr.CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT, space=self.space,
            eye_visibility=xr.EyeVisibility.BOTH, sub_image=sc.sub_image,
            pose=xr.Posef(orientation=xr.Quaternionf(0, 0, 0, 1), position=xr.Vector3f(x, y, z)),
            size=xr.Extent2Df(w, h))

    def wait_frame(self):
        st = self.xr.wait_frame(self.session, self.xr.FrameWaitInfo())
        self.xr.begin_frame(self.session, self.xr.FrameBeginInfo())
        return st

    def end_frame(self, display_time, quads):
        xr = self.xr
        xr.end_frame(self.session, xr.FrameEndInfo(
            display_time=display_time, environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
            layers=[ctypes.byref(q) for q in quads]))

    def close(self):
        xr, EGL = self.xr, self.EGL
        for sc in self.swapchains:
            try:
                sc.destroy()
            except Exception:                    # noqa: BLE001
                pass
        for handle, fn in ((self.action_set, xr.destroy_action_set), (self.session, xr.destroy_session),
                           (self.instance, xr.destroy_instance)):
            if handle is not None:
                try:
                    fn(handle)
                except Exception:                # noqa: BLE001
                    pass
        try:                                     # the display may be shared with others: no eglTerminate
            EGL.eglMakeCurrent(self.display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
            EGL.eglDestroyContext(self.display, self.context)
            EGL.eglDestroySurface(self.display, self.surface)
        except Exception:                        # noqa: BLE001
            pass


# ---- the calibration flow ----------------------------------------------------------------
class HeadsetLauncher:
    """Starts and stops the headset calibration of one runtime; the window, the tray and the
    `opencymple calibrate precise` command share it."""

    def __init__(self, runtime):
        self.rt = runtime
        self.current = None

    @property
    def running(self):
        return self.current is not None and self.current.running

    def start(self):
        """None when started (or already running), else why not."""
        if self.running:
            return None
        if self.rt.calibrator.coarse is not None:
            return "the coarse calibration is running"
        if self.rt.calibrator.mesh is not None:
            return "a 13-point calibration is already running on the desktop"
        why = available()
        if why:
            return "headset calibration unavailable: %s" % why
        if not self.rt.calib.precise_gaze:
            self.rt.set_precise(True)                # the 13 points belong to precise mode
        self.current = HeadsetCalibration(self.rt)
        self.current.start()
        return None

    def stop(self, wait=0.0):
        if self.current is not None:
            self.current.stop()
            if wait:
                self.current.join(wait)

    def status(self):
        """The overlay's status while it runs or after it failed, else None."""
        c = self.current
        if c is not None and (c.running or c.error):
            return c.status
        return None


class HeadsetCalibration:
    """Runs the overlay in its own thread. `status` describes the current step for the window."""

    def __init__(self, runtime):
        self.rt = runtime
        y, s = runtime.calib.overlay_geometry
        self.geometry = G.OverlayGeometry(y, s)
        self.state = ADJUST
        self.status = "starting the headset overlay"
        self.message, self.message_color = "", TEXT_DIM
        self.error = None
        self.saved = None
        self.ready = (False, False)
        self.done_at = None
        self._stop = threading.Event()
        self._thread = None
        self._prep = None
        self._trigger = [False, False]
        self._back = False
        self.main_dirty = self.text_dirty = True

    # ---- control from the window ----------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, name="xr-calibration", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    # ---- thread ---------------------------------------------------------------------------
    def _run(self):
        x = None
        try:
            x = XrOverlay()
            self.status = "headset overlay open: step 1, place the dots"
            self._loop(x)
            if self.state != DONE:
                self.status = "headset calibration closed"
        except Exception as e:                   # noqa: BLE001
            log.exception("headset calibration failed")
            self.error = str(e) or type(e).__name__
            self.status = "headset overlay failed: %s" % self.error
        finally:
            if self._prep is not None:
                self._prep.join()
            if self.state in (PREPARING, CAPTURE):
                self.rt.cancel_mesh()
            if x is not None:
                x.close()

    def _loop(self, x):
        main, glow, text = x.swapchain(MAIN_PX, MAIN_PX), x.swapchain(GLOW_PX, GLOW_PX), x.swapchain(TEXT_W, TEXT_H)
        glow.upload(paint_glow())
        last = time.monotonic()
        views_at = ready_at = text_at = 0.0
        while x.poll():
            now = time.monotonic()
            closing = self._stop.is_set() or (self.done_at is not None and now >= self.done_at)
            if closing:
                if not x.running:
                    break
                x.request_exit()
            if not x.running:
                time.sleep(0.05)
                continue
            fs = x.wait_frame()
            now = time.monotonic()
            dt, last = now - last, now
            quads = []
            if fs.should_render and not closing:
                if now - views_at > 1.0:
                    eyes = x.eye_views(fs.predicted_display_time)
                    if eyes:
                        self.geometry.set_eyes(eyes)
                        views_at = now
                if now - ready_at > 0.2:
                    ready, ready_at = self.rt.eye_readiness(), now
                    if ready != self.ready:
                        self.ready, self.text_dirty = ready, True
                self._prepared()
                self.handle_input(x, x.read_input() if x.focused else None, dt, now)
                if self.main_dirty:
                    main.upload(self.paint_main())
                    self.main_dirty = False
                if self.text_dirty and now - text_at >= 1.0 / TEXT_HZ:
                    text.upload(paint_text(self.text_lines()))
                    self.text_dirty, text_at = False, now
                quads = self.layout(x, main, glow, text, now)
            x.end_frame(fs.predicted_display_time, quads)

    # ---- flow -----------------------------------------------------------------------------
    def _edges(self, inp):
        """Trigger presses per hand (with hysteresis) and a back press since the last poll."""
        _size, _height, trig, back = inp
        pressed = [False, False]
        for h in (0, 1):
            if not self._trigger[h] and trig[h] >= TRIGGER_ON:
                self._trigger[h] = pressed[h] = True
            elif self._trigger[h] and trig[h] <= TRIGGER_OFF:
                self._trigger[h] = False
        any_back = any(back)
        back_pressed = any_back and not self._back
        self._back = any_back
        return pressed, back_pressed

    def handle_input(self, x, inp, dt, now):
        if inp is None:
            return
        size, height, _trig, _back = inp
        pressed, back = self._edges(inp)
        if self.state == ADJUST:
            g = self.geometry
            if g.adjust_coverage(size, dt) | g.adjust_height(height, dt):
                self.text_dirty = True
            if pressed[1]:
                self._accept(x)
            elif back:
                self._stop.set()
        elif self.state == CAPTURE:
            hand = 1 if pressed[1] else 0 if pressed[0] else None
            if hand is not None:
                self._capture(x, hand, now)
            elif back:
                self.rt.cancel_mesh()
                self._set_state(ADJUST, "", TEXT_DIM)
        elif back:
            self._stop.set()

    def _set_state(self, state, message, color):
        self.state, self.message, self.message_color = state, message, color
        self.main_dirty = self.text_dirty = True
        self.status = {ADJUST: "headset overlay: step 1, place the dots",
                       PREPARING: "headset overlay: finding the eyes",
                       CAPTURE: "headset overlay: step 2, capturing the dots",
                       DONE: "headset calibration: %s" % message}[state]

    def _accept(self, x):
        y, s = self.geometry.y, self.geometry.scale
        self.rt.set_overlay_geometry(y, s)
        x.pulse(1, 0.5, 0.04)
        self._set_state(PREPARING, "Finding the eyes...", TEXT_DIM)

        def prep():
            try:
                self.rt.start_mesh(overlay=(y, s))
            except Exception:                    # noqa: BLE001
                log.exception("starting the 13-point capture failed")
        self._prep = threading.Thread(target=prep, name="xr-calibration-prep", daemon=True)
        self._prep.start()

    def _prepared(self):
        if self.state == PREPARING and self._prep is not None and not self._prep.is_alive():
            self._prep = None
            if self.rt.calibrator.mesh is None:
                self._set_state(ADJUST, "Could not start the capture", TEXT_BAD)
            else:
                self._set_state(CAPTURE, "Look at the yellow dot, then pull the trigger", TEXT_DIM)

    def _capture(self, x, hand, now):
        rep = self.rt.capture_mesh_point()
        if rep is None:
            return
        n = rep["point"] + 1
        if "skipped" in rep:
            self.message, self.message_color = "Point %d not captured: keep the eyes open and try again" % n, TEXT_BAD
            x.pulse(hand, 0.9, 0.15)
        else:
            missing = [name for e, name in enumerate(("left", "right")) if e not in rep["eyes"]]
            self.message = "Point %d captured%s" % (n, " (no %s eye)" % missing[0] if missing else "")
            self.message_color = TEXT_DIM if missing else TEXT_GOOD
            x.pulse(hand, 0.5, 0.04)
        if rep.get("done"):
            self.saved = rep["saved"]
            msg = "saved for " + (" and ".join(n for e, n in enumerate(("left", "right")) if self.saved.get(e))
                                  or "no eye (points missing)")
            self._set_state(DONE, msg, TEXT_GOOD if any(self.saved.values()) else TEXT_BAD)
            self.done_at = now + DONE_CLOSE_S
        self.main_dirty = self.text_dirty = True

    # ---- drawing --------------------------------------------------------------------------
    def current_point(self):
        m = self.rt.calibrator.mesh
        return m.step if self.state == CAPTURE and m is not None and m.step < len(MESH_TARGETS) else None

    def paint_main(self):
        m = self.rt.calibrator.mesh
        captured = set()
        if m is not None:
            captured = {i for eye in (0, 1) for i, v in enumerate(m.raw[eye]) if v is not None}
        return paint_main(self.current_point(), captured, adjusting=self.state in (ADJUST, PREPARING))

    def text_lines(self):
        g = self.geometry
        eyes = "Left eye %s    Right eye %s" % tuple("tracked" if r else "not tracked" for r in self.ready)
        eye_color = TEXT_GOOD if all(self.ready) else TEXT_BAD
        if self.state == ADJUST:
            return [("Gaze calibration, step 1 of 2", TEXT, 46),
                    ("Left stick: size   Right stick: height   Right trigger: start", TEXT, 30),
                    ("Outer dots where your eyes can still comfortably look   B: close", TEXT_DIM, 26),
                    ("size %.2f (%.0f deg)   height %+.2f m%s" % (
                        g.scale, g.ring_degrees(), g.y, "   " + self.message if self.message else ""),
                     self.message_color if self.message else TEXT_DIM, 26)]
        if self.state == PREPARING:
            return [("Gaze calibration", TEXT, 46), (self.message, TEXT_DIM, 30)]
        if self.state == DONE:
            return [("Calibration %s" % self.message, self.message_color, 44), ("Closing...", TEXT_DIM, 28)]
        k = self.current_point()
        return [("Point %d of %d" % ((k if k is not None else 12) + 1, len(MESH_TARGETS)), TEXT, 46),
                (self.message or "Look at the yellow dot, then pull the trigger", self.message_color, 28),
                (eyes, eye_color, 26), ("B: back to step 1", TEXT_DIM, 24)]

    def layout(self, x, main, glow, text, now):
        g = self.geometry
        width = G.overlay_width(g.scale)
        quads = [x.quad(main, 0.0, g.y, -G.DISTANCE_M, width, width)]
        k = self.current_point()
        if k is not None:
            px, py, pz = G.dot_position(k, g.y, g.scale)
            pulse = 1.0 + 0.12 * abs(((now * 1.6) % 2.0) - 1.0)
            s = width * 0.0625 * pulse
            quads.append(x.quad(glow, px, py, pz + 0.005, s, s))
        r = G.ring_radius(g.scale)
        tw = G.clamp(0.585 * r, 0.5, 0.9) if self.state == CAPTURE else 1.0
        ty = g.y - 0.225 * r if self.state == CAPTURE and k == 0 else g.y
        quads.append(x.quad(text, 0.0, ty, -G.DISTANCE_M + 0.01, tw, tw * TEXT_H / TEXT_W))
        return quads
