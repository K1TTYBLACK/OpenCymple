"""Load Cymple's obfuscated ONNX models straight from the shipped model/*.bin files.

Scheme (cympleface.dll sub_18004d7f0 / sub_180049900 / sub_180049d50):
  file  = salt[16] || ciphertext
  h     = FNV-1a-64(passphrase || salt)
  state = [h, h ^ 0x9e3779b97f4a7c15]; discard 10 xorshift128+ outputs
  plain = ciphertext XOR little-endian xorshift128+ stream (last partial word: low bytes)

Decrypted bytes only ever live in memory and are handed to ONNX Runtime directly, as the
original does. The weights are Cymple's; do not write or share decrypted copies.
"""
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

M64 = (1 << 64) - 1


@dataclass(frozen=True)
class ModelSpec:
    file: str
    passphrase: str
    input_size: int


SPECS = {
    "eye_roi": ModelSpec("model0.bin", "Project_Cymple_Face_Eye_Roi", 64),
    "eye_landmark": ModelSpec("model2.bin", "Project_Cymple_Face_Eye_Landmark", 96),
    "mouth_roi": ModelSpec("model3.bin", "Project_Cymple_Face_MouthV1_Roi", 96),
    "mouth_v1": ModelSpec("model4.bin", "Cymple_Face_Mouth_Blendshapes_V1", 96),
    "mouth_v2": ModelSpec("model6.bin", "Cymple_Face_Mouth_Blendshapes_V2", 96),
}

# Friendly names for --provider; "auto" stays on CPU because these models are 64-96 px and
# 1.3-4.9 M parameters, where GPU dispatch costs more than it saves.
PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
}


def decrypt(data, passphrase):
    salt, body = data[:16], data[16:]
    h = 0xCBF29CE484222325
    for b in passphrase.encode() + salt:
        h = ((h ^ b) * 0x100000001B3) & M64
    s0, s1 = h, h ^ 0x9E3779B97F4A7C15

    def stream():
        nonlocal s0, s1
        while True:
            a, b = s0, s1
            s0 = b
            a ^= (a << 23) & M64
            s1 = a ^ b ^ (a >> 17) ^ (b >> 26)
            yield (s1 + b) & M64

    ks = stream()
    for _ in range(10):
        next(ks)
    out = bytearray(body)
    n = len(out) // 8
    words = struct.unpack_from("<%dQ" % n, out)
    struct.pack_into("<%dQ" % n, out, 0, *(w ^ k for w, k in zip(words, ks)))
    if n * 8 < len(out):
        tail = struct.pack("<Q", next(ks))
        for i in range(n * 8, len(out)):
            out[i] ^= tail[i - n * 8]
    return bytes(out)


def resolve_providers(name):
    available = ort.get_available_providers()
    if name in (None, "auto", "cpu"):
        return ["CPUExecutionProvider"]
    ep = PROVIDERS.get(name, name)
    if ep not in available:
        raise RuntimeError("ONNX Runtime provider %s is not available (have: %s). Install the matching "
                           "onnxruntime build, e.g. onnxruntime-gpu or onnxruntime-rocm."
                           % (ep, ", ".join(available)))
    return [ep, "CPUExecutionProvider"]


class Model:
    """One ORT session. `run` is safe to call from several threads (ORT sessions are)."""

    def __init__(self, name, model_dir, providers=("CPUExecutionProvider",), threads=1):
        self.name, self.spec = name, SPECS[name]
        path = os.path.join(model_dir, self.spec.file)
        with open(path, "rb") as f:
            plain = decrypt(f.read(), self.spec.passphrase)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(plain, sess_options=opts, providers=list(providers))
        except Exception as e:
            raise RuntimeError("%s (%s) did not load; wrong file or passphrase?" % (path, e)) from e
        del plain
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.provider = self.session.get_providers()[0]
        self._lock = threading.Lock()
        self.calls, self.total_ms = 0, 0.0

    def run(self, tensor):
        t = time.perf_counter()
        outs = self.session.run(None, {self.input_name: tensor})
        dt = (time.perf_counter() - t) * 1000.0
        with self._lock:
            self.calls += 1
            self.total_ms += dt
        return dict(zip(self.output_names, outs))

    def describe(self):
        i = self.session.get_inputs()[0]
        return "%-12s %s  input %s %s -> %s  [%s]" % (
            self.name, self.spec.file, i.name, i.shape,
            ", ".join("%s %s" % (o.name, o.shape) for o in self.session.get_outputs()), self.provider)


def load_models(model_dir, names, provider="auto", threads=1):
    providers = resolve_providers(provider)
    out = {}
    for name in names:
        t = time.perf_counter()
        out[name] = Model(name, model_dir, providers, threads)
        log.info("loaded %s in %.2fs on %s", name, time.perf_counter() - t, out[name].provider)
    return out


def warmup(model, n=3):
    x = np.zeros((1, 3, model.spec.input_size, model.spec.input_size), np.float32)
    for _ in range(n):
        model.run(x)
