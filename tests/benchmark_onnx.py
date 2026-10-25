import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

candidates = [
    Path("models/yolov8n-face.onnx"),
    Path("yolov8n-face.onnx"),
    Path("yolov8n.onnx"),
]

model_path = next((p for p in candidates if p.exists()), None)

if model_path is None:
    print("ERROR: No ONNX model found.")
    print("Available ONNX files:")
    for p in Path(".").rglob("*.onnx"):
        print(p)
    raise SystemExit(1)

session = ort.InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"],
)

input_info = session.get_inputs()[0]
shape = [
    dim if isinstance(dim, int) and dim > 0 else default
    for dim, default in zip(input_info.shape, [1, 3, 640, 640])
]
dummy_input = np.zeros(shape, dtype=np.float32)

print("Model:", model_path, flush=True)
print("Input shape:", shape, flush=True)
print("Warming up...", flush=True)

for _ in range(2):
    session.run(None, {input_info.name: dummy_input})

print("Benchmarking...", flush=True)
times = []

for i in range(10):
    start = time.perf_counter()
    session.run(None, {input_info.name: dummy_input})
    times.append((time.perf_counter() - start) * 1000)
    print(f"Completed run {i + 1}/10", flush=True)

avg_latency = sum(times) / len(times)

print(f"Average latency: {avg_latency:.2f} ms")
print(f"Approximate FPS: {1000 / avg_latency:.2f}")
print("Note: This uses blank input, not real face images.")
