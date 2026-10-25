import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/yolov8n-face.onnx"
WARMUP_RUNS = 3
BENCHMARK_RUNS = 20

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

model_input = session.get_inputs()[0]
shape = [
    dim if isinstance(dim, int) and dim > 0 else default
    for dim, default in zip(model_input.shape, [1, 3, 640, 640])
]
dummy_input = np.zeros(shape, dtype=np.float32)

print("Model:", MODEL_PATH)
print("Input shape:", shape)
print("Execution provider: CPU")

for _ in range(WARMUP_RUNS):
    session.run(None, {model_input.name: dummy_input})

times_ms = []
for _ in range(BENCHMARK_RUNS):
    start = time.perf_counter()
    session.run(None, {model_input.name: dummy_input})
    times_ms.append((time.perf_counter() - start) * 1000)

average_ms = sum(times_ms) / len(times_ms)

print(f"Average latency: {average_ms:.2f} ms")
print(f"Approximate throughput: {1000 / average_ms:.2f} FPS")
print("Note: Blank inputs; accuracy and camera-to-result latency are not measured.")
