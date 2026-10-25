# Face Recognition System Architecture

## 1. Overview
The project is organized into modules for video ingestion, face detection,
face recognition, database operations, and dashboard functionality.

## 2. Current Pipeline Prototype
The current Python prototype simulates video-frame ingestion and detection
processing using separate threads.

## 3. Thread-Safe Queue
A bounded queue with a maximum capacity of 30 frames transfers frames
from ingestion to the detection worker. The queue is thread-safe.

## 4. Worker Coordination
The ingestion thread produces simulated frames.
The detection worker consumes queued frames.
A threading Event signals when ingestion has finished.
Queue task tracking helps the main thread wait for queued work to finish.

## 5. ONNX Acceleration
The project includes an ONNX export test and an inference benchmark script.
Benchmark results depend on the model, machine, input shape, and runtime.

## 6. Benchmark Limitation
The recorded benchmark uses a blank tensor. It measures model execution
time, not face-detection accuracy or full camera-to-result latency.

## 7. Current Limitations
- Video ingestion is simulated; a real camera stream is not yet connected.
- The detection worker currently simulates processing rather than detecting faces.
- Face recognition, database integration, and dashboard functionality
  are not implemented in this prototype.
- Face-specific ONNX performance must be verified separately.

## 8. Future Work
- Integrate real video sources.
- Connect a face-detection model to the pipeline.
- Measure latency and throughput with the intended model.
- Add error handling, logging, and tests.
