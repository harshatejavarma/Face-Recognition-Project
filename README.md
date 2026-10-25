# Face Recognition System

## Project Overview
This project aims to develop a face recognition system with a modular architecture for video-frame ingestion, face detection, face recognition, database management, and dashboard integration.

## My Role: System Architecture & Acceleration
My responsibility is to establish the project structure, develop the initial processing pipeline, and investigate model acceleration using ONNX Runtime.

## Work Completed

### 1. Repository Setup
- Created the central GitHub repository.
- Configured `.gitignore` to exclude Python cache files, virtual environments, and large model-weight files.
- Organized the project into separate modules.

### 2. Project Structure
- `ingestion/` — Video and frame input.
- `detection/` — Face detection.
- `recognition/` — Face recognition.
- `database/` — Data storage.
- `dashboard/` — User interface.
- `models/` — Local model files.
- `tests/` — Export and performance benchmark scripts.
- `main.py` — Initial pipeline prototype.

### 3. Pipeline Prototype
Implemented a Python pipeline prototype using:
- A thread-safe queue to transfer frames between processing stages.
- Separate ingestion and detection-worker threads.
- A bounded queue with a maximum capacity of 30 frames.
- Thread synchronization and queue task tracking.

**Current limitation:** The prototype uses simulated frames. Real camera ingestion and actual face detection have not yet been integrated into this pipeline.

### 4. ONNX Export and Performance Testing
Exported a YOLO face-detection model to ONNX format and tested inference using ONNX Runtime.

**Benchmark results**
- Model: `models/yolov8n-face.onnx`
- Input shape: `1 × 3 × 640 × 640`
- Execution provider: CPU
- Average inference latency: 33.03 ms
- Approximate throughput: 30.27 FPS

The benchmark used blank input tensors, three warm-up runs, and 20 timed runs. These results measure model execution speed, not face-detection accuracy or complete camera-to-result latency.

### 5. Documentation
Created architecture documentation and recorded benchmark results to support reproducibility and future integration.

## How to Run the Pipeline Prototype

Install Python 3.10 or later and the project dependencies listed in `requirements.txt`.

Run:

```bash
python main.py
```

The prototype should print simulated frames being queued and processed.

## How to Run the Face ONNX Benchmark

Ensure the face model is available locally at `models/yolov8n-face.onnx` and the required dependencies are installed.

Run:

```bash
python tests/benchmark_face_onnx.py
```

The benchmark uses blank input tensors and CPU inference.

## Limitations and Future Work
- Integrate real video or camera streams.
- Connect the pipeline to an actual face-detection model.
- Evaluate accuracy using appropriate test images or video.
- Measure end-to-end latency and throughput.
- Integrate recognition, database, and dashboard modules.

## Contribution Summary
My contribution establishes the initial architecture and processing pipeline, organizes the repository, and evaluates ONNX inference performance for a face-detection model. This provides a foundation for integration and further testing by the project team.
