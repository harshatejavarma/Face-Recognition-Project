from pathlib import Path
from ultralytics import YOLO

def main():
    # Small standard YOLO model used only to test ONNX export tooling.
    # This is NOT a face-specific detection model.
    model = YOLO("yolov8n.pt")

    output_path = Path("models/yolov8n.onnx")
    model.export(format="onnx")
    print("Export command completed.")
    print("Check the generated ONNX file in the project directory.")

if __name__ == "__main__":
    main()
