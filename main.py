import threading
from queue import Queue, Empty
from time import sleep

# Shared, thread-safe queue for video frames
frame_queue = Queue(maxsize=30)

# Signals the worker when ingestion is finished
stop_event = threading.Event()


def ingestion():
    """Simulate incoming video frames."""
    for frame_number in range(1, 11):
        if stop_event.is_set():
            break

        frame = {
            "frame_id": frame_number,
            "source": "test_stream"
        }

        frame_queue.put(frame)
        print(f"[Ingestion] Frame {frame_number} queued")
        sleep(0.1)

    stop_event.set()


def detection_worker():
    """Receive frames and pass them to the detection stage."""
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=0.2)
        except Empty:
            continue

        try:
            print(
                f"[Detection] Processing frame "
                f"{frame['frame_id']} from {frame['source']}"
            )
        finally:
            frame_queue.task_done()


def main():
    print("Face Recognition Project")
    print("Starting ingestion and detection workers...\n")

    worker = threading.Thread(
        target=detection_worker,
        name="DetectionWorker"
    )

    producer = threading.Thread(
        target=ingestion,
        name="IngestionWorker"
    )

    worker.start()
    producer.start()

    producer.join()
    frame_queue.join()
    worker.join()

    print("\nPipeline test completed successfully.")


if __name__ == "__main__":
    main()