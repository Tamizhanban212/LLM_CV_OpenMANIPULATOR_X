import os
import cv2
import torch
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Suppress warnings and logs
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def detect_objects_with_display(model_id, text, frame_end=20):
    """
    Detect objects described by the input text (one at a time) from a webcam stream,
    display bounding boxes and centroids for 2 seconds, and return the centroids.

    Args:
        model_id: The model ID for Grounding DINO Tiny.
        text: The description of the object to detect (e.g., "a cylindrical coil").
        frame_end: The frame number to stop processing.

    Returns:
        Tuple containing centroid [x, y], confidence level, and processing time in seconds.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def process_frame(frame, text):
        frame_copy = frame.copy()
        original_height, original_width = frame_copy.shape[:2]
        image = Image.fromarray(frame_copy)

        inputs = processor(images=image, text=text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        start_time = time.time()

        with torch.no_grad():
            outputs = model(**inputs)

        processing_time = time.time() - start_time

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=0.5,
            text_threshold=0.4,
            target_sizes=[(original_height, original_width)]
        )

        centroids = []
        highest_confidence_score = 0
        highest_confidence_centroid = None

        if results and len(results[0]["boxes"]) > 0:
            for score, box in zip(results[0]["scores"], results[0]["boxes"]):
                if score > highest_confidence_score:
                    x_min, y_min, x_max, y_max = map(int, box[:4])
                    centroid_x = (x_min + x_max) // 2
                    centroid_y = (y_min + y_max) // 2
                    highest_confidence_centroid = [centroid_x, centroid_y]
                    highest_confidence_score = score.item()

                    frame_copy = frame.copy()  # Reset frame copy to remove previous drawings
                    cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.circle(frame_copy, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
                    cv2.putText(
                        frame_copy,
                        f"Centroid: ({centroid_x}, {centroid_y})",
                        (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1
                    )

        return highest_confidence_centroid, highest_confidence_score, processing_time, frame_copy

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        raise RuntimeError("Error: Unable to access the webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    selected_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame_count += 1

        if frame_count > frame_end:
            selected_frame = frame
            break

    cap.release()
    cv2.destroyAllWindows()

    if selected_frame is not None:
        print(f"Processing: {text}")
        centroid, confidence, processing_time, processed_frame = process_frame(selected_frame, text)

        # Display result for the current object
        cv2.imshow(f"Object Detection Result: {text}", processed_frame)
        cv2.waitKey(2000)
        cv2.destroyWindow(f"Object Detection Result: {text}")

        return centroid, confidence, processing_time
    else:
        raise RuntimeError("No frame selected for processing.")

# Example usage:
if __name__ == "__main__":
    try:
        model_id = "IDEA-Research/grounding-dino-tiny"
        centroid, confidence, processing_time = detect_objects_with_display(model_id, "a screwdriver.", 20)
        print("Detected centroid:", centroid)
        print("Confidence level:", confidence)
        print("Processing time (seconds):", processing_time)
    except Exception as e:
        print(str(e))
