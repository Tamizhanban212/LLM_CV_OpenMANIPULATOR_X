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

def detect_objects(model_id, object_text, frame_end=20, camera_index=2):
    """
    Detect objects described by the input text from a webcam stream,
    display bounding boxes and centroids for 2 seconds, and return the centroid,
    confidence, and processing time.

    Args:
        model_id: The model ID for Grounding DINO Tiny.
        object_text: The description of the object to detect (e.g., "a cylindrical coil").
        frame_end: The frame number to stop processing.
        camera_index: The index of the camera to use.

    Returns:
        Tuple containing centroid [x, y], confidence level, and processing time in seconds.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def process_frame(frame, text):
        frame_copy = frame.copy()
        original_height, original_width = frame_copy.shape[:2]
        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

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

        highest_confidence_centroid = None
        highest_confidence_score = 0

        if results and len(results[0]["boxes"]) > 0:
            for box, score in zip(results[0]["boxes"], results[0]["scores"]):
                if len(box) >= 4:
                    x_min, y_min, x_max, y_max = map(int, box[:4])
                    centroid_x = (x_min + x_max) // 2
                    centroid_y = (y_min + y_max) // 2

                    color = (0, 255, 0)  # Green color for regular bounding boxes
                    if score > highest_confidence_score:
                        highest_confidence_centroid = [centroid_x, centroid_y]
                        highest_confidence_score = score.item()
                        color = (255, 0, 0)  # Blue color for the highest confidence bounding box

                    cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.circle(frame_copy, (centroid_x, centroid_y), 5, color, -1)
                    cv2.putText(
                        frame_copy,
                        f"Centroid: ({centroid_x}, {centroid_y})",
                        (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )
                else:
                    print(f"Skipping invalid box: {box.tolist()}")
        else:
            print(f"No boxes detected for text: {text}")

        return highest_confidence_centroid, highest_confidence_score, processing_time, frame_copy

    cap = cv2.VideoCapture(camera_index)
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
        print(f"Processing: {object_text}")
        centroid, confidence, processing_time, processed_frame = process_frame(selected_frame, object_text)

        # Display results for the current object
        cv2.imshow(f"Object Detection Result: {object_text}", processed_frame)
        cv2.waitKey(2000)
        cv2.destroyWindow(f"Object Detection Result: {object_text}")

        return centroid, confidence, processing_time
    else:
        raise RuntimeError("No frame selected for processing.")

# Example usage:
if __name__ == "__main__":
    try:
        model_id = "IDEA-Research/grounding-dino-tiny"
        centroid, confidence, processing_time = detect_objects(model_id, "a blue circular object.", 20, 2)
        print("Detected centroid:", centroid)
        print("Confidence level:", confidence)
        print("Processing time (seconds):", processing_time)
    except Exception as e:
        print(str(e))
