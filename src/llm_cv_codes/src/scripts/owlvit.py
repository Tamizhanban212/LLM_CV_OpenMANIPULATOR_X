import os
import cv2
import torch
import time
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Suppress warnings and logs
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def detect_objects_owlvit(model_id, object_text, frame_end=20, camera_index=2):
    """
    Detect objects described by the input text from a webcam stream,
    display bounding boxes and centroids for 2 seconds, and return the centroid,
    confidence, and processing time.

    Args:
        model_id: The model ID for OWL-ViT.
        object_text: The description of the object to detect (e.g., "a photo of a cat").
        frame_end: The frame number to stop processing.
        camera_index: The index of the camera to use.

    Returns:
        Tuple containing centroid [x, y], confidence level, and processing time in seconds.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id).to(device)

    def process_frame(frame, text):
        frame_copy = frame.copy()
        original_height, original_width = frame_copy.shape[:2]
        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        inputs = processor(text=[[text]], images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        start_time = time.time()

        with torch.no_grad():
            outputs = model(**inputs)

        processing_time = time.time() - start_time

        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)[0]

        highest_confidence_centroid = None
        highest_confidence_score = 0
        highest_confidence_box = None

        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x_min, y_min, x_max, y_max = map(int, box.tolist())
            centroid_x = (x_min + x_max) // 2
            centroid_y = (y_min + y_max) // 2

            if score > highest_confidence_score:
                highest_confidence_centroid = [centroid_x, centroid_y]
                highest_confidence_score = score.item()
                highest_confidence_box = [x_min, y_min, x_max, y_max]

            cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(frame_copy, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
            cv2.putText(
                frame_copy,
                f"Centroid: ({centroid_x}, {centroid_y})",
                (x_min, y_max + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        if highest_confidence_box:
            x_min, y_min, x_max, y_max = highest_confidence_box
            centroid_x, centroid_y = highest_confidence_centroid
            cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue color for highest confidence
            cv2.circle(frame_copy, (centroid_x, centroid_y), 5, (255, 0, 0), -1)  # Blue color for highest confidence
            cv2.putText(
                frame_copy,
                f"Highest Conf: ({centroid_x}, {centroid_y})",
                (x_min, y_max + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

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
        model_id = "google/owlvit-base-patch32"
        centroid, confidence, processing_time = detect_objects_owlvit(model_id, "a circular ball bearing.", 30, 0)
        print("Detected centroid:", centroid)
        print("Confidence level:", confidence)
        print("Processing time (seconds):", processing_time)
    except Exception as e:
        print(str(e))
