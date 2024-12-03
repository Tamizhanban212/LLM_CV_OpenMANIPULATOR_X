import cv2
import torch
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO/WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the Grounding DINO model and processor
model_id = "IDEA-Research/grounding-dino-tiny"  # Use the regular model
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


def detect_objects_with_display(text, frame_start=5, frame_end=10):
    """
    Detect objects described by the input text from a webcam stream, display bounding boxes
    and centroids for 2 seconds, and return the centroids.

    Args:
        text: The description of objects to detect (e.g., "a spanner. a black marker.").
        frame_start: The frame number to start processing from.
        frame_end: The frame number to stop processing.

    Returns:
        centroids: A list of centroids [(x1, y1), (x2, y2), ...] for all detected objects.
    """
    def process_frame(frame, text):
        original_height, original_width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Process inputs
        inputs = processor(images=image, text=text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move tensors to device

        # Perform object detection
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process the results
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=0.5,
            text_threshold=0.4,
            target_sizes=[(original_height, original_width)]  # Match to original frame size
        )

        centroids = []

        if results and len(results[0]["boxes"]) > 0:
            for box in results[0]["boxes"]:
                x_min, y_min, x_max, y_max = map(int, box.tolist())

                # Calculate the centroid
                centroid_x = (x_min + x_max) // 2
                centroid_y = (y_min + y_max) // 2
                centroids.append((centroid_x, centroid_y))

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Draw centroid
                cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

                # Display label and centroid
                cv2.putText(
                    frame,
                    f"Centroid: ({centroid_x}, {centroid_y})",
                    (x_min, y_max + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1
                )

        return centroids, frame

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Error: Unable to access the webcam.")

    # Set the resolution to 720x720
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

        if frame_count < frame_start:
            continue

        if frame_start <= frame_count <= frame_end:
            selected_frame = frame

        if frame_count > frame_end:
            break

    cap.release()
    cv2.destroyAllWindows()

    if selected_frame is not None:
        centroids, processed_frame = process_frame(selected_frame, text)

        # Display the processed frame with bounding boxes and centroids for 2 seconds
        cv2.imshow("Object Detection Result", processed_frame)
        cv2.waitKey(2000)  # Display for 2 seconds
        cv2.destroyWindow("Object Detection Result")

        return centroids
    else:
        raise RuntimeError("No frame selected for processing.")


# Example usage:
if __name__ == "__main__":
    object_description = "a coil. a tape."
    try:
        detected_centroids = detect_objects_with_display(object_description)
        print("Detected centroids:", detected_centroids)
    except Exception as e:
        print(str(e))
