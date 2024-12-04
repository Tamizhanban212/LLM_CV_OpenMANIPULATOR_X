import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import time

def detect_and_display_grounded_objects(
    object_text,
    frame_start=10,
    frame_end=20,
    delay=5,
    camera_index=0,
    model_id="IDEA-Research/grounding-dino-tiny"
):
    """
    Detects objects in a live video feed using GroundingDINO and displays bounding boxes and centroids.

    Args:
        object_text (str): Text query for object detection. Must be lowercase and end with a dot.
        frame_start (int): Frame number to start processing.
        frame_end (int): Frame number to stop processing.
        delay (int): Delay in seconds before starting detection.
        camera_index (int): Camera index for OpenCV video capture.
        model_id (str): Model ID for GroundingDINO.

    Returns:
        tuple: Average centroid (x, y) of detected objects, or None if no objects are detected.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # Initialize video capture
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return None

    print(f"Starting camera... waiting for {delay} seconds.")
    time.sleep(delay)  # Delay before starting detection

    centroids = []
    bounding_boxes = []
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame_count += 1

            # Skip frames before frame_start
            if frame_count < frame_start:
                continue

            # Stop processing after frame_end
            if frame_count > frame_end:
                break

            # Convert the frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Process the frame and make predictions
            inputs = processor(images=image, text=object_text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process results
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )

            # Collect bounding boxes and centroids
            for box in results[0]["boxes"]:
                if len(box) >= 4:  # Ensure valid box
                    x1, y1, x2, y2 = map(float, box[:4])  # Safely unpack first 4 values
                    centroid_x = (x1 + x2) / 2
                    centroid_y = (y1 + y2) / 2
                    centroids.append((centroid_x, centroid_y))
                    bounding_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                    print(f"Frame {frame_count}: Detected object at centroid ({centroid_x:.2f}, {centroid_y:.2f})")
                else:
                    print(f"Skipping invalid box: {box.tolist()}")

        # Compute the average centroid
        if centroids:
            avg_x = sum(x for x, y in centroids) / len(centroids)
            avg_y = sum(y for x, y in centroids) / len(centroids)
            avg_centroid = (avg_x, avg_y)

            # Display the last processed frame with bounding boxes and centroids
            for (x1, y1, x2, y2) in bounding_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (int(avg_x), int(avg_y)), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Avg Centroid: ({avg_x:.2f}, {avg_y:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Detected Objects and Centroid", frame)
            cv2.waitKey(2000)  # Display for 2 seconds

            return avg_centroid
        else:
            print("No objects detected.")
            return None

    finally:
        cap.release()
        cv2.destroyAllWindows()



# Example usage
object_text = "a screwdriver"  # Text query for detection (lowercase, ends with a dot)
average_centroid = detect_and_display_grounded_objects(object_text, frame_start=10, frame_end=20, delay=1, camera_index=2)

if average_centroid:
    print(f"Average centroid of detected objects: {average_centroid}")
else:
    print("No objects detected.")
