import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, OmDetTurboForObjectDetection
import time

def detect_and_display_with_omdet(
    object_text,
    num_frames=10,
    delay=5,
    camera_index=0,
    model_id="omlab/omdet-turbo-swin-tiny-hf"
):
    """
    Detects a single object in a live video feed using OmDet-Turbo and displays bounding boxes and centroids.

    Args:
        object_text (str): Text query for object detection (single object, e.g., "cardboard box").
        num_frames (int): Number of frames to process.
        delay (int): Delay in seconds before starting detection.
        camera_index (int): Camera index for OpenCV video capture.
        model_id (str): Model ID for OmDet-Turbo.

    Returns:
        tuple: Average centroid (x, y) of detected object, or None if no object is detected.
    """
    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_id)
    model = OmDetTurboForObjectDetection.from_pretrained(model_id)

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
        while frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Convert the frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Process the frame and make predictions
            inputs = processor(image, text=[object_text], return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process results
            results = processor.post_process_grounded_object_detection(
                outputs,
                classes=[object_text],
                target_sizes=[image.size[::-1]],
                score_threshold=0.3,
                nms_threshold=0.3,
            )[0]

            # Collect bounding boxes and centroids
            for score, class_name, box in zip(
                results["scores"], results["classes"], results["boxes"]
            ):
                x1, y1, x2, y2 = box.tolist()
                centroid_x = (x1 + x2) / 2
                centroid_y = (y1 + y2) / 2
                centroids.append((centroid_x, centroid_y))
                bounding_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                print(
                    f"Frame {frame_count + 1}: Detected {class_name} with confidence "
                    f"{round(score.item(), 2)} at centroid ({centroid_x:.2f}, {centroid_y:.2f})"
                )

            frame_count += 1

        # Compute the average centroid
        if centroids:
            avg_x = sum(x for x, y in centroids) / len(centroids)
            avg_y = sum(y for x, y in centroids) / len(centroids)
            avg_centroid = (avg_x, avg_y)

            # Display the last processed frame with bounding boxes and centroids
            ret, frame = cap.read()
            if ret:
                for (x1, y1, x2, y2) in bounding_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(avg_x), int(avg_y)), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"Avg Centroid: ({avg_x:.2f}, {avg_y:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Detected Object and Centroid", frame)
                cv2.waitKey(2000)  # Display for 2 seconds

            return avg_centroid
        else:
            print("No object detected.")
            return None

    finally:
        cap.release()
        cv2.destroyAllWindows()


# Example usage
object_text = "cardboard box"  # Text input for a single object
average_centroid = detect_and_display_with_omdet(object_text, num_frames=2, delay=3, camera_index=2)

if average_centroid:
    print(f"Average centroid of detected object: {average_centroid}")
else:
    print("No object detected.")
