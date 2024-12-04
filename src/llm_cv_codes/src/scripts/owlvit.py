import cv2
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import time

def detect_and_display_average_centroid(object_to_detect, num_frames=10, delay=5, camera_index=0):
    """
    Detects the average centroid of the specified object in a live camera feed and displays it.

    Args:
        object_to_detect (str): The object to detect.
        num_frames (int): Number of frames to process after delay.
        delay (int): Delay in seconds before starting detection.
        camera_index (int): Index of the camera (default is 0).

    Returns:
        tuple: Average centroid (x, y) of the detected object, or None if no centroids found.
    """
    model_id = "google/owlvit-base-patch32"

    # Load the model and processor
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id)

    # Initialize video capture
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return None

    print(f"Starting camera... waiting for {delay} seconds.")
    time.sleep(delay)  # Wait before starting detection

    centroids = []
    bounding_boxes = []
    frame_count = 0

    try:
        while frame_count < num_frames:  # Process specified number of frames
            # Capture a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Convert the frame (BGR to RGB) and to PIL Image format
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Process the image and make predictions
            inputs = processor(text=[[object_to_detect]], images=image, return_tensors="pt")
            outputs = model(**inputs)
            
            # Target image sizes (height, width) to rescale box predictions
            target_sizes = torch.Tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs=outputs, threshold=0.1, target_sizes=target_sizes
            )
            
            # Retrieve predictions for the first image
            boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
            
            # Collect centroids and bounding boxes
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                centroid_x = (x1 + x2) / 2
                centroid_y = (y1 + y2) / 2
                centroids.append((centroid_x, centroid_y))
                bounding_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                print(f"Frame {frame_count + 1}: Detected '{object_to_detect}' at centroid ({centroid_x:.2f}, {centroid_y:.2f})")
            
            frame_count += 1

        # Compute the average centroid
        if centroids:
            avg_x = sum(x for x, y in centroids) / len(centroids)
            avg_y = sum(y for x, y in centroids) / len(centroids)
            avg_centroid = (avg_x, avg_y)

            # Display the bounding boxes and average centroid
            ret, frame = cap.read()
            if ret:
                for (x1, y1, x2, y2) in bounding_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(avg_x), int(avg_y)), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Avg Centroid: ({avg_x:.2f}, {avg_y:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Detected Objects and Centroid", frame)
                cv2.waitKey(2000)  # Display for 2 seconds

            return avg_centroid
        else:
            print("No centroids detected.")
            return None

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


# Example usage
object_to_detect = "a cardboard box"
average_centroid = detect_and_display_average_centroid(object_to_detect, num_frames=2, delay=3, camera_index=2)

if average_centroid:
    print(f"Average centroid of detected object: {average_centroid}")
else:
    print("No object detected.")
