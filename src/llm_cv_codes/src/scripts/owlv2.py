import cv2
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import time

def detect_and_display_average_centroid_owlv2(object_to_detect, num_frames=10, delay=5, camera_index=0):
    """
    Detects the average centroid of the specified object in a live camera feed using OWLv2
    and displays the bounding boxes and centroid on the final frame for 2 seconds.

    Args:
        object_to_detect (str): The object to detect.
        num_frames (int): Number of frames to process after delay.
        delay (int): Delay in seconds before starting detection.
        camera_index (int): Index of the camera (default is 0).

    Returns:
        tuple: Average centroid (x, y) of the detected object, or None if no centroids found.
    """
    # Load the model and processor
    model_id = "google/owlv2-base-patch16-ensemble"
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id)

    # Initialize video capture
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return None

    print(f"Starting camera... waiting for {delay} seconds.")
    time.sleep(delay)  # Wait before starting detection

    centroids = []
    frame_count = 0
    final_frame = None

    try:
        while frame_count < num_frames:  # Process specified number of frames
            # Capture a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Convert the frame (BGR to RGB) and to PIL Image format
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Prepare text input for the processor
            texts = [[object_to_detect]]
            
            # Process the image and make predictions
            inputs = processor(text=texts, images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            # Target image sizes to rescale box predictions
            target_sizes = torch.Tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.1
            )[0]
            
            # Retrieve predictions for the target object
            for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                predicted_label = texts[0][label]
                if predicted_label.lower() == object_to_detect.lower():
                    box = box.tolist()
                    x1, y1, x2, y2 = box
                    centroid_x = (x1 + x2) / 2
                    centroid_y = (y1 + y2) / 2
                    centroids.append((centroid_x, centroid_y))
                    print(f"Frame {frame_count + 1}: Detected '{object_to_detect}' at centroid ({centroid_x:.2f}, {centroid_y:.2f})")
                    
                    # Draw bounding box and centroid
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(centroid_x), int(centroid_y)), 5, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"Centroid: ({centroid_x:.2f}, {centroid_y:.2f})",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            final_frame = frame  # Save the final processed frame
            frame_count += 1

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    # Compute the average centroid
    if centroids:
        avg_x = sum(x for x, y in centroids) / len(centroids)
        avg_y = sum(y for x, y in centroids) / len(centroids)

        # Display the final frame for 2 seconds with bounding boxes and the average centroid
        if final_frame is not None:
            cv2.circle(final_frame, (int(avg_x), int(avg_y)), 10, (255, 0, 0), -1)
            cv2.putText(
                final_frame,
                f"Avg Centroid: ({avg_x:.2f}, {avg_y:.2f})",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.imshow("Bounding Boxes and Centroid", final_frame)
            cv2.waitKey(2000)  # Display for 2 seconds
            cv2.destroyAllWindows()

        return avg_x, avg_y
    else:
        print("No centroids detected.")
        return None

# Example usage
object_to_detect = "a cardboard box"
average_centroid = detect_and_display_average_centroid_owlv2(object_to_detect, num_frames=2, delay=3, camera_index=2)

if average_centroid:
    print(f"Average centroid of detected object: {average_centroid}")
else:
    print("No object detected.")
