import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Load the Grounding DINO model and processor
model_id = "IDEA-Research/grounding-dino-tiny"  # Use the regular model
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

def process_frame(frame, text):
    """
    Detects the objects described by `text` in the given frame, and returns an array of centroids.

    Args:
        frame: The input frame from the webcam.
        text: The text describing the objects to detect.

    Returns:
        centroids: A list of centroids [(x1, y1), (x2, y2), ...] for all detected objects.
        results: The detection results including bounding boxes.
    """
    # Convert frame to PIL format for processing
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

    # Initialize a list for centroids
    centroids = []

    # Collect results
    if results and len(results[0]["boxes"]) > 0:
        for box in results[0]["boxes"]:
            # Bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, box.tolist())

            # Calculate the centroid
            centroid_x = (x_min + x_max) // 2
            centroid_y = (y_min + y_max) // 2
            centroids.append((centroid_x, centroid_y))

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw centroid
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

            # Display label and centroid on the frame
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

def main():
    # Open the webcam
    cap = cv2.VideoCapture(4)  # Adjust index if multiple webcams are connected
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    text = "a spanner. a box."  # Objects to detect
    frame_start = 5  # Start processing from the 5th frame
    frame_end = 15  # Stop after processing the 15th frame
    frame_count = 0

    print("Capturing selected frames...")

    selected_frame = None  # To store the last processed frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame_count += 1

        # Skip frames before the start frame
        if frame_count < frame_start:
            continue

        # Process and save the frame between frame_start and frame_end
        if frame_start <= frame_count <= frame_end:
            selected_frame = frame

        # Stop processing after frame_end
        if frame_count > frame_end:
            break

    print("Processing the selected frame...")

    # Process the last selected frame
    if selected_frame is not None:
        centroids, processed_frame = process_frame(selected_frame, text)

        # Print the array of centroids
        print("Centroids of detected objects:", centroids)

        # Display the processed frame
        cv2.imshow("Object Detection Result", processed_frame)

        # Close the window automatically after 10 seconds
        cv2.waitKey(10000)  # 10 seconds in milliseconds
        cv2.destroyWindow("Object Detection Result")
    else:
        print("No frame selected for processing.")

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
