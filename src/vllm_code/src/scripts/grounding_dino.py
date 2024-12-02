import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Load the Grounding DINO model and processor
model_id = "IDEA-Research/grounding-dino-tiny"  # Regular model
device = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically use GPU if available
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

def process_frame(frame, text):
    """
    Detects the objects described by `text` in the given frame, and returns the updated frame
    with bounding boxes and centroids drawn, as well as the centroid coordinates.

    Args:
        frame: The input frame from the webcam.
        text: The text describing the objects to detect.

    Returns:
        frame: The frame with bounding boxes and centroids drawn.
        centroids: A dictionary with detected objects as keys and lists of centroids as values.
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

    # Initialize a dictionary for centroids
    centroids = {}

    # Draw results on the frame
    if results and len(results[0]["boxes"]) > 0:
        for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
            # If label is not already a string, decode it
            label_text = label if isinstance(label, str) else processor.tokenizer.decode([label]).strip()

            # Bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, box.tolist())

            # Calculate the centroid
            centroid_x = (x_min + x_max) // 2
            centroid_y = (y_min + y_max) // 2
            centroid = (centroid_x, centroid_y)

            # Add centroid to the dictionary
            if label_text not in centroids:
                centroids[label_text] = []
            centroids[label_text].append(centroid)

            # Draw the bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw the centroid
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

            # Display label and centroid
            cv2.putText(
                frame,
                f"{label_text} ({score:.2f})",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
            cv2.putText(
                frame,
                f"Centroid: ({centroid_x}, {centroid_y})",
                (x_min, y_max + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )
    else:
        cv2.putText(
            frame,
            "No objects detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return frame, centroids

def main():
    # Open the webcam
    cap = cv2.VideoCapture(4)  # Adjust index if multiple webcams are available
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    text = "a green marker. a black marker."  # Objects to detect
    frame_start = 5  # Start processing from the 5th frame
    frame_end = 15  # Stop after processing the 15th frame
    frame_count = 0

    print("Processing selected frames...")

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

        # Process and save the selected frame
        if frame_start <= frame_count <= frame_end:
            selected_frame = frame

        # Stop processing after the frame limit is reached
        if frame_count > frame_end:
            break

    print("Capturing frame for detection...")

    # Process the last selected frame
    if selected_frame is not None:
        frame, centroids = process_frame(selected_frame, text)

        # Print the centroids for each detected object
        for obj, obj_centroids in centroids.items():
            print(f"Object: {obj}, Centroids: {obj_centroids}")

        # Display the processed frame
        cv2.imshow("Object Detection Result", frame)

        # Add a proper key listener to close the window
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
    else:
        print("No frame selected for processing.")

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
