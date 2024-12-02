import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Load the Grounding DINO model and processor
model_id = "IDEA-Research/grounding-dino-tiny"  # Regular model
device = "cuda"  # Change to "cuda" if you have a GPU
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

def process_frame(frame, text):
    """
    Detects the object described by `text` in the given frame, and returns the updated frame
    with bounding box and centroid drawn, as well as the centroid coordinates.

    Args:
        frame: The input frame from the webcam.
        text: The text describing the object to detect.

    Returns:
        frame: The frame with bounding boxes and centroids drawn.
        centroid: A tuple (centroid_x, centroid_y) representing the centroid coordinates of the object.
                  Returns None if the object is not detected.
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

    # Initialize centroid as None
    centroid = None

    # Draw results on the frame
    if results and len(results[0]["boxes"]) > 0:
        for box, score in zip(results[0]["boxes"], results[0]["scores"]):
            # Bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, box.tolist())

            # Draw the bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Calculate the centroid
            centroid_x = (x_min + x_max) // 2
            centroid_y = (y_min + y_max) // 2
            centroid = (centroid_x, centroid_y)

            # Draw the centroid
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

            # Display label and centroid
            cv2.putText(
                frame,
                f"Score: {score:.2f}",
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
            "Object not detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return frame, centroid


def main():
    # Open the webcam
    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    text = "a green marker."  # Object to detect

    print("Press 'q' to quit.")
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Process the frame
        frame, centroid = process_frame(frame, text)

        # Print the centroid if available
        if centroid:
            print(f"Centroid of '{text}': {centroid}")
        else:
            print(f"'{text}' not detected.")

        # Display the frame
        cv2.imshow("Real-Time Object Detection with Centroid", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
