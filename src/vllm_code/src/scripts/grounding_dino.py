import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Load the Grounding DINO model and processor
model_id = "IDEA-Research/grounding-dino-tiny"  # Regular model
device = "cpu"  # Change to "cuda" if you have a GPU
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    # Command for Grounding DINO
    text = "a spanner."
    frame_counter = 0  # To skip frames

    print("Press 'q' to quit.")
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Skip frames to reduce lag
        if frame_counter % 10 != 0:  # Process every 10th frame
            frame_counter += 1
            continue
        frame_counter += 1

        # Resize frame for faster processing
        frame = cv2.resize(frame, (480, 360))  # Smaller resolution

        # Convert frame to RGB and PIL image
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
            box_threshold=0.5,  # Increase box threshold
            text_threshold=0.4,  # Increase text threshold
            target_sizes=[image.size[::-1]]
        )

        # Draw results on the frame
        if results:
            for label, box, score in zip(results[0]["labels"], results[0]["boxes"], results[0]["scores"]):
                # Bounding box coordinates
                x_min, y_min, x_max, y_max = map(int, box.tolist())
                
                # Draw the bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green bounding box
                
                # Calculate the centroid
                centroid_x = (x_min + x_max) // 2
                centroid_y = (y_min + y_max) // 2
                
                # Draw the centroid
                cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Red dot
                
                # Display label and centroid (optional for reduced lag)
                cv2.putText(
                    frame,
                    f"{label}: {score:.2f}",
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

        # Display the frame with bounding boxes and centroids
        cv2.imshow("Real-Time Object Detection with Centroid", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
