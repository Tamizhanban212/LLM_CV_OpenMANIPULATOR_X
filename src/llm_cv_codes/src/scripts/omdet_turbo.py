import requests
import cv2
import torch
import time
from PIL import Image
from transformers import AutoProcessor, OmDetTurboForObjectDetection

def detect_objects(model_id, end_frame, camera_index, object_text):
    """
    Detect objects described by the input text from a webcam stream,
    display the bounding box and centroid of the object with the highest confidence
    for 2 seconds, and return the centroid, confidence level, and processing time.

    Args:
        model_id: The model ID for OmDet Turbo.
        end_frame: The frame number to stop processing.
        camera_index: The index of the camera to use.
        object_text: The description of the object to detect (e.g., "yellow screwdriver").

    Returns:
        Tuple containing centroid [x, y], confidence level, and processing time in seconds.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = OmDetTurboForObjectDetection.from_pretrained(model_id).to(device)

    def process_frame(frame, text):
        frame_copy = frame.copy()
        original_height, original_width = frame_copy.shape[:2]
        image = Image.fromarray(frame_copy)

        inputs = processor(images=image, text=[text], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(**inputs)

        processing_time = time.time() - start_time

        results = processor.post_process_grounded_object_detection(
            outputs,
            classes=[text],
            target_sizes=[(original_height, original_width)],
            score_threshold=0.3,
            nms_threshold=0.3
        )[0]

        highest_confidence_centroid = None
        highest_confidence_score = 0

        for score, class_name, box in zip(results["scores"], results["classes"], results["boxes"]):
            if score > highest_confidence_score:
                box = [round(i, 1) for i in box.tolist()]
                x_min, y_min, x_max, y_max = map(int, box[:4])
                centroid_x = (x_min + x_max) // 2
                centroid_y = (y_min + y_max) // 2
                highest_confidence_centroid = [centroid_x, centroid_y]
                highest_confidence_score = score.item()

                frame_copy = frame.copy()  # Reset frame copy to remove previous drawings
                cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.circle(frame_copy, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame_copy,
                    f"Centroid: ({centroid_x}, {centroid_y})",
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

        if frame_count > end_frame:
            selected_frame = frame
            break

    cap.release()
    cv2.destroyAllWindows()

    if selected_frame is not None:
        print(f"Processing: {object_text}")
        centroid, confidence, processing_time, processed_frame = process_frame(selected_frame, object_text)

        # Display result for the current object
        cv2.imshow(f"Object Detection Result: {object_text}", processed_frame)
        cv2.waitKey(2000)
        cv2.destroyWindow(f"Object Detection Result: {object_text}")

        return centroid, confidence, processing_time
    else:
        raise RuntimeError("No frame selected for processing.")

# Example usage:
if __name__ == "__main__":
    try:
        centroid, confidence, processing_time = detect_objects("omlab/omdet-turbo-swin-tiny-hf", 50, 2, "blue circular object")
        print("Detected centroid:", centroid)
        print("Confidence level:", confidence)
        print("Processing time (seconds):", processing_time)
    except Exception as e:
        print(str(e))
