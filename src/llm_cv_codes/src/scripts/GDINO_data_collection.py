import os
import cv2
import torch
import time
import csv
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Suppress warnings and logs
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def detect_objects_from_image(model_id, image_path):
    """
    Detect objects described by the image file name from a local image file,
    display bounding boxes for 5 seconds in a fixed window size, and return the confidence and processing time.

    Args:
        model_id: The model ID for Grounding DINO Tiny.
        image_path: Path to the local image file (e.g., "path/to/image.jpg").

    Returns:
        Tuple containing input text, confidence level, and processing time in seconds.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def process_image(image, text):
        frame_copy = image.copy()
        original_height, original_width = frame_copy.shape[:2]
        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        inputs = processor(images=image_pil, text=text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        start_time = time.time()

        with torch.no_grad():
            outputs = model(**inputs)

        processing_time = time.time() - start_time

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=0.5,
            text_threshold=0.4,
            target_sizes=[(original_height, original_width)]
        )

        highest_confidence_score = 0
        highest_confidence_box = None

        if results and len(results[0]["boxes"]) > 0:
            for box, score in zip(results[0]["boxes"], results[0]["scores"]):
                if len(box) >= 4:
                    x_min, y_min, x_max, y_max = map(int, box[:4])

                    if score > highest_confidence_score:
                        highest_confidence_score = score.item()
                        highest_confidence_box = [x_min, y_min, x_max, y_max]

                    cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            if highest_confidence_box:
                x_min, y_min, x_max, y_max = highest_confidence_box
                cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue color for highest confidence
        else:
            print(f"No boxes detected for text: {text}")

        return highest_confidence_score, processing_time, frame_copy

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError("Error: Unable to read the image file.")

    # Extract object description from image file name
    file_name = os.path.basename(image_path)
    object_text = "a " + file_name.replace("_", " ").rsplit(".", 1)[0] + "."

    print(f"Processing: {object_text}")
    confidence, processing_time, processed_frame = process_image(image, object_text)

    # Display results for 5 seconds in a fixed window size
    window_name = "Object Detection Result"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 720, 720)
    cv2.imshow(window_name, processed_frame)
    cv2.waitKey(3500)
    cv2.destroyWindow(window_name)

    return file_name, object_text, confidence, processing_time

def process_folder(model_id, folder_path, csv_file_path):
    """
    Iterate over all JPG images in the folder, process each image, and store the results in a CSV file.

    Args:
        model_id: The model ID for Grounding DINO Tiny.
        folder_path: Path to the folder containing JPG images.
        csv_file_path: Path to the CSV file to store the results.
    """
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image file name", "input text", "1/0", "confidence", "processing time"])

        for image_file in os.listdir(folder_path):
            if image_file.endswith(".jpg"):
                image_path = os.path.join(folder_path, image_file)
                file_name, object_text, confidence, processing_time = detect_objects_from_image(model_id, image_path)
                try:
                    display_parameter = int(input(f"1 or 0 for {file_name}? "))  # Set this to 0 or 1 based on user input   
                except ValueError:
                    display_parameter = int(input(f"1 or 0 for {file_name}? "))  # Set this to 0 or 1 based on user input
                writer.writerow([file_name, object_text, display_parameter, confidence, processing_time])

# Example usage:
if __name__ == "__main__":
    try:
        model_id = "IDEA-Research/grounding-dino-tiny"
        folder_path = "/home/tamizhanban/Downloads/sample_data/Electrical"  # Update with the path to your folder
        csv_file_path = "GDINO_electrical_tools.csv"  # Path to the output CSV file
        process_folder(model_id, folder_path, csv_file_path)
    except Exception as e:
        print(str(e))
