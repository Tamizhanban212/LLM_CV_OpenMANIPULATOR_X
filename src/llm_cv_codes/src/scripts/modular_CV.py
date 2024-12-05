import os
import cv2
import torch
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, OmDetTurboForObjectDetection, Owlv2Processor, Owlv2ForObjectDetection, OwlViTProcessor, OwlViTForObjectDetection

# Suppress warnings and logs
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def detect_objects(model_id, object_text, frame_end=30, camera_index=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_id == 1:
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    elif model_id == 2:
        model_id = "omlab/omdet-turbo-swin-tiny-hf"
        processor = AutoProcessor.from_pretrained(model_id)
        model = OmDetTurboForObjectDetection.from_pretrained(model_id).to(device)
    
    elif model_id == 3:
        model_id = "google/owlv2-base-patch16-ensemble"
        processor = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)
    
    elif model_id == 4:
        model_id = "google/owlvit-base-patch32"
        processor = OwlViTProcessor.from_pretrained(model_id)
        model = OwlViTForObjectDetection.from_pretrained(model_id).to(device)
    
    else:
        raise ValueError("Invalid model ID. Please choose from 1, 2, 3, or 4.")
    
    def process_frame_dino(frame, text):
        frame_copy = frame.copy()
        original_height, original_width = frame_copy.shape[:2]
        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        inputs = processor(images=image, text=text, return_tensors="pt")
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

        highest_confidence_centroid = None
        highest_confidence_score = 0
        highest_confidence_box = None

        if results and len(results[0]["boxes"]) > 0:
            for box, score in zip(results[0]["boxes"], results[0]["scores"]):
                if len(box) >= 4:
                    x_min, y_min, x_max, y_max = map(int, box[:4])
                    centroid_x = (x_min + x_max) // 2
                    centroid_y = (y_min + y_max) // 2

                    if score > highest_confidence_score:
                        highest_confidence_centroid = [centroid_x, centroid_y]
                        highest_confidence_score = score.item()
                        highest_confidence_box = [x_min, y_min, x_max, y_max]

                    cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.circle(frame_copy, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
                    cv2.putText(
                        frame_copy,
                        f"Centroid: ({centroid_x}, {centroid_y})",
                        (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

            if highest_confidence_box:
                x_min, y_min, x_max, y_max = highest_confidence_box
                centroid_x, centroid_y = highest_confidence_centroid
                cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue color for highest confidence
                cv2.circle(frame_copy, (centroid_x, centroid_y), 5, (255, 0, 0), -1)  # Blue color for highest confidence
                cv2.putText(
                    frame_copy,
                    f"Highest Conf: ({centroid_x}, {centroid_y})",
                    (x_min, y_max + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1
                )
        else:
            print(f"No boxes detected for text: {text}")

        return highest_confidence_centroid, highest_confidence_score, processing_time, frame_copy

    def process_frame_omdet(frame, text):
        frame_copy = frame.copy()
        original_height, original_width = frame_copy.shape[:2]
        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)        
        image = Image.fromarray(frame_rgb)

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
        highest_confidence_box = None

        for score, class_name, box in zip(results["scores"], results["classes"], results["boxes"]):
            x_min, y_min, x_max, y_max = map(int, box[:4])
            centroid_x = (x_min + x_max) // 2
            centroid_y = (y_min + y_max) // 2

            if score > highest_confidence_score:
                highest_confidence_centroid = (centroid_x, centroid_y)
                highest_confidence_score = score.item()
                highest_confidence_box = (x_min, y_min, x_max, y_max)

            cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(frame_copy, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
            cv2.putText(
                frame_copy,
                f"Centroid: ({centroid_x}, {centroid_y})",
                (x_min, y_max + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        if highest_confidence_box:
            x_min, y_min, x_max, y_max = highest_confidence_box
            centroid_x, centroid_y = highest_confidence_centroid
            cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue color for highest confidence
            cv2.circle(frame_copy, (centroid_x, centroid_y), 5, (255, 0, 0), -1)  # Blue color for highest confidence
            cv2.putText(
                frame_copy,
                f"Highest Conf: ({centroid_x}, {centroid_y})",
                (x_min, y_max + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        return highest_confidence_centroid, highest_confidence_score, processing_time, frame_copy

    def process_frame_owlv2(frame, text):
        frame_copy = frame.copy()
        original_height, original_width = frame_copy.shape[:2]
        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        inputs = processor(text=[[text]], images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        start_time = time.time()

        with torch.no_grad():
            outputs = model(**inputs)

        processing_time = time.time() - start_time

        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)[0]

        highest_confidence_centroid = None
        highest_confidence_score = 0
        highest_confidence_box = None

        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x_min, y_min, x_max, y_max = map(int, box.tolist())
            centroid_x = (x_min + x_max) // 2
            centroid_y = (y_min + y_max) // 2

            if score > highest_confidence_score:
                highest_confidence_centroid = [centroid_x, centroid_y]
                highest_confidence_score = score.item()
                highest_confidence_box = [x_min, y_min, x_max, y_max]

            cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(frame_copy, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
            cv2.putText(
                frame_copy,
                f"Centroid: ({centroid_x}, {centroid_y})",
                (x_min, y_max + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        if highest_confidence_box:
            x_min, y_min, x_max, y_max = highest_confidence_box
            centroid_x, centroid_y = highest_confidence_centroid
            cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue color for highest confidence
            cv2.circle(frame_copy, (centroid_x, centroid_y), 5, (255, 0, 0), -1)  # Blue color for highest confidence
            cv2.putText(
                frame_copy,
                f"Highest Conf: ({centroid_x}, {centroid_y})",
                (x_min, y_max + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        return highest_confidence_centroid, highest_confidence_score, processing_time, frame_copy

    def process_frame_owlvit(frame, text):
        frame_copy = frame.copy()
        original_height, original_width = frame_copy.shape[:2]
        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        inputs = processor(text=[[text]], images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        start_time = time.time()

        with torch.no_grad():
            outputs = model(**inputs)

        processing_time = time.time() - start_time

        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)[0]

        highest_confidence_centroid = None
        highest_confidence_score = 0
        highest_confidence_box = None

        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x_min, y_min, x_max, y_max = map(int, box.tolist())
            centroid_x = (x_min + x_max) // 2
            centroid_y = (y_min + y_max) // 2

            if score > highest_confidence_score:
                highest_confidence_centroid = [centroid_x, centroid_y]
                highest_confidence_score = score.item()
                highest_confidence_box = [x_min, y_min, x_max, y_max]

            cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(frame_copy, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
            cv2.putText(
                frame_copy,
                f"Centroid: ({centroid_x}, {centroid_y})",
                (x_min, y_max + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        if highest_confidence_box:
            x_min, y_min, x_max, y_max = highest_confidence_box
            centroid_x, centroid_y = highest_confidence_centroid
            cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue color for highest confidence
            cv2.circle(frame_copy, (centroid_x, centroid_y), 5, (255, 0, 0), -1)  # Blue color for highest confidence
            cv2.putText(
                frame_copy,
                f"Highest Conf: ({centroid_x}, {centroid_y})",
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

        if frame_count > frame_end:
            selected_frame = frame
            break

    cap.release()
    cv2.destroyAllWindows()    

    if selected_frame is not None:
        print(f"Processing: {object_text}")
        if model_id == "IDEA-Research/grounding-dino-tiny":
            centroid, confidence, processing_time, processed_frame = process_frame_dino(selected_frame, object_text)
        
        elif model_id == "omlab/omdet-turbo-swin-tiny-hf":
            centroid, confidence, processing_time, processed_frame = process_frame_omdet(selected_frame, object_text)

        elif model_id == "google/owlv2-base-patch16-ensemble":
            centroid, confidence, processing_time, processed_frame = process_frame_owlv2(selected_frame, object_text)
    
        elif model_id == "google/owlvit-base-patch32":
            centroid, confidence, processing_time, processed_frame = process_frame_owlvit(selected_frame, object_text)        
        
        else:
            raise ValueError("Something went wrong. Please try again.")
        
        # Display results for the current object
        cv2.imshow(f"Object Detection Result: {object_text}", processed_frame)
        cv2.waitKey(2000)
        cv2.destroyWindow(f"Object Detection Result: {object_text}")

        return centroid, confidence, processing_time, processed_frame
    else:
        raise RuntimeError("No frame selected for processing.")
    
    return None

model_id = 4

centroid, confidence, processing_time, processed_frame = detect_objects(model_id, "a white circular object.", 30, 2)
print(f"Centroid: {centroid}, Confidence: {confidence}, Processing Time: {processing_time}")

centroid, confidence, processing_time, processed_frame = detect_objects(model_id, "a cardboard box.", 30, 2)
print(f"Centroid: {centroid}, Confidence: {confidence}, Processing Time: {processing_time}")
