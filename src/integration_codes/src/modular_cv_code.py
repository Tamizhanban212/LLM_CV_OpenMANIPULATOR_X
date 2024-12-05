import cv2
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection, OmDetTurboForObjectDetection, AutoProcessor, AutoModelForZeroShotObjectDetection, Owlv2Processor, Owlv2ForObjectDetection
import os
from PIL import Image
from openai import OpenAI
import time

# Suppress warnings and logs
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key="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"  # Replace with your actual Hugging Face API key
)

def analyze_instruction(speech_text):
    """
    Uses the OpenAI API to analyze the instruction and generate a response.

    Args:
        speech_text: Natural language instruction.

    Returns:
        Response from the LLM.
    """
    messages = [
        {
            "role": "user",
            "content": f"\"{speech_text}\". Give me only the objects and their properties mentioned in the previous sentence in correct order of appearance, separated by commas. Don't mention any action or positions."
        }
    ]
    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct", 
            messages=messages, 
            max_tokens=500
        )
        output_content = completion.choices[0].message.content
        return output_content
    except Exception as e:
        print(f"Error during LLM inference: {e}")
        return "Error processing instruction."

def process_speech_text(speech_text):
    """
    Processes the given speech text and generates a 'dino_string'.

    Args:
        speech_text: Input speech text.

    Returns:
        Dino string (a list of objects in speech_text) or None if a termination keyword is found.
    """
    stop_keywords = ["stop", "shut", "close", "shutdown", "off", "enough", "terminate", "end"]
    
    if any(keyword in speech_text.lower() for keyword in stop_keywords):
        print("Termination keyword detected in speech_text. Exiting.")
        return None

    response = analyze_instruction(speech_text)
    if "Error" in response:
        print("Failed to process instruction.")
        return None

    response_list = response.split(",")
    response_list = [r.strip() for r in response_list]
    response_list = list(dict.fromkeys(response_list))
    ordered_response_list = sorted(response_list, key=lambda x: speech_text.find(x))

    return ordered_response_list

def detect_and_return_centroids(
    object_text,
    num_frames=2,
    camera_index=2,
    model_id=4
):

    if model_id == 1:
        model_id = "IDEA-Research/grounding-dino-tiny"
    elif model_id == 2:
        model_id = "omlab/omdet-turbo-swin-tiny-hf"
    elif model_id == 3:
        model_id = "google/owlv2-base-patch16-ensemble"
    elif model_id == 4:
        model_id = "google/owlvit-base-patch32"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    if model_id == "IDEA-Research/grounding-dino-tiny":
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    elif model_id == "omlab/omdet-turbo-swin-tiny-hf":
        processor = AutoProcessor.from_pretrained(model_id)
        model = OmDetTurboForObjectDetection.from_pretrained(model_id)
    elif model_id == "google/owlv2-base-patch16-ensemble":
        processor = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id)
    elif model_id == "google/owlvit-base-patch32":
        processor = OwlViTProcessor.from_pretrained(model_id)
        model = OwlViTForObjectDetection.from_pretrained(model_id)

    # Initialize video capture
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return None

    centroids = []
    bounding_boxes = []
    frame_count = 0

    if model_id == "IDEA-Research/grounding-dino-tiny":
        try:
            while frame_count < num_frames:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break

                # Convert the frame to PIL Image
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Process the frame and make predictions
                inputs = processor(images=image, text=object_text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)

                # Post-process results
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=[image.size[::-1]]
                )

                # Collect bounding boxes and centroids
                for box in results[0]["boxes"]:
                    x1, y1, x2, y2 = box.tolist()
                    centroid_x = (x1 + x2) / 2
                    centroid_y = (y1 + y2) / 2
                    centroids.append((centroid_x, centroid_y))
                    bounding_boxes.append((int(x1), int(y1), int(x2), int(y2)))

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
                    cv2.putText(frame, f"Avg Centroid: ({avg_x:.2f}, {avg_y:.2f})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Detected Objects and Centroid", frame)
                    cv2.waitKey(2000)  # Display for 2 seconds

                return avg_centroid
            else:
                print("No objects detected.")
                return None

        finally:
            cap.release()
            cv2.destroyAllWindows()

    elif model_id == "omlab/omdet-turbo-swin-tiny-hf":
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

    elif model_id == "google/owlv2-base-patch16-ensemble":
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
                texts = [[object_text]]
                
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
                    if predicted_label.lower() == object_text.lower():
                        box = box.tolist()
                        x1, y1, x2, y2 = box
                        centroid_x = (x1 + x2) / 2
                        centroid_y = (y1 + y2) / 2
                        centroids.append((centroid_x, centroid_y))
                        
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
        
    elif model_id == "google/owlvit-base-patch32":
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
                inputs = processor(text=[[object_text]], images=image, return_tensors="pt")
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

def get_available_models():
    """
    Lists the available object detection models.
    """
    print("Available Object Detection Models:")
    print("1. IDEA-Research/grounding-dino-tiny")
    print("2. omlab/omdet-turbo-swin-tiny-hf")
    print("3. google/owlv2-base-patch16-ensemble")
    print("4. google/owlvit-base-patch32")

    return int(input("Enter the model ID to use: "))

def main():
    object_text = "a blue colored object."
    model_id = get_available_models()
    centroid = detect_and_return_centroids(object_text, model_id=model_id)
    print(f"Detected centroid for '{object_text}': {centroid}")

if __name__ == "__main__":
    main()
