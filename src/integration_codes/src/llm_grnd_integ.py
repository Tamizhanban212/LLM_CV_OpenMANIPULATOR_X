import os
import cv2
import torch
import speech_recognition as sr
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from openai import OpenAI

# Suppress warnings and logs
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key="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"  # Replace with your actual Hugging Face API key
)

# Load the Grounding Dino model and processor
model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

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

def detect_objects_with_display(ordered_list, frame_start=10, frame_end=20):
    """
    Detect objects described by the input text (one at a time) from a webcam stream,
    display bounding boxes and centroids for 2 seconds, and return the centroids.

    Args:
        ordered_list: A list of object descriptions (e.g., ["object1", "object2"]).
        frame_start: The frame number to start processing from.
        frame_end: The frame number to stop processing.

    Returns:
        centroids: A list of centroids [(x1, y1), (x2, y2), ...] for all detected objects.
    """
    def process_frame(frame, text):
        # Clone the frame to ensure bounding boxes for only the current object
        frame_copy = frame.copy()
        original_height, original_width = frame_copy.shape[:2]
        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        inputs = processor(images=image, text=text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=0.5,
            text_threshold=0.4,
            target_sizes=[(original_height, original_width)]
        )

        centroids = []

        if results and len(results[0]["boxes"]) > 0:
            for box in results[0]["boxes"]:
                # Safely unpack box values
                if len(box) >= 4:  # Ensure the box has at least 4 values
                    x_min, y_min, x_max, y_max = map(int, box[:4])  # Take only the first 4 values
                    centroid_x = (x_min + x_max) // 2
                    centroid_y = (y_min + y_max) // 2
                    centroids.append((centroid_x, centroid_y))
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
                else:
                    print(f"Skipping invalid box: {box.tolist()}")
        else:
            print(f"No boxes detected for text: {text}")

        return centroids, frame_copy


    cap = cv2.VideoCapture(0)
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

        if frame_count < frame_start:
            continue

        if frame_start <= frame_count <= frame_end:
            selected_frame = frame

        if frame_count > frame_end:
            break

    cap.release()
    cv2.destroyAllWindows()

    if selected_frame is not None:
        all_centroids = []
        
        for obj in ordered_list:
            text = f"a {obj}."
            print(f"Processing: {text}")
            centroids, processed_frame = process_frame(selected_frame, text)
            all_centroids.extend(centroids)

            # Display results for the current object
            cv2.imshow(f"Object Detection Result: {obj}", processed_frame)
            cv2.waitKey(2000)
            cv2.destroyWindow(f"Object Detection Result: {obj}")

        return all_centroids
    else:
        raise RuntimeError("No frame selected for processing.")



def main():
    speech_text = "look at the cardboard box"
    ordered_list = process_speech_text(speech_text)

    if ordered_list is None:
        print("Terminating program.")
        return

    print(f"\nProcessed LLM Response: {ordered_list}")
    try:
        detected_centroids = detect_objects_with_display(ordered_list)
        print("Detected centroids:", detected_centroids)
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()
