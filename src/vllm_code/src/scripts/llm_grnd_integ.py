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
model_id = "IDEA-Research/grounding-dino-tiny"
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
            "content": f"\"{speech_text}\". Give me only the objects and their properties mentioned in the previous sentence in correct order of appearance, separated by commas."
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
    stop_keywords = ["stop", "enough", "terminate", "end"]
    
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
    dino_string = " ".join(f"a {r}." for r in ordered_response_list).strip()

    if any(keyword in dino_string.lower() for keyword in stop_keywords):
        print("Termination keyword detected in dino_string. Exiting.")
        return None

    return dino_string

def detect_objects_with_display(text, frame_start=5, frame_end=10):
    """
    Detect objects described by the input text from a webcam stream, display bounding boxes
    and centroids for 2 seconds, and return the centroids.

    Args:
        text: The description of objects to detect (e.g., "a spanner. a black marker.").
        frame_start: The frame number to start processing from.
        frame_end: The frame number to stop processing.

    Returns:
        centroids: A list of centroids [(x1, y1), (x2, y2), ...] for all detected objects.
    """
    def process_frame(frame, text):
        original_height, original_width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                x_min, y_min, x_max, y_max = map(int, box.tolist())
                centroid_x = (x_min + x_max) // 2
                centroid_y = (y_min + y_max) // 2
                centroids.append((centroid_x, centroid_y))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
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
        centroids, processed_frame = process_frame(selected_frame, text)
        cv2.imshow("Object Detection Result", processed_frame)
        cv2.waitKey(2000)
        cv2.destroyWindow("Object Detection Result")
        return centroids
    else:
        raise RuntimeError("No frame selected for processing.")

def main():
    recognizer = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            print("Listening... Please speak your instruction.")
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                speech_text = recognizer.recognize_google(audio)
                print(f"Recognized Speech: {speech_text}")
            except sr.WaitTimeoutError:
                print("Listening timed out. Please try again.")
                continue
            except sr.UnknownValueError:
                print("Could not understand the audio. Please try again.")
                continue
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")
                continue

        dino_string = process_speech_text(speech_text)
        if dino_string is None:
            break

        print(f"\nProcessed LLM Response: {dino_string}")
        try:
            detected_centroids = detect_objects_with_display(dino_string)
            print("Detected centroids:", detected_centroids)
        except Exception as e:
            print(str(e))

if __name__ == "__main__":
    main()
