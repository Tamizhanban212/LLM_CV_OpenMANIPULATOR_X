import cv2
import torch
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Load a small LLM for text analysis
llm_model_name = "google/flan-t5-base"  # Replace with other suitable LLM models
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

# Load the Grounding DINO model for object detection
dino_model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(dino_model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id)

# Device configuration
device = "cpu"  # Change to "cuda" if you have a GPU
llm_model.to(device)
model.to(device)


def analyze_instruction(instruction):
    """
    Uses a small LLM to analyze the instruction and generate prompts for Grounding DINO.

    Args:
        instruction: Natural language instruction.

    Returns:
        A list of objects to detect based on the instruction.
    """
    # Tokenize and generate response
    input_ids = llm_tokenizer.encode(instruction, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = llm_model.generate(input_ids, max_new_tokens=50)

    # Decode the output
    response = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"LLM Response: {response}")

    # Extract objects from the response
    objects_to_detect = []
    if "green marker" in response:
        objects_to_detect.append("a green marker.")
    if "black marker" in response:
        objects_to_detect.append("a black marker.")

    # Ensure both objects are included
    if not objects_to_detect:
        print("LLM failed to identify objects correctly. Using defaults.")
        return ["a green marker.", "a black marker."]
    
    return objects_to_detect


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
    original_height, original_width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    # Debug: Print the input text
    print(f"Processing frame with text: {text}")

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
            x_min, y_min, x_max, y_max = map(int, box.tolist())
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw bounding box
            centroid_x = (x_min + x_max) // 2
            centroid_y = (y_min + y_max) // 2
            centroid = (centroid_x, centroid_y)
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Draw centroid
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
        print(f"No objects detected for text: {text}")
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    instruction = "Place the black marker on top of the green marker."
    print(f"Instruction: {instruction}")

    # Analyze the instruction using LLM
    objects_to_find = analyze_instruction(instruction)
    print(f"Objects to find: {objects_to_find}")

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        positions = {}
        for obj in objects_to_find:
            frame, centroid = process_frame(frame, obj)
            positions[obj] = centroid

        # Display results on the frame
        for obj, centroid in positions.items():
            if centroid:
                print(f"{obj.capitalize()} detected at {centroid}")
            else:
                print(f"{obj.capitalize()} not detected.")

        # Prepare feedback prompt for the LLM
        if all(positions[obj] is not None for obj in objects_to_find):
            centroid_feedback = ". ".join([f"{obj.capitalize()} detected at {positions[obj]}" for obj in objects_to_find]) + "."
            print(f"Feedback to LLM: {centroid_feedback}")

            # Feed feedback back to LLM
            input_ids = llm_tokenizer.encode(centroid_feedback, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = llm_model.generate(input_ids, max_new_tokens=50)

            # Decode the LLM response
            llm_response = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"LLM Response: {llm_response}")

        cv2.imshow("Real-Time Object Detection and LLM Execution", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
