import cv2
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_name)
# Load the model with GPU support
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map="auto" if device.type == "cuda" else None
).to(device)

print(f"Model loaded on: {device}")

def get_webcam_image():
    """
    Captures an image from the webcam.
    """
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return None

    ret, frame = cap.read()
    cap.release()  # Release the webcam
    if not ret:
        print("Error: Unable to capture an image.")
        return None

    # Convert to RGB for compatibility with PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)  # Convert to PIL image

def main():
    # Capture an image from the webcam
    print("Capturing an image from the webcam...")
    image = get_webcam_image()
    if image is None:
        return

    # Prepare the prompt using the earlier template
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Prepare inputs
    inputs = processor(
        images=image,  # Single image from webcam
        text=prompt,  # Structured prompt
        return_tensors="pt"
    ).to(model.device, torch.float16)

    # Generate outputs
    print("Running inference...")
    outputs = model.generate(**inputs, max_new_tokens=30)

    # Decode the result
    result = processor.decode(outputs[0], skip_special_tokens=True)

    # Print the result
    print(f"Description: {result}")

if __name__ == "__main__":
    main()
