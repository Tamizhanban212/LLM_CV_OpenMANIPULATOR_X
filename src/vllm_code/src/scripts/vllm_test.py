import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests

# Model and processor setup
model_name = "llava-hf/llava-1.5-7b-hf"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the processor
processor = AutoProcessor.from_pretrained(model_name)

# Load the model with mixed precision
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map="auto"
)

def infer_with_llava(image_path, question):
    """
    Use LLaVA model to infer the answer to a question about an image.

    Args:
        image_path: Path or URL to the input image.
        question: The question to ask about the image.

    Returns:
        The generated response from the model.
    """
    try:
        # Load the image from path or URL
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")

        # Prepare a structured prompt for the input image
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Apply the structured chat template
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Preprocess the input
        inputs = processor(
            images=[image],
            text=[prompt],
            return_tensors="pt",
            padding=True
        ).to(device, torch.float16 if device.type == "cuda" else torch.float32)

        # Debug input shapes
        print(f"Pixel Values Shape: {inputs['pixel_values'].shape}")
        print(f"Input Text Tokens Shape: {inputs['input_ids'].shape}")

        # Generate the output with reduced max_length
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                 max_new_tokens=30
            )

        # Decode the response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error during inference: {e}")
        return "Error occurred during processing."


def main():
    # Path or URL to the input image
    image_path = "/home/aarav/LLM_CV_OpenMANIPULATOR_X/src/vllm_code/src/scripts/table.jpeg"  # Replace with your image path or URL
    question = "What is shown in this image?"

    # Infer with LLaVA
    response = infer_with_llava(image_path, question)
    print(f"LLaVA Response: {response}")


if __name__ == "__main__":
    torch.cuda.empty_cache()  # Clear GPU memory before starting
    main()
