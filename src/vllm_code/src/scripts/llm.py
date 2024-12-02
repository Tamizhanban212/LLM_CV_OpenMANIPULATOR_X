import cv2
import torch
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Load a small LLM for text analysis
llm_model_name = "google/flan-t5-base"  # Replace with other suitable LLM models
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)



# Device configuration
device = "cpu"  # Change to "cuda" if you have a GPU
llm_model.to(device)



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



def main():
    
    instruction = "Place the black marker on top of the green marker."
    print(f"Instruction: {instruction}")

    # Analyze the instruction using LLM
    objects_to_find = analyze_instruction(instruction)
    print(f"Objects to find: {objects_to_find}")

    


if __name__ == "__main__":
    main()
