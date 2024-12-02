import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO/WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load a small LLM for text analysis
llm_model_name = "google/flan-t5-base"  # Replace with other suitable LLM models
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
llm_model.to(device)

def analyze_instruction(instruction):
    """
    Uses a small LLM to analyze the instruction and generate a response.

    Args:
        instruction: Natural language instruction.

    Returns:
        Response from the LLM.
    """
    # Tokenize and generate response
    input_ids = llm_tokenizer.encode(instruction, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = llm_model.generate(input_ids, max_new_tokens=50)

    # Decode the output
    response = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def main():
    instruction = "\"Place the red marker on top of the my hand.\" Give all the objects and its properties in the sentence separated by commas in the correct order of appearance in the sentence. Dont give me the action words."
    print(f"\nInstruction: {instruction}")

    # Analyze the instruction using LLM
    response = analyze_instruction(instruction)
    response = response.split(",")
    response = [r.strip() for r in response]
    dino_string = ""
    for r in response:
        dino_string += "a " + r + ". "
    dino_string = dino_string.strip()
    print(f"\nLLM Response: {dino_string}")

if __name__ == "__main__":
    main()
