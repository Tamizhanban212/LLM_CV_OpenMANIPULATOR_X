import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import speech_recognition as sr  # For speech recognition

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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

def get_speech_text():
    """
    Captures speech input and converts it to text using SpeechRecognition library.

    Returns:
        Transcribed speech text.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Please speak your instruction.")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            speech_text = recognizer.recognize_google(audio)
            print(f"Recognized Speech: {speech_text}")
            return speech_text
        except sr.WaitTimeoutError:
            print("Listening timed out. Please try again.")
            return None
        except sr.UnknownValueError:
            print("Could not understand the audio. Please try again.")
            return None
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
            return None

def main():
    while True:
        speech_text = get_speech_text()
        if not speech_text:
            continue  # Retry if no valid speech text is obtained
        
        # Check for stop keywords
        if any(keyword in speech_text.lower() for keyword in ["stop", "enough", "terminate", "end"]):
            print("Termination keyword detected. Exiting.")
            break

        instruction = f"\"{speech_text}\". Give me only the objects mentioned in the previous sentence separated by commas."
        print(f"\nInstruction: {instruction}")

        # Analyze the instruction using LLM
        response = analyze_instruction(instruction)
        response_list = response.split(",")
        response_list = [r.strip() for r in response_list]

        # Remove repetitions
        response_list = list(dict.fromkeys(response_list))

        # Order response_list according to the object order in speech_text
        ordered_response_list = sorted(response_list, key=lambda x: speech_text.find(x))

        dino_string = ""
        for r in ordered_response_list:
            dino_string += "a " + r + ". "
        dino_string = dino_string.strip()

        print(f"\nLLM Response: {dino_string}")

if __name__ == "__main__":
    main()
