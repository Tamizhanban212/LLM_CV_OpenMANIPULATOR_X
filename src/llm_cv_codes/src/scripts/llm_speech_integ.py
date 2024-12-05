import os
import speech_recognition as sr  # For speech recognition
from openai import OpenAI

# Suppress warnings and logs for a cleaner console
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize the OpenAI client
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
            "content": f"\"{speech_text}\". Give me only the objects and its properties mentioned in the previous sentence in correct order of appearance, separated by commas."
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
    
    # Check for stop keywords in the input speech text
    if any(keyword in speech_text.lower() for keyword in stop_keywords):
        print("Termination keyword detected in speech_text. Exiting.")
        return None

    # Analyze the instruction using the LLM
    response = analyze_instruction(speech_text)
    if "Error" in response:
        print("Failed to process instruction.")
        return None

    # Process the LLM response
    response_list = response.split(",")
    response_list = [r.strip() for r in response_list]

    # Remove repetitions
    response_list = list(dict.fromkeys(response_list))

    # Order response_list according to the object order in speech_text
    ordered_response_list = sorted(response_list, key=lambda x: speech_text.find(x))

    # Create the dino string
    dino_string = " ".join(f"a {r}." for r in ordered_response_list).strip()

    # Check for stop keywords in the dino string
    if any(keyword in dino_string.lower() for keyword in stop_keywords):
        print("Termination keyword detected in dino_string. Exiting.")
        return None

    return dino_string

def main():
    recognizer = sr.Recognizer()
    while True:
        # Get speech input
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

        # Process the speech text
        dino_string = process_speech_text(speech_text)
        if dino_string is None:  # Stop if termination keywords are found
            break

        print(f"\nLLM Response: {dino_string}")

if __name__ == "__main__":
    main()
