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
        except sr.UnknownValueError:
            print("Could not understand the audio. Please try again.")
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

        print(f"\nInstruction: {speech_text}")

        # Analyze the instruction using LLM
        response = analyze_instruction(speech_text)
        if "Error" in response:
            print("Failed to process instruction. Please try again.")
            continue

        response_list = response.split(",")
        response_list = [r.strip() for r in response_list]

        # Remove repetitions
        response_list = list(dict.fromkeys(response_list))

        # Order response_list according to the object order in speech_text
        ordered_response_list = sorted(response_list, key=lambda x: speech_text.find(x))

        dino_string = " ".join(f"a {r}." for r in ordered_response_list).strip()

        print(f"\nLLM Response: {dino_string}")

if __name__ == "__main__":
    main()
