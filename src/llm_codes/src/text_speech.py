import pyttsx3

def text_to_speech(text):
    """
    Converts the given text to speech.

    Args:
        text (str): The text to convert to speech.
    """
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()
    
    # Set properties (optional)
    # You can adjust the voice, rate, and volume
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # Selects the first voice
    engine.setProperty('rate', 150)           # Speed of speech
    engine.setProperty('volume', 0.9)         # Volume (0.0 to 1.0)
    
    # Speak the text
    engine.say(text)
    engine.runAndWait()

# Example usage
if __name__ == "__main__":
    feedback = "The task has been completed successfully. Please proceed to the next step."
    text_to_speech(feedback)
