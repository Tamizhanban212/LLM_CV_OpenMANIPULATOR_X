import pyttsx3

def text_to_speech(text):
    """
    Converts the given text to speech using the Gujarati voice.

    Args:
        text (str): The text to convert to speech.
    """
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()
    
    # Set properties
    voices = engine.getProperty('voices')
    
    # Set Gujarati voice (based on your index 40)
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 100)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    
    # Speak the text
    engine.say(text)
    engine.runAndWait()

# Example usage
if __name__ == "__main__":
    feedback = "Task has been Accomplished successfully, please proceed to the next step." 
    text_to_speech(feedback)


# import pyttsx3

# def list_voices():
#     engine = pyttsx3.init()
#     voices = engine.getProperty('voices')
#     for index, voice in enumerate(voices):
#         print(f"Voice {index}: {voice.name} ({voice.languages})")

# list_voices()


# engine = pyttsx3.init()
# voices = engine.getProperty('voices')
# for voice in voices:
#    engine.setProperty('voice', voice.id)
#    engine.say('The quick brown fox jumped over the lazy dog.')
# engine.runAndWait()
