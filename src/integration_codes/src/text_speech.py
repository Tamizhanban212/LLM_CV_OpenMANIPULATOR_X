from gtts import gTTS
from playsound import playsound
import os

def text_to_speech(text, lang='en'):
    try:
        # Generate speech from text
        tts = gTTS(text=text, lang=lang, slow=False)

        # Save the audio file
        audio_file = "src/integration_codes/src/voices/output.mp3"
        tts.save(audio_file)

        # Play the audio directly
        playsound(audio_file)

        # Remove the audio file after playing
        os.remove(audio_file)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    # Input text
    text = input("Enter text to convert to speech: ")
    text_to_speech(text)
