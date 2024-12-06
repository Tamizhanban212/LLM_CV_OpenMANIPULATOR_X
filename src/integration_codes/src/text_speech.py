from gtts import gTTS
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
import os

def change_speed(audio_file, speed_factor):
    # Load the audio file
    audio = AudioSegment.from_file(audio_file)

    # Change the playback speed
    new_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed_factor)
    }).set_frame_rate(audio.frame_rate)

    # Save the modified audio
    new_audio.export(audio_file, format="mp3")

def text_to_speech(text, lang='en', speed_factor=1.1):
    try:
        # Generate speech from text
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_file = "src/integration_codes/src/voices/output.mp3"
        tts.save(audio_file)

        # Adjust the speed of the speech
        if speed_factor != 1.0:
            change_speed(audio_file, speed_factor)

        # Play the audio
        playsound(audio_file)

        # Remove the audio file after playing
        os.remove(audio_file)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    # Input text and speed factor
    text = "Please speak now!"
    text_to_speech(text)
