import whisper
import sounddevice as sd
import numpy as np
import torch

# Load the Whisper model and move it to GPU
model = whisper.load_model("medium").to("cpu")

def record_audio(duration=8, samplerate=16000):
    """
    Records audio from the microphone for a specified duration.
    """
    print("Recording... Please speak.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to finish
    print("Recording complete.")
    return (audio.flatten() * 32767).astype(np.int16)  # Normalize to int16 format

def transcribe_audio(audio_data, samplerate=16000):
    """
    Transcribes audio data using the Whisper model.
    """
    import tempfile
    import soundfile as sf

    # Save the audio to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        sf.write(temp_audio_file.name, audio_data, samplerate)
        print("Transcribing...")
        result = model.transcribe(temp_audio_file.name)
        return result["text"]

# Main script
if __name__ == "__main__":
    # Record audio from the microphone
    audio_data = record_audio(duration=10)  # Record for 10 seconds

    # Transcribe the audio using Whisper
    transcription = transcribe_audio(audio_data)
    print(f"Transcription: {transcription}")

