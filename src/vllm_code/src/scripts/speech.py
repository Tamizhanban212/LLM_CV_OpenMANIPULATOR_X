# from vosk import Model, KaldiRecognizer
# import sounddevice as sd
# import queue
# import json

# # Initialize the audio stream
# def audio_stream_to_text():
#     model_path = "src/vllm_code/src/scripts/model"  # Path to Vosk model directory
#     model = Model(model_path)
#     recognizer = KaldiRecognizer(model, 16000)
    
#     # Queue to hold audio data
#     audio_queue = queue.Queue()

#     # Callback function to capture audio from the microphone
#     def audio_callback(indata, frames, time, status):
#         if status:
#             print(status, flush=True)
#         audio_queue.put(bytes(indata))
    
#     # Configure sounddevice stream
#     with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
#                            channels=1, callback=audio_callback):
#         print("Listening... Press Ctrl+C to stop.")
#         try:
#             while True:
#                 data = audio_queue.get()
#                 if recognizer.AcceptWaveform(data):
#                     result = json.loads(recognizer.Result())
#                     print("You said:", result.get("text", ""))
#         except KeyboardInterrupt:
#             print("\nStopped by user")
#         except Exception as e:
#             print(f"Error: {e}")

# # Run the function
# if __name__ == "__main__":
#     audio_stream_to_text()

import whisper
import sounddevice as sd
import numpy as np
import torch

# Load the Whisper model and move it to GPU
model = whisper.load_model("turbo").to("cuda")

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

