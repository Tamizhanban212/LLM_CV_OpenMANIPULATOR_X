from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json

# Initialize the audio stream
def audio_stream_to_text():
    model_path = "src/vllm_code/src/scripts/model"  # Path to Vosk model directory
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)
    
    # Queue to hold audio data
    audio_queue = queue.Queue()

    # Callback function to capture audio from the microphone
    def audio_callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        audio_queue.put(bytes(indata))
    
    # Configure sounddevice stream
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                           channels=1, callback=audio_callback):
        print("Listening... Press Ctrl+C to stop.")
        try:
            while True:
                data = audio_queue.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    print("You said:", result.get("text", ""))
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error: {e}")

# Run the function
if __name__ == "__main__":
    audio_stream_to_text()
