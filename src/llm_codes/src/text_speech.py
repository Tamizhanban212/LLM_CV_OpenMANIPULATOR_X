import vlc
import time

# Function to play an audio file
def play_audio(file_path):
    try:
        player = vlc.MediaPlayer(file_path)
        player.play()
        
        # Wait for the file to finish playing
        while True:
            state = player.get_state()
            if state in [vlc.State.Ended, vlc.State.Stopped, vlc.State.Error]:
                break
            time.sleep(1)  # Check state every second
    except Exception as e:
        print(f"An error occurred while playing {file_path}: {e}")

# Individual functions for specific tasks
def task_completed_speech(audio_file):
    """Plays the 'task completed' speech."""
    play_audio(audio_file)

def system_shutdown_speech(audio_file):
    """Plays the 'system shutdown' speech."""
    play_audio(audio_file)

def error_occurred_speech(audio_file):
    """Plays the 'error occurred' speech."""
    play_audio(audio_file)

def listening_timed_out_speech(audio_file):
    """Plays the 'listening timed out' speech."""
    play_audio(audio_file)

def could_not_understand_speech(audio_file):
    """Plays the 'could not understand' speech."""
    play_audio(audio_file)

def speech_recognized(audio_file):
    """Plays the 'speech recognized' speech."""
    play_audio(audio_file)
