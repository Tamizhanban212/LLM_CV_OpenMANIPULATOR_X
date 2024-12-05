import os
import speech_recognition as sr
import rospy
import efficient_IK as ik
import pick_place as pp
import llm_grnd_integ as llm
import warnings
from text_speech import (
    task_completed_speech,
    system_shutdown_speech,
    error_occurred_speech,
    listening_timed_out_speech,
    could_not_understand_speech,speech_recognized
)

warnings.filterwarnings("ignore")

# Suppress warnings and logs
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Get the relative path to the src/voices directory
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
voices_dir = os.path.join(base_dir, "voices")          # Path to the voices directory within src

# Define paths to the audio files
recognized_speech_audio = os.path.join(voices_dir, "speech_recognized.mp3")
listening_timeout_audio = os.path.join(voices_dir, "listening_timeout.mp3")
could_not_understand_audio = os.path.join(voices_dir, "try_again.mp3")
error_occurred_audio = os.path.join(voices_dir, "object_not_detected.mp3")
system_shutdown_audio = os.path.join(voices_dir, "shut_down.mp3")
system_task_completed_audio = os.path.join(voices_dir, "task_successfull.mp3")


def main():
    recognizer = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            print("Listening... Please speak your instruction.")
            try:
                # Capture audio input
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                # Recognize speech with Google Web Speech API using English (India)
                speech_text = recognizer.recognize_google(audio, language="en-IN")
                print(f"Recognized Speech: {speech_text}")
                speech_recognized(recognized_speech_audio)  # Play "Recognized Speech" audio
            except sr.WaitTimeoutError:
                print("Listening timed out. Please try again.")
                listening_timed_out_speech(listening_timeout_audio)  # Play "Listening Timeout" audio
                continue
            except sr.UnknownValueError:
                print("Could not understand the audio. Please try again.")
                could_not_understand_speech(could_not_understand_audio)  # Play "Could Not Understand" audio
                continue
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")
                error_occurred_speech(error_occurred_audio)  # Play "Error Occurred" audio
                continue

        # Process the transcribed text with LLM
        ordered_list = llm.process_speech_text(speech_text)
        if ordered_list is None:
            print("No actionable commands detected. Exiting.")
            system_shutdown_speech(system_shutdown_audio)  # Play "System Shutdown" audio
            
            pp.switch_off()
            break

        print(f"\nProcessed LLM Response: {ordered_list}")

        # Perform object detection and handle the response
        try:
            detected_centroids = llm.detect_objects_with_display(ordered_list)
            xfrom, yfrom = ik.transform_pixels(detected_centroids[0][0], detected_centroids[0][1])
            xto, yto = ik.transform_pixels(detected_centroids[1][0], detected_centroids[1][1])
            print("Detected centroids:", detected_centroids)
            pp.pick_and_place(xfrom, yfrom, xto, yto)
            print(xfrom, yfrom, xto, yto)
            task_completed_speech(system_task_completed_audio)  # Play "Task Completed
            rospy.sleep(1)
        except Exception as e:
            print(f"Error during object detection: {e}")
            error_occurred_speech(error_occurred_audio)  # Play "Error Occurred" audio

if __name__ == "__main__":
    main()
