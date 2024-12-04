#!/usr/bin/env python3
import speech_recognition as sr
import rospy
import os
import efficient_IK as ik
import pick_place as pp
import llm_grnd_integ as llm
import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")

# Suppress warnings and logs
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
            except sr.WaitTimeoutError:
                print("Listening timed out. Please try again.")
                continue
            except sr.UnknownValueError:
                print("Could not understand the audio. Please try again.")
                continue
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")
                continue

        # Process the transcribed text with LLM
        ordered_list = llm.process_speech_text(speech_text)
        if ordered_list is None:
            print("No actionable commands detected. Exiting.")
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
        except Exception as e:
            print(f"Error during object detection: {e}")

if __name__ == "__main__":
    main()
