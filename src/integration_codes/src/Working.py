#!/usr/bin/env python3
import speech_recognition as sr
import rospy
import os
import efficient_IK as ik
import pick_place as pp
import modular_CV_llm as MCVLLM
import warnings
import text_speech as ts

warnings.filterwarnings("ignore")

# Suppress warnings and logs
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

detected_centroids = []

def main():
    ts.text_to_speech("Hello! I am ready to assist you.")
    ts.text_to_speech("Please select the required CV model.")
    model_id = MCVLLM.get_available_models()
    recognizer = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            print("Listening... Please speak your instruction.")
            ts.text_to_speech("Please speak now!")
            try:
                # Capture audio input
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                # Recognize speech with Google Web Speech API using English (India)
                speech_text = recognizer.recognize_google(audio, language="en-IN")
                print(f"Recognized Speech: {speech_text}")
                ts.text_to_speech("Speech recognized.")
            except sr.WaitTimeoutError:
                print("Listening timed out. Please try again.")
                ts.text_to_speech("Please try again.")
                continue
            except sr.UnknownValueError:
                print("Could not understand the audio. Please try again.")
                ts.text_to_speech("Could not understand the audio. Please try again.")
                continue
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")
                ts.text_to_speech("Error with speech recognition service. Please try again.")
                continue

        # Process the transcribed text with LLM
        ordered_list = MCVLLM.process_speech_text(speech_text)
        if ordered_list is None:
            print("No actionable commands detected. Exiting.")
            ts.text_to_speech("No actionable commands detected. Shutting down.")
            pp.switch_off()
            break

        print(f"\nProcessed LLM Response: {ordered_list}")

        # Perform object detection and handle the response
        try:
            for i in ordered_list:
                object_text = "a " + i + "."  # Add "a" before the object name
                centroid, confidence, processing_time, processed_frame = MCVLLM.detect_objects(model_id, object_text, 30, 2)
                detected_centroids.append(centroid)
                print(f"Detected {object_text} with confidence {confidence*100}% at centroid {centroid} in {processing_time:.2f} seconds.")
            
            xfrom, yfrom = ik.transform_pixels(detected_centroids[0][0], detected_centroids[0][1])
            xto, yto = ik.transform_pixels(detected_centroids[1][0], detected_centroids[1][1])
            print("Detected centroids:", detected_centroids)
            pp.pick_and_place(xfrom, yfrom, xto, yto)
            completed_text = MCVLLM.complete_action_text(speech_text)
            ts.text_to_speech(completed_text)

        except Exception as e:
            print(f"Error during object detection: {e}")
            ts.text_to_speech("Error during object detection. Please try again.")

if __name__ == "__main__":
    main()
