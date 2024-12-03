import speech_recognition as sr
import llm_grnd_integ as llm
import efficient_IK as ik
import pick_place as pp

def main():
    recognizer = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            print("Listening... Please speak your instruction.")
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                speech_text = recognizer.recognize_google(audio)
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

        dino_string = llm.process_speech_text(speech_text)
        if dino_string is None:
            break

        print(f"\nProcessed LLM Response: {dino_string}")
        try:
            detected_centroids = llm.detect_objects_with_display(dino_string)
            print("Detected centroids:", detected_centroids)
        except Exception as e:
            print(str(e))

if __name__ == "__main__":
    main()