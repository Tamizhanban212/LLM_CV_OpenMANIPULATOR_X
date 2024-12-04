import tkinter as tk
from tkinter import messagebox
import threading
import rospy
import speech_recognition as sr
import os
import warnings
import efficient_IK as ik
import pick_place as pp
import llm_grnd_integ as llm

# Suppress warnings and logs
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Recognition Robot Controller")
        self.root.geometry("600x400")

        self.recognizer = sr.Recognizer()
        self.stop_flag = False

        # UI Components
        self.instructions_label = tk.Label(
            root, text="Press 'Start' to listen for instructions.", font=("Arial", 14)
        )
        self.instructions_label.pack(pady=10)

        self.start_button = tk.Button(
            root, text="Start Listening", command=self.start_listening, font=("Arial", 12)
        )
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(
            root, text="Stop Listening", command=self.stop_listening, font=("Arial", 12)
        )
        self.stop_button.pack(pady=10)

        self.speech_label = tk.Label(root, text="Recognized Speech:", font=("Arial", 12))
        self.speech_label.pack(pady=5)

        self.speech_output = tk.Text(root, height=5, width=50)
        self.speech_output.pack(pady=10)

        self.result_label = tk.Label(root, text="LLM Processed Commands:", font=("Arial", 12))
        self.result_label.pack(pady=5)

        self.result_output = tk.Text(root, height=5, width=50)
        self.result_output.pack(pady=10)

        self.exit_button = tk.Button(root, text="Exit", command=root.quit, font=("Arial", 12))
        self.exit_button.pack(pady=10)

    def start_listening(self):
        self.stop_flag = False
        threading.Thread(target=self.listen_for_commands, daemon=True).start()

    def stop_listening(self):
        self.stop_flag = True
        messagebox.showinfo("Info", "Stopped listening.")

    def listen_for_commands(self):
        with sr.Microphone() as source:
            while not self.stop_flag:
                self.instructions_label.config(text="Listening... Speak your instruction.")
                try:
                    # Capture audio input
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

                    # Recognize speech with Google Web Speech API using English (India)
                    speech_text = self.recognizer.recognize_google(audio, language="en-IN")
                    self.speech_output.delete(1.0, tk.END)
                    self.speech_output.insert(tk.END, speech_text)

                    # Process the transcribed text with LLM
                    ordered_list = llm.process_speech_text(speech_text)
                    if ordered_list is None:
                        self.result_output.delete(1.0, tk.END)
                        self.result_output.insert(tk.END, "No actionable commands detected.")
                        pp.switch_off()
                        break

                    self.result_output.delete(1.0, tk.END)
                    self.result_output.insert(tk.END, f"Commands: {ordered_list}")

                    # Perform object detection and handle the response
                    try:
                        detected_centroids = llm.detect_objects_with_display(ordered_list)
                        xfrom, yfrom = ik.transform_pixels(detected_centroids[0][0], detected_centroids[0][1])
                        xto, yto = ik.transform_pixels(detected_centroids[1][0], detected_centroids[1][1])
                        pp.pick_and_place(xfrom, yfrom, xto, yto)
                    except Exception as e:
                        self.result_output.insert(tk.END, f"\nError during object detection: {e}")

                except sr.WaitTimeoutError:
                    self.speech_output.delete(1.0, tk.END)
                    self.speech_output.insert(tk.END, "Listening timed out. Please try again.")
                except sr.UnknownValueError:
                    self.speech_output.delete(1.0, tk.END)
                    self.speech_output.insert(tk.END, "Could not understand the audio.")
                except sr.RequestError as e:
                    self.speech_output.delete(1.0, tk.END)
                    self.speech_output.insert(tk.END, f"Service Error: {e}")
                except Exception as e:
                    self.speech_output.delete(1.0, tk.END)
                    self.speech_output.insert(tk.END, f"Error: {e}")

        self.instructions_label.config(text="Press 'Start' to listen for instructions again.")


if __name__ == "__main__":
    rospy.init_node("speech_recognition_ui", anonymous=True)
    root = tk.Tk()
    app = SpeechRecognitionApp(root)
    root.mainloop()
