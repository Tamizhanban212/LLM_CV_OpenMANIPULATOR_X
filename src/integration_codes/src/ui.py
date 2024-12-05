import os
import threading
from tkinter import Tk, Label, Button, Text, Scrollbar, Canvas, VERTICAL, END, messagebox, Frame
import speech_recognition as sr
import rospy
import efficient_IK as ik
import pick_place as pp
import llm_grnd_integ as llm
from text_speech import (
    task_completed_speech,
    system_shutdown_speech,
    error_occurred_speech,
    listening_timed_out_speech,
    could_not_understand_speech,
    speech_recognized,
)

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


class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Recognition Robot Controller")
        self.root.geometry("1024x800")
        self.root.configure(bg="#2E4053")

        self.recognizer = sr.Recognizer()
        self.stop_flag = False

        # Title
        self.title_label = Label(
            root, text="Speech Recognition Robot Controller", font=("Arial", 20, "bold"), bg="#1ABC9C", fg="#FFFFFF"
        )
        self.title_label.pack(fill="x", pady=10)

        # Image display area
        self.canvas = Canvas(root, width=720, height=480, bg="#34495E", highlightthickness=1, highlightbackground="#1ABC9C")
        self.canvas.pack(pady=10)

        # Status label
        self.status_label = Label(
            root, text="Status: Ready", font=("Arial", 14), bg="#2E4053", fg="#FFFFFF"
        )
        self.status_label.pack(pady=5)

        # Logs text area
        self.logs = Text(root, wrap="word", height=10, width=100, state="normal", bg="#1C2833", fg="#EAECEE", font=("Arial", 10))
        self.logs.pack(pady=10)
        self.logs_scrollbar = Scrollbar(
            root, orient=VERTICAL, command=self.logs.yview
        )
        self.logs_scrollbar.pack(side="right", fill="y")
        self.logs.configure(yscrollcommand=self.logs_scrollbar.set)

        # Button Frame for Centered Buttons
        self.button_frame = Frame(root, bg="#2E4053")
        self.button_frame.pack(pady=20)

        # Buttons
        self.start_button = Button(
            self.button_frame, text="Start Listening", command=self.start_listening, font=("Arial", 12), bg="#2980B9", fg="#FFFFFF"
        )
        self.start_button.grid(row=0, column=0, padx=20)

        self.stop_button = Button(
            self.button_frame, text="Stop Listening", command=self.stop_listening, font=("Arial", 12), bg="#E74C3C", fg="#FFFFFF"
        )
        self.stop_button.grid(row=0, column=1, padx=20)

        self.exit_button = Button(
            self.button_frame, text="Exit", command=self.exit_app, font=("Arial", 12), bg="#34495E", fg="#FFFFFF"
        )
        self.exit_button.grid(row=0, column=2, padx=20)

    def log_message(self, message):
        """Log messages to the text area."""
        self.logs.insert(END, f"{message}\n")
        self.logs.see(END)

    def start_listening(self):
        """Start the speech recognition process."""
        self.stop_flag = False
        self.status_label.config(text="Status: Listening...", bg="#3498DB")
        threading.Thread(target=self.listen_for_commands, daemon=True).start()

    def stop_listening(self):
        """Stop the speech recognition process."""
        self.stop_flag = True
        self.status_label.config(text="Status: Stopped", bg="#E74C3C")
        messagebox.showinfo("Info", "Listening stopped.")

    def exit_app(self):
        """Exit the application."""
        self.stop_flag = True  # Stop the listening thread
        rospy.signal_shutdown("Application exited by user")  # Shut down ROS node
        self.root.quit()  # Stop Tkinter main loop
        os._exit(0)  # Forcefully terminate all threads

    def listen_for_commands(self):
        """Main speech recognition loop."""
        with sr.Microphone() as source:
            while not self.stop_flag:
                try:
                    self.log_message("Listening... Please speak your instruction.")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    speech_text = self.recognizer.recognize_google(audio, language="en-IN")
                    self.log_message(f"Recognized Speech: {speech_text}")
                    speech_recognized(recognized_speech_audio)

                    # Process the transcribed text with LLM
                    ordered_list = llm.process_speech_text(speech_text)
                    if ordered_list is None:
                        self.log_message("No actionable commands detected. Exiting.")
                        system_shutdown_speech(system_shutdown_audio)
                        pp.switch_off()
                        self.exit_app()
                        break

                    self.log_message(f"Processed LLM Response: {ordered_list}")

                    # Perform object detection and handle the response
                    try:
                        detected_centroids = llm.detect_objects_with_display(ordered_list)
                        xfrom, yfrom = ik.transform_pixels(
                            detected_centroids[0][0], detected_centroids[0][1]
                        )
                        xto, yto = ik.transform_pixels(
                            detected_centroids[1][0], detected_centroids[1][1]
                        )
                        pp.pick_and_place(xfrom, yfrom, xto, yto)
                        self.log_message(
                            f"Pick and place completed: {xfrom, yfrom} -> {xto, yto}"
                        )
                        task_completed_speech(system_task_completed_audio)
                    except Exception as e:
                        self.log_message(f"Error during object detection: {e}")
                        error_occurred_speech(error_occurred_audio)

                except sr.WaitTimeoutError:
                    self.log_message("Listening timed out. Please try again.")
                    listening_timed_out_speech(listening_timeout_audio)
                except sr.UnknownValueError:
                    self.log_message("Could not understand the audio. Please try again.")
                    could_not_understand_speech(could_not_understand_audio)
                except sr.RequestError as e:
                    self.log_message(f"Error with speech recognition service: {e}")
                    error_occurred_speech(error_occurred_audio)
                except Exception as e:
                    self.log_message(f"Unexpected Error: {e}")
                    error_occurred_speech(error_occurred_audio)

        self.status_label.config(text="Status: Ready", bg="#2E4053")


if __name__ == "__main__":
    def start_tkinter():
        """Start the Tkinter GUI."""
        root = Tk()
        app = SpeechRecognitionApp(root)
        root.mainloop()

    # Initialize the ROS node in the main thread
    rospy.init_node("speech_recognition_ui", anonymous=True)

    # Run Tkinter in a separate thread
    tkinter_thread = threading.Thread(target=start_tkinter, daemon=True)
    tkinter_thread.start()

    # Keep the ROS node running
    rospy.spin()


# import os
# import threading
# from tkinter import Tk, Label, Button, Text, Scrollbar, Canvas, VERTICAL, END, messagebox, Frame
# import speech_recognition as sr
# import rospy
# import efficient_IK as ik
# import pick_place as pp
# import llm_grnd_integ as llm
# from text_speech import (
#     task_completed_speech,
#     system_shutdown_speech,
#     error_occurred_speech,
#     listening_timed_out_speech,
#     could_not_understand_speech,
#     speech_recognized,
# )

# # Suppress warnings and logs
# os.environ["PYTHONWARNINGS"] = "ignore"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TOKENIZERS_PARALLELISM"] = "true"

# # Get the relative path to the src/voices directory
# base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
# voices_dir = os.path.join(base_dir, "voices")          # Path to the voices directory within src

# # Define paths to the audio files
# recognized_speech_audio = os.path.join(voices_dir, "speech_recognized.mp3")
# listening_timeout_audio = os.path.join(voices_dir, "listening_timeout.mp3")
# could_not_understand_audio = os.path.join(voices_dir, "try_again.mp3")
# error_occurred_audio = os.path.join(voices_dir, "object_not_detected.mp3")
# system_shutdown_audio = os.path.join(voices_dir, "shut_down.mp3")
# system_task_completed_audio = os.path.join(voices_dir, "task_successfull.mp3")


# class SpeechRecognitionApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Speech Recognition Robot Controller")
#         self.root.geometry("1024x800")
#         self.root.configure(bg="#2E4053")

#         self.recognizer = sr.Recognizer()
#         self.stop_flag = False

#         # Title
#         self.title_label = Label(
#             root, text="Speech Recognition Robot Controller", font=("Arial", 20, "bold"), bg="#1ABC9C", fg="#FFFFFF"
#         )
#         self.title_label.pack(fill="x", pady=10)

#         # Image display area
#         self.canvas = Canvas(root, width=720, height=720, bg="#34495E", highlightthickness=1, highlightbackground="#1ABC9C")
#         self.canvas.pack(pady=10)

#         # Status label
#         self.status_label = Label(
#             root, text="Status: Ready", font=("Arial", 14), bg="#2E4053", fg="#FFFFFF"
#         )
#         self.status_label.pack(pady=5)

#         # Logs text area
#         self.logs = Text(root, wrap="word", height=10, width=100, state="normal", bg="#1C2833", fg="#EAECEE", font=("Arial", 10))
#         self.logs.pack(pady=10)
#         self.logs_scrollbar = Scrollbar(
#             root, orient=VERTICAL, command=self.logs.yview
#         )
#         self.logs_scrollbar.pack(side="right", fill="y")
#         self.logs.configure(yscrollcommand=self.logs_scrollbar.set)

#         # Button Frame for Centered Buttons
#         self.button_frame = Frame(root, bg="#2E4053")
#         self.button_frame.pack(pady=20)

#         # Buttons
#         self.start_button = Button(
#             self.button_frame, text="Start Listening", command=self.start_listening, font=("Arial", 12), bg="#2980B9", fg="#FFFFFF"
#         )
#         self.start_button.grid(row=0, column=0, padx=20)

#         self.stop_button = Button(
#             self.button_frame, text="Stop Listening", command=self.stop_listening, font=("Arial", 12), bg="#E74C3C", fg="#FFFFFF"
#         )
#         self.stop_button.grid(row=0, column=1, padx=20)

#         self.exit_button = Button(
#             self.button_frame, text="Exit", command=self.exit_app, font=("Arial", 12), bg="#34495E", fg="#FFFFFF"
#         )
#         self.exit_button.grid(row=0, column=2, padx=20)

#     def log_message(self, message):
#         """Log messages to the text area."""
#         self.logs.insert(END, f"{message}\n")
#         self.logs.see(END)

#     def start_listening(self):
#         """Start the speech recognition process."""
#         self.stop_flag = False
#         self.status_label.config(text="Status: Listening...", bg="#3498DB")
#         threading.Thread(target=self.listen_for_commands, daemon=True).start()

#     def stop_listening(self):
#         """Stop the speech recognition process."""
#         self.stop_flag = True
#         self.status_label.config(text="Status: Stopped", bg="#E74C3C")
#         messagebox.showinfo("Info", "Listening stopped.")

#     def exit_app(self):
#         """Exit the application."""
#         self.stop_flag = True  # Stop the listening thread
#         rospy.signal_shutdown("Application exited by user")  # Shut down ROS node
#         self.root.quit()  # Stop Tkinter main loop
#         os._exit(0)  # Forcefully terminate all threads

#     def listen_for_commands(self):
#         """Main speech recognition loop."""
#         with sr.Microphone() as source:
#             while not self.stop_flag:
#                 try:
#                     self.log_message("Listening... Please speak your instruction.")
#                     audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
#                     speech_text = self.recognizer.recognize_google(audio, language="en-IN")
#                     self.log_message(f"Recognized Speech: {speech_text}")
#                     speech_recognized(recognized_speech_audio)

#                     # Process the transcribed text with LLM
#                     ordered_list = llm.process_speech_text(speech_text)
#                     if ordered_list is None:
#                         self.log_message("No actionable commands detected. Exiting.")
#                         system_shutdown_speech(system_shutdown_audio)
#                         pp.switch_off()
#                         self.exit_app()
#                         break

#                     self.log_message(f"Processed LLM Response: {ordered_list}")

#                     # Perform object detection and handle the response
#                     try:
#                         detected_centroids = llm.detect_objects_with_display(ordered_list)
#                         xfrom, yfrom = ik.transform_pixels(
#                             detected_centroids[0][0], detected_centroids[0][1]
#                         )
#                         xto, yto = ik.transform_pixels(
#                             detected_centroids[1][0], detected_centroids[1][1]
#                         )
#                         pp.pick_and_place(xfrom, yfrom, xto, yto)
#                         self.log_message(
#                             f"Pick and place completed: {xfrom, yfrom} -> {xto, yto}"
#                         )
#                         task_completed_speech(system_task_completed_audio)
#                     except Exception as e:
#                         self.log_message(f"Error during object detection: {e}")
#                         error_occurred_speech(error_occurred_audio)

#                 except sr.WaitTimeoutError:
#                     self.log_message("Listening timed out. Please try again.")
#                     listening_timed_out_speech(listening_timeout_audio)
#                 except sr.UnknownValueError:
#                     self.log_message("Could not understand the audio. Please try again.")
#                     could_not_understand_speech(could_not_understand_audio)
#                 except sr.RequestError as e:
#                     self.log_message(f"Error with speech recognition service: {e}")
#                     error_occurred_speech(error_occurred_audio)
#                 except Exception as e:
#                     self.log_message(f"Unexpected Error: {e}")
#                     error_occurred_speech(error_occurred_audio)

#         self.status_label.config(text="Status: Ready", bg="#2E4053")


# if __name__ == "__main__":
#     def start_tkinter():
#         """Start the Tkinter GUI."""
#         root = Tk()
#         app = SpeechRecognitionApp(root)
#         root.mainloop()

#     # Initialize the ROS node in the main thread
#     rospy.init_node("speech_recognition_ui", anonymous=True)

#     # Run Tkinter in a separate thread
#     tkinter_thread = threading.Thread(target=start_tkinter, daemon=True)
#     tkinter_thread.start()

#     # Keep the ROS node running
#     rospy.spin()


