#!/usr/bin/env python3

# import tkinter as tk
# from tkinter import ttk
# from tkinter import messagebox
# import threading
# import cv2
# import os
# import queue
# import speech_recognition as sr
# import modular_CV_llm as MCVLLM
# import text_speech as ts
# import pick_place as pp
# import efficient_IK as ik
# import warnings
# import time
# from PIL import Image, ImageTk  # Import PIL for image handling

# warnings.filterwarnings("ignore")

# # Suppress warnings and logs
# os.environ["PYTHONWARNINGS"] = "ignore"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TOKENIZERS_PARALLELISM"] = "true"

# # Define global variables
# model_id = 1  # Selected model (1-based index)
# processed_frame = None

# # A queue to manage log messages
# log_queue = queue.Queue()

# def start_listening():
#     log("Code started.")
#     # Start the main process in a separate thread to avoid freezing the UI
#     threading.Thread(target=main_process, daemon=True).start()

# def exit_ui():
#     if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
#         pp.switch_off()
#         ts.text_to_speech("Shutting down. Goodbye!")
#         root.destroy()

# # Utility function to update the frame label
# def show_processed_frame(frame, duration=2000):
#     """Display the processed frame in the UI for a given duration (milliseconds)."""
#     if frame is not None:
#         # Convert the OpenCV frame (BGR) to an image compatible with tkinter (RGB)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(frame_rgb)
#         img_tk = ImageTk.PhotoImage(image=img)

#         # Update the label with the image
#         frame_label.img_tk = img_tk  # Keep a reference to prevent garbage collection
#         frame_label.configure(image=img_tk)

#         # Schedule clearing the frame after the specified duration
#         root.after(duration, lambda: frame_label.configure(image=""))
#     else:
#         frame_label.configure(image="")


# def main_process():
#     global processed_frame, model_id
#     ts.text_to_speech("Starting with the selected model.")
#     ts.text_to_speech("Please speak now!")
#     time.sleep(0.5)
#     recognizer = sr.Recognizer()

#     log("Listening... Please speak your instruction.")
#     with sr.Microphone() as source:
#         try:
#             # Capture audio input
#             audio = recognizer.listen(source, timeout=6, phrase_time_limit=10)
#             speech_text = recognizer.recognize_google(audio, language="en-IN")
#             log(f"Recognized Speech: {speech_text}")
#             ts.text_to_speech("Speech recognized.")
#         except sr.WaitTimeoutError:
#             log("Listening timed out. Please try again.")
#             ts.text_to_speech("Please try again.")
#             return
#         except sr.UnknownValueError:
#             log("Could not understand the audio. Please try again.")
#             ts.text_to_speech("Could not understand the audio. Please try again.")
#             return
#         except sr.RequestError as e:
#             log(f"Error with speech recognition service: {e}")
#             ts.text_to_speech("Error with speech recognition service. Please try again.")
#             return

#     ordered_list = MCVLLM.process_speech_text(speech_text)
#     log(f"Processed LLM Response: {ordered_list}")
#     if ordered_list is None:
#         log("No actionable commands detected. Exiting.")
#         ts.text_to_speech("No actionable commands detected. Shutting down.")
#         pp.switch_off()
#         return

#     detected_centroids = []

#     try:
#         for i in ordered_list:
#             log(f"Processing: {i}")
#             object_text = "a " + i + "."
#             try:
#                 centroid, confidence, processing_time, processed_frame = MCVLLM.detect_objects(
#                     model_id, object_text, 30, 0
#                 )
#                 detected_centroids.append(centroid)

#                 # Show the processed frame in the UI for 2 seconds
#                 show_processed_frame(processed_frame, duration=2000)
#                 log(f"Detected {object_text} with confidence {confidence*100}% at centroid {centroid} in {processing_time:.2f} seconds.")
                
#             except Exception as e:
#                 log(f"Error during detection: {e}")
#                 ts.text_to_speech("Error during object detection. Please try again.")
#                 return
        
#         xfrom, yfrom = ik.transform_pixels(detected_centroids[0][0], detected_centroids[0][1])
#         xto, yto = ik.transform_pixels(detected_centroids[1][0], detected_centroids[1][1])
#         pp.pick_and_place(xfrom, yfrom, xto, yto)
#         completed_text = MCVLLM.complete_action_text(speech_text)
#         log(completed_text)
#         ts.text_to_speech(completed_text)

#     except Exception as e:
#         log(f"Error during object detection: {e}")
#         ts.text_to_speech("Error during object detection. Please try again.")


# # Utility function to log messages
# def log(message):
#     log_queue.put(message)

# def update_log_window():
#     while not log_queue.empty():
#         message = log_queue.get_nowait()
#         log_text.insert(tk.END, message + "\n")
#         log_text.see(tk.END)
#     root.after(100, update_log_window)

# # Tkinter UI setup
# root = tk.Tk()
# root.title("Speech-Driven Object Detection")

# # Frame for processed frame display
# frame_canvas = tk.Canvas(root, width=720, height=720, bg="black")
# frame_canvas.pack(pady=10)

# # Define a label to display the processed frame (within the canvas)
# frame_label = tk.Label(frame_canvas, bg="black")  # Black background for the label
# frame_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Center the label in the canvas

# # Dropdown menu for model selection
# model_label = tk.Label(root, text="Select Model:")
# model_label.pack()

# model_dropdown = ttk.Combobox(
#     root,
#     state="readonly",
#     values=[
#         "IDEA-Research/grounding-dino-tiny",
#         "omlab/omdet-turbo-swin-tiny-hf",
#         "google/owlv2-base-patch16-ensemble",
#         "google/owlvit-base-patch32"
#     ],
#     width=30  # Adjust the width of the dropdown menu
# )
# model_dropdown.pack()
# model_dropdown.current(0)

# def set_model():
#     global model_id
#     model_id = model_dropdown.current() + 1  # Adjusting for 1-based indexing
#     log(f"Selected Model ID: {model_id}")

# model_dropdown.bind("<<ComboboxSelected>>", lambda e: set_model())

# # Buttons
# button_frame = tk.Frame(root)
# button_frame.pack(pady=10)

# start_button = tk.Button(button_frame, text="Start Listening", command=start_listening)
# start_button.pack(side=tk.LEFT, padx=5)

# exit_button = tk.Button(button_frame, text="Exit", command=exit_ui)
# exit_button.pack(side=tk.LEFT, padx=5)

# # Log window
# log_label = tk.Label(root, text="Log:")
# log_label.pack()

# log_text = tk.Text(root, height=15, width=100, state="normal", wrap="word")
# log_text.pack(pady=5)

# # Start updating the log window
# update_log_window()
# ts.text_to_speech("Hello! I am ready to assist you.")
# # Run the Tkinter event loop
# root.mainloop()

#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import cv2
import os
import queue
import speech_recognition as sr
import modular_CV_llm as MCVLLM
import text_speech as ts
import pick_place as pp
import efficient_IK as ik
import warnings
import time
from PIL import Image, ImageTk  # Import PIL for image handling

warnings.filterwarnings("ignore")

# Suppress warnings and logs
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Define global variables
model_id = 1  # Selected model (1-based index)
processed_frame = None
capturing = True  # To control the capturing of the live feed
frame_queue = queue.Queue()  # Queue to handle frames

# A queue to manage log messages
log_queue = queue.Queue()

def start_listening():
    log("Code started.")
    # Start the main process in a separate thread to avoid freezing the UI
    threading.Thread(target=main_process, daemon=True).start()

def exit_ui():
    global capturing
    if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
        capturing = False
        pp.switch_off()
        ts.text_to_speech("Shutting down. Goodbye!")
        root.destroy()

# Utility function to update the frame label
def show_processed_frame(frame, duration=2000):
    """Display the processed frame in the UI for a given duration (milliseconds)."""
    if frame is not None:
        # Convert the OpenCV frame (BGR) to an image compatible with tkinter (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the label with the image
        frame_label.img_tk = img_tk  # Keep a reference to prevent garbage collection
        frame_label.configure(image=img_tk)

        # Schedule clearing the frame after the specified duration
        root.after(duration, lambda: frame_label.configure(image=""))

def capture_live_feed():
    global capturing
    cap = cv2.VideoCapture(0)  # Open the default camera
    while capturing:
        ret, frame = cap.read()
        if ret:
            # Put the frame in the queue
            frame_queue.put(frame)
        time.sleep(0.033)  # Update approximately every 33ms (30fps)
    cap.release()

def update_frame_label():
    """Update the frame label with the latest frame from the queue."""
    if not frame_queue.empty():
        frame = frame_queue.get()
        if frame is not None:
            # Convert the OpenCV frame (BGR) to an image compatible with tkinter (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            # Update the label with the image
            frame_label.img_tk = img_tk  # Keep a reference to prevent garbage collection
            frame_label.configure(image=img_tk)
    # Schedule the next update
    root.after(10, update_frame_label)

def resume_live_feed():
    """Resume capturing and displaying the live feed."""
    global capturing
    capturing = True
    threading.Thread(target=capture_live_feed, daemon=True).start()

def main_process():
    global processed_frame, model_id, capturing

    resume_live_feed()

    ts.text_to_speech("Starting with the selected model.")
    ts.text_to_speech("Please speak now!")
    time.sleep(0.5)
    recognizer = sr.Recognizer()

    log("Listening... Please speak your instruction.")
    with sr.Microphone() as source:
        try:
            # Capture audio input
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=10)
            speech_text = recognizer.recognize_google(audio, language="en-IN")
            log(f"Recognized Speech: {speech_text}")
            ts.text_to_speech("Speech recognized.")
        except sr.WaitTimeoutError:
            log("Listening timed out. Please try again.")
            ts.text_to_speech("Please try again.")
            resume_live_feed()
            return
        except sr.UnknownValueError:
            log("Could not understand the audio. Please try again.")
            ts.text_to_speech("Could not understand the audio. Please try again.")
            resume_live_feed()
            return
        except sr.RequestError as e:
            log(f"Error with speech recognition service: {e}")
            ts.text_to_speech("Error with speech recognition service. Please try again.")
            resume_live_feed()
            return

    ordered_list = MCVLLM.process_speech_text(speech_text)
    log(f"Processed LLM Response: {ordered_list}")
    if ordered_list is None:
        log("No actionable commands detected. Exiting.")
        ts.text_to_speech("No actionable commands detected. Shutting down.")
        pp.switch_off()
        resume_live_feed()
        return

    detected_centroids = []

    # Stop capturing live feed before processing frames
    capturing = False
    time.sleep(0.1)  # Allow a moment for the feed to stop

    try:
        for i in ordered_list:
            log(f"Processing: {i}")
            object_text = "a " + i + "."
            try:
                centroid, confidence, processing_time, processed_frame = MCVLLM.detect_objects(
                    model_id, object_text, 30, 0
                )
                detected_centroids.append(centroid)

                # Show the processed frame in the UI for 2 seconds
                show_processed_frame(processed_frame, duration=2000)
                log(f"Detected {object_text} with confidence {confidence*100}% at centroid {centroid} in {processing_time:.2f} seconds.")
                
            except Exception as e:
                log(f"Error during detection: {e}")
                ts.text_to_speech("Error during object detection. Please try again.")
                resume_live_feed()
                return

        # Resume live feed after all processed frames are displayed
        resume_live_feed()

        xfrom, yfrom = ik.transform_pixels(detected_centroids[0][0], detected_centroids[0][1])
        xto, yto = ik.transform_pixels(detected_centroids[1][0], detected_centroids[1][1])
        pp.pick_and_place(xfrom, yfrom, xto, yto)
        completed_text = MCVLLM.complete_action_text(speech_text)
        log(completed_text)
        ts.text_to_speech(completed_text)

    except Exception as e:
        log(f"Error during object detection: {e}")
        ts.text_to_speech("Error during object detection. Please try again.")

    # Resume capturing the live feed after all processing is done
    resume_live_feed()

# Utility function to log messages
def log(message):
    log_queue.put(message)

def update_log_window():
    while not log_queue.empty():
        message = log_queue.get_nowait()
        log_text.insert(tk.END, message + "\n")
        log_text.see(tk.END)
    root.after(100, update_log_window)

# Tkinter UI setup
root = tk.Tk()
root.title("Speech-Driven Object Detection")

# Frame for processed frame display
frame_canvas = tk.Canvas(root, width=720, height=720, bg="black")
frame_canvas.pack(pady=10)

# Define a label to display the processed frame (within the canvas)
frame_label = tk.Label(frame_canvas, bg="black")  # Black background for the label
frame_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Center the label in the canvas

# Dropdown menu for model selection
model_label = tk.Label(root, text="Select Model:")
model_label.pack()

model_dropdown = ttk.Combobox(
    root,
    state="readonly",
    values=[
        "IDEA-Research/grounding-dino-tiny",
        "omlab/omdet-turbo-swin-tiny-hf",
        "google/owlv2-base-patch16-ensemble",
        "google/owlvit-base-patch32"
    ],
    width=30  # Adjust the width of the dropdown menu
)
model_dropdown.pack()
model_dropdown.current(0)

def set_model():
    global model_id
    model_id = model_dropdown.current() + 1  # Adjusting for 1-based indexing
    log(f"Selected Model ID: {model_id}")

model_dropdown.bind("<<ComboboxSelected>>", lambda e: set_model())

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

start_button = tk.Button(button_frame, text="Start Listening", command=start_listening)
start_button.pack(side=tk.LEFT, padx=5)

exit_button = tk.Button(button_frame, text="Exit", command=exit_ui)
exit_button.pack(side=tk.LEFT, padx=5)

# Log window
log_label = tk.Label(root, text="Log:")
log_label.pack()

log_text = tk.Text(root, height=15, width=100, state="normal", wrap="word")
log_text.pack(pady=5)

# Start updating the log window
update_log_window()

threading.Thread(target=capture_live_feed, daemon=True).start()
# Start updating the frame label
update_frame_label()
ts.text_to_speech("Hello! I am ready to assist you.")
# Run the Tkinter event loop
root.mainloop()




