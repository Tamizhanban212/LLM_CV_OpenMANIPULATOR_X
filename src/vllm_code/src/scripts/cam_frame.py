import cv2

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or adjust for other devices

    # Set the desired frame size
    frame_width = 720
    frame_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    print("Press 'q' to exit the camera feed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Display the camera feed
        cv2.imshow("Camera Feed", frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
