
import serial
import numpy as np
import cv2
import dlib
import time
import os

# --------------------- Serial Setup ---------------------
BAUD_RATE = 921600      # Try 1,000,000 if supported
SERIAL_PORT = "COM7"    # Adjust based on your setup

# Open serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)

def read_frame():
    """
    Reads a full JPEG frame from the serial stream.
    Waits for the start marker, then reads until the end marker.
    """
    buffer = bytearray()

    # Wait for start marker (ignores any extra output)
    while True:
        chunk = ser.readline()
        if b"---START---" in chunk:
            break

    # Read image data until the end marker is found
    while True:
        chunk = ser.read(1024)  # Read in chunks
        if b"---END---" in chunk:
            # Append only the data before the end marker
            buffer += chunk.split(b"---END---")[0]
            break
        buffer += chunk

    return np.frombuffer(buffer, dtype=np.uint8)

# --------------------- dlib Setup ---------------------
# Initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor. Make sure the .dat file is in the working directory.
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.isfile(predictor_path):
    raise FileNotFoundError("Facial landmark predictor file not found. "
                            "Download 'shape_predictor_68_face_landmarks.dat' from "
                            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "
                            "and place it in the current directory.")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def shape_to_np(shape, dtype="int"):
    """
    Convert a dlib shape object to a NumPy array with (x, y)-coordinates.
    """
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_aspect_ratio(eye):
    """
    Compute the eye aspect ratio (EAR) for a given eye.
    The eye should be a NumPy array with 6 (x, y)-coordinates.
    """
    # Compute the Euclidean distances between the vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Compute the distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --------------------- Blink Detection Parameters ---------------------
EYE_AR_THRESH = 0.25  # Threshold for EAR to indicate a blink
blink_in_progress = False  # Tracks whether eyes are currently closed
blink_count = 0            # Total blink count

# --------------------- OpenCV Window ---------------------
cv2.namedWindow("ArduCAM Stream", cv2.WINDOW_AUTOSIZE)
print("Streaming video... Press 'q' to exit.")

while True:
    start_time = time.time()

    # Read and decode the frame from the serial stream
    frame_data = read_frame()
    if frame_data.size == 0:
        print("⚠️ No image data received.")
        continue

    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
    if frame is None:
        print("⚠️ Failed to decode image.")
        continue

    # Convert the frame to grayscale for dlib detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)
    
    # Process each detected face (here, we process only the first face)
    for face in faces:
        # Get facial landmarks
        shape = predictor(gray, face)
        shape_np = shape_to_np(shape)

        # Extract the coordinates of the left and right eye.
        # dlib's 68-point model: Right eye = points 36-41, Left eye = points 42-47.
        rightEye = shape_np[36:42]
        leftEye = shape_np[42:48]

        # Compute the EAR for both eyes
        rightEAR = eye_aspect_ratio(rightEye)
        leftEAR  = eye_aspect_ratio(leftEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw contours for the eyes
        rightEyeHull = cv2.convexHull(rightEye)
        leftEyeHull  = cv2.convexHull(leftEye)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)

        # Blink detection logic:
        if ear < EYE_AR_THRESH:
            if not blink_in_progress:
                blink_count += 1
                blink_in_progress = True
                cv2.putText(frame, "Blink Detected!", (face.left(), face.top()-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            blink_in_progress = False

        # Optionally, draw a rectangle around the face
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()),
                      (255, 0, 0), 2)
        # Display the computed EAR on the frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (face.left(), face.bottom()+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        # Process only the first face detected
        break

    # Overlay the total blink count on the frame
    cv2.putText(frame, f"Blink Count: {blink_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the video frame
    cv2.imshow("ArduCAM Stream", frame)

    # Calculate and print FPS
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    print(f"FPS: {fps:.2f}")

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

ser.close()
cv2.destroyAllWindows()
