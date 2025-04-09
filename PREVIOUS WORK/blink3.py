import serial
import numpy as np
import cv2
from scipy.spatial import distance as dist
import dlib

# Serial configuration
serial_port = 'COM6'  # Confirm this is the correct port
baud_rate = 921600
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# Constants for frame size and EAR threshold
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
EAR_THRESHOLD = 0.2
EAR_CONSEC_FRAMES = 3

# Initialize Dlib
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    """Compute the Eye Aspect Ratio (EAR)."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def read_frame_from_serial():
    """Read a single frame from the Serial connection."""
    frame_data = b''
    while True:
        byte = ser.read(1)
        if not byte:
            continue
        frame_data += byte
        if frame_data.endswith(b'\xFF\xD9'):  # JPEG end-of-file marker
            break
    return frame_data

def process_live_video_from_serial():
    """Process video frames from Serial, perform blink detection, and display the video."""
    COUNTER = 0
    TOTAL = 0

    while True:
        # Read frame from Serial
        frame_data = read_frame_from_serial()

        # Decode JPEG frame
        np_frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        rects = dlib_detector(gray, 0)

        for rect in rects:
            shape = dlib_predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            leftEye = shape[42:48]
            rightEye = shape[36:42]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye regions and EAR on the frame
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Check for blinks
            if ear < EAR_THRESHOLD:
                COUNTER += 1
            else:
                if COUNTER >= EAR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0

            cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Video Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
            break

    cv2.destroyAllWindows()

# Start processing
process_live_video_from_serial()
