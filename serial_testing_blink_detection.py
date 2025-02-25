import serial
import numpy as np
import cv2
import dlib
import time
import os

# --------------------- Serial Setup ---------------------
BAUD_RATE = 921600  # Try 1,000,000 if supported
SERIAL_PORT = "COM6"  # Adjust based on your setup
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)  # Reduced timeout for faster reading

def read_frame():
    buffer = bytearray()
    while True:
        chunk = ser.readline()
        if b"---START---" in chunk:
            break
    while True:
        chunk = ser.read_until(b"---END---")  # Ensure full frame capture
        buffer += chunk.split(b"---END---")[0]
        if b"---END---" in chunk:
            break
    frame_data = np.frombuffer(buffer, dtype=np.uint8)
    if len(frame_data) < 1000:  # Avoid decoding very small/incomplete frames
        return None
    return frame_data

# --------------------- dlib Setup ---------------------
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.isfile(predictor_path):
    raise FileNotFoundError("Facial landmark predictor file not found. Download it from dlib.net.")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# --------------------- Virtual Button Press Parameters ---------------------
EYE_AR_THRESH = 0.25
VBP_DURATION = 2.0  # Virtual button press duration threshold in seconds
vbp_count = 0
blink_start_time = None
vbp_active = False

# --------------------- OpenCV Window ---------------------
cv2.namedWindow("ArduCAM Stream", cv2.WINDOW_AUTOSIZE)
print("Streaming video... Press 'q' to exit.")

while True:
    start_time = time.time()
    frame_data = read_frame()
    if frame_data is None:
        continue
    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
    if frame is None:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    for face in faces:
        shape = predictor(gray, face)
        shape_np = shape_to_np(shape)
        rightEye = shape_np[36:42]
        leftEye = shape_np[42:48]
        rightEAR = eye_aspect_ratio(rightEye)
        leftEAR = eye_aspect_ratio(leftEye)
        ear = (leftEAR + rightEAR) / 2.0
        if ear < EYE_AR_THRESH:
            if not vbp_active:
                blink_start_time = time.time()
                vbp_active = True
        else:
            if vbp_active and (time.time() - blink_start_time >= VBP_DURATION):
                vbp_count += 1
            vbp_active = False
        cv2.putText(frame, f"EAR: {ear:.2f}", (face.left(), face.bottom()+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()),
                      (255, 0, 0), 2)
        rightEyeHull = cv2.convexHull(rightEye)
        leftEyeHull = cv2.convexHull(leftEye)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        break
    cv2.putText(frame, f"VBP Count: {vbp_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("ArduCAM Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
ser.close()
cv2.destroyAllWindows()
