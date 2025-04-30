import serial
import time
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Target dimensions must match the training dimensions
IMG_WIDTH = 180
IMG_HEIGHT = 108

# EAR threshold and minimum consecutive closed frames for blink detection
EAR_THRESHOLD = 0.25
MIN_CLOSED_FRAMES = 2

def compute_EAR(landmarks):
    """
    Compute the Eye Aspect Ratio (EAR) from 6 landmark points.
    landmarks: an array of 12 numbers [x1, y1, ..., x6, y6].
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1 = np.array([landmarks[0], landmarks[1]])
    p2 = np.array([landmarks[2], landmarks[3]])
    p3 = np.array([landmarks[4], landmarks[5]])
    p4 = np.array([landmarks[6], landmarks[7]])
    p5 = np.array([landmarks[8], landmarks[9]])
    p6 = np.array([landmarks[10], landmarks[11]])
    
    dist_26 = np.linalg.norm(p2 - p6)
    dist_35 = np.linalg.norm(p3 - p5)
    dist_14 = np.linalg.norm(p1 - p4)
    
    ear = (dist_26 + dist_35) / (2.0 * dist_14)
    return ear

def read_image_from_serial(ser):
    """
    Reads one complete JPEG image from the serial stream using the protocol:
      1. A line starting with "FIFO length:" containing the length.
      2. A line with the start marker: "---START---"
      3. Exactly FIFO-length bytes of JPEG data.
      4. A line with the end marker: "---END---"
    Returns the JPEG bytes (as a byte array) or None if reading fails.
    """
    fifo_length = None

    # Wait for a line that starts with "FIFO length:"
    while True:
        line = ser.readline().strip()
        if not line:
            continue
        try:
            decoded = line.decode('utf-8', errors='ignore')
        except Exception as e:
            continue
        if decoded.startswith("FIFO length:"):
            parts = decoded.split(":")
            if len(parts) >= 2:
                try:
                    fifo_length = int(parts[1].strip())
                    break
                except Exception as e:
                    print("Could not parse FIFO length:", e)
                    continue

    if fifo_length is None:
        return None

    # Wait for the start marker line
    while True:
        line = ser.readline().strip()
        try:
            if line.decode('utf-8', errors='ignore') == "---START---":
                break
        except Exception:
            continue

    # Read exactly fifo_length bytes of image data
    jpeg_bytes = ser.read(fifo_length)

    # Read and verify the end marker
    end_marker = ser.readline().strip()
    if end_marker.decode('utf-8', errors='ignore') != "---END---":
        print("End marker not found.")
        return None

    return jpeg_bytes

def main():
    # Open serial port (adjust COM port as needed)
    ser = serial.Serial("COM7", 921600, timeout=1)
    time.sleep(2)  # Allow connection to initialize

    # Load the trained CNN model (make sure "eye_landmark_model.h5" is available)
    model = load_model("eye_landmark_model.h5")
    print("Model loaded.")

    closed_frames = 0
    blink_count = 0

    while True:
        # Optional: read and print any non-image status messages
        while ser.in_waiting:
            try:
                status_line = ser.readline().strip()
                decoded = status_line.decode('utf-8', errors='ignore')
                if decoded and not decoded.startswith("FIFO length:") and \
                   decoded != "---START---" and decoded != "---END---":
                    print("Status:", decoded)
            except Exception:
                break

        # Read one image from serial
        jpeg_data = read_image_from_serial(ser)
        if jpeg_data is None or len(jpeg_data) == 0:
            print("No image data received.")
            continue

        print(f"Received image with {len(jpeg_data)} bytes.")

        # Decode JPEG image using OpenCV
        np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("Failed to decode image.")
            continue

        # Resize the image to match training dimensions and normalize
        resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        input_img = resized.astype('float32') / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        # Predict landmarks (expects 12 outputs: six (x,y) pairs)
        landmarks = model.predict(input_img)[0]
        print(landmarks)
        print("Landmarks shape:", landmarks.shape)
        # Define the original display resolution
        ORIG_WIDTH = 320
        ORIG_HEIGHT = 240

        # Calculate scaling factors from model input to display image
        scale_x = ORIG_WIDTH / IMG_WIDTH  # IMG_WIDTH = 180
        scale_y = ORIG_HEIGHT / IMG_HEIGHT  # IMG_HEIGHT = 108

        # Scale and clip the landmarks
        landmarks_scaled = []
        for i in range(0, len(landmarks), 2):
            x = landmarks[i] * scale_x
            y = landmarks[i+1] * scale_y
            # Ensure coordinates are within the display boundaries:
            x = np.clip(x, 0, ORIG_WIDTH - 1)
            y = np.clip(y, 0, ORIG_HEIGHT - 1)
            landmarks_scaled.extend([x, y])


        ear = compute_EAR(landmarks)
        print(f"EAR: {ear:.2f}")

        # Draw the six landmarks on the resized image
        for i in range(0, len(landmarks), 2):
            x = int(landmarks[i])
            y = int(landmarks[i+1])
            cv2.circle(resized, (x, y), 2, (0, 255, 0), -1)

        # Overlay landmark coordinates
        coords_text = ""
        for i in range(0, len(landmarks), 2):
            coords_text += f"p{i//2+1}: ({landmarks[i]:.1f}, {landmarks[i+1]:.1f})  "
        cv2.putText(resized, coords_text, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)

        # Overlay EAR
        cv2.putText(resized, f"EAR: {ear:.2f}", (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Blink detection logic
        if ear < EAR_THRESHOLD:
            closed_frames += 1
        else:
            if closed_frames >= MIN_CLOSED_FRAMES:
                blink_count += 1
                cv2.putText(resized, "Blink Detected!", (5, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            closed_frames = 0

        cv2.putText(resized, f"Blink Count: {blink_count}", (5, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Display the image
        cv2.namedWindow("Live Blink Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live Blink Detection", ORIG_WIDTH, ORIG_HEIGHT)
        cv2.imshow("Live Blink Detection", resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
