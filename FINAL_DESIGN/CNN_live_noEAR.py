import serial
import time
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import math

##############################################
# Serial Setup for Arducam
##############################################
BAUD_RATE = 921600  # Use your appropriate baud rate
SERIAL_PORT = "COM7"  # Adjust based on your configuration

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Allow serial connection to initialize
except Exception as e:
    print("Error opening serial port:", e)
    exit()

##############################################
# Function: Read One Complete JPEG Image from Serial
##############################################
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
    # Wait for FIFO length line.
    while True:
        line = ser.readline().strip()
        if not line:
            continue
        try:
            decoded = line.decode('utf-8', errors='ignore')
        except Exception:
            continue
        if decoded.startswith("FIFO length:"):
            parts = decoded.split(":")
            if len(parts) >= 2:
                try:
                    fifo_length = int(parts[1].strip())
                    break
                except Exception as e:
                    print("Parsing FIFO length error:", e)
                    continue

    if fifo_length is None:
        return None

    # Wait for start marker.
    while True:
        line = ser.readline().strip()
        try:
            if line.decode('utf-8', errors='ignore') == "---START---":
                break
        except Exception:
            continue

    # Read exactly fifo_length bytes of JPEG image data.
    jpeg_bytes = ser.read(fifo_length)

    # Read and verify the end marker.
    end_marker = ser.readline().strip()
    try:
        if end_marker.decode('utf-8', errors='ignore') != "---END---":
            print("End marker not found.")
            return None
    except Exception:
        return None

    return jpeg_bytes

##############################################
# Step 1: Load the Pre-trained Blink Detection Model
##############################################
model = load_model("blink_detection_model.h5")
print("Blink detection model loaded successfully.")

##############################################
# Step 2: Preprocessing Function
##############################################
# The model was trained on images with dimensions 128x128.
IMG_WIDTH = 128
IMG_HEIGHT = 128

def preprocess_frame(frame):
    """
    Resize the captured frame to 128x128, convert to float32, and normalize to [0,1].
    """
    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    return img

##############################################
# Step 3: Blink Counting Setup
##############################################
VBP_count = 0          # Total blinks counted.
closed_frames = 0        # Count of consecutive frames with eye classified as "Closed."
MIN_VBP_FRAMES = 120    # If the eye is closed for at least 2 consecutive frames, register one blink.

##############################################
# Step 4: Live Blink Detection Loop (Using Arducam Serial)
##############################################
print("Starting live blink detection from Arducam. Press 'q' to exit.")

while True:
    # Read a JPEG image from the Arducam over serial
    jpeg_data = read_image_from_serial(ser)
    if jpeg_data is None or len(jpeg_data) == 0:
        print("No image data received.")
        continue

    print(f"Received image with {len(jpeg_data)} bytes.")
    
    # Decode the JPEG image using OpenCV.
    np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        print("Failed to decode image.")
        continue

    # Preprocess the image for the CNN.
    processed_img = preprocess_frame(frame)
    input_img = np.expand_dims(processed_img, axis=0)  # Create a batch of 1.

    # Use the CNN model to predict blink state.
    # The model outputs a probability; threshold > 0.5 indicates "Open".
    prediction_prob = model.predict(input_img)[0][0]
    status = "Open" if prediction_prob > 0.5 else "Closed"

    # Blink counting logic.
    if status == "Closed":
        closed_frames += 1
    else:
        if closed_frames >= MIN_VBP_FRAMES:
            VBP_count += 1
        closed_frames = 0  # Reset the counter when eye is open.

    # Annotate the frame with the blink state and blink count.
    label_color = (0, 255, 0) if status == "Open" else (0, 0, 255)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
    cv2.putText(frame, f"VBP Count: {VBP_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display the frame.
    cv2.imshow("Arducam Blink Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up: close serial and windows.
ser.close()
cv2.destroyAllWindows()
print("Exiting Blink Detection. Total blinks:", blink_count)
