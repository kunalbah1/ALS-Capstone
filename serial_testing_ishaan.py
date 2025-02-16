import serial
import numpy as np
import cv2
import time
import os

# Serial configuration (must match your Arduino settings)
BAUD_RATE = 921600
SERIAL_PORT = "COM7"  # Adjust as needed

# Open the serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)

# Adjust the minimum expected JPEG size (in bytes)
MIN_JPEG_SIZE = 3500

def read_frame():
    """
    Reads a full JPEG frame from the serial stream.
    1. Reads byte-by-byte until the start marker (---START---) is found.
    2. Reads data in chunks (with an overlapping buffer) until the end marker (---END---) is detected.
    """
    buffer = bytearray()
    start_marker = b"---START---"
    end_marker   = b"---END---"
    
    # --- Step 1: Read until start marker is found ---
    marker_buf = bytearray()
    while True:
        byte = ser.read(1)
        if not byte:
            continue  # No byte read; keep waiting
        marker_buf += byte
        # Keep only the last len(start_marker) bytes
        if len(marker_buf) > len(start_marker):
            marker_buf = marker_buf[-len(start_marker):]
        if start_marker in marker_buf:
            break

    # --- Step 2: Read JPEG data until the end marker is detected ---
    overlap = bytearray()
    while True:
        chunk = ser.read(1024)
        if not chunk:
            continue  # No data; try again
        # Prepend any leftover bytes (overlap) from the previous read
        chunk = overlap + chunk
        marker_index = chunk.find(end_marker)
        if marker_index != -1:
            # Append data only up to the end marker and exit loop.
            buffer += chunk[:marker_index]
            break
        else:
            buffer += chunk
            # Save the last few bytes to catch markers that may span chunks.
            overlap = chunk[-(len(end_marker)-1):] if len(chunk) >= (len(end_marker)-1) else chunk
    return np.frombuffer(buffer, dtype=np.uint8)

def trim_to_valid_jpeg(data_bytes):
    """
    Trims the provided data so that it starts with the JPEG SOI marker (0xFFD8)
    and ends with the JPEG EOI marker (0xFFD9).
    Returns the trimmed bytes if valid, or None if the markers are missing.
    """
    start_index = data_bytes.find(b'\xff\xd8')
    end_index   = data_bytes.rfind(b'\xff\xd9')
    print(f"Debug: SOI index: {start_index}, EOI index: {end_index}")
    
    if start_index == -1:
        print("Error: Missing JPEG SOI marker")
        return None
    if end_index == -1:
        print("Error: Missing JPEG EOI marker")
        return None
    if end_index < start_index:
        print("Error: EOI marker found before SOI marker")
        return None

    trimmed = data_bytes[start_index:end_index+2]
    if not trimmed.endswith(b'\xff\xd9'):
        print("Error: Trimmed JPEG does not properly end with EOI marker")
        return None
    return trimmed

# Directory for saving frames that fail to decode (for debugging)
failed_dir = "failed_frames"
if not os.path.exists(failed_dir):
    os.makedirs(failed_dir)

# Initialize OpenCV window
cv2.namedWindow("ArduCAM Stream", cv2.WINDOW_AUTOSIZE)
print("Streaming video... Press 'q' to exit.")

frame_number = 0
while True:
    start_time = time.time()

    # Read the raw frame data from the serial stream
    frame_data = read_frame()
    data_length = len(frame_data)
    print(f"Received frame data of length: {data_length} bytes")

    if data_length == 0:
        print("⚠️ No image data received.")
        continue

    # Convert the data to bytes (the markers should now be clean)
    frame_bytes = frame_data.tobytes()
    trimmed_bytes = trim_to_valid_jpeg(frame_bytes)
    if trimmed_bytes is None:
        print("Frame discarded due to invalid JPEG markers.")
        continue

    # Check if the trimmed frame is too small (likely incomplete)
    if len(trimmed_bytes) < MIN_JPEG_SIZE:
        print(f"Frame discarded: size {len(trimmed_bytes)} bytes is smaller than expected minimum {MIN_JPEG_SIZE} bytes.")
        continue

    # Convert the trimmed bytes to a NumPy array and decode as JPEG
    frame_array = np.frombuffer(trimmed_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    if frame is None:
        print("⚠️ Failed to decode image.")
        # Save the raw frame for debugging purposes
        failed_filename = os.path.join(failed_dir, f"failed_frame_{frame_number}.jpg")
        with open(failed_filename, "wb") as f:
            f.write(trimmed_bytes)
        frame_number += 1
        continue

    # Display the decoded frame
    cv2.imshow("ArduCAM Stream", frame)
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    print(f"FPS: {fps:.2f}")

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

ser.close()
cv2.destroyAllWindows()
