import serial
import numpy as np
import cv2
import time

# Increase baud rate for faster transmission
BAUD_RATE = 1000000 # Try 1,000,000 if supported
SERIAL_PORT = "COM6"  # Adjust based on your setup

# Open serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)


def read_frame():
    """ Reads a full frame from the serial stream. """
    buffer = bytearray()

    # Find start marker
    while True:
        chunk = ser.readline()
        if b"---START---" in chunk:
            break

    # Read image data until end marker
    while True:
        chunk = ser.read(1024)  # Read in chunks of 1024 bytes
        if b"---END---" in chunk:
            buffer += chunk.split(b"---END---")[0]
            break
        buffer += chunk

    return np.frombuffer(buffer, dtype=np.uint8)


# Initialize OpenCV window
cv2.namedWindow("ArduCAM Stream", cv2.WINDOW_AUTOSIZE)

print("Streaming video... Press 'q' to exit.")

while True:
    start_time = time.time()

    frame_data = read_frame()

    if frame_data.size == 0:
        print("⚠️ No image data received.")
        continue

    # Decode the JPEG frame
    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
    if frame is None:
        print("⚠️ Failed to decode image.")
        continue

    # Display frame
    cv2.imshow("ArduCAM Stream", frame)

    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    print(f"FPS: {fps:.2f}")

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

ser.close()
cv2.destroyAllWindows()
