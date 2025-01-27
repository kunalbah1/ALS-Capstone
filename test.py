import cv2
import serial
import numpy as np

# Define the serial port and baud rate
serial_port = 'COM6'  # Replace with your actual COM port
baud_rate = 115200

# Open the serial port
ser = serial.Serial(serial_port, baud_rate, timeout=1)
print(f"Connected to {serial_port} at {baud_rate} baud.")

# Frame delimiters
FRAME_START = b'FRAME START\n'
FRAME_END = b'FRAME END\n'

def read_frame():
    """Reads a single frame from the serial port."""
    frame_data = bytearray()
    in_frame = False

    while True:
        line = ser.readline()

        # Check for start of frame
        if line == FRAME_START:
            in_frame = True
            frame_data = bytearray()
            continue

        # Check for end of frame
        if line == FRAME_END:
            break

        # Collect frame data
        if in_frame:
            frame_data.extend(line)

    return frame_data

def main():
    while True:
        try:
            # Read a frame from the serial port
            frame_data = read_frame()

            # Decode the frame (MJPEG format)
            frame_array = np.asarray(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is not None:
                # Display the frame
                cv2.imshow("ArduCAM Video Feed", frame)

                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Failed to decode frame.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

    # Cleanup
    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
