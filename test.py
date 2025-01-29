import cv2
import serial
import numpy as np

# Define the serial port and baud rate
serial_port = 'COM6'  # Replace with your actual COM port
baud_rate = 921600

# Open the serial port
ser = serial.Serial(serial_port, baud_rate, timeout=5)
print(f"Connected to {serial_port} at {baud_rate} baud.")

# Frame delimiters
FRAME_START = b'FRAME START\n'
FRAME_END = b'FRAME END\n'


def read_frame():
    frame_data = bytearray()
    in_frame = False

    while True:
        chunk = ser.read(1024)  # Read 1024 bytes at a time
        print(f"Chunk Data (first 50 bytes): {chunk[:50]}")

        if not chunk:
            break  # Stop if nothing is received

        print(f"Received chunk of size {len(chunk)}")  # Debugging

        if FRAME_START in chunk:
            in_frame = True
            frame_data = bytearray()
            chunk = chunk.split(FRAME_START)[1]  # Remove everything before FRAME START

        if FRAME_END in chunk:
            frame_data.extend(chunk.split(FRAME_END)[0])  # Save only data before FRAME END
            break  # Exit when FRAME END is found

        if in_frame:
            frame_data.extend(chunk)

    if len(frame_data) == 0:
        print("⚠️ Warning: Empty frame received!")

    return frame_data


def main():
    while True:
        try:
            # Read a frame from the serial port
            frame_data = read_frame()


            # Decode the frame (MJPEG format)
            frame_array = np.asarray(frame_data, dtype=np.uint8)

            if frame_array.size == 0:
                # print("Received empty frame data. Skipping...")
                continue

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
