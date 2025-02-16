        import serial
        import numpy as np
        import cv2
        import struct
        import time

        # Configuration parameters (adjust as needed)
        BAUD_RATE = 921600  # Should match Arduino
        SERIAL_PORT = "COM7"  # Adjust based on your system

        # Open the serial connection with a longer timeout
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        ser.reset_input_buffer()  # Clear any leftover data

        def read_exactly(num_bytes):
            """Read exactly num_bytes from the serial port."""
            data = b''
            while len(data) < num_bytes:
                chunk = ser.read(num_bytes - len(data))
                if not chunk:
                    break
                data += chunk
            return data

        def read_frame():
            """
            Reads a full frame:
            1. Reads a 4-byte header (big-endian integer) specifying the frame length.
            2. Reads exactly that many bytes for the JPEG data.
            """
            # Read 4-byte header
            header = read_exactly(4)
            if len(header) != 4:
                print("⚠️ Header read failed.")
                return None

            fifo_length = struct.unpack('>I', header)[0]
            print(f"Frame length from header: {fifo_length}")
            
            # Read the JPEG data
            frame_data = read_exactly(fifo_length)
            if len(frame_data) != fifo_length:
                print(f"⚠️ Incomplete frame data: expected {fifo_length}, got {len(frame_data)}")
                return None
            return np.frombuffer(frame_data, dtype=np.uint8)

        def trim_to_valid_jpeg(data):
            """
            Searches for the JPEG Start Of Image (SOI, 0xFFD8)
            and End Of Image (EOI, 0xFFD9) markers in data,
            returning only the portion of the data that forms a valid JPEG.
            """
            arr = np.frombuffer(data, dtype=np.uint8)
            start = None
            for i in range(len(arr)-1):
                if arr[i] == 0xFF and arr[i+1] == 0xD8:
                    start = i
                    break
            end = None
            for j in range(len(arr)-1, 0, -1):
                if arr[j-1] == 0xFF and arr[j] == 0xD9:
                    end = j
                    break
            if start is None or end is None or end <= start:
                return None
            return arr[start:end+1]

        cv2.namedWindow("ArduCAM Stream", cv2.WINDOW_AUTOSIZE)
        print("Streaming video... Press 'q' to exit.")


        # read byte by byte, until the full string below has been seen, there could be random stuff before but tstop this process after we see the next string
        starting_string = b"START"

        iteration = 0
        while True:
            start_time = time.time()
            if iteration != 0:
                read_exactly(len(starting_string))
            frame_data = read_frame()
            if frame_data is None or frame_data.size == 0:
                print("⚠️ No image data received.")
                continue

            # Trim extra bytes to get a valid JPEG image
            trimmed_frame = trim_to_valid_jpeg(frame_data.tobytes())
            if trimmed_frame is None or trimmed_frame.size == 0:
                print("⚠️ Failed to extract valid JPEG data.")
                continue

            # Decode the JPEG frame
            frame = cv2.imdecode(trimmed_frame, cv2.IMREAD_COLOR)
            if frame is None:
                print("⚠️ Failed to decode image.")
                with open("debug_frame.jpg", "wb") as f:
                    f.write(trimmed_frame.tobytes())
                continue

            cv2.imshow("ArduCAM Stream", frame)
            
            elapsed_time = time.time() - start_time
            fps = 1 / elapsed_time if elapsed_time > 0 else 0
            print(f"FPS: {fps:.2f}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            iteration += 1

        ser.close()
        cv2.destroyAllWindows()
