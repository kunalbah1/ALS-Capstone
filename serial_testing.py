import serial
import cv2
import numpy as np

# Configure the serial port
ser = serial.Serial('COM6', 115200, timeout=1)


def read_image_from_serial():
    image_data = bytearray()
    start_found = False

    while True:
        line = ser.readline()

        if not start_found:
            if b'---START---' in line:
                start_found = True
                print("📸 Start of image detected!")
                continue
        elif b'---END---' in line:
            print("✅ End of image detected!")
            break
        elif start_found:
            image_data.extend(line)

    return bytes(image_data)


def main():
    while True:
        image_bytes = read_image_from_serial()

        if len(image_bytes) > 0:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                cv2.imshow('Video Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("⚠️ Failed to decode image.")
        else:
            print("⚠️ No image data received.")

    ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
