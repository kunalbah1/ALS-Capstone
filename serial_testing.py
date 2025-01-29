import serial

serial_port = 'COM6'  # Adjust this
baud_rate = 921600

ser = serial.Serial(serial_port, baud_rate, timeout=1)

while True:
    line = ser.readline()
    if line:
        print(f"Received: {line}")
