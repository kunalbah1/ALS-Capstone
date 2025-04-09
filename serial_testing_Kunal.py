import serial
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import os
import CNN_eye_landmarks as CNN  # This includes SoftArgmax2D

##############################################
# Serial Setup
##############################################
BAUD_RATE = 921600
SERIAL_PORT = "COM6"
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)


def read_frame():
    buffer = bytearray()
    while True:
        chunk = ser.readline()
        if b"---START---" in chunk:
            break
    while True:
        chunk = ser.read_until(b"---END---")
        buffer += chunk.split(b"---END---")[0]
        if b"---END---" in chunk:
            break
    frame_data = np.frombuffer(buffer, dtype=np.uint8)
    if len(frame_data) < 1000:
        return None
    return frame_data


##############################################
# Load the Saved Model
##############################################
input_size = (250, 250)
model = CNN.StackedHourglass()
model_path = "eye_landmark_model2.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize(input_size),  # 🔧 Ensure consistency with training
    transforms.ToTensor()
])

soft_argmax = CNN.SoftArgmax2D(normalize=True)  # 🔧 Ensure normalized output


##############################################
# Blink Detection Function
##############################################
def detect_blink(landmarks, threshold=0.1):
    upper = landmarks[2, 1]
    lower = landmarks[5, 1]
    vertical_distance = lower - upper
    return vertical_distance < threshold


##############################################
# Live Blink Detection Loop
##############################################
cv2.namedWindow("CNN Blink Detection", cv2.WINDOW_AUTOSIZE)
blink_count = 0
blink_active = False
VBP_DURATION = 0.2
blink_start_time = None

print("Starting live blink detection. Press 'q' to exit.")

while True:
    frame_data = read_frame()
    if frame_data is None:
        continue

    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
    if frame is None:
        continue

    # 🔧 Preprocess frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    img_tensor = transform(pil_img).unsqueeze(0)  # shape: [1, 3, 100, 100]

    with torch.no_grad():
        pred_heatmaps = model(img_tensor)  # shape: [1, L, H, W]
        landmarks = soft_argmax(pred_heatmaps)[0]  # shape: [L, 2], normalized coords

    # 🔧 Convert normalized to pixel coordinates
    landmarks_px = landmarks.clone()
    landmarks_px[:, 0] *= (input_size[0] - 1)
    landmarks_px[:, 1] *= (input_size[1] - 1)
    landmarks_px = landmarks_px.cpu().numpy().astype(int)

    # 🔧 Resize frame for display and overlay landmarks
    frame_resized = cv2.resize(frame, input_size)
    for (x, y) in landmarks_px:
        cv2.circle(frame_resized, (x, y), 2, (0, 255, 0), -1)  # Green dots

    # Blink logic
    blink = detect_blink(landmarks)
    if blink:
        if not blink_active:
            blink_start_time = time.time()
            blink_active = True
    else:
        if blink_active and (time.time() - blink_start_time >= VBP_DURATION):
            blink_count += 1
        blink_active = False

    cv2.putText(frame_resized, f"Blink Count: {blink_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("CNN Blink Detection", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

ser.close()
cv2.destroyAllWindows()
