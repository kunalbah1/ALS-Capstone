import serial
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import os
import CNN_eye_landmarks as CNN

##############################################
# Serial Setup
##############################################
BAUD_RATE = 921600  # or 1,000,000 if supported
SERIAL_PORT = "COM6"  # Adjust based on your setup
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)


def read_frame():
    buffer = bytearray()
    # Wait for the start marker.
    while True:
        chunk = ser.readline()
        if b"---START---" in chunk:
            break
    # Read until end marker.
    while True:
        chunk = ser.read_until(b"---END---")
        buffer += chunk.split(b"---END---")[0]
        if b"---END---" in chunk:
            break
    frame_data = np.frombuffer(buffer, dtype=np.uint8)
    if len(frame_data) < 1000:  # Filter out incomplete frames.
        return None
    return frame_data


##############################################
# Load the Saved Model
##############################################

# Ensure input_size match those used during training.
input_size = (100, 100)  # (width, height) as used during training.
model = CNN.StackedHourglass()
model_path = "eye_landmark_model.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Transformation for input images.
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor()
])
soft_argmax = CNN.SoftArgmax2D()


##############################################
# Blink Detection Function
##############################################

# Here, we assume that landmark indices 2 and 5 correspond to an upper and lower eyelid point.
def detect_blink(landmarks, threshold=0.1):
    # landmarks: tensor of shape [num_landmarks, 2] in normalized heatmap space.
    upper = landmarks[2, 1]
    lower = landmarks[5, 1]
    vertical_distance = lower - upper
    if vertical_distance < threshold:
        return True
    return False


##############################################
# Live Blink Detection Loop
##############################################

cv2.namedWindow("CNN Blink Detection", cv2.WINDOW_AUTOSIZE)
blink_count = 0
blink_active = False
VBP_DURATION = 0.2  # Minimum duration (in seconds) to register a blink.
blink_start_time = None

print("Starting live blink detection. Press 'q' to exit.")

while True:
    frame_data = read_frame()
    if frame_data is None:
        continue
    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
    if frame is None:
        continue
    # Convert frame from BGR to RGB and then to a PIL Image.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    img_tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension.

    with torch.no_grad():
        pred_heatmaps = model(img_tensor)
        # Convert heatmaps to landmark coordinates.
        landmarks = soft_argmax(pred_heatmaps)[0]  # Shape: [num_landmarks, 2]
        # print("Raw landmarks (normalized):", landmarks)

    # Map normalized landmarks to pixel coordinates of the resized image.
    h_img, w_img = input_size[1], input_size[0]
    landmarks_px = landmarks.clone()
    landmarks_px[:, 0] = landmarks_px[:, 0] * (w_img - 1)
    landmarks_px[:, 1] = landmarks_px[:, 1] * (h_img - 1)
    landmarks_px = landmarks_px.cpu().numpy().astype(int)

    # Visualize: Resize the frame to input_size.
    frame_resized = cv2.resize(frame, input_size)
    for (x, y) in landmarks_px:
        cv2.circle(frame_resized, (x, y), 2, (0, 255, 0), -1)

    # Blink detection using the CNN-based landmarks.
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
