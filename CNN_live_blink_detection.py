import serial 
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import time
import os

##############################################
# Serial Setup
##############################################
BAUD_RATE = 921600  # or 1,000,000 if supported
SERIAL_PORT = "COM6"  # Adjust based on your setup
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)

def read_frame():
    buffer = bytearray()
    # Wait for start marker.
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
# Model Definitions (must match training)
##############################################
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
            
    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class Hourglass(nn.Module):
    def __init__(self, channels):
        super(Hourglass, self).__init__()
        self.res1 = Residual(channels, channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.res2 = Residual(channels, channels)
        # Instead of scale_factor, we use interpolate with explicit size.
        self.upsample = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        skip = self.res1(x)
        down = self.pool(x)
        down = self.res2(down)
        up = self.upsample(down, size=skip.shape[2:])
        return skip + up

class StackedHourglass(nn.Module):
    def __init__(self, in_channels=3, num_features=64, num_hourglass=3, num_landmarks=16):
        """
        num_landmarks: number of output heatmaps.
        Adjust this number to match your training configuration.
        """
        super(StackedHourglass, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=7, stride=2, padding=3)
        self.bn1   = nn.BatchNorm2d(num_features)
        self.relu  = nn.ReLU(inplace=True)
        self.res_init = Residual(num_features, num_features)
        self.hourglass_modules = nn.ModuleList([Hourglass(num_features) for _ in range(num_hourglass)])
        self.conv_out = nn.Conv2d(num_features, num_landmarks, kernel_size=1)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res_init(x)
        for hg in self.hourglass_modules:
            x = hg(x)
        heatmaps = self.conv_out(x)
        return heatmaps

class SoftArgmax2D(nn.Module):
    def __init__(self, normalize=True):
        super(SoftArgmax2D, self).__init__()
        self.normalize = normalize
        
    def forward(self, heatmaps):
        # heatmaps: [B, num_landmarks, H, W]
        b, n, h, w = heatmaps.size()
        heatmaps = heatmaps.view(b, n, -1)
        softmax = F.softmax(heatmaps, dim=-1)
        indices = torch.arange(h * w, device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, -1)
        coords = torch.sum(softmax * indices, dim=-1)
        x = coords % w
        y = coords // w
        if self.normalize:
            x = x / (w - 1)
            y = y / (h - 1)
        coords = torch.stack([x, y], dim=-1)  # Shape: [B, num_landmarks, 2]
        return coords

##############################################
# Load the Saved Model
##############################################
# Ensure num_landmarks and input_size match those used during training.
num_landmarks = 16  # Update if necessary.
input_size = (150, 90)  # (width, height) as used during training.
model = StackedHourglass(in_channels=3, num_features=64, num_hourglass=3, num_landmarks=num_landmarks)
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
soft_argmax = SoftArgmax2D(normalize=True)

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
