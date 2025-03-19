import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader

#####################################
# Utility Functions
#####################################

def parse_landmark_string(s):
    # Expects a string like "(346.3854, 270.2328, 9.1174)" and returns the first two floats.
    s = s.strip().strip('()')
    parts = s.split(',')
    if len(parts) < 2:
        raise ValueError("Unexpected landmark string format: " + s)
    x = float(parts[0])
    y = float(parts[1])
    return x, y

def generate_heatmap(center, heatmap_size, sigma):
    """
    Generates a 2D Gaussian heatmap.
    center: (x, y) in heatmap coordinate space.
    heatmap_size: (height, width)
    sigma: standard deviation for the Gaussian.
    """
    h, w = heatmap_size
    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    cx, cy = center
    dist_sq = (grid_x - cx)**2 + (grid_y - cy)**2
    exponent = -dist_sq / (2 * sigma**2)
    heatmap = np.exp(exponent)
    heatmap = heatmap / np.max(heatmap)
    return heatmap

#####################################
# Custom Dataset: Aggregating All Keys
#####################################

class EyeLandmarkDataset(Dataset):
    def __init__(self, data_dir, input_size=(150, 90), sigma=1.5):
        """
        data_dir: directory containing jpg and json files.
        input_size: desired size for network input (width, height).
        sigma: standard deviation for heatmap generation.
        """
        self.data_dir = data_dir
        self.input_size = input_size  # (width, height)
        self.sigma = sigma

        # List all jpg files in the directory.
        self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.jpg')]
        self.image_files.sort()  # for reproducibility

        # Transform: resize image to input_size and convert to tensor.
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image and JSON filenames.
        img_name = self.image_files[idx]
        json_name = os.path.splitext(img_name)[0] + ".json"
        img_path = os.path.join(self.data_dir, img_name)
        json_path = os.path.join(self.data_dir, json_name)
        
        # Load and transform image.
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size  # (width, height)
        image_resized = self.transform(image)
        target_h, target_w = self.input_size
        
        # Load JSON.
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Loop over every key and extract values that are lists of coordinate strings.
        landmarks = []
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], str) and val[0].strip().startswith("("):
                landmarks.extend(val)
        
        if len(landmarks) == 0:
            raise ValueError(f"No valid landmark entries found in {json_path}")

        # Parse and scale landmarks from original image coordinates to the resized image.
        scaled_landmarks = []
        for s in landmarks:
            x, y = parse_landmark_string(s)
            # Scale coordinates from original image to resized image.
            x = x * (target_w / orig_w)
            y = y * (target_h / orig_h)
            scaled_landmarks.append((x, y))
        
        # Assume the network downsamples by a factor of 2.
        heatmap_size = (target_h // 2, target_w // 2)  # (height, width)
        # Scale landmarks to heatmap space.
        heatmap_landmarks = [(x / 2.0, y / 2.0) for (x, y) in scaled_landmarks]
        
        num_landmarks = len(heatmap_landmarks)
        heatmaps = np.zeros((num_landmarks, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
        for i, center in enumerate(heatmap_landmarks):
            heatmaps[i] = generate_heatmap(center, heatmap_size, self.sigma)
        
        heatmaps = torch.from_numpy(heatmaps)
        return image_resized, heatmaps

#####################################
# Network Architecture
#####################################

# Basic Residual Block.
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

# Simplified Hourglass Module.
class Hourglass(nn.Module):
    def __init__(self, channels):
        super(Hourglass, self).__init__()
        self.res1 = Residual(channels, channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.res2 = Residual(channels, channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        skip = self.res1(x)
        down = self.pool(x)
        down = self.res2(down)
        # Use F.interpolate with size set to skip's spatial dimensions:
        up = F.interpolate(down, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return skip + up


# Stacked Hourglass Network.
class StackedHourglass(nn.Module):
    def __init__(self, in_channels=3, num_features=64, num_hourglass=3, num_landmarks=1):
        """
        num_landmarks: number of output heatmaps.
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

# Soft-argmax Layer (Optional): Converts heatmaps to (x, y) coordinates.
class SoftArgmax2D(nn.Module):
    def __init__(self, normalize=True):
        super(SoftArgmax2D, self).__init__()
        self.normalize = normalize
        
    def forward(self, heatmaps):
        # heatmaps shape: [B, num_landmarks, H, W]
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
        coords = torch.stack([x, y], dim=-1)
        return coords

#####################################
# Training and Saving the Model
#####################################

if __name__ == '__main__':
    # Set parameters.
    data_dir = "imgs"  # Replace with your dataset directory.
    input_size = (90, 150)  # (width, height)
    sigma = 1.5
    
    # Create dataset and dataloader.
    dataset = EyeLandmarkDataset(data_dir, input_size=input_size, sigma=sigma)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Determine number of landmarks from one sample.
    sample_img, sample_heatmaps = dataset[0]
    num_landmarks = sample_heatmaps.shape[0]
    print("Number of landmarks:", num_landmarks)
    
    # Instantiate the model, loss function, and optimizer.
    model = StackedHourglass(in_channels=3, num_features=64, num_hourglass=3, num_landmarks=num_landmarks)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    # Training loop.
    model.train()
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for imgs, gt_heatmaps in dataloader:
            optimizer.zero_grad()
            pred_heatmaps = model(imgs)
            loss = criterion(pred_heatmaps, gt_heatmaps)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}')
    
    # Save the trained model.
    torch.save(model.state_dict(), "eye_landmark_model.pth")
    print("Model saved to eye_landmark_model.pth")
    
