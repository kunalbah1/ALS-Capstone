import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib

matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'Agg', etc., depending on your system

import matplotlib.pyplot as plt

gol_input_size = (250, 250)


def visualize_heatmap(heatmap):
    """Display a single heatmap."""
    plt.imshow(heatmap, cmap='jet')  # 'jet' gives a nice color gradient
    plt.colorbar()
    plt.title("Landmark Heatmap")
    plt.show()


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
    dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    exponent = -dist_sq / (2 * sigma ** 2)
    heatmap = np.exp(exponent)
    heatmap = heatmap / np.max(heatmap)
    # visualize_heatmap(heatmap)
    return heatmap


#####################################
# Custom Dataset: Aggregating All Keys
#####################################

class EyeLandmarkDataset(Dataset):
    def __init__(self, data_dir, input_size=gol_input_size, sigma=1.5):
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
        image_resized = self.transform(
            image)  # Transform will shift the synthesized image in reference to the input_size
        target_h, target_w = self.input_size

        # Load JSON.
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract only 'interior_margin_2d' landmarks.
        if "interior_margin_2d" not in data or not isinstance(data["interior_margin_2d"], list):
            raise ValueError(f"Missing or invalid 'interior_margin_2d' in {json_path}")

        landmarks = data["interior_margin_2d"]  # Expecting a list of coordinate strings

        # Parse and scale landmarks from original image coordinates to the resized image.
        scaled_landmarks = []
        for s in landmarks:
            x, y = parse_landmark_string(s)
            # Scale coordinates from original image to resized image.
            x = x * (target_w / orig_w)
            y = y * (target_h / orig_h)
            scaled_landmarks.append((x, y))  # Correct for the input_size that you have

        # Assume the network downsamples by a factor of 2.
        heatmap_size = (target_h // 2, target_w // 2)  # (height, width)
        # Scale landmarks to heatmap space.
        heatmap_landmarks = [(x / 2.0, y / 2.0) for (x, y) in scaled_landmarks]

        num_interior = len(heatmap_landmarks)
        heatmaps = np.zeros((num_interior, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1)  # Sets up a 2D 3 x 3 convolution matrix to do stuff
        self.bn1 = nn.BatchNorm2d(out_channels)  # idk but like its important
        self.relu = nn.ReLU(inplace=True)  # same
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1)  # Sets up another 2D 3 x 3 convolution matrix (layer 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:  # If there isn't a 1 to 1 input to output ratio something is wrong.
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
    def __init__(self, channels):  # The channels refer to the depth of the feature space
        super(Hourglass, self).__init__()
        self.res1 = Residual(channels, channels)  # The 3 x 3 conv matrix for layer 1
        self.pool = nn.MaxPool2d(2, 2)
        self.res2 = Residual(channels, channels)  # Layer 2
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    # Down sampling: Reducing spatial resolution while maintaining key info
    # Example: Pooling via maximum value to create a 2x2 matrix from a 4x4

    # Up sampling: Increasing spatial resolution to try and restore details

    def forward(self, x):  # Defines dataflow through the hourglass
        skip = self.res1(x)
        down = self.pool(x)
        down = self.res2(down)
        # Use F.interpolate with size set to skip's spatial dimensions:
        up = F.interpolate(down, size=skip.shape[2:], mode='bilinear',
                           align_corners=True)  # Return the down sample to its original size and add it back
        return skip + up


# Stacked Hourglass Network.
class StackedHourglass(nn.Module):
    def __init__(self, in_channels=3, num_features=64, num_hourglass=3, num_landmarks=16):
        """
        num_landmarks: number of output heatmaps.
        """
        super(StackedHourglass, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
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
        temperature = 0.1
        # heatmaps shape: [B, num_landmarks, H, W]
        b, c, h, w = heatmaps.shape  # b = batch size, n = num_landmarks, h & w is height and width
        heatmaps_flat = heatmaps.view(b, c, -1) / temperature
        softmaxed = F.softmax(heatmaps_flat, dim=2)

        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), dim=-1)  # (H, W, 2)
        coords = coords.reshape(-1, 2).to(heatmaps.device).float()  # (H*W, 2)

        expected_coords = torch.einsum("bcn,nk->bck", softmaxed, coords)  # (B, C, 2)
        return expected_coords

        # heatmaps = heatmaps.view(b, n, -1)
        # softmax = F.softmax(heatmaps, dim=-1)
        # indices = torch.arange(h * w, device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, -1)
        # coords = torch.sum(softmax * indices, dim=-1)
        # x = coords % w
        # y = coords // w
        # if self.normalize:
        #     x = x / (w - 1)
        #     y = y / (h - 1)
        # coords = torch.stack([x, y], dim=-1)
        # return coords


#####################################
# Training and Saving the Model
#####################################

if __name__ == '__main__':
    # Set parameters.
    data_dir = "test_imgs"  # Replace with your dataset directory.
    input_size = gol_input_size  # (width, height)
    sigma = 1.5

    # Create dataset and dataloader.
    dataset = EyeLandmarkDataset(data_dir, input_size=input_size, sigma=sigma)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    softargmax = SoftArgmax2D()

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

            # ------------------------------------------- GT HEATMAP -------------------------------------------

            # for i in range(gt_heatmaps.shape[1]):
            #     plt.imshow(gt_heatmaps[0, i].detach().cpu(), cmap='jet')
            #     plt.title(f'Pred Heatmap {i}')
            #     plt.colorbar()
            #     plt.show()

            # ------------------------------------------- GT HEATMAP -------------------------------------------

            # ------------------------------------------- PRED HEATMAP -------------------------------------------

            # for i in range(pred_heatmaps.shape[1]):
            #     plt.imshow(pred_heatmaps[0, i].detach().cpu(), cmap='jet')
            #     plt.title(f'Pred Heatmap {i}')
            #     plt.colorbar()
            #     plt.show()

            # ------------------------------------------- PRED HEATMAP -------------------------------------------

            # ------------------------------------------- DEBUGGING -------------------------------------------

            # pred_coords = softargmax(pred_heatmaps)[0].detach().cpu().numpy()
            # gt_coords = softargmax(gt_heatmaps)[0].detach().cpu().numpy()
            #
            # pred_x, pred_y = pred_coords[:, 1], pred_coords[:, 0]
            # gt_x, gt_y = gt_coords[:, 1], gt_coords[:, 0]
            #
            # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            #
            # axs[0].imshow(pred_heatmaps[0].detach().cpu().mean(dim=0).numpy(), cmap="hot")
            # axs[0].scatter(pred_x, pred_y, color='blue', s=40, label="Pred")
            # axs[0].set_title("Predicted")
            #
            # axs[1].imshow(gt_heatmaps[0].detach().cpu().mean(dim=0).numpy(), cmap="hot",)
            # axs[1].scatter(gt_x, gt_y, color='green', s=40, label="GT")
            # axs[1].set_title("Ground Truth")
            #
            # for ax in axs:
            #     ax.legend()
            #
            # plt.tight_layout()
            # plt.show()
            #
            # print("Heatmap stats:", pred_heatmaps.max().item(), pred_heatmaps.min().item(), pred_heatmaps.mean().item())

            # ------------------------------------------- DEBUGGING -------------------------------------------

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}')

    # Save the trained model.
    torch.save(model.state_dict(), "eye_landmark_model.pth")
    print("Model saved to eye_landmark_model.pth")
