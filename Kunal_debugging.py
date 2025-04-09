# Testing the trained model on one training image
import cv2

from CNN_eye_landmarks import EyeLandmarkDataset, StackedHourglass, SoftArgmax2D
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


if __name__ == "__main__":
    from torchvision.transforms import ToPILImage

    dataset = EyeLandmarkDataset("test_imgs")
    img, gt_heatmaps = dataset[0]  # Just test the first image
    model = StackedHourglass(num_landmarks=gt_heatmaps.shape[0])
    model.load_state_dict(torch.load("eye_landmark_model2.pth", map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        pred_heatmaps = model(img.unsqueeze(0))
        soft_argmax = SoftArgmax2D(normalize=True)
        coords = soft_argmax(pred_heatmaps)[0].cpu().numpy()

    # Convert normalized coords to pixels
    print(pred_heatmaps.max(), pred_heatmaps.min(), pred_heatmaps.mean())

    coords[:, 0] *= 99
    coords[:, 1] *= 99

    # Visualize
    img_np = ToPILImage()(img).convert("RGB")
    img_cv = np.array(img_np)
    for (x, y) in coords.astype(int):
        cv2.circle(img_cv, (x, y), 2, (0, 255, 0), -1)
    cv2.imshow("Debug Landmark Prediction", img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
