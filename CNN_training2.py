import os
import re
import json
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define the target image dimensions used for training (you can adjust these)
IMG_WIDTH = 180
IMG_HEIGHT = 108

def parse_point(point_str):
    """
    Parse a string of the form "(346.3854, 270.2328, 9.1174)" and return (x, y).
    We ignore the third value.
    """
    # Extract floating point numbers using a regex
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", point_str)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    else:
        return None, None

def extract_six_landmarks(json_path):
    """
    Load the JSON file and extract six landmarks from the "interior_margin_2d" field.
    Adjust the indices below as needed.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Choose indices for six landmarks (example indices)
    indices = [0, 4, 7, 8, 11, 15]
    points = data.get("interior_margin_2d", [])
    landmarks = []
    for i in indices:
        if i < len(points):
            x, y = parse_point(points[i])
            if x is None or y is None:
                continue
            landmarks.extend([x, y])
    # If we don't get exactly 12 numbers, return None to skip the sample
    return landmarks if len(landmarks) == 12 else None

def load_data(image_dir, json_dir):
    data = []
    labels = []
    # Assume that every JPG in image_dir has a corresponding JSON file in json_dir with the same base name.
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(".jpg"):
            base = os.path.splitext(filename)[0]
            img_path = os.path.join(image_dir, filename)
            json_path = os.path.join(json_dir, base + ".json")
            if not os.path.exists(json_path):
                continue  # Skip if no corresponding JSON file
            
            # Load and resize image
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype('float32') / 255.0  # Normalize pixel values
            
            # Extract landmarks from JSON
            landmarks = extract_six_landmarks(json_path)
            if landmarks is None:
                continue  # Skip samples with invalid/missing labels

            data.append(img)
            labels.append(landmarks)
    return np.array(data), np.array(labels)

# Paths to your image and JSON directories (adjust these paths)
image_directory = "imgs"
json_directory = "imgs"

X, y = load_data(image_directory, json_directory)
print("Loaded", len(X), "samples.")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dense(12, activation='linear')  # 12 outputs for 6 (x,y) landmark pairs
    ])
    return model

model = create_model()
model.compile(optimizer=Adam(1e-3), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=25, batch_size=64)

# Save the model for later use in live detection
model.save("eye_landmark_model.h5")
