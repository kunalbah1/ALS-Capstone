import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Set the image dimensions for training.
# You can increase these dimensions if you wish, but larger sizes mean more computation.
IMG_WIDTH = 128
IMG_HEIGHT = 128

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize image to the target dimensions
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                # Convert image to float32 and normalize pixel values to [0, 1]
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(label)
    return images, labels

# Define folder paths (update these paths to match your dataset organization)
closed_left_folder = "dataset_B_Eye_Images\closedLeftEyes"
closed_right_folder = "dataset_B_Eye_Images\closedRightEyes"
open_left_folder = "dataset_B_Eye_Images\openLeftEyes"
open_right_folder = "dataset_B_Eye_Images\openRightEyes"

# Load images from all four folders and assign labels:
# Closed eyes are labeled 0, Open eyes are labeled 1.
X = []
y = []

imgs, labs = load_images_from_folder(closed_left_folder, 0)
X.extend(imgs)
y.extend(labs)

imgs, labs = load_images_from_folder(closed_right_folder, 0)
X.extend(imgs)
y.extend(labs)

imgs, labs = load_images_from_folder(open_left_folder, 1)
X.extend(imgs)
y.extend(labs)

imgs, labs = load_images_from_folder(open_right_folder, 1)
X.extend(imgs)
y.extend(labs)

X = np.array(X)
y = np.array(y)

print("Loaded", len(X), "images.")

# Split data into training and validation sets (90% training, 10% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Define a simple CNN for binary classification (open vs closed)
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary output: 0 for closed, 1 for open
    ])
    return model

model = create_model()
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=50, batch_size=32)  # Increase epochs if needed

# Save the model for later use in live blink detection
model.save("blink_detection_model.h5")
print("Model saved as blink_detection_model.h5")
