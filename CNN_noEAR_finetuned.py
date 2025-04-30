import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Image dimensions
IMG_WIDTH = 128
IMG_HEIGHT = 128

def load_images_from_folder(folder, label):
    """
    Loads all images from `folder`, resizes to IMG_WIDTH×IMG_HEIGHT,
    normalizes to [0,1], and returns lists of images and labels.
    """
    images, labels = [], []
    for fn in os.listdir(folder):
        if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(folder, fn))
            if img is not None:
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(label)
    return images, labels

# 1. Load your pre-trained blink-detection model
model = load_model("blink_detection_model.h5")

# 2. Freeze the convolutional base (all but the last 3 layers)
for layer in model.layers[:-3]:
    layer.trainable = False

# 3. Re-compile with a low learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Load your occluded-eye dataset
#    Update these paths to point at your folders of occluded closed/open images
occluded_paths = {
    0: ["dataset_occluded_eyes/occludedLeftClosed", "dataset_occluded_eyes/occludedRightClosed"],
    1: ["dataset_occluded_eyes/occludedLeftOpen",   "dataset_occluded_eyes/occludedRightOpen"]
}

X_new, y_new = [], []
for label, folders in occluded_paths.items():
    for folder in folders:
        imgs, labs = load_images_from_folder(folder, label)
        X_new.extend(imgs)
        y_new.extend(labs)

# Optionally: mix in a subset of your original normal-eyes data here
# X_norm, y_norm = load_images_from_folder("dataset_norm/openEyes", 1)
# X_new.extend(X_norm); y_new.extend(y_norm)

X_new = np.array(X_new)
y_new = np.array(y_new)

# 5. Split into train/validation sets (90/10)
X_train, X_val, y_train, y_val = train_test_split(
    X_new, y_new, test_size=0.1, random_state=42, shuffle=True
)

# 6. Fine-tune the classifier head on occluded data
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,        # small number to avoid overfitting
    batch_size=16
)

# 7. (Optional) Unfreeze the last conv block and do a gentle pass
for layer in model.layers[-6:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-6),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=16
)

# 8. Save your fine-tuned model
model.save("blink_detection_finetuned.h5")
print("Fine-tuned model saved as blink_detection_finetuned.h5")
