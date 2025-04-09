import os
model_path = "eye_landmark_model.pth"
if os.path.exists(model_path):
    print(f"File size: {os.path.getsize(model_path)} bytes")
else:
    print("Model file does not exist!")
