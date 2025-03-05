import cv2
import time
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

class TactileCNN(torch.nn.Module):
    def __init__(self):
        super(TactileCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 28 * 28, 128),  # Changed from 512 to 128 to match original
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),  
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def set_camera_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_EXPOSURE, -50)

def center_crop_and_resize(img, target_size=(224, 224)):
    """
    Crops the center of a 640x480 image to 480x480 and then resizes it to target_size.
    """
    h, w = img.shape[:2]  # Get image dimensions (480, 640)
    
    # Calculate cropping box for center crop
    crop_size = min(h, w)  # Make it square (480, 480)
    start_x = (w - crop_size) // 2  # Center horizontally
    start_y = (h - crop_size) // 2  # Center vertically

    cropped_img = img[start_y:start_y+crop_size, start_x:start_x+crop_size]  # Crop to 480x480
    resized_img = cv2.resize(cropped_img, target_size)  # Resize to 224x224

    return resized_img

# Set up device and transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model
model_path = "/home/pi/lab_3/fine_grained.pth"
model = TactileCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Loaded model from {model_path}")

# Initialize the USB webcam feed
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Set camera resolution
set_camera_resolution(cap, 640, 480)

# Allow camera to warm up
time.sleep(1)

# Set the target frequency for predictions (10 Hz)
prediction_interval = 1.0 / 10.0  # 0.1 seconds between predictions
last_prediction_time = time.time()

# Initialize prediction status
last_status = None
last_confidence = 0.0
status_text = ""
status_color = (0, 0, 255)  # Default to red (no contact)

print("Starting classification. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    # Get current time
    current_time = time.time()

    # Check if it's time for a new prediction (5Hz)
    if current_time - last_prediction_time >= prediction_interval:
        # Preprocess the frame for the model exactly as in training
        processed_frame = center_crop_and_resize(frame)  # Crop and resize to 224x224
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        processed_frame = transform(processed_frame).unsqueeze(0).to(device)  # Apply transforms and add batch dimension

        # Get prediction
        with torch.no_grad():
            prediction = model(processed_frame).item()
        
        contact_status = "Contact" if prediction > 0.5 else "No Contact"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Only update display text if status changes or confidence changes significantly
        if last_status != contact_status or abs(confidence - last_confidence) > 0.05:
            status_text = f"{contact_status} ({confidence:.2f})"
            status_color = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)
            last_status = contact_status
            last_confidence = confidence

        # Update last prediction time
        last_prediction_time = current_time

    # Display the persistent status on frame
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                status_color, 2)

    # Display the frame
    cv2.imshow('Tactile Sensor Classifier', frame)

    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()
print("Classification completed.") 