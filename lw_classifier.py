import cv2
import numpy as np
import time

def set_camera_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_EXPOSURE, -50)

def center_crop_and_resize(img, target_size=(224, 224)):
    """
    Crops the center of a 640x480 image to 480x480 and then resizes it to target_size.
    """
    h, w = img.shape[:2]  # Get image dimensions
    
    # Calculate cropping box for center crop
    crop_size = min(h, w)  # Make it square
    start_x = (w - crop_size) // 2  # Center horizontally
    start_y = (h - crop_size) // 2  # Center vertically

    cropped_img = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
    resized_img = cv2.resize(cropped_img, target_size)

    return resized_img

def image_difference(img1, img2):
    """
    Calculate the mean absolute difference between two images.
    Lower values indicate more similarity.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert to grayscale for simpler comparison
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        
    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2
    
    # Calculate absolute difference
    diff = cv2.absdiff(img1_gray, img2_gray)
    
    # Return mean difference (lower is more similar)
    return np.mean(diff)

# Paths to the reference images
contact_reference_image_paths = [
    "contact_reference/images/image_20250305_005203.jpg",
    "contact_reference/images/image_20250305_005206.jpg",
    "contact_reference/images/image_20250305_005207.jpg",
    "contact_reference/images/image_20250305_005209.jpg",
    "contact_reference/images/image_20250305_005210.jpg",
]

no_contact_reference_image_paths = [
    "contact_reference/images/image_20250304_233311.jpg",
    "contact_reference/images/image_20250304_233312.jpg",
    "contact_reference/images/image_20250304_233313.jpg",
    "contact_reference/images/image_20250304_233314.jpg",
    "contact_reference/images/image_20250304_233315.jpg"
]

# Load contact reference images
contact_images = []
try:
    for path in contact_reference_image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load image: {path}")
            continue
        
        # Preprocess reference image
        processed_img = center_crop_and_resize(img)
        contact_images.append(processed_img)
    
    print(f"Loaded {len(contact_images)} contact reference images successfully")
except Exception as e:
    print(f"Error loading contact reference images: {e}")

# Load no-contact reference images
no_contact_images = []
try:
    for path in no_contact_reference_image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load image: {path}")
            continue
        
        # Preprocess reference image
        processed_img = center_crop_and_resize(img)
        no_contact_images.append(processed_img)
    
    print(f"Loaded {len(no_contact_images)} no-contact reference images successfully")
except Exception as e:
    print(f"Error loading no-contact reference images: {e}")

# Check if we have enough reference images to proceed
if len(contact_images) == 0 or len(no_contact_images) == 0:
    print("Error: Not enough reference images were loaded")
    exit()

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

print("Starting classification. Press Ctrl+C to quit.")

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        # Get current time
        current_time = time.time()

        # Check if it's time for a new prediction (10Hz)
        if current_time - last_prediction_time >= prediction_interval:
            # Preprocess the frame
            processed_frame = center_crop_and_resize(frame)
            
            # Calculate differences to all contact reference images
            contact_diffs = []
            for ref_img in contact_images:
                diff = image_difference(processed_frame, ref_img)
                contact_diffs.append(diff)
            
            # Calculate differences to all no-contact reference images
            no_contact_diffs = []
            for ref_img in no_contact_images:
                diff = image_difference(processed_frame, ref_img)
                no_contact_diffs.append(diff)
            
            # Get the minimum difference for each category
            min_contact_diff = min(contact_diffs)
            min_no_contact_diff = min(no_contact_diffs)
            
            # Get the average difference for each category (alternative approach)
            avg_contact_diff = sum(contact_diffs) / len(contact_diffs)
            avg_no_contact_diff = sum(no_contact_diffs) / len(no_contact_diffs)
            
            # Determine classification based on minimum difference
            if min_contact_diff < min_no_contact_diff:
                status = "Contact"
                confidence = 1.0 - (min_contact_diff / (min_contact_diff + min_no_contact_diff))
                best_match_index = contact_diffs.index(min_contact_diff)
                best_match_path = contact_reference_image_paths[best_match_index]
            else:
                status = "No Contact"
                confidence = 1.0 - (min_no_contact_diff / (min_contact_diff + min_no_contact_diff))
                best_match_index = no_contact_diffs.index(min_no_contact_diff)
                best_match_path = no_contact_reference_image_paths[best_match_index]
            
            # Only update terminal output if status changes or significant change in confidence
            if last_status != status or True:  # Always print for now
                print(f"Status: {status} (confidence: {confidence:.2f})")
                print(f"Best match: {best_match_path}")
                print(f"Min contact diff: {min_contact_diff:.2f}, Min no-contact diff: {min_no_contact_diff:.2f}")
                print(f"Avg contact diff: {avg_contact_diff:.2f}, Avg no-contact diff: {avg_no_contact_diff:.2f}")
                print("-" * 50)
                last_status = status

            # Update last prediction time
            last_prediction_time = current_time

except KeyboardInterrupt:
    print("\nInterrupted by user")

# Release the camera when done
cap.release()
print("Classification completed.")