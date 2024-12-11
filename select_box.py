import cv2
from ultralytics import SAM

# Load the SAM model
model = SAM("sam2.1_b.pt")

# Display model information (optional)
model.info()

# Function to select a bounding box using the mouse
def select_bbox(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image. Check the image path.")
    
    # Open a window to select the ROI
    roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # Return the bounding box in the required format
    x, y, w, h = roi
    return [x, y, x + w, y + h]

# Path to the input image
image_path = "sample.jpg"

# Select the bounding box with the mouse
bbox = select_bbox(image_path)
print(f"Selected bbox: {bbox}")

# Run inference with the selected bounding box
results = model(image_path, bboxes=bbox, save=True, project='result', name='seg')

print("Inference completed. Results saved.")
