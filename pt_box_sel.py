import cv2
import numpy as np
from ultralytics import SAM

# Load the SAM model
model = SAM("sam2.1_b.pt")

# Initialize global variables
global bboxes, points, image, temp_image
bboxes = []
points = []

def draw_annotations():
    global temp_image
    temp_image = image.copy()
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(temp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for point in points:
        cv2.circle(temp_image, point, 5, (0, 0, 255), -1)

def select_bbox(event, x, y, flags, param):
    global temp_bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        temp_bbox = [x, y, x, y]
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        temp_bbox[2:] = [x, y]
        draw_annotations()
        cv2.rectangle(temp_image, (temp_bbox[0], temp_bbox[1]), (x, y), (255, 0, 0), 2)
        cv2.imshow("Image", temp_image)
    elif event == cv2.EVENT_LBUTTONUP:
        temp_bbox[2:] = [x, y]
        bboxes.append(temp_bbox)
        draw_annotations()
        cv2.imshow("Image", temp_image)

def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        draw_annotations()
        cv2.imshow("Image", temp_image)

def delete_bbox(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for bbox in bboxes[:]:
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                bboxes.remove(bbox)
                break
        draw_annotations()
        cv2.imshow("Image", temp_image)

def delete_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for point in points[:]:
            px, py = point
            if abs(px - x) <= 5 and abs(py - y) <= 5:
                points.remove(point)
                break
        draw_annotations()
        cv2.imshow("Image", temp_image)

def analyze():
    print("Selected bboxes:", bboxes)
    print("Selected points:", points)
    if bboxes or points:
        if bboxes:
            results = model(image_path, bboxes=bboxes, save=True, project='result', name='seg')
            print(f"bboxes Results saved.")
            bbox_result_image = cv2.imread(results[0].save_dir + '\\sample.jpg')
            if bbox_result_image is not None:
                cv2.imshow(f"Bbox Results", bbox_result_image)
        # for bbox in bboxes:
        #     results = model(image_path, bboxes=bbox, save=True, project='result', name='seg')
        #     print(f"Results for bbox {bbox} saved.")
        #     bbox_result_image = cv2.imread(results[0].save_dir + '\\sample.jpg')
        #     if bbox_result_image is not None:
        #         cv2.namedWindow(f"BBox Result {bbox}", cv2.WINDOW_NORMAL)
        #         cv2.resizeWindow(f"BBox Result {bbox}", 1024, 768)
        #         cv2.imshow(f"BBox Result {bbox}", bbox_result_image)
        if points:
            results = model(image_path, points=points, labels=[1]*len(points), save=True, project='result', name='seg')
            print("Point analysis results saved.")
            point_result_image = cv2.imread(results[0].save_dir + '\\sample.jpg')
            if point_result_image is not None:
                cv2.imshow("Point Results", point_result_image)
    else:
        print("No bboxes or points to analyze.")
    cv2.putText(temp_image, "Analysis Complete", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Image", temp_image)

def clear_all():
    global bboxes, points
    bboxes.clear()
    points.clear()
    draw_annotations()
    cv2.imshow("Image", temp_image)

def show_startup_commands():
    commands = [
        "Commands:",
        "q - Quit",
        "b - Draw bounding boxes",
        "d - Delete bounding boxes",
        "p - Add points",
        "x - Delete points",
        "a - Analyze",
        "c - Clear all annotations"
    ]
    command_image = np.zeros((300, 700, 3), dtype=np.uint8)
    for i, command in enumerate(commands):
        cv2.putText(command_image, command, (10, 40 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Commands", command_image)
    # cv2.waitKey(3000)
    # cv2.destroyWindow("Commands")

# Load the image
image_path = "sample.jpg"
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Could not read the image. Check the image path.")

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1024, 768)
draw_annotations()
cv2.imshow("Image", image)

cv2.namedWindow(f"Bbox Results", cv2.WINDOW_NORMAL)
cv2.resizeWindow(f"Bbox Results", 1024, 768)
cv2.namedWindow("Point Results", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Point Results", 1024, 768)
show_startup_commands()

def reset_mouse_callback():
    cv2.setMouseCallback("Image", lambda *args: None)

while True:
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('b'):
        print("Draw bounding boxes. Press 'b' to stop.")
        reset_mouse_callback()
        cv2.setMouseCallback("Image", select_bbox)
    elif key == ord('d'):
        print("Delete bounding boxes. Press 'd' to stop.")
        reset_mouse_callback()
        cv2.setMouseCallback("Image", delete_bbox)
    elif key == ord('p'):
        print("Add points. Press 'p' to stop.")
        reset_mouse_callback()
        cv2.setMouseCallback("Image", select_point)
    elif key == ord('x'):
        print("Delete points. Press 'x' to stop.")
        reset_mouse_callback()
        cv2.setMouseCallback("Image", delete_point)
    elif key == ord('a'):
        print("Analyzing...")
        analyze()
    elif key == ord('c'):
        print("Clearing all annotations...")
        clear_all()

cv2.destroyAllWindows()