import cv2
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
zone_file = os.path.join(BASE_DIR, "zone1.json")
# Global variables
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1


# Load saved zone if exists
def load_zone():
    global ix, iy, fx, fy
    if os.path.exists(zone_file):
        with open(zone_file, "r") as f:
            data = json.load(f)
            ix, iy, fx, fy = data["ix"], data["iy"], data["fx"], data["fy"]

# Save zone to file
def save_zone():
    data = {
        "ix": ix,
        "iy": iy,
        "fx": fx,
        "fy": fy
    }
    with open(zone_file, "w") as f:
        json.dump(data, f)

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        save_zone()
        print("Zone Saved!")

# Load saved zone at startup
load_zone()

# Open webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", draw_rectangle)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw rectangle if exists
    if ix != -1 and iy != -1 and fx != -1 and fy != -1:
        cv2.rectangle(frame, (ix, iy), (fx, fy), (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
