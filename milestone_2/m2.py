import cv2
import json
import os
from datetime import datetime
import random
import atexit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZONE_FILE = os.path.join(BASE_DIR, "zone2.json")
# ---------------- PEOPLE DETECTION SETUP ----------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())



zones = []
drawing = False
start_point = None
fullscreen = False
selected_zone_index = None

# Predefined color palette
COLORS = [
    (0,255,0),
    (255,0,0),
    (0,0,255),
    (255,255,0),
    (255,0,255),
    (0,255,255)
]

# ================= SAVE & LOAD =================

def save_zones():
    with open(ZONE_FILE, "w") as f:
        json.dump(zones, f, indent=4)

def load_zones():
    global zones
    if os.path.exists(ZONE_FILE):
        with open(ZONE_FILE, "r") as f:
            zones = json.load(f)

# Auto-save when program exits
atexit.register(save_zones)

# ================= MOUSE FUNCTION =================

def mouse_events(event, x, y, flags, param):
    global drawing, start_point, zones, selected_zone_index

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if clicking inside an existing zone (for selection)
        for i, zone in enumerate(zones):
            x1, y1 = zone["start"]
            x2, y2 = zone["end"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_zone_index = i
                return

        # Otherwise start drawing
        drawing = True
        start_point = (x, y)
        selected_zone_index = None

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        end_point = (x, y)

        zone_name = f"Zone {len(zones)+1}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = COLORS[len(zones) % len(COLORS)]

        zone = {
            "name": zone_name,
            "start": start_point,
            "end": end_point,
            "color": color,
            "created_at": timestamp
        }

        zones.append(zone)
        print(f"{zone_name} created at {timestamp}")
        save_zones()

# ================= MAIN =================

load_zones()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera failed to open.")
    exit()

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

cv2.setMouseCallback("Video", mouse_events)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame.")
        break
     # ================= PEOPLE DETECTION =================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(8,8),
        padding=(8,8),
        scale=1.05
    )

    people_count = 0

    for (x, y, w, h) in boxes:
        people_count += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    # Display total people count
    cv2.putText(frame, f"Total People: {people_count}",
                (frame.shape[1] - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2)
    

    # Draw zones
    for i, zone in enumerate(zones):
        x1, y1 = zone["start"]
        x2, y2 = zone["end"]
        color = tuple(zone["color"])

        thickness = 4 if i == selected_zone_index else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label
        cv2.putText(frame, zone["name"], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Count placeholder
        cv2.putText(frame, "Count: 0", (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Instructions overlay
    instructions = [
        "Draw Zone: Click & Drag",
        "Click Zone + X: Delete Selected",
        "D: Delete Last Zone",
        "R: Reset Zones",
        "P: Save Screenshot",
        "F: Fullscreen",
        "Q: Quit"
    ]

    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, 20 + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    cv2.imshow("Video", frame)

    key = cv2.waitKey(1) & 0xFF

    # Delete last zone
    if key == ord('d'):
        if zones:
            removed = zones.pop()
            print(f"{removed['name']} deleted")
            save_zones()

    # Delete selected zone
    if key == ord('x') and selected_zone_index is not None:
        removed = zones.pop(selected_zone_index)
        print(f"{removed['name']} deleted (selected)")
        selected_zone_index = None
        save_zones()

    # Reset all zones
    if key == ord('r'):
        zones.clear()
        print("All zones cleared")
        save_zones()

    # Save screenshot
    if key == ord('p'):
        filename = f"screenshot_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")

    # Fullscreen toggle
    if key == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty("Video",
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Video",
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL)

    # Quit safely
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
