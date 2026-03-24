import cv2
import json
import os
import csv
from datetime import datetime
import atexit
from ultralytics import YOLO

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZONE_FILE = os.path.join(BASE_DIR, "zone3.json")
CSV_FILE = os.path.join(BASE_DIR, "count_data.csv")

# ================= YOLO TRACKING SETUP =================
model = YOLO(os.path.join(BASE_DIR, "..", "yolov8n.pt"))

zones = []
drawing = False
start_point = None
selected_zone_index = None
fullscreen = False

entry_count = {}
exit_count = {}
id_zone_side = {}

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

    for zone in zones:
        entry_count[zone["name"]] = 0
        exit_count[zone["name"]] = 0

atexit.register(save_zones)

# ================= CSV LOGGING =================
def log_to_csv(zone_name):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            zone_name,
            entry_count[zone_name],
            exit_count[zone_name]
        ])

# ================= MOUSE EVENTS =================
def mouse_events(event, x, y, flags, param):
    global drawing, start_point, selected_zone_index

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, zone in enumerate(zones):
            x1, y1 = zone["start"]
            x2, y2 = zone["end"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_zone_index = i
                return

        drawing = True
        start_point = (x, y)
        selected_zone_index = None

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        end_point = (x, y)

        zone_name = f"Zone {len(zones)+1}"
        color = COLORS[len(zones) % len(COLORS)]

        zone = {
            "name": zone_name,
            "start": start_point,
            "end": end_point,
            "color": color
        }

        zones.append(zone)
        entry_count[zone_name] = 0
        exit_count[zone_name] = 0
        save_zones()

# ================= MAIN =================
load_zones()

cap = cv2.VideoCapture(0)
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Video", mouse_events)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO + ByteTrack
    results = model.track(frame, persist=True, verbose=False)

    active_ids = set()

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:

            if box.id is None:
                continue

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls != 0 or conf < 0.5:
                continue

            person_id = int(box.id[0])
            active_ids.add(person_id)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {person_id}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,255,0), 2)

            # ===== ENTRY / EXIT =====
            for zone in zones:
                zone_name = zone["name"]
                zx1, zy1 = zone["start"]
                zx2, zy2 = zone["end"]

                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:

                    line_x = (zx1 + zx2) // 2
                    current_side = "left" if cx < line_x else "right"

                    if person_id not in id_zone_side:
                        id_zone_side[person_id] = {}

                    if zone_name not in id_zone_side[person_id]:
                        id_zone_side[person_id][zone_name] = current_side
                    else:
                        previous_side = id_zone_side[person_id][zone_name]

                        if previous_side == "left" and current_side == "right":
                            entry_count[zone_name] += 1
                            log_to_csv(zone_name)

                        elif previous_side == "right" and current_side == "left":
                            exit_count[zone_name] += 1
                            log_to_csv(zone_name)

                        id_zone_side[person_id][zone_name] = current_side

    # ===== DRAW ZONES =====
    for i, zone in enumerate(zones):
        zx1, zy1 = zone["start"]
        zx2, zy2 = zone["end"]
        color = tuple(zone["color"])
        zone_name = zone["name"]

        thickness = 4 if i == selected_zone_index else 2

        cv2.rectangle(frame, (zx1,zy1), (zx2,zy2), color, thickness)
        line_y = (zy1 + zy2) // 2
        cv2.line(frame, (zx1, line_y), (zx2, line_y), (255,255,255), 2)
        cv2.putText(frame,
                    f"{zone_name} | In:{entry_count[zone_name]} | Out:{exit_count[zone_name]}",
                    (zx1, zy1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    # ===== DASHBOARD =====
    cv2.putText(frame, f"Active People: {len(active_ids)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,255,255), 2)

    # ===== INSTRUCTIONS =====
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
        cv2.putText(frame, text,
                    (10, frame.shape[0] - 140 + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,255), 1)

    cv2.imshow("Video", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('d') and zones:
        zones.pop()
        save_zones()

    if key == ord('x') and selected_zone_index is not None:
        zones.pop(selected_zone_index)
        selected_zone_index = None
        save_zones()

    if key == ord('r'):
        zones.clear()
        entry_count.clear()
        exit_count.clear()
        id_zone_side.clear()
        save_zones()

    if key == ord('p'):
        filename = f"screenshot_{datetime.now().strftime('%H%M%S')}.jpg"
        filepath = os.path.join(BASE_DIR, filename)
        cv2.imwrite(filepath, frame)

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

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()