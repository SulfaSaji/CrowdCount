import cv2
import json
import os
import csv
from datetime import datetime
import atexit
from ultralytics import YOLO

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ZONE_FILE = os.path.join(BASE_DIR, "zone4.json")
DATA_FILE = os.path.join(BASE_DIR, "data", "crowd_data.csv")
LOG_FILE = os.path.join(BASE_DIR, "system_log.txt")
ALERT_FOLDER = os.path.join(BASE_DIR, "alert")

os.makedirs(ALERT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
# ================= YOLO =================
model = YOLO(os.path.join(BASE_DIR, "..", "yolov8n.pt"))

MAX_PEOPLE = 8  # overcrowding limit

zones = []
drawing = False
start_point = None
selected_zone_index = None
fullscreen = False

entry_count = {}
exit_count = {}
current_count = {}
id_zone_side = {}

COLORS = [
    (0,255,0),
    (255,0,0),
    (0,0,255),
    (255,255,0),
    (255,0,255),
    (0,255,255)
]

# ================= SYSTEM LOG =================
def log_event(text):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} : {text}\n")

log_event("System Started")

# ================= SAVE / LOAD =================
def save_zones():
    with open(ZONE_FILE, "w") as f:
        json.dump(zones, f, indent=4)

def load_zones():
    global zones

    if os.path.exists(ZONE_FILE):
        with open(ZONE_FILE, "r") as f:
            zones = json.load(f)

    for zone in zones:
        name = zone["name"]
        entry_count[name] = 0
        exit_count[name] = 0
        current_count[name] = 0

atexit.register(save_zones)

# ================= DATA STORAGE =================
def store_crowd_data(zone_name):

    with open(DATA_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            zone_name,
            entry_count[zone_name],
            exit_count[zone_name],
            current_count[zone_name]
        ])

# ================= ALERT =================
def trigger_alert(frame, zone_name):

    log_event(f"ALERT Triggered in {zone_name}")

    filename = f"alert_{zone_name}_{datetime.now().strftime('%H%M%S')}.jpg"
    filepath = os.path.join(ALERT_FOLDER, filename)

    cv2.imwrite(filepath, frame)

# ================= MOUSE =================
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
        current_count[zone_name] = 0

        save_zones()

        log_event(f"{zone_name} Created")

# ================= MAIN =================
load_zones()

cap = cv2.VideoCapture(os.path.join(BASE_DIR, "demo.mp4"))

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Video", mouse_events)

log_event("Camera Started")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False)

    active_ids = set()

    zone_people = {z["name"]:0 for z in zones}

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

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            cx = (x1+x2)//2
            cy = (y1+y2)//2

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,165,255),2)

            cv2.putText(frame,f"ID {person_id}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

            # ===== ZONE CHECK =====
            for zone in zones:

                name = zone["name"]

                zx1,zy1 = zone["start"]
                zx2,zy2 = zone["end"]

                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:

                    zone_people[name] += 1

                    line_y = (zy1 + zy2) // 2

                    current_side = "top" if cy < line_y else "bottom"

                    if person_id not in id_zone_side:
                        id_zone_side[person_id] = {}

                    if name not in id_zone_side[person_id]:

                        id_zone_side[person_id][name] = current_side

                    else:

                        prev = id_zone_side[person_id][name]

                        if prev == "top" and current_side == "bottom":

                            entry_count[name] += 1
                            current_count[name] += 1
                            store_crowd_data(name)

                        elif prev == "bottom" and current_side == "top":

                            if current_count[name] > 0:

                                exit_count[name] += 1
                                current_count[name] -= 1
                                store_crowd_data(name)

                        id_zone_side[person_id][name] = current_side

    # ===== DRAW ZONES =====
    for i,zone in enumerate(zones):

        zx1,zy1 = zone["start"]
        zx2,zy2 = zone["end"]

        color = tuple(zone["color"])
        name = zone["name"]

        cv2.rectangle(frame,(zx1,zy1),(zx2,zy2),color,2)

        line_y = (zy1+zy2)//2

        cv2.line(frame,(zx1,line_y),(zx2,line_y),(255,255,255),2)

        current = zone_people[name]

        cv2.putText(frame,
                    f"{name} | In:{entry_count[name]} Out:{exit_count[name]} Now:{current}",
                    (zx1,zy1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,color,2)

        # ===== ALERT =====
        if current > MAX_PEOPLE:

            alert_text = f"OVERCROWDING IN {name}"

            (text_w, text_h), _ = cv2.getTextSize(alert_text,
                                                  cv2.FONT_HERSHEY_SIMPLEX,1,3)

            x = (frame.shape[1] - text_w) // 2

            cv2.putText(frame,
                        alert_text,
                        (x, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)

            trigger_alert(frame,name)

    # ===== DASHBOARD =====
    cv2.putText(frame,
                f"Active People: {len(active_ids)}",
                (20,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(0,255,255),2)

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

# ✅ OUTSIDE LOOP
log_event("System Shutdown")

cap.release()
cv2.destroyAllWindows()