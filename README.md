# Crowd Counting using Video Analytics - Infosys Springboard Virtual Internship 6.0 
A computer vision-based project designed to monitor and manage crowd density using real-time video analytics. The system leverages the ultralytics YOLOv8 object detection model to track individuals, count entries/exits within customizable zones, and alert users when overcrowding limits are exceeded.

## Features

- **Real-Time Object Tracking**: Uses YOLOv8 for accurate, real-time person detection and tracking.
- **Dynamic Zones (ROI)**: Interactively draw, delete, and manage multiple Regions of Interest (zones) directly on the video feed.
- **Entry & Exit Counting**: Accurately counts how many people enter and exit defined zones.
- **Overcrowding Alerts**: Triggers visual warnings and saves alert snapshot images automatically if a zone exceeds the maximum allowed capacity.
- **Data Logging & Analytics**: Logs system events and exports crowd data to a CSV file.
- **Automated Reporting**: Generates textual and visual reports (bar charts and time-series plots) from the logged crowd data.
- **Dashboard Overlay**: Real-time Heads-Up Display showing active people counts, zone status, and interactive commands.

## System Requirements

- Python 3.8+
- OpenCV (`cv2`)
- Ultralytics YOLO (`ultralytics`)
- Pandas (`pandas`)
- Matplotlib (`matplotlib`)

## Project Structure

The project was developed in multiple milestones, culminating in the fully-featured `milestone_4`.

- `milestone_4/m4.py`: The main video analytics and tracking script.
- `milestone_4/crowd_report.py`: Script to generate analytical reports from collected data.
- `documentation/`: Contains the project presentation and reports.
- `yolov8n.pt`: YOLOv8 nano model (downloaded automatically if missing).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SulfaSaji/CrowdCount.git
   cd CrowdCount
   ```

2. Install the required dependencies:
   ```bash
   pip install opencv-python ultralytics pandas matplotlib
   ```

## Usage

### 1. Running the Real-Time Tracker

Navigate to the `milestone_4` directory and run the main application:

```bash
cd milestone_4
python m4.py
```

#### Interactive Controls:
- **Click & Drag**: Draw a new monitoring zone.
- **D**: Delete the last created zone.
- **X**: Delete the currently selected zone (click inside a zone to select it).
- **R**: Reset/clear all zones.
- **P**: Save a screenshot of the current frame.
- **F**: Toggle Fullscreen mode.
- **Q**: Quit the application.

### 2. Generating Reports

After running the tracker, data is saved inside the `milestone_4/data/` folder. To generate insightful visual reports, run:

```bash
python crowd_report.py
```

This will output:
- Textual summaries (Total Visitors, Peak Crowd, Most Crowded Zone) in `milestone_4/reports/`.
- Graphical plots (Crowd Trends, Zone Usage Comparison) in `milestone_4/graphs/`.

## Logs and Data

- **System Logs**: All system events and alerts are logged in `milestone_4/system_log.txt`.
- **Alert Snapshots**: Saved automatically to `milestone_4/alert/` during overcrowding events.
- **Crowd Metrics**: Real-time traffic is appended to `milestone_4/data/crowd_data.csv`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
