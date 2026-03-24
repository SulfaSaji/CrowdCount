import pandas as pd
import matplotlib.pyplot as plt
import os

# ================= BASE PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= FILE PATH =================
DATA_FILE = os.path.join(BASE_DIR, "data", "crowd_data.csv")

# ================= CREATE FOLDERS =================
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "graphs"), exist_ok=True)

# ================= CHECK DATA FILE =================
if not os.path.exists(DATA_FILE):
    print("No crowd data found. Run the monitoring system first.")
    exit()

# ================= LOAD DATA =================
data = pd.read_csv(DATA_FILE)

data.columns = ["timestamp", "zone", "entry", "exit", "current"]

data["timestamp"] = pd.to_datetime(data["timestamp"])

print("\n===== CROWD REPORT =====\n")

# ================= TOTAL VISITORS =================
zone_entries = data.groupby("zone")["entry"].max()

total_visitors = zone_entries.sum()

print("Total Visitors:", total_visitors)

# ================= PEAK CROWD =================
peak_row = data.loc[data["current"].idxmax()]

peak_time = peak_row["timestamp"]
peak_count = peak_row["current"]

print("Peak Crowd Time:", peak_time)
print("Peak Crowd Count:", peak_count)

# ================= MOST CROWDED ZONE =================
most_crowded_zone = zone_entries.idxmax()

print("Most Crowded Zone:", most_crowded_zone)

# ================= SAVE REPORT =================
report_path = os.path.join(BASE_DIR, "reports", "crowd_report.txt")

with open(report_path, "w") as f:

    f.write("===== CROWD REPORT =====\n\n")

    f.write(f"Total Visitors: {total_visitors}\n")
    f.write(f"Peak Crowd Time: {peak_time}\n")
    f.write(f"Peak Crowd Count: {peak_count}\n")
    f.write(f"Most Crowded Zone: {most_crowded_zone}\n")

print("\nReport saved to:", report_path)

# ================= GRAPH 1 : CROWD TREND =================
plt.figure()

plt.plot(data["timestamp"], data["current"])

plt.xlabel("Time")
plt.ylabel("People Count")
plt.title("Crowd Trend Over Time")

plt.xticks(rotation=45)

plt.tight_layout()

graph1 = os.path.join(BASE_DIR, "graphs", "crowd_trend.png")

plt.savefig(graph1)

print("Graph saved:", graph1)

# ================= GRAPH 2 : ZONE COMPARISON =================
plt.figure()

zone_entries.plot(kind="bar")

plt.xlabel("Zone")
plt.ylabel("Visitors")
plt.title("Zone Usage Comparison")

plt.tight_layout()

graph2 = os.path.join(BASE_DIR, "graphs", "zone_usage.png")

plt.savefig(graph2)

print("Graph saved:", graph2)

print("\nAll results stored successfully.")