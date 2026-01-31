import cv2 as cv
from datetime import datetime
import logging
import csv
import os
import json
import time

import log_config

import face_prescence_module
import head_pose_module
import eye_gaze_module


# -----------------------------
# FILES FOR STREAMLIT CONTROL
# -----------------------------
CONTROL_FILE = "control.json"
STATUS_FILE = "status.json"
SUMMARY_FILE = "summary.json"
DASHBOARD_CSV = "dashboard.csv"


def write_status(state: str, message: str = ""):
    payload = {
        "state": state,
        "message": message,
        "ts": time.time()
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def read_command():
    if not os.path.exists(CONTROL_FILE):
        return None
    try:
        with open(CONTROL_FILE, "r") as f:
            return json.load(f).get("command")
    except Exception:
        return None


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h {m}m {s}s"


# -----------------------------
# LOGS
# -----------------------------
debug_log = log_config.setup_logger("microscope.log", logging.INFO)

# -----------------------------
# SESSION FLAGS
# -----------------------------
session_started = False
session_ended = False

session_start = None
session_end = None

# -----------------------------
# TIME COUNTERS
# -----------------------------
attentive_seconds = 0.0
distracted_seconds = 0.0
away_seconds = 0.0

last_frame_time = None

# -----------------------------
# CSV LOGGING SETUP (created only when session starts)
# -----------------------------
csv_file = None
csv_writer = None
last_csv_write_time = None


def start_csv():
    global csv_file, csv_writer, last_csv_write_time

    csv_exists = os.path.exists(DASHBOARD_CSV)
    csv_file = open(DASHBOARD_CSV, mode="a", newline="")
    csv_writer = csv.writer(csv_file)

    if not csv_exists:
        csv_writer.writerow([
            "timestamp",
            "presence_label",
            "head_pose_label",
            "eye_gaze_label",
            "final_state",
            "attentive_seconds",
            "distracted_seconds",
            "away_seconds"
        ])

    last_csv_write_time = None


def close_csv():
    global csv_file
    if csv_file:
        csv_file.close()
        csv_file = None


def write_summary():
    global session_start, session_end
    total_seconds = attentive_seconds + distracted_seconds + away_seconds
    focus_percent = (attentive_seconds / total_seconds * 100) if total_seconds > 0 else 0.0

    summary = {
        "session_start": session_start.strftime("%Y-%m-%d %H:%M:%S") if session_start else "N/A",
        "session_end": session_end.strftime("%Y-%m-%d %H:%M:%S") if session_end else "N/A",
        "total_seconds": int(total_seconds),
        "attentive_seconds": int(attentive_seconds),
        "distracted_seconds": int(distracted_seconds),
        "away_seconds": int(away_seconds),
        "focus_percent": round(focus_percent, 2),
        "csv_path": DASHBOARD_CSV
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== FocusOS SESSION SUMMARY ==========")
    print(f"Session Start : {summary['session_start']}")
    print(f"Session End   : {summary['session_end']}")
    print(f"Total Time    : {format_time(total_seconds)}")

    print("\n--- Breakdown ---")
    print(f"✅ Attentive   : {format_time(attentive_seconds)}")
    print(f"❌ Distracted  : {format_time(distracted_seconds)}")
    print(f"🚫 Away        : {format_time(away_seconds)}")

    print("\n--- Score ---")
    print(f"🎯 Focus %     : {focus_percent:.1f}%")

    print("\nDashboard saved to: dashboard.csv")
    print("============================================\n")


# -----------------------------
# CAMERA INIT
# -----------------------------
capture = cv.VideoCapture(0)

write_status("IDLE", "Waiting for Streamlit commands...")

last_cmd = None

while True:
    cmd = read_command()

    # Only react when command changes
    if cmd is not None and cmd != last_cmd:
        last_cmd = cmd

        # START SESSION
        if cmd == "START_SESSION":
            if not session_started:
                session_started = True
                session_start = datetime.now()

                attentive_seconds = 0.0
                distracted_seconds = 0.0
                away_seconds = 0.0
                last_frame_time = None

                start_csv()
                write_status("RUNNING", "Session started ✅")
                debug_log.info("Session started from Streamlit.")

            else:
                write_status("RUNNING", "Session already running 💀")

        # CALIBRATE
        elif cmd == "CALIBRATE":
            if session_started and not session_ended:
                write_status("CALIBRATING", "Calibration requested 🎯")
                debug_log.info("Calibration requested from Streamlit.")
                # You already use key='c' in modules, so we simulate that:
                # We'll send key=ord('c') for one cycle
            else:
                write_status("IDLE", "Cannot calibrate. Start session first.")

        # END SESSION
        elif cmd == "END_SESSION":
            if session_started and not session_ended:
                session_ended = True
                session_end = datetime.now()
                write_status("ENDED", "Session ended. Writing summary... 🏁")
                debug_log.info("Session ended from Streamlit.")
                break
            else:
                write_status("IDLE", "No session running to end.")

    # If session hasn't started, just chill and keep window alive
    if not session_started:
        time.sleep(0.1)
        continue

    # -----------------------------
    # READ FRAME
    # -----------------------------
    ret, frame = capture.read()
    now = datetime.now()

    if not ret:
        write_status("ERROR", "Camera read failed.")
        break

    # Fake key input for calibration
    key = 0
    if last_cmd == "CALIBRATE":
        key = ord("c")  # one-shot calibration trigger
        # reset so it doesn't keep calibrating forever
        last_cmd = "RUNNING_AFTER_CALIBRATE"
        write_status("RUNNING", "Calibration triggered ✅")

    flipped_frame = cv.flip(frame, 1)

    # -----------------------------
    # DEFAULT STATES
    # -----------------------------
    presence_label = "AWAY"
    head_pose_label = "NO_FACE"
    eye_gaze_label = "NOT_CALIBRATED"
    final_state = "AWAY"

    # -----------------------------
    # 1) FACE PRESENCE
    # -----------------------------
    presence_label = face_prescence_module.update(frame, now)

    if presence_label == "PRESENT":
        # -----------------------------
        # 2) HEAD POSE
        # -----------------------------
        head_pose_label = head_pose_module.update(frame, now, key)

        if "ATTENTIVE" in head_pose_label:
            # -----------------------------
            # 3) EYE GAZE
            # -----------------------------
            eye_gaze_label = eye_gaze_module.update(frame, key, now)

            if "attentive" in eye_gaze_label.lower():
                final_state = "ATTENTIVE"
            else:
                final_state = "DISTRACTED"
        else:
            final_state = "DISTRACTED"
    else:
        final_state = "AWAY"

    # -----------------------------
    # TIME COUNTING
    # -----------------------------
    if last_frame_time is None:
        last_frame_time = now

    dt = (now - last_frame_time).total_seconds()
    last_frame_time = now

    if final_state == "ATTENTIVE":
        attentive_seconds += dt
    elif final_state == "DISTRACTED":
        distracted_seconds += dt
    else:
        away_seconds += dt

    # -----------------------------
    # DEBUG LOG
    # -----------------------------
    debug_log.info(
        f"[{final_state}] presence={presence_label} head={head_pose_label} eyes={eye_gaze_label} "
        f"t_att={attentive_seconds:.1f}s t_dis={distracted_seconds:.1f}s t_away={away_seconds:.1f}s"
    )

    # -----------------------------
    # CSV LOGGING (1 row/sec)
    # -----------------------------
    if csv_writer is not None:
        if last_csv_write_time is None:
            last_csv_write_time = now

        elapsed = (now - last_csv_write_time).total_seconds()

        if elapsed >= 1.0:
            csv_writer.writerow([
                now.strftime("%Y-%m-%d %H:%M:%S"),
                presence_label,
                head_pose_label,
                eye_gaze_label,
                final_state,
                round(attentive_seconds, 2),
                round(distracted_seconds, 2),
                round(away_seconds, 2)
            ])
            csv_file.flush()
            last_csv_write_time = now

    # -----------------------------
    # OPENCV UI (optional)
    # -----------------------------
    cv.putText(flipped_frame, f"Presence: {presence_label}", (20, 40),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv.putText(flipped_frame, f"Head: {head_pose_label}", (20, 70),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv.putText(flipped_frame, f"Eyes: {eye_gaze_label}", (20, 100),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv.putText(flipped_frame, f"FINAL: {final_state}", (20, 140),
               cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv.putText(flipped_frame, f"Attentive: {attentive_seconds:.1f}s", (20, 180),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv.putText(flipped_frame, f"Distracted: {distracted_seconds:.1f}s", (20, 210),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv.putText(flipped_frame, f"Away: {away_seconds:.1f}s", (20, 240),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv.putText(flipped_frame, "Controlled by Streamlit (Start/Calibrate/End)", (20, 280),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv.imshow("FocusOS V1", flipped_frame)

    # Optional emergency quit (still keep this because you're clumsy)
    if cv.waitKey(1) & 0xFF == ord("q"):
        write_status("ENDED", "Ended from OpenCV window (q).")
        session_ended = True
        session_end = datetime.now()
        break


# -----------------------------
# CLEANUP
# -----------------------------
capture.release()
close_csv()

if session_start is None:
    session_start = datetime.now()
if session_end is None:
    session_end = datetime.now()

write_summary()
cv.destroyAllWindows()
write_status("DONE", "Summary written ✅")
