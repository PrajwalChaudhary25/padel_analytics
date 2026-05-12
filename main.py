import cv2
import numpy as np
import pandas as pd
import json
from ultralytics import YOLO

# ── models ───────────────────────────────────────────
pose_model   = YOLO("models/yolo26n-pose.pt")   # players + keypoints
detect_model = YOLO("models/yolo26n.pt")        # ball + racket

# ── config ───────────────────────────────────────────
VIDEO_PATH   = "input/infernce_sample_video.mp4"
OUTPUT_VIDEO = "output/annotated_output.mp4"
OUTPUT_JSON  = "output/shots.json"
OUTPUT_CSV   = "output/shots.csv"

L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW,    R_ELBOW    = 7, 8
L_WRIST,    R_WRIST    = 9, 10

BALL_CLASS_ID   = 32
RACKET_CLASS_ID = 38

VELOCITY_THRESHOLD = 55
MAX_VELOCITY       = 200
COOLDOWN_FRAMES    = 15

SHOT_COLORS = {
    "forehand": (0, 255, 0),
    "backhand": (255, 165, 0),
    "smash"   : (0, 0, 255),
}

cap    = cv2.VideoCapture(VIDEO_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print(f"FPS: {fps}  |  Resolution: {width}x{height}  |  Total frames: {total}")
print("Processing video...\n")

# ── helper functions ──────────────────────────────────
def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def classify_shot(keypoints):
    ls = keypoints[L_SHOULDER]
    rs = keypoints[R_SHOULDER]
    re = keypoints[R_ELBOW]
    rw = keypoints[R_WRIST]
    body_midline = (ls[0] + rs[0]) / 2
    if rw[1] < rs[1] - 20:
        return "smash"
    elif rw[0] > body_midline:
        return "forehand"
    else:
        return "backhand"

# ── tracking variables ────────────────────────────────
shot_log        = []
prev_wrists     = {}
last_shot_frame = {}
last_shot_label = {}
frame_idx       = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── run both models ───────────────────────────────
    pose_results   = pose_model(frame, verbose=False)
    detect_results = detect_model(frame, verbose=False)

    # ── draw skeleton on all players ──────────────────
    annotated = pose_results[0].plot()

    # ── ball and racket detection ─────────────────────
    for box in detect_results[0].boxes:
        class_id   = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if class_id == BALL_CLASS_ID and confidence > 0.2:
            radius = max((x2 - x1), (y2 - y1)) // 2
            cv2.circle(annotated, (cx, cy), radius + 5, (0, 255, 255), 2)
            cv2.circle(annotated, (cx, cy), 3, (0, 255, 255), -1)
            cv2.putText(annotated, f"BALL {confidence:.2f}",
                        (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

        elif class_id == RACKET_CLASS_ID and confidence > 0.2:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(annotated, f"RACKET {confidence:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 255), 2)

    # ── shot classification ───────────────────────────
    keypoints_all = pose_results[0].keypoints.xy.cpu().numpy()

    for player_id, keypoints in enumerate(keypoints_all):
        rw = keypoints[R_WRIST]

        if rw[0] == 0 and rw[1] == 0:
            continue

        if player_id in prev_wrists:
            velocity = np.linalg.norm(rw - prev_wrists[player_id])
        else:
            velocity = 0.0
        prev_wrists[player_id] = rw.copy()

        frames_since_last = frame_idx - last_shot_frame.get(player_id, -999)

        if VELOCITY_THRESHOLD < velocity < MAX_VELOCITY and frames_since_last > COOLDOWN_FRAMES:
            shot      = classify_shot(keypoints)
            timestamp = round(frame_idx / fps, 2)

            shot_log.append({
                "frame"         : frame_idx,
                "timestamp"     : timestamp,
                "player_id"     : player_id,
                "shot_type"     : shot,
                "wrist_velocity": round(float(velocity), 2)
            })

            last_shot_frame[player_id] = frame_idx
            last_shot_label[player_id] = shot

            print(f"  Frame {frame_idx:>5} | Player {player_id} | {shot:<10} | velocity: {round(velocity,1)}")

        # show shot label for 20 frames after detection
        if player_id in last_shot_label:
            if frame_idx - last_shot_frame.get(player_id, 0) < 20:
                shot  = last_shot_label[player_id]
                color = SHOT_COLORS[shot]
                x, y  = int(rw[0]), int(rw[1])
                label = f"P{player_id}: {shot.upper()}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(annotated, (x - 5, y - th - 30), (x + tw + 5, y - 5), color, -1)
                cv2.putText(annotated, label,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 0), 2)

    # frame counter
    cv2.putText(annotated, f"Frame: {frame_idx}  |  Shots: {len(shot_log)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    out_video.write(annotated)
    frame_idx += 1

    if frame_idx % 100 == 0:
        print(f"--- {frame_idx}/{total} frames processed ---")

cap.release()
out_video.release()

# ── save outputs ──────────────────────────────────────
with open(OUTPUT_JSON, "w") as f:
    json.dump(shot_log, f, indent=2)

df = pd.DataFrame(shot_log)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'='*45}")
print(f"DONE — {frame_idx} frames processed")
print(f"Total shots detected: {len(shot_log)}")
if len(shot_log) > 0:
    print(f"\nShot breakdown:")
    print(df["shot_type"].value_counts().to_string())
    print(f"\nPer player:")
    print(df.groupby(["player_id","shot_type"]).size().to_string())
print(f"\nFiles saved:")
print(f"  {OUTPUT_VIDEO}")
print(f"  {OUTPUT_JSON}")
print(f"  {OUTPUT_CSV}")

