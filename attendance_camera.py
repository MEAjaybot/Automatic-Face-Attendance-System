
import cv2
import os
import numpy as np
from utils import (
    ensure_dirs, 
    get_face_embedding_from_bgr,
    load_all_embeddings, 
    mark_attendance_once
)
import pandas as pd

ensure_dirs()
print("Loading embeddings...")

known = load_all_embeddings()
print(f"Loaded {len(known)} embeddings.")


THRESH = 1.15   

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

print("Starting camera. Press 'q' to quit.")

# Load student data 
students_df = pd.read_csv("Students.csv")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    emb = get_face_embedding_from_bgr(frame)

    name = "Unknown"
    sid = None

    if emb is not None and len(known) > 0:

        # Find  embedding
        min_d = 999
        for k, v in known.items():
            d = np.linalg.norm(emb - v)
            if d < min_d:
                min_d = d
                sid = k

       
        if min_d < THRESH:
            
            row = students_df[students_df["student_id"].astype(str) == str(sid)]
            if not row.empty:
                name = row.iloc[0]["name"]

                #attendance
                marked = mark_attendance_once(sid, name)
                status_text = "Present" if marked else "Already marked"

                cv2.putText(frame, f"{name} ({status_text})",
                            (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2)

        else:
            name = "Unknown"

    
    cv2.putText(frame, name, (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Attendance Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
