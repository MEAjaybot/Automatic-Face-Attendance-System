import cv2
import torch
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on:", device)

# Load models
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load student CSV
csv_path = "Students.csv"
df = pd.read_csv(csv_path)

# Extract face embedding function
def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image missing:", image_path)
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)

    if boxes is None:
        print("No face found in:", image_path)
        return None

    x1, y1, x2, y2 = boxes[0]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = int(x2), int(y2)

    face = img_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None

    face = torch.tensor(face.transpose(2, 0, 1) / 255.,
                        dtype=torch.float32).unsqueeze(0).to(device)

    return resnet(face).detach()

# Encode all known faces
known_embeddings = {}

for index, row in df.iterrows():
    name = row["name"]
    img_path = row["image_path"]
    embedding = get_embedding(img_path)

    if embedding is not None:
        known_embeddings[name] = embedding
        print(f"[OK] Loaded: {name}")
    else:
        print(f"[ERROR] Face not found for {name}")

# Create daily attendance CSV
def create_attendance_csv():
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance_{today}.csv"

    if not os.path.exists(filename):
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Date", "Time", "Status"])

    return filename

# Mark attendance (only once)
marked_names = set()

def mark_attendance(name, csv_file):
    if name in marked_names:
        return

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, date, time, "Present"])

    marked_names.add(name)
    print(f"[MARKED] {name} at {time}")

# Start camera
cap = cv2.VideoCapture(0)
csv_file = create_attendance_csv()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(frame_rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [max(0, int(b)) for b in box]
            face = frame_rgb[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_tensor = torch.tensor(face.transpose(2, 0, 1) / 255.,
                                       dtype=torch.float32).unsqueeze(0).to(device)

            embedding = resnet(face_tensor).detach()

            # Compare
            min_dist = 1.0     # BEST THRESHOLD FOR FACENET
            name = "Unknown"

            for k, v in known_embeddings.items():
                dist = (embedding - v).norm().item()
                if dist < min_dist:
                    name = k
                    min_dist = dist

            # Draw UI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if name != "Unknown":
                mark_attendance(name, csv_file)

    cv2.imshow("Face Attendance (Deep Learning)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
