
import os
import csv
import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


_mtcnn = MTCNN(keep_all=False, device=device)
_resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def ensure_dirs():
    os.makedirs("images", exist_ok=True)
    os.makedirs("encodings", exist_ok=True)
    os.makedirs("attendance", exist_ok=True)

def student_csv_path():
    return "Students.csv"

def load_students():
    path = student_csv_path()
    if not os.path.exists(path):
        
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "image_path"])
        return []
    import pandas as pd
    return pd.read_csv(path).to_dict(orient="records")

def add_student_row(student_id, name, image_path):
    path = student_csv_path()
    
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([student_id, name, image_path])

def get_face_embedding_from_bgr(bgr_img):

    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    boxes, _ = _mtcnn.detect(rgb)
    if boxes is None:
        return None
    x1, y1, x2, y2 = [max(0, int(v)) for v in boxes[0]]
    face = rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face_t = torch.tensor(face.transpose(2,0,1)/255., dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = _resnet(face_t)
    return emb.detach().cpu().numpy()

def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    return get_face_embedding_from_bgr(img)

def save_embedding(student_id, embedding):
    path = os.path.join("encodings", f"{student_id}.npy")
    np.save(path, embedding)

def load_all_embeddings():
    embeddings = {}
    for fn in os.listdir("encodings"):
        if fn.endswith(".npy"):
            sid = os.path.splitext(fn)[0]
            emb = np.load(os.path.join("encodings", fn))
            embeddings[sid] = emb
    return embeddings

def today_attendance_filename():
    today = datetime.now().strftime("%Y-%m-%d")
    name = f"attendance/attendance_{today}.csv"
    if not os.path.exists(name):
        with open(name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "date", "time", "status"])
    return name

def mark_attendance_once(student_id, name):
    fn = today_attendance_filename()
    
    existing = set()
    import pandas as pd
    try:
        df = pd.read_csv(fn)
        existing = set(df["id"].astype(str).tolist())
    except Exception:
        existing = set()
    sid = str(student_id)
    if sid in existing:
        return False
    now = datetime.now()
    with open(fn, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([student_id, name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), "Present"])
    return True
