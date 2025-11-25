import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize face detector (MTCNN) and embedding model (FaceNet)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print(torch.cuda.is_available())