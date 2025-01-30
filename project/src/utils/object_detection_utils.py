
import socket
import keyboard
import struct
import numpy as np
import cv2
import torch
import time
import queue
from threading import Thread
import pathlib
import sys
import os
import torchvision.transforms as T







def ensure_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image

def preprocess_image(image, target_size=640):
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((target_size, target_size, 3), dtype=np.float32)
    top, left = (target_size - new_h) // 2, (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized_image / 255.0
    return canvas,scale, top, left


def preprocess_image_torch(image, target_size=640, device='cpu'):

    h, w = image.shape[:2]


    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 

    transform = T.Compose([
        T.Resize((new_h, new_w)),  
    ])
    image_tensor = transform(image_tensor)

    canvas = torch.zeros((3, target_size, target_size), dtype=torch.float32)
    top, left = (target_size - new_h) // 2, (target_size - new_w) // 2
    canvas[:, top:top + new_h, left:left + new_w] = image_tensor

    canvas = canvas.unsqueeze(0).to(device)

    return canvas, top, left


def post_process(predictions, scale, top, left, orig_width, orig_height):
    predictions = predictions[predictions[:, 4] > 0.7]
    boxes = predictions[:, :4].cpu().numpy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    return boxes.astype(int).tolist()

def draw_detection(frame, bbox, conf, class_id):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"Detected {class_id}: {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

def draw_tracking(frame, track, movement_detected=False):
    x1, y1, w, h = map(int, track.to_tlwh())
    x2, y2 = x1 + w, y1 + h
    track_id = track.track_id
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1 - 30), (x2, y1), (0, 255, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.putText(frame, f"Track ID: {track_id}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display "Movement Detected" message if movement is detected
    if movement_detected:
        cv2.putText(frame, "Movement Detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return frame
