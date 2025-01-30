# import socket
# import keyboard
# import struct
# import numpy as np
# import cv2
# import torch
# import time
# import queue
# from threading import Thread
# import pathlib
# import sys
# import os
# sys.path.append(os.path.dirname(__file__))

# from src.summon_logic.audio_processing import process_voice_command, text_to_speech, process_file_command
# from src.functionalities.execution import match_command
# from src.utils.object_detection_utils import post_process, preprocess_image, draw_detection, draw_tracking, ensure_rgb
# from src.functionalities.functionalities_definition import ai_image_analysis_and_summary
# from deep_sort_realtime.deepsort_tracker import DeepSort


# frame_for_summary = None
# summary_generated = None
# detected_item = None
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
# from ultralytics import YOLO
# model = YOLO('best_tools.pt')

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# import onnxruntime as ort

# # Initialize models
# model = YOLO('best_tools.pt')  # YOLO model
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # MiDaS ONNX model
# depth_session = ort.InferenceSession("dpt_large_384.pt", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# # DeepSort tracker
# deep_sort_tracker = DeepSort(max_age=35, n_init=3, nn_budget=100)

# # Global tracker and depth settings
# MOVEMENT_THRESHOLD = 10
# STATIONARY_THRESHOLD = 5
# object_movement_data = {}

# def preprocess_depth(image, size=(256, 256)):
#     """Prepare the input image for MiDaS."""
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
#     img = cv2.resize(img, size)
#     img = np.transpose(img, (2, 0, 1))[np.newaxis, :].astype(np.float32)
#     return img

# def get_depth_map(frame):
#     """Generate a depth map using MiDaS."""
#     input_image = preprocess_depth(frame)
#     depth_map = depth_session.run(None, {"input": input_image})[0]
#     depth_map = cv2.resize(depth_map.squeeze(), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
#     return depth_map

# def draw_detection(frame, bbox, conf, class_id, depth_value=None):
#     """Draw bounding boxes and optionally annotate depth."""
#     x1, y1, x2, y2 = bbox
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     label = f"Class {class_id}: {conf:.2f}"
#     if depth_value is not None:
#         label += f" | Depth: {depth_value:.2f}m"
#     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#     return frame

# def draw_tracking(frame, track, depth_value=None):
#     """Draw tracking bounding boxes and depth information."""
#     x1, y1, w, h = map(int, track.to_tlwh())
#     x2, y2 = x1 + w, y1 + h
#     track_id = track.track_id
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     label = f"ID {track_id}"
#     if depth_value is not None:
#         label += f" | Depth: {depth_value:.2f}m"
#     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#     return frame

# class FrameProcessor(Thread):
#     def __init__(self, frame_queue):
#         super().__init__()
#         self.frame_queue = frame_queue
#         self.running = True

#     def run(self):
#         while self.running:
#             try:
#                 frame_data = self.frame_queue.get(timeout=1.0)
#                 if frame_data is None:
#                     continue

#                 frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
#                 if frame is None:
#                     continue

#                 # Depth estimation
#                 depth_map = get_depth_map(frame)

#                 # Object detection
#                 results = model(frame)
#                 detections = results[0].boxes.data  # YOLOv8 outputs
#                 detections = detections[detections[:, 4] > 0.45]  # Filter confidence

#                 # Tracker input preparation
#                 tracker_inputs = []
#                 for *xyxy, conf, class_id in detections.cpu().numpy():
#                     x1, y1, x2, y2 = map(int, xyxy)
#                     depth_value = np.mean(depth_map[y1:y2, x1:x2]) if y2 > y1 and x2 > x1 else 0
#                     tracker_inputs.append([[x1, y1, x2 - x1, y2 - y1], float(conf), str(int(class_id))])
#                     frame = draw_detection(frame, (x1, y1, x2, y2), conf, class_id, depth_value)

#                 # Tracking
#                 tracks = deep_sort_tracker.update_tracks(tracker_inputs, frame=frame)
#                 for track in tracks:
#                     if not track.is_confirmed() or track.time_since_update > 1:
#                         continue
#                     x, y, w, h = map(int, track.to_tlwh())
#                     depth_value = np.mean(depth_map[y:y + h, x:x + w]) if h > 0 and w > 0 else 0
#                     frame = draw_tracking(frame, track, depth_value)

#                 # Display frame
#                 cv2.imshow("Object Detection and Tracking", frame)
#                 if cv2.waitKey(10) == ord('q'):
#                     self.running = False
#                     break

#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 print(f"Error in frame processing: {e}")

#         cv2.destroyAllWindows()

# def start_server(host='0.0.0.0', port=12345):
#     """Socket server to receive frames."""
#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     server_socket.bind((host, port))
#     server_socket.listen(1)

#     frame_queue = queue.Queue(maxsize=5)
#     processor = FrameProcessor(frame_queue)
#     processor.start()

#     try:
#         while True:
#             client_socket, addr = server_socket.accept()
#             try:
#                 if client_socket.recv(6) != b'START\n':
#                     continue
#                 while True:
#                     size_data = client_socket.recv(4)
#                     if not size_data or len(size_data) != 4:
#                         break
#                     frame_size = struct.unpack('>I', size_data)[0]
#                     frame_data = bytearray()
#                     while frame_size > 0:
#                         chunk = client_socket.recv(min(65536, frame_size))
#                         if not chunk:
#                             break
#                         frame_data.extend(chunk)
#                         frame_size -= len(chunk)
#                     if len(frame_data) == struct.unpack('>I', size_data)[0]:
#                         try:
#                             frame_queue.put_nowait(bytes(frame_data))
#                         except queue.Full:
#                             pass
#             finally:
#                 client_socket.close()

#     except KeyboardInterrupt:
#         print("Shutting down...")
#     finally:
#         processor.running = False
#         processor.join()
#         server_socket.close()

# if __name__ == "__main__":
#     start_server()



