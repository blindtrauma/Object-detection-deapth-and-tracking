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
sys.path.append(os.path.dirname(__file__))
from src.summon_logic.audio_processing import process_voice_command, text_to_speech, process_file_command
from src.functionalities.execution import match_command
from src.utils.object_detection_utils import post_process, preprocess_image, draw_detection, draw_tracking, ensure_rgb
from src.functionalities.functionalities_definition import ai_image_analysis_and_summary
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.transforms import Compose, Resize, ToTensor
import torch.hub
from PIL import Image
import numpy as np
import pywavefront



def save_frame_as_image(frame, output_dir='output_images1', filename="image.jpg"):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    pil_image.save(file_path, format="JPEG")
    print(file_path)

    return file_path

def load_obj_file(file_path):

    scene = pywavefront.Wavefront(file_path, collect_faces=True)
    vertices = np.array(scene.vertices)
    faces = np.array(scene.mesh_list[0].faces)
    return vertices, faces

def project_3d_to_2d(vertices, projection_matrix, transformation_matrix):

    vertices = np.dot(vertices, transformation_matrix.T) 
    vertices = np.dot(vertices, projection_matrix.T)    
    vertices /= vertices[:, -1:]                     
    return vertices[:, :2].astype(int)

def draw_obj_on_frame(frame, vertices_2d, faces):
    """
    Draw the projected 2D model on the frame.
    """
    for face in faces:
        points = vertices_2d[face]
        cv2.fillPoly(frame, [points], (0, 255, 0))  # Fill with green





def load_obj(filepath):
    vertices = []
    faces = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex positions
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
            elif line.startswith('f '):  # Faces
                parts = line.strip().split()[1:]
                face = [int(p.split('/')[0]) - 1 for p in parts]  # Only vertex indices
                faces.append(face)
    return np.array(vertices), faces
    



def draw_3d_overlay(frame, projected_vertices, faces=None, color=(0, 255, 0)):

    if faces:
        for face in faces:
            pts = np.array([projected_vertices[vertex_idx - 1] for vertex_idx in face], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)
            cv2.fillPoly(frame, [pts], color=color)
    else:
        # If no faces are provided, draw vertices as points
        for vertex in projected_vertices:
            x, y = int(vertex[0]), int(vertex[1])
            cv2.circle(frame, (x, y), radius=2, color=color, thickness=-1)





def transform_and_project(vertices, transformation_matrix, projection_matrix):

    vertices_homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])

    transformed_vertices = vertices_homogeneous @ transformation_matrix.T  

    projected_vertices = transformed_vertices @ projection_matrix.T 
    projected_2d = projected_vertices[:, :2] / projected_vertices[:, 2:3] 
    return projected_2d





def transform_and_project_exact(vertices, bbox, depth):
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2


    perspective_scale = 1 / (depth + 1e-6)  
    size_muliplier = 10
    scale = min(bbox_width, bbox_height) * perspective_scale*10

    tx, ty = box_center_x, box_center_y
    tz = depth  


    transformation_matrix = np.array([
        [scale, 0,     0,     tx],
        [0,     scale, 0,     ty],
        [0,     0,     scale, tz],
        [0,     0,     0,     1 ]
    ])

    transformed_vertices = []
    for vertex in vertices:
        homogenous_vertex = np.append(vertex, 1)  
        transformed_vertex = transformation_matrix @ homogenous_vertex
        transformed_vertex /= transformed_vertex[3]  
        transformed_vertices.append(transformed_vertex[:2]) 
    return np.array(transformed_vertices)



def overlay_model(frame, vertices, faces, color=(0, 255, 0), thickness=1):
    for face in faces:
        for i in range(len(face)):
            start = tuple(vertices[face[i - 1]].astype(int))  # Previous vertex
            end = tuple(vertices[face[i]].astype(int))  # Current vertex
            cv2.line(frame, start, end, color, thickness)




frame_for_summary = None
summary_generated = None
detected_item = None
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from ultralytics import YOLO
'''add path to your desired model'''
model = YOLO(r'') 

# Load MiDaS model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

''' ADD MIDAS MODEL PATH E.G.  dpt_large_384.pt'''
midas_model_path = "" 
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large") 
midas.load_state_dict(torch.load(midas_model_path)) 
midas.to(device)
midas.eval()

# MiDaS transforms
midas_transforms = Compose([
    Resize((384, 384)),  
    ToTensor()
])


def estimate_depth(image):
    # Ensure the image is in the right format for the MiDaS model
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    input_img = midas_transforms(image).unsqueeze(0).to(device)


    with torch.no_grad():
        depth = midas(input_img)

    # Convert depth map to numpy array
    depth_np = depth.squeeze().cpu().numpy()
    return depth_np

def get_depth_for_object(depth_map, bbox):
    x1, y1, x2, y2 = bbox
    # Take the depth at the center of the bounding box
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    depth_value = depth_map[center_y, center_x]
    return depth_value

def get_transformation_matrix(x1,y1,x2,y2, depth):

    width, height = x2 - x1, y2 - y1
    scale = depth / 1000.0  # Normalize depth
    translation = np.array([x1 + width / 2, y1 + height / 2, depth])  # Center the object
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] *= scale  # Scale
    transformation_matrix[:3, 3] = translation  # Translate
    return transformation_matrix



print(device)
# model = torch.hub.load('ultralytics/yolov8', 'custom', path='best_tools.pt').to(device)
model(torch.zeros(1, 3, 640, 640).to(device))
deep_sort_tracker = DeepSort(max_age=35, n_init=3, nn_budget=100)

STATIONARY_THRESHOLD = 5  
MOVEMENT_THRESHOLD = 10  
object_movement_data = {}  

def terminal_task(frame):
    global summary_generated
    if frame is None:
        print("Frame is None in terminal_task")
        return
    texttt = process_voice_command()
    # print(textt)
    # if textt is None:
    # texttt = process_file_command(r'')
    textt = str("root question is"+texttt+"this is what is detected by the model" +str(detected_item)+" give it more importance if you do not identify anythingbut this has something")
    a, b = match_command(textt)
    if b == 'ai_image_analysis_and_summary':
        summary = ai_image_analysis_and_summary(frame, textt)
        summary_generated = summary

        print(summary, "this is the summary")
        text_to_speech(summary)


def draw_detection(frame, bbox, conf, class_id):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"Detected {class_id}: {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    print('still not out')
    if summary_generated is not None:
        cv2.putText(frame, f"{summary_generated}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    
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

def update_object_movement(track_id, x, y, frame_num):
    current_time = time.time()
    movement_detected = False  # Flag to indicate movement status
    
    if track_id not in object_movement_data:
        object_movement_data[track_id] = {
            "start_position": (x, y),
            "start_frame": frame_num,
            "last_position": (x, y),
            "last_movement_time": current_time,
            "stationary_time": 0
        }
    else:
        movement = object_movement_data[track_id]
        last_x, last_y = movement["last_position"]
        distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5

        if distance < MOVEMENT_THRESHOLD:
            movement["stationary_time"] += current_time - movement["last_movement_time"]
            if movement["stationary_time"] >= STATIONARY_THRESHOLD:
                movement["start_position"] = (x, y)
                movement["start_frame"] = frame_num
                movement["stationary_time"] = 0
        else:
            movement["stationary_time"] = 0
            movement_detected = True  # Movement detected if distance > threshold

        movement["last_position"] = (x, y)
        movement["last_movement_time"] = current_time

    return movement_detected

vertices, faces = load_obj("hh.obj")
def normalize_vertices(vertices):
    min_vals = vertices.min(axis=0)
    max_vals = vertices.max(axis=0)
    return (vertices - min_vals) / (max_vals - min_vals)  # Scale to [0, 1]
vertices = normalize_vertices(vertices)

normalized_vertices = normalize_vertices(vertices)
import cv2
import numpy as np

def apply_custom_colormap_cv2(depth_map):
    # Mask to ignore zero values in depth
    mask = depth_map != 0
    
    disp_map = np.zeros_like(depth_map, dtype=np.uint8)
    if mask.any():  # Check if there are non-zero depth values
        vmax = np.percentile(depth_map[mask], 95)
        vmin = np.percentile(depth_map[mask], 5)
        depth_normalized = np.clip((depth_map - vmin) / (vmax - vmin), 0, 1)
        disp_map = (depth_normalized * 255).astype(np.uint8)


    depth_colored = cv2.applyColorMap(disp_map, cv2.COLORMAP_JET)
    
    depth_colored[~mask] = [255, 255, 255]

    return depth_colored



class FrameProcessor(Thread):
    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = True
        self.current_frame_data = None

    def run(self):
        frame = 0
        while self.running:
            try:
                global detected_item
                frame_data = self.frame_queue.get(timeout=1.0)
                if frame_data is None:
                    continue

                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    frame_rgb = ensure_rgb(frame)
                    
                    # cv2.imshow("normal",frame_rgb)
                    preprocessed_frame, scale, top, left = preprocess_image(frame_rgb)
                    input_tensor = torch.from_numpy(preprocessed_frame).permute(2, 0, 1).unsqueeze(0).to(device)
                    with torch.no_grad():
                        results = model(input_tensor)
                        print(results, "these are reults")
                        
                    
                    # Accessing detections from results
                    predictions = results[0].boxes.data 
                    

                    detected_item = predictions # YOLOv8 output structure
                    

                    predictions = predictions[predictions[:, 4] > 0.59]
                    tracker_inputs = []
                    
                    object_list = []
                    for *xyxy, conf, class_id in predictions.cpu().numpy():
                        # Save the input tensor to a file
                        input_tensor_file = "input_tensor.pt"
                        torch.save(input_tensor, input_tensor_file)
                        print(f"Input tensor saved to {input_tensor_file}")

                        depth_map = estimate_depth(frame_rgb)
                        
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        x, y = min(x1, x2), min(y1, y2)
                        w, h = abs(x2 - x1), abs(y2 - y1)
                        
                        tracker_inputs.append([[x, y, w, h], float(conf), str(int(class_id))])
                        print('were in ')
                        draw_detection(frame, (x1, y1, x2, y2), conf, class_id)
                        
                        depth = get_depth_for_object(depth_map, (x1, y1, x2, y2))
                        object_data = {
                                "id": len(object_list),  # unique ID
                                "bbox": (x1, y1, x2, y2),
                                "depth": depth,
                                "class_id": int(class_id),
                                "confidence": float(conf)
                            }
                        save_frame_as_image(frame)
                        transformed_vertices = transform_and_project(normalize_vertices, (x1, y1, x2, y2), depth, frame_rgb.shape)
                        projected_vertices = transform_and_project_exact(
                                vertices,(x1, y1, x2, y2), depth
                            )
                        draw_3d_overlay(frame, projected_vertices, faces=faces, color=(123, 235, 113))




                        
                        depth_colored =apply_custom_colormap_cv2(depth_map)
                        cv2.imshow("Depth Map", depth_colored)

                        object_list.append(object_data)
                        print(object_list, "this is object_list")

                    print('were out')
                    global frame_for_summary
                    frame_for_summary = frame_rgb
                    if keyboard.is_pressed('space'):
                        print("Space key pressed, executing terminal_task...")
                        terminal_task(self.current_frame_data)



                    tracked_objects = deep_sort_tracker.update_tracks(tracker_inputs, frame=frame)
                    for track in tracked_objects:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue

                        x1, y1, w, h = map(int, track.to_tlwh())
                        movement_detected = update_object_movement(track.track_id, x1, y1, frame)
                        # update_object_movement(track.track_id, x1, y1, frame)
                        frame = draw_tracking(frame, track, movement_detected)
                    global summary_generated
                    print("summary generates", summary_generated)

                    cv2.imshow("Object Detection Stream", frame)
                    if cv2.waitKey(10) == ord('q'):
                        self.running = False
                        break

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")

        cv2.destroyAllWindows()
def start_server(host='0.0.0.0', port=12345):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    frame_queue = queue.Queue(maxsize=2)
    processor = FrameProcessor(frame_queue)
    processor.start()

    try:
        while True:
            client_socket, addr = server_socket.accept()
            try:
                if client_socket.recv(6) != b'START\n':
                    continue
                while True:
                    size_data = client_socket.recv(4)
                    if not size_data or len(size_data) != 4:
                        break
                    frame_size = struct.unpack('>I', size_data)[0]
                    frame_data = bytearray()
                    while frame_size > 0:
                        chunk = client_socket.recv(min(65536, frame_size))
                        if not chunk:
                            break
                        frame_data.extend(chunk)
                        frame_size -= len(chunk)
                    if len(frame_data) == struct.unpack('>I', size_data)[0]:
                        try:
                            frame_queue.put_nowait(bytes(frame_data))
                        except queue.Full:
                            pass
            finally:
                client_socket.close()

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        processor.running = False
        processor.join()
        server_socket.close()

def start_key_listener():
    keyboard.add_hotkey('space', lambda: (
        print("Space key pressed, checking frame availability"),
        terminal_task(frame_for_summary)))  # Assign 'space' key to trigger the function
    keyboard.wait('esc')

if __name__ == "__main__":
    Thread(target=start_key_listener, daemon=True).start()
    start_server()




# import torch
# print(torch.version.cuda)
 # Should return True if GPU is accessible

