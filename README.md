# Real-Time AI-Driven Object Detection & 3D Overlay System

This project is a sophisticated real-time video processing pipeline that integrates advanced computer vision and deep learning models to detect, track, and overlay 3D objects on live video streams. It also incorporates audio processing for interactive voice commands and uses multi-threaded socket communication to ensure high performance and low latency.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Operating Instructions](#operating-instructions)
- [Using WebSocket for Video Input](#using-websocket-for-video-input)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Instructions to Run the Code]

---

## Overview

This project combines state-of-the-art technologies to process video streams in real time. The pipeline consists of:
- **Object Detection & Tracking:**  
  Uses YOLO for object detection and DeepSort for tracking objects across frames.
- **Depth Estimation:**  
  Employs the MiDaS model to generate depth maps from video frames.
- **3D Overlay:**  
  Projects 3D models (loaded from OBJ files) onto detected objects by applying transformation and projection matrices.
- **Audio & Interaction:**  
  Processes voice commands and utilizes text-to-speech to provide interactive feedback.
- **Multi-threaded Processing:**  
  Utilizes Python’s threading and socket modules to concurrently process frames and handle incoming video data.

---

## Features

- **Real-Time Processing:**  
  Processes live video streams with low latency.
- **Advanced Object Detection:**  
  Integrates YOLO for robust object detection with confidence filtering.
- **Precision Tracking:**  
  Uses DeepSort to maintain accurate tracking of objects.
- **Dynamic 3D Overlay:**  
  Projects 3D models onto live video frames using custom transformation techniques.
- **Depth Estimation:**  
  Implements MiDaS for accurate depth mapping.
- **Interactive Audio Commands:**  
  Integrates voice command processing and text-to-speech for enhanced interactivity.
- **Scalable Multi-threaded Architecture:**  
  Handles socket communication and concurrent frame processing efficiently.
- **WebSocket Integration:**  
  (Optional) Can accept video files via WebSocket for flexible deployment scenarios.

---

## Prerequisites

- **Python 3.8+**
- **CUDA-enabled GPU (recommended)** for deep learning inference
- Required Python packages:
  - OpenCV
  - PyTorch
  - TorchVision
  - NumPy
  - PyWavefront
  - DeepSort (deep_sort_realtime)
  - ultralytics (for YOLO)
  - keyboard
  - Pillow
  - websocket-client (if using WebSocket client for video input)
  - Other dependencies listed in `requirements.txt`

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

1. **Model Files:**
   - **YOLO Model:**  
     Update the path to your YOLO model file in the source code where indicated (e.g., `model = YOLO(r'<YOUR_MODEL_PATH>')`).
   - **MiDaS Model:**  
     Set the MiDaS model path in the variable `midas_model_path` (e.g., `midas_model_path = "dpt_large_384.pt"`).

2. **OBJ Files:**  
   Ensure that your 3D model files (e.g., `hh.obj`) are placed in the correct directory and the path is updated accordingly in the code.

3. **Environment:**  
   Confirm that your environment has access to a GPU if available; otherwise, the system will run on CPU.

---

## Running the Project

1. **Start the Server:**  
   The project includes a socket server to receive video frames.
   
   ```bash
   python your_project_script.py
   ```

   The server listens on the default host (`0.0.0.0`) and port (`12345`).

2. **Key Listener:**  
   A key listener is active—press `space` to trigger a terminal task (e.g., process the current frame for a summary) and `esc` to exit.

3. **Monitor the Output:**  
   The system opens several OpenCV windows:
   - **"Object Detection Stream":** Shows the processed video with detections, tracking, and overlays.
   - **"Depth Map":** Displays the colored depth map.

---

## Operating Instructions

1. **Sending Video Frames:**
   - The server expects a specific protocol:
     - The client should first send the string `START\n`.
     - Each video frame is prefixed with a 4-byte size header (big-endian format) followed by the frame data (JPEG-encoded).
   - Ensure your client follows this protocol when transmitting frames.

2. **Using Audio Commands:**
   - When a video frame is displayed, you can use your microphone to input a voice command.
   - The system processes the command and, if applicable, executes the corresponding function (e.g., generating a summary or adjusting the overlay).

3. **Real-Time Interaction:**
   - Press the `space` key to capture the current frame and trigger the terminal task.
   - The system’s text-to-speech component will provide audible feedback based on the processed command.

---

## Using WebSocket for Video Input

To enable video input via WebSocket instead of a raw TCP socket, follow these steps:

1. **WebSocket Server Setup:**
   - You can use libraries like `websockets` or `websocket-server` in Python to create a WebSocket server.
   - Below is an example snippet using the `websockets` library:

   ```python
   import asyncio
   import websockets
   import struct

   async def video_receiver(websocket, path):
       print("Client connected")
       try:
           async for message in websocket:
               # Assume the first 4 bytes indicate frame size in big-endian format
               if len(message) < 4:
                   continue
               frame_size = struct.unpack('>I', message[:4])[0]
               frame_data = message[4:]
               if len(frame_data) == frame_size:
                   # Put the frame data into the processing queue
                   frame_queue.put_nowait(frame_data)
               else:
                   print("Incomplete frame received")
       except websockets.ConnectionClosed:
           print("Client disconnected")

   async def main():
       async with websockets.serve(video_receiver, "0.0.0.0", 8765):
           await asyncio.Future()  # Run forever

   if __name__ == "__main__":
       asyncio.run(main())
   ```

2. **WebSocket Client:**
   - Create a client that reads a video file, encodes frames as JPEG, and sends them to the WebSocket server.
   - Ensure each frame is sent with a 4-byte header indicating its size.

3. **Integration:**
   - Modify your main server script to optionally accept WebSocket connections on a dedicated port (e.g., `8765`).
   - Use a shared queue (`frame_queue`) to handle frames from both TCP and WebSocket sources.

4. **Testing:**
   - Use a tool like `websocat` or a custom Python script to send video frames via WebSocket.
   - Confirm that frames are received correctly and processed similarly to the TCP-based approach.

---

## Troubleshooting

- **Frame Not Displaying:**  
  Check if the frame decoding via OpenCV (`cv2.imdecode`) is successful. Verify that the video stream is correctly encoded in JPEG format.
- **Model Loading Issues:**  
  Ensure that the paths for YOLO and MiDaS models are correctly configured and that the models are compatible with your PyTorch version.
- **Performance Bottlenecks:**  
  Monitor GPU/CPU usage. Adjust the frame queue size or processing thread parameters if the system lags.
- **WebSocket Connection Issues:**  
  Validate network connectivity and ensure the WebSocket client follows the correct protocol format.

---

*For any issues or contributions, please open an issue or submit a pull request on the GitHub repository.*



## FINALLY INSTRUCTIONS AND PYTHON CODE TO RUN THE CODE,

```python
import cv2
import asyncio
import websockets
import struct

async def send_video(uri, video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return
    async with websockets.connect(uri) as websocket:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Could not encode frame.")
                continue
            frame_data = buffer.tobytes()
            header = struct.pack('>I', len(frame_data))
            message = header + frame_data
            await websocket.send(message)
            await asyncio.sleep(1/30)
        cap.release()

if __name__ == "__main__":
    uri = "ws://localhost:8765"
    asyncio.run(send_video(uri, video_source=0))
```

THIS THIS PYTHON CODE TO SEND VIDEO TO THE URL BY RUNNING THIS ON YOUR TERMINAL AND ON THE OTHER TERMINAL RUN tracking_summary.py


