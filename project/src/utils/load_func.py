import json
import cv2
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os

import os
import cv2
from PIL import Image
import cv2
import numpy as np
import pywavefront

def save_frame_as_image(frame, output_dir='output_images', filename="image.jpg"):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    pil_image.save(file_path, format="JPEG")
    print(file_path)

    return file_path


def load_json_file(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
        return data


def load_obj_file(file_path):
    """
    Load the OBJ file and return vertices and faces.
    """
    scene = pywavefront.Wavefront(file_path, collect_faces=True)
    vertices = np.array(scene.vertices)
    faces = np.array(scene.mesh_list[0].faces)
    return vertices, faces