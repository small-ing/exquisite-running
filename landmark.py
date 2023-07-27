# imports

# need python 3.7.9 for mediapipe
import mediapipe as mp # pip install mediapipe==0.10.2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2


def calculate_angle(start, shared, end):
    vec_1, vec_2 = shared - start, end - shared # (x, y) vectors
    vec_1_magnitude, vec_2_magnitude = np.linalg.norm(vec_1), np.linalg.norm(vec_2)
    dot_prd = np.dot(vec_1, vec_2)
    angle = np.arccos(dot_prd / (vec_1_magnitude * vec_2_magnitude))
    return angle
    

# Path: landmark.py
model_path = 'pose_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

with PoseLandmarker.create_from_options(options) as landmarker:
        # Load the input image from an image file.
    mp_image = mp.Image.create_from_file('image.png')
    pose_landmarker_result = landmarker.detect(mp_image)