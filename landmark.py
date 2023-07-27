# imports

# need python 3.7.9 for mediapipe
import mediapipe as mp # pip install mediapipe==0.10.2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2


def calculate_angle(start, shared, end):
    vec_1, vec_2 = shared - start, end - shared # (x, y) vectors
    vec_1_magnitude, vec_2_magnitude = np.linalg.norm(vec_1), np.linalg.norm(vec_2)
    dot_prd = np.dot(vec_1, vec_2)
    angle = np.arccos(dot_prd / (vec_1_magnitude * vec_2_magnitude))
    return angle
    
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

# Path: landmark.py
model_path = 'pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

with PoseLandmarker.create_from_options(options) as landmarker:
        # Load the input image from an image file.
    mp_image = mp.Image.create_from_file('bad running form169.jpg')
    pose_landmarker_result = landmarker.detect(mp_image)
    #save pose_landmarker_result to a file
    #print(pose_landmarker_result)
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
    while True:
        cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()