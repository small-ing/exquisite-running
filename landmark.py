# imports

# need python 3.10.8 for mediapipe
import mediapipe as mp # pip install mediapipe==0.10.2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

class Tracker():
    def __init__(self, mode='IMAGE'):
        self.model_path = 'pose_landmarker_lite.task'
        self.base_options = mp.tasks.BaseOptions
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker
        self.landmarker_options = mp.tasks.vision.PoseLandmarkerOptions
        self.running_mode = mp.tasks.vision.RunningMode
        if mode == 'IMAGE':
            self.options = self.landmarker_options(
                base_options=self.base_options(model_asset_path=self.model_path),
                running_mode=self.running_mode.IMAGE)
        else:
            self.options = self.landmarker_options(
                base_options=self.base_options(model_asset_path=self.model_path),
                running_mode=self.running_mode.VIDEO)
        self.final_landmarker = self.pose_landmarker.create_from_options(self.options)
                
        
    def calculate_angle(start, shared, end):
        vec_1, vec_2 = shared - start, end - shared # (x, y) vectors
        vec_1_magnitude, vec_2_magnitude = np.linalg.norm(vec_1), np.linalg.norm(vec_2)
        dot_prd = np.dot(vec_1, vec_2)
        angle = np.arccos(dot_prd / (vec_1_magnitude * vec_2_magnitude))
        return angle
        
    def draw_landmarks_on_image(self, rgb_image, detection_result):
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

    def detect_and_draw(self, image):
        marks = self.final_landmarker.detect(image)
        return self.draw_landmarks_on_image(image.numpy_view(), marks)
    
    def detect_and_draw_frame(self, image):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        marks = self.final_landmarker.detect(image)
        return self.draw_landmarks_on_image(image.numpy_view(), marks)
    
# Path: landmark.py

if __name__ == "__main__":
    pTracker = Tracker()
    with pTracker.final_landmarker as landmarker:
            # Load the input image from an image file.
        mp_image = mp.Image.create_from_file('data/image.png')
        # print(pTracker.final_landmarker.detect(mp_image).pose_landmarks[0])
        annotated_image = pTracker.detect_and_draw(mp_image)
        while True:
            cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()