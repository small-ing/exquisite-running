# imports

# need python 3.10.8 for mediapipe
import mediapipe as mp # pip install mediapipe==0.10.2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import torch
import cv2

class Tracker():
    def __init__(self, model='LITE', mode='IMAGE'):
        if mode == 'LITE':
            self.model_path = 'model/pose_landmarker_lite.task'
        elif mode == 'FULL':
            self.model_path = 'model/pose_landmarker_full.task'
        else:
            self.model_path = 'model/pose_landmarker_heavy.task'
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
    
    def calculate_center_mass(l_shoulder, r_shoulder, l_hip, r_hip):
        shoulders = [l_shoulder, r_shoulder]
        hips = [l_hip, r_hip]

        #extract positional data from landmarks
        for index, shld in enumerate(shoulders):
            if index == 0:
                shoulder_pos_x =[]
                shoulder_pos_y =[]
            else:
                None
            shoulder_pos_x.append(shld.x)
            shoulder_pos_y.append(shld.y)
        for i, hip in enumerate(hips):
            if i ==0:
                hip_pos_x = []
                hip_pos_y = []
            else:
                None
            hip_pos_x.append(hip.x)
            hip_pos_y.append(hip.y)

        #calculate midpoints for shoulders/hips
        mid_shoulder_x, mid_shoulder_y  = np.average(shoulder_pos_x), np.average(shoulder_pos_y)
        mid_hip_x, mid_hip_y  = np.average(hip_pos_x), np.average(hip_pos_y)

        #creates vector from midpoints
        center_of_mass = (mid_shoulder_x - mid_hip_x, mid_shoulder_y - mid_hip_y)
        return center_of_mass


        
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

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        
        self.layer1 = torch.nn.Linear(128, 1056, bias=True)
        self.layer2 = torch.nn.Linear(128, 128, bias=True)
        self.layer3 = torch.nn.Linear(128, 2, bias=True)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        keep_prob = 0.5
        # L1 ImgIn shape=(?, 33, 4, 1)
        # Conv -> (?, 33, 4, 32)
        # Pool -> (?, 16, 2, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, 16, 2, 32)
        # Conv      ->(?, 16, 2, 64)
        # Pool      ->(?, 8, 1, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, 7, 7, 64)
        # Conv ->(?, 5, 0, 128)
        # Pool ->(?, 2, 0, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Dropout(p=1 - keep_prob))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(2304, 625, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(625, 625, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(625, 625, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 2, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer4(out)
        out = self.fc2(out)
        return out
    
    def predict_frame(self):
        pass
        '''
        form = torch.topk(self.model) etc.
        '''
# Path: landmark.py

if __name__ == "__main__":
    pTracker = Tracker(model="HEAVY")
    with pTracker.final_landmarker as landmarker:
            # Load the input image from an image file.
        image = cv2.imread("data/9189211C-DEBD-4D0B-8C4A-39C3BD30261F.png")
        image = pTracker.detect_and_draw_frame(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("data/9189211C-DEBD-4D0B-8C4A-39C3BD30261F_LANDMARKED.png", image)
        while True:
            cv2.imshow("Image", image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()