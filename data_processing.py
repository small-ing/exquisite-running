from landmark import Tracker, CNN #if CNN
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import os
import mediapipe as mp
import time
import cv2
from alive_progress import alive_bar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
landmarker = Tracker(model="HEAVY")

# read in images, pass them through the landmark, and save tensors that store the landmark data
def collect_data():
    folder_path = 'data'
    '''
    Each landmark has 4 values:
        X, Y both normalized with respect to the image size
        
    X coordinate
    Y coordinate
    Z coordinate (Depth with respect to hips) [CLOSER TO CAMERA = NEGATIVE]
        Z is roughly normalized to same scale as X
    Visibility between 0-1
    '''
    empty_frames = 0
    counter = 0
    joint_counter = 0
    empty_vid_landmarks = np.zeros((1, 33, 4)) # 33 landmarks, 4 values per landmark
    empty_labels = np.zeros((1)) # 1 label per image
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Get a list of image files in the folder
    good_image_files = [file for file in os.listdir(folder_path + "/good") if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]
    good_video_files = [file for file in os.listdir(folder_path + "/good") if file.lower().endswith((".mp4", ".mov", ".avi"))]
    
    bad_image_files = [file for file in os.listdir(folder_path + "/bad") if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]
    bad_video_files = [file for file in os.listdir(folder_path + "/bad") if file.lower().endswith((".mp4"))]
    
    image_files = good_image_files + bad_image_files
    video_files = good_video_files + bad_video_files
    empty_im_landmarks = np.zeros((len(image_files), 33, 4)) # 33 landmarks, 4 values per landmark
    # Process each image in the folder
    for video_file in video_files:
        with alive_bar(len(video_files)) as bar:
            #print(os.path.join(folder_path, video_file))
            if video_file in good_video_files:
                vid = cv2.VideoCapture(os.path.join(folder_path + '/good', video_file))
            else:
                vid = cv2.VideoCapture(os.path.join(folder_path + '/bad', video_file))

            '''
            stride_time = "whatever"
            def calc_stride():
                pass
            #some logic tracking same foot hitting the ground
            #stride_time = calc_stride()
            #pass some var stride time to the tensor as a fifth value
            '''
            
            frame = 0
            success = 1 
            while success:
                success, image = vid.read()
                if image is None:
                    pass
                else:
                    # if frame number is the same as the number of frames in the landmark tensor, recreate it with more space
                    if frame == len(empty_vid_landmarks):
                        empty_vid_landmarks = np.concatenate((empty_vid_landmarks, np.zeros((1, 33, 4))))
                        empty_labels = np.concatenate((empty_labels, np.zeros((1))))
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    try:
                        for landmark in landmarker.final_landmarker.detect(image).pose_landmarks[0]:
                            empty_vid_landmarks[frame][joint_counter][0], empty_vid_landmarks[frame][joint_counter][1] = landmark.x, landmark.y
                            empty_vid_landmarks[frame][joint_counter][2], empty_vid_landmarks[frame][joint_counter][3] = landmark.z, landmark.visibility
                        if video_file in good_video_files:
                            empty_labels[frame] = 1 # 0 for bad, 1 for good
                        else:
                            empty_labels[frame] = 0
                        frame += 1
                    except:
                        empty_frames += 1
            vid.release()
            bar()
    print("Total empty frames: " + str(empty_frames))
                    
            
    offset = len(empty_labels) -1
    empty_labels = np.concatenate((empty_labels, np.zeros((len(image_files)))))
    for image_file in image_files:
        with alive_bar(len(image_files)) as bar:
            if image_file in good_image_files:
                image_path = os.path.join(folder_path + "/good", image_file)
            else:
                image_path = os.path.join(folder_path + "/bad", image_file)
            mp_image = mp.Image.create_from_file(image_path)
            
            for landmark in landmarker.final_landmarker.detect(mp_image).pose_landmarks[0]: # this is the landmark data we need to save in the tensors
                empty_im_landmarks[counter][joint_counter][0], empty_im_landmarks[counter][joint_counter][1] = landmark.x, landmark.y
                empty_im_landmarks[counter][joint_counter][2], empty_im_landmarks[counter][joint_counter][3] = landmark.z, landmark.visibility
                if joint_counter < 32:
                    joint_counter += 1
                else: 
                    joint_counter = 0
            if image_file in good_image_files:
                empty_labels[offset + counter] = 1
            else:        
                empty_labels[offset + counter] = 0 # 0 for bad, 1 for good
            counter += 1
            bar()
    
    
    # combine the image and video landmarks
    filled_landmarks = np.concatenate((empty_im_landmarks, empty_vid_landmarks))
    return filled_landmarks, empty_labels
    
def train_model(model, train_loader, loss_fn, optimizer, epochs, test_images, test_labels):
    should_save = False
    for i in range(epochs):
        with alive_bar(len(train_loader), title=i) as bar:
            for img, label in train_loader:
                img = img.to(device)
                img = img.to(torch.float)
                label = label.to(device) 
                pred = model(img)
                bar()

                loss = loss_fn(pred, label)
                optimizer.zero_grad()    
                loss.backward()
                optimizer.step()
            test_images = test_images.to(torch.float).to(device)
            pred = model(test_images)
            digit = torch.argmax(pred, dim=1)
            test_labels = test_labels.to(device)
            acc = torch.sum(digit == test_labels)/len(test_labels)
            if acc > 0.92 and loss < 0.2:
                if not should_save:
                    print("Good enough to save")
                should_save = True
                if acc > 0.95 and loss < 0.10:
                    print(f"Accuracy - {acc} and Loss - {loss} are ideal")
                    print("Model is Ideal, saving now...")
                    break
        print(f"Epoch {i+1}: loss: {loss}, test accuracy: {acc}")
    return should_save

# take the generated tensors, and pass them through a CNN to generate a prediction
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img, label):
        self.img = img
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]

if __name__ == "__main__":
    print("Starting...")
    start_time = time.time()
    landmarks, labels = collect_data()
    test_marks, test_labels = landmarks[-100:], labels[-100:]
    
    print("Time to collect data: ", time.time() - start_time)
    
    landmarks, test_marks = torch.from_numpy(landmarks), torch.from_numpy(test_marks)
    landmarks, test_marks = landmarks.reshape(-1, 1, 33, 4), test_marks.reshape(-1, 1, 33, 4)
    landmarks, test_marks = landmarks.float(), test_marks.float()
    
    labels, test_labels = torch.from_numpy(labels), torch.from_numpy(test_labels)
    labels, test_labels = labels.long(), test_labels.long()
    
    data = ImageDataset(landmarks, labels)
    data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, num_workers=0)
    
    print("Time to load data: ", time.time() - start_time)
    
    model = CNN()
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criteron = nn.CrossEntropyLoss()
    
    print("Time to initialize model: ", time.time() - start_time)
    
    train_model(model, data_loader, criteron, optimizer, 100, test_marks, test_labels)
    
    print("Ending... Time elapsed: ", time.time() - start_time)
    # try:
    #     # Save the array to the text file
    #     np.savetxt('output.txt', landmarks)
    #     print(f"Array saved successfully.")
    # except Exception as e:
    #     print(f"Error occurred while saving the array: {e}")