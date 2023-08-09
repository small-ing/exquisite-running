from landmark import Tracker, CNN #if CNN
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.init
import torch.optim as optim
import torchvision
import numpy as np
import os
import mediapipe as mp
import time
import cv2
from alive_progress import alive_bar
import random
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
landmarker = Tracker(model="HEAVY")

# read in images, pass them through the landmark, and save tensors that store the landmark data
def collect_data():
    folder_path = 'data'
    branch = ['/test', '/train']
    '''
    Each landmark has 4 values:
        X, Y both normalized with respect to the image size
        
    X coordinate
    Y coordinate
    Z coordinate (Depth with respect to hips) [CLOSER TO CAMERA = NEGATIVE]
        Z is roughly normalized to same scale as X
    Visibility between 0-1
    '''
    joint_counter = 0
    
    filled_test_landmarks = None
    test_labels = None
    
    filled_train_landmarks = None
    train_labels = None
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Get a list of image files in the folder
    for b in branch:
        good_image_files = [file for file in os.listdir(folder_path + b + "/good") if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]
        bad_image_files = [file for file in os.listdir(folder_path + b + "/bad") if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]
        print("Good Images: ", len(good_image_files))
        print("Bad Images: ", len(bad_image_files))
        image_files = good_image_files + bad_image_files
        print("Total Images: ", len(image_files))
        empty_labels = np.zeros(len(image_files)) # 1 label per image
        for num in range(len(good_image_files)):
            empty_labels[num] = 1
        empty_im_landmarks = np.zeros((len(image_files), 36, 4)) # 33 landmarks, 4 values per landmark

        offset = 0
        image_length = len(image_files)
        image = 0
        with alive_bar(image_length, title=b) as bar:
            while image < image_length:
                try:
                    # print("image: " + str(image))
                    # print("counter: " + str(counter))
                    # print("image length: " + str(image_length))
                    if image_files[image] in good_image_files:
                        # print("file in good images")
                        image_path = os.path.join(folder_path + b + "/good", image_files[image])
                    else:
                        # print("file in bad images")
                        image_path = os.path.join(folder_path + b + "/bad", image_files[image])
                    mp_image = mp.Image.create_from_file(image_path)
                    
                    for landmark in landmarker.final_landmarker.detect(mp_image).pose_landmarks[0]: # this is the landmark data we need to save in the tensors
                        empty_im_landmarks[image][joint_counter][0], empty_im_landmarks[image][joint_counter][1] = landmark.x, landmark.y
                        empty_im_landmarks[image][joint_counter][2], empty_im_landmarks[image][joint_counter][3] = landmark.z, landmark.visibility
                        if joint_counter < 32:
                            joint_counter += 1
                        else: 
                            joint_counter = 0
                            if image_files[image] in good_image_files:
                                empty_labels[image+offset] = 1
                            else:
                                empty_labels[image+offset] = 0
                except Exception as e:
                    # print(e)
                    empty_labels[image+offset] = -1
                    pass
                    # print("deleting index ", counter+offset-1, " because of error")
                    # empty_im_landmarks = np.delete(empty_im_landmarks, counter+offset-1, 0)
                    # empty_labels = np.delete(empty_labels, counter+offset-1, 0)
                    # image_length -= 1
                    # counter -= 1
                    # print("new length: ", image_length)
                image += 1
                bar()
        
        
        # combine the image and video landmarks
        # filled_landmarks = np.concatenate((empty_im_landmarks, empty_vid_landmarks))
        failed_indexes = np.where(empty_labels == -1)
        empty_im_landmarks = np.delete(empty_im_landmarks, failed_indexes, 0)
        empty_labels = np.delete(empty_labels, failed_indexes, 0)
        if b == '/test':
            filled_test_landmarks = empty_im_landmarks
            test_labels = empty_labels
        else:
            filled_train_landmarks = empty_im_landmarks
            train_labels = empty_labels
    return filled_test_landmarks, test_labels.astype(int), filled_train_landmarks, train_labels.astype(int)

def create_data(landmarks, height=72):
    '''
    BATCH, 36, 4
    [center of mass, stride length, max_stride_length, height]
    [left elbow angle, right elbow angle, left hip angle, right hip angle]
    [left knee angle, right knee angle, left ankle angle, right ankle angle]
    
    rewrite
    1 -> left elbow angle   (11, 13, 15)
    2 -> right elbow angle  (12, 14, 16)
    3 -> left hip angle     (24, 23, 25)
    4 -> right hip angle    (23, 24, 26)
    5 -> left knee angle    (23, 25, 27)
    6 -> right knee angle   (24, 26, 28)
    9 -> left ankle angle   (25, 27, 31)
    10 -> right ankle angle (26, 28, 32)
    
    Append Center of Mass to the end of each frame
    Append Stride length to the end of each frame
        Stride length is the distance between the left and right ankle
    Height is user-input in feet, we then take further of both heels, and find distance from that to the average of the eyes, then multiply by (14/13)
        this should give us a pixel to feet ratio
    '''
    #for loop iterate through the frames
    landmark_length = len(landmarks)
    for i in range(landmark_length):
        # calculate the center of mass
        num = random.randint(1, 5000) 
        lshoulder, rshoulder = [landmarks[i][11][0], landmarks[i][11][1]], [landmarks[i][12][0], landmarks[i][12][1]]
        lhip, rhip = [landmarks[i][23][0], landmarks[i][23][1]], [landmarks[i][24][0], landmarks[i][24][1]]
        
        ang, com = landmarker.calculate_center_mass(lshoulder, rshoulder, lhip, rhip)
        if num == 1:
            print("Center of Mass Angle: ", ang)
        landmarks[i][33][0] = ang
        # calculate the stride length
        stride_length, pixel_height = landmarker.stride_length(landmarks[i], height)
        landmarks[i][33][1] = stride_length
        landmarks[i][33][3] = height
        # if i != 0:
        #     if stride_length > landmarks[i-1][33][2]:
        #         landmarks[i][33][2] = stride_length
        landmarks[i][33][2] = 0
        
        # calculate the angles
        elbows_and_hips = [[11, 13, 15], [12, 14, 16], [24, 23, 25], [23, 24, 26]]
        knees_and_ankles = [[23, 25, 27], [24, 26, 28], [25, 27, 31], [26, 28, 32]]
        angle_set_1 = []
        for joint in elbows_and_hips:
            angle_set_1.append(landmarker.calculate_angle(landmarks[i][joint[0]], landmarks[i][joint[1]], landmarks[i][joint[2]]))
        angle_set_2 = []
        for joint in knees_and_ankles:
            angle_set_2.append(landmarker.calculate_angle(landmarks[i][joint[0]], landmarks[i][joint[1]], landmarks[i][joint[2]]))
        if num == 2:
            print("Random Angle 1: ", angle_set_1[random.randint(0, 3)])
            print("Random Angle 2: ", angle_set_2[random.randint(0, 3)])
        landmarks[i][34][0], landmarks[i][34][1], landmarks[i][34][2], landmarks[i][34][3] = angle_set_1[0], angle_set_1[1], angle_set_1[2], angle_set_1[3]
        # print(landmarks[i][34])
        landmarks[i][35][0], landmarks[i][35][1], landmarks[i][35][2], landmarks[i][35][3] = angle_set_2[0], angle_set_2[1], angle_set_2[2], angle_set_2[3]
        # print(landmarks[i][35])
    return landmarks        

def train_model(model, train_loader, loss_fn, optimizer, epochs, test_images, test_labels):
    should_save = False
    for i in range(epochs):
        with alive_bar(len(train_loader), title=i) as bar:
            for img, label in train_loader:
                bar()
                img = img.to(device)
                img = img.to(torch.float)
                label = label.to(device) 
                # print("img: ", img)
                # print("label: ", label)
                pred = model(img)
                # print("pred: ", pred)

                loss = loss_fn(pred, label)
                optimizer.zero_grad()    
                loss.backward()
                optimizer.step()
            test_images = test_images.to(torch.float).to(device)
            pred = model(test_images)
            digit = torch.argmax(pred, dim=1)
            test_labels = test_labels.to(device)
            acc = torch.sum(digit == test_labels)/len(test_labels)
            wandb.log({"loss": loss, "accuracy": acc})
            if acc > 0.9 and loss < 0.1:
                if not should_save:
                    print("Good enough to save")
                should_save = True
                if acc > 0.925 and loss < 0.01:
                    print(f"Accuracy - {acc} and Loss - {loss} are ideal")
                    print("Model is Ideal, saving now...")
                    break
                if acc > 0.94 and loss < 0.1:
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

def draw_annotations(landmarks, image):
    angles = [[11, 13, 15], [12, 14, 16], [24, 23, 25], [23, 24, 26], [23, 25, 27], [24, 26, 28], [25, 27, 31], [26, 28, 32]]
    for i in range(len(angles)):
        cv2.line(image, (int(angles[i][0]), int(angles[i][0])), (int(angles[i][1])), (0, 255, 0), 2)

def model_train(learning_rate, epochs):
    print("Starting...")
    start_time = time.time()
    # start a new wandb run to track this script
    wandb.init(
    # set the wandb project where this run will be logged
        project="stridesense",  
    # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "CNN",
        "dataset": "CUSTOM-STRIDESENSE-EXPANDED",
        "epochs": epochs,
        }
    )
    test_marks, test_labels, landmarks, labels = collect_data()
    # print(landmarks.shape)
    landmarks = create_data(landmarks)
    test_marks = create_data(test_marks)
    # print(landmarks.shape)
    # print(labels.shape)
    
    print("Time to collect data: ", time.time() - start_time)
    
    landmarks, test_marks = torch.from_numpy(landmarks), torch.from_numpy(test_marks)
    landmarks, test_marks = landmarks.reshape(-1, 1, 36, 4), test_marks.reshape(-1, 1, 36, 4)
    landmarks, test_marks = landmarks.float(), test_marks.float()
    
    labels, test_labels = torch.from_numpy(labels), torch.from_numpy(test_labels)
    labels, test_labels = labels.long(), test_labels.long()
    # print("Labels: ", labels.shape)
    # print("Test Labels: ", test_labels.shape)
    test_label_count = 0
    for i in range(len(labels)):
        if labels[i] == 0:
           #print(labels[i])
            test_label_count += 1
    print("Out of the ", len(labels), " labels, ", test_label_count, " of the images are bad form")
        
    data = ImageDataset(landmarks, labels)
    data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, num_workers=2)
    
    print("Time to load data: ", time.time() - start_time)
    
    model = CNN()
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criteron = nn.CrossEntropyLoss()
    test_label_count = 0
    for i in range(len(test_labels)):
        if test_labels[i] == 0:
           #print(labels[i])
            test_label_count += 1
    print("Out of the ", len(test_labels), " labels, ", test_label_count, " of the test set images are bad form")
    print("Time to initialize model: ", time.time() - start_time)
    
    
    res = train_model(model, data_loader, criteron, optimizer, epochs, test_marks, test_labels)
    
    
    print("Should the model be saved?: ", res)
    if res:
        torch.save(model, "stride_sense_model_ut.pth")
        torch.save(model.state_dict(), "stride_sense_model_state_dict_ut.pth")
        
    print("Ending...\nTotal Time elapsed: ", time.time() - start_time)


if __name__ == "__main__":
    model_train(0.0005, 100)
