from landmark import Tracker
import torch
import torch.nn as nn
import numpy as np
import os
import mediapipe as mp

landmarker = Tracker()
# read in images, pass them through the landmark, and save tensors that store the landmark data
def collect_data(batch_size):
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
    counter = 0
    joint_counter = 0
    empty_landmarks = np.zeros((batch_size, 33, 4)) # 33 landmarks, 4 values per landmark
    empty_labels = np.zeros((batch_size)) # 1 label per image
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Get a list of image files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]

    # Process each image in the folder
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        mp_image = mp.Image.create_from_file(image_path)
        
        for landmark in landmarker.final_landmarker.detect(mp_image).pose_landmarks[0]: # this is the landmark data we need to save in the tensors
            empty_landmarks[counter][joint_counter][0], empty_landmarks[counter][joint_counter][1] = landmark.x, landmark.y
            empty_landmarks[counter][joint_counter][2], empty_landmarks[counter][joint_counter][3] = landmark.z, landmark.visibility
            if joint_counter < 32:
                joint_counter += 1
            else: 
                joint_counter = 0
        counter += 1
    return empty_landmarks, empty_labels
    


# take the generated tensors, and pass them through a CNN to generate a prediction


if __name__ == "__main__":
    landmarks, labels = collect_data(3)
    # try:
    #     # Save the array to the text file
    #     np.savetxt('output.txt', landmarks)
    #     print(f"Array saved successfully.")
    # except Exception as e:
    #     print(f"Error occurred while saving the array: {e}")