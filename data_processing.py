from landmark import Tracker
import torch
import torch.nn as nn
import numpy as np
import os
import mediapipe as mp

landmarker = Tracker()
# read in images, pass them through the landmark, and save tensors that store the landmark data
def collect_data(batch_size):
    pass
    folder_path = 'data'
    '''
    Each landmark has 4 values:
        X, Y both normalized with respect to the image size
        
    X coordinate
    Y coordinate
    Z coordinate (Depth with respect to hips)
    Visibility between 0-1
    '''
    
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
        
        landmarker.final_landmarker.detect(mp_image).pose_landmarks[0] # this is the landmark data we need to save in the tensors
    
    
    


# take the generated tensors, and pass them through a CNN to generate a prediction


if __name__ == "__main__":
    