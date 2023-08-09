#With correct python version, run 
# pip install -r requirements.txt
import cv2
from landmark import *
from data_processing import create_data

# Path: main.py
cap = cv2.VideoCapture("istockphoto-811241150-640_adpp_is (2).mp4")
landmark_tracker = Tracker(model='HEAVY', detect=True)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

print("Dimensions") 
print("Frame width: ", frame_width)
print("Frame height: ", frame_height)
marks = np.zeros((1, 36, 4))
frame_counter = -1
joint_counter = 0
done = False
frames = []

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
    # print("Starting Video Writer Loop")
    try:
        success, frame = cap.read()
        frames.append(frame)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        joint_counter = 0
        frame_counter += 1
        print("Frame Read was successful")
    except:
        print("There is no camera")
        pass
    #insert any model calls here
    try:
        print("Detecting and Drawing on Frame")
        if not done:
            if frame_counter == len(marks) - 1:
                # print("breaking after this")
                marks = np.concatenate((marks, np.zeros((1, 36, 4))))
                
            for landmark in landmark_tracker.final_landmarker.detect(mp_frame).pose_landmarks[0]: # this is the landmark data we need to save in the tensors
                marks[frame_counter][joint_counter][0], marks[frame_counter][joint_counter][1] = landmark.x, landmark.y
                marks[frame_counter][joint_counter][2], marks[frame_counter][joint_counter][3] = landmark.z, landmark.visibility
                if joint_counter < 32:
                    joint_counter += 1
                else: 
                    joint_counter = 0
            
            frames[frame_counter], landmarks = landmark_tracker.detect_and_draw_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames[frame_counter] = cv2.cvtColor(frames[frame_counter], cv2.COLOR_RGB2BGR)
            marks = create_data(marks)
            print(len(frames))
            pass
    except IndexError as ie:
        pass
    except Exception as e:
        print(e)
        print("No more frames")
        done = True
    try:
        if done:
            print("Evaluating Landmarks")
            marks = torch.from_numpy(marks).float()
            print("Marks shape: ", marks.shape)
            marks = marks.reshape(-1, 1, 36, 4)
            print("Marks shape: ", marks.shape)
            avg = 0
            blank_frame = 0
            for i in range(len(frames)):      
                print("Frame: ", i)          
                res = landmark_tracker.stride_model(marks[i].reshape(1,1,36,4)).softmax(dim=1).detach().numpy()
                marks[i][0][33][2] = res[0][1]*100
                cv2.putText(frames[i], str(res[0][1]*100) + "%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                if marks[i][0][33][2].item() > 0 and not np.isnan(marks[i][0][33][2].item()):
                    avg += marks[i][0][33][2].item()
                else:
                    blank_frame += 1
                if marks[i][0][33][2] < 90:
                    print("...  Frame: ", i, " needs further evaluation")
                    # print("Score: ", marks[i][0][33][2].item())
                    # print("Elbow Angles: ", marks[i][0][34][:2])
                    # print("Knee Angles: ", marks[i][0][35][:2])
                    # print("Hip Angles: ", marks[i][0][34][2:])
                    # print("Ankle Angles: ", marks[i][0][35][2:])
                out.write(frames[i])

            print("Average Score (0-100): ", avg/(len(frames)-blank_frame))
            
            cv2.imshow("Camera", frames[i])        
    except Exception as e:
        # print(e)
        print("Done writing video")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows()