#With correct python version, run 
# pip install -r requirements.txt
import cv2
import os
from landmark import *
from data_processing import create_data
from matplotlib.figure import Figure
from io import BytesIO
import base64

# Path: main.py
class VideoWriter():
    def __init__(self):
        self.landmark_tracker = Tracker(model='HEAVY', detect=True)
        self.marks = np.zeros((1, 36, 4))
        self.frame_counter = -1
        self.joint_counter = 0
        self.done = False
        self.frames = []
        self.problems = []
        self.avg = 0
        print("Video Writer Initialized")
        

    def write(self):
        print("Starting Video Writer")
        loop = True
        videos = [file.rsplit('.', 1)[0] for file in os.listdir("static/uploads") if file.lower().endswith((".mp4"))]
        print("Videos: ", videos)
        self.cap = cv2.VideoCapture("static/uploads/" + videos[0] + ".mp4")
        print("Video Capture Initialized ", videos[0])
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("Starting Video Writer")
        self.out = cv2.VideoWriter(f'static/uploads/{videos[0]}_w.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (self.frame_width,self.frame_height))
        # return []
        while loop:
            # print("Starting Video Writer Loop")
            try:
                success, frame = self.cap.read()
                self.frames.append(frame)
                mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                self.joint_counter = 0
                self.frame_counter += 1
                print("Frame Read was successful")
            except Exception as e:
                print(e)
                print("There is no camera")
                pass
            #insert any model calls here
            try:
                print("Detecting and Drawing on Frame")
                if not self.done:
                    if self.frame_counter == len(self.marks) - 1:
                        # print("breaking after this")
                        self.marks = np.concatenate((self.marks, np.zeros((1, 36, 4))))
                        
                    for landmark in self.landmark_tracker.final_landmarker.detect(mp_frame).pose_landmarks[0]: # this is the landmark data we need to save in the tensors
                        self.marks[self.frame_counter][self.joint_counter][0], self.marks[self.frame_counter][self.joint_counter][1] = landmark.x, landmark.y
                        self.marks[self.frame_counter][self.joint_counter][2], self.marks[self.frame_counter][self.joint_counter][3] = landmark.z, landmark.visibility
                        if self.joint_counter < 32:
                            self.joint_counter += 1
                        else: 
                            self.joint_counter = 0
                    
                    self.frames[self.frame_counter], landmarks = self.landmark_tracker.detect_and_draw_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.frames[self.frame_counter] = cv2.cvtColor(self.frames[self.frame_counter], cv2.COLOR_RGB2BGR)
                    self.marks = create_data(self.marks)
                    print(len(self.frames))
                    pass
            except IndexError as ie:
                pass
            except Exception as e:
                print(e)
                print("No more frames")
                self.done, loop = True, False
        try:
            if self.done:
                print("Evaluating Landmarks")
                self.marks = torch.from_numpy(self.marks).float()
                print("Marks shape: ", self.marks.shape)
                self.marks = self.marks.reshape(-1, 1, 36, 4)
                print("Marks shape: ", self.marks.shape)
                avg = 0
                blank_frame = 0
                for i in range(len(self.frames)):      
                    print("Frame: ", i)          
                    res = self.landmark_tracker.stride_model(self.marks[i].reshape(1,1,36,4)).softmax(dim=1).detach().numpy()
                    self.marks[i][0][33][2] = res[0][1]*100
                    cv2.putText(self.frames[i], "StrideScore " + str(round(res[0][1]*100),3) + "%", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                    if self.marks[i][0][33][2].item() > 0 and not np.isnan(self.marks[i][0][33][2].item()):
                        avg += self.marks[i][0][33][2].item()
                    else:
                        blank_frame += 1
                    if self.marks[i][0][33][2] < 90:
                        print("...  Frame: ", i, " needs further evaluation")
                        self.problems.append(i)
                        # print("Score: ", marks[i][0][33][2].item())
                        # print("Elbow Angles: ", marks[i][0][34][:2])
                        # print("Knee Angles: ", marks[i][0][35][:2])
                        # print("Hip Angles: ", marks[i][0][34][2:])
                        # print("Ankle Angles: ", marks[i][0][35][2:])
                    self.out.write(self.frames[i])

                print("Average Score (0-100): ", avg/(len(self.frames)-blank_frame)) 
                self.avg = avg/(len(self.frames)-blank_frame)
                self.convert_write(f'static/uploads/{videos[0]}_w.avi', f'static/uploads/{videos[0]}_w')
                # os.remove(f'uploads/{videos[0]}_w.avi')
                print("Done writing video")
                
                return self.metric_graph(self.marks) # returns the array of landmarks for every frame, with the score in i, 0, 33, 2
                
                    
        except Exception as e:
            print(e)
                # print("Breaking or something")
    def convert_write(self, avi_file_path, output_name):
        os.popen("ffmpeg -i {input} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {output}.mp4".format(input = avi_file_path, output = output_name))
        return True
    
    def metric_graph(self, landmarks):
        # using matplot lib, generate a figure with the x axis being the number of frames, and the y axis being the average score that frame
        # save the picture as an image, and encode it as a base64 string
        # return the string
        print("Creating Data")
        print(landmarks.shape)
        data_y = []
        for i in range(len(landmarks)):
            data_y.append(landmarks[i][0][33][2])
        
        for i in range(len(data_y)):
            if np.isnan(data_y[i]) and i != 0 and i != len(data_y)-1:
                if np.isnan(data_y[i+1]):
                    data_y[i] = 0
                else:
                    data_y[i] = (data_y[i-1] + data_y[i+1])/2
        
        sum = 0
        counter = 1
        data_y_avg = []
        for item in data_y:
            sum += item
            data_y_avg.append(sum/counter)
            counter += 1
        data_y = np.array(data_y)
        data_x = [i for i in range(len(data_y))]
        
        print("Creating Graph")
        fig = Figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(data_x, data_y, marker="o", linestyle='-', color='blue')
        ax.plot(data_x, data_y_avg, linestyle='--', color='red')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Score')
        ax.legend(['Frame Score', 'Average Score'])
        print("Saving Graph")
        fig.savefig("static/uploads/stride-sense-analysis.png", format="png")
        return str(self.avg), self.problems, data_y
    