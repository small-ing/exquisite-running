#With correct python version, run 
# pip install -r requirements.txt

import cv2
from landmark import Tracker

# Path: main.py
cap = cv2.VideoCapture("data/good/istockphoto.mp4")
landmark_tracker = Tracker(mode='IMAGE')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
    try:
        success, frame = cap.read()
    except:
        print("There is no camera")
        break
    
    #insert any model calls here
    frame = landmark_tracker.detect_and_draw_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows()