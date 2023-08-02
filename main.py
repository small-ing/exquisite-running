#With correct python version, run 
# pip install -r requirements.txt

import cv2
from landmark import Tracker

# Path: main.py
camera = cv2.VideoCapture(0)
landmark_tracker = Tracker(mode='IMAGE')

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)




while True:
    success, frame = camera.read()
    #try:
     #   success, frame = camera.read()
    #except:
     #   print("There is no camera")
      #  break
    
    #insert any model calls here
    frame = landmark_tracker.detect_and_draw_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()