import cv2
import landmark

# Path: main.py
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



while True:
    try:
        success, frame = camera.read()
    except:
        print("There is no camera")
        break
    
    #insert any model calls here

    
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()