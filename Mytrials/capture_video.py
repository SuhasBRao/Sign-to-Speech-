import cv2
import numpy as np

cam = cv2.VideoCapture(0)
num_frames = 0

ROI_top = 100
ROI_bottom = 400
ROI_right = 300
ROI_left = 600

while True:
    ret, frame = cam.read()

    # flipping the frame to prevent inverted image of captured frame...
    
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    # ROI from the frame
    
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
    
    cv2.putText(frame_copy, "Now you can gesture", (80, 400), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                   
    cv2.imwrite('C:/Users/suhas/GIT_HUB/Sign-to-Speech-/images/roi.jpg',roi)

    # incrementing the number of frames for tracking
    num_frames += 1

    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,ROI_bottom), (255,128,0), 3)
    
    cv2.putText(frame_copy, "Sign language recognition_ _ _",
    (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    
    cv2.imshow("Sign Detection", frame_copy)
    
    
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()