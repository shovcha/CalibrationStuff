# import the opencv library
import matplotlib
import cv2
from cv2 import *
  
  
# define a video capture object
vidR = cv2.VideoCapture('rtsp://10.6.10.162/live_stream')
vidL = cv2.VideoCapture('rtsp://10.6.10.161/live_stream')
  
resultR, imageR = vidR.read()
resultL, imageL = vidL.read()
  
# If image will detected without any error, 
# show result
if resultL:
  
    # showing result, it take frame name and image 
    # output
    #imshow("GeeksForGeeks", image)
  
    # saving image in local storage
    cv2.imwrite("oneemR.png", imageR)
    cv2.imwrite("oneemL.png", imageL)
  
    # If keyboard interrupt occurs, destroy image 
    # window
    waitKey(0)
    #destroyWindow("GeeksForGeeks")
  
# If captured image is corrupted, moving to else part
else:
    print("No image detected. Please! try again")