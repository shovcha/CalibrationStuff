import cv2
import numpy as np
from skimage.util import random_noise
 
# Load an image
im_arr = cv2.imread("fishLL.png")
im_arr2 = cv2.imread("fishRR.png")
# Add salt and pepper noise to the image
#noise_img = random_noise(im_arr, mode="s&p",amount=0.3)
cropped_imageL = im_arr[200:700, 800:1500]
cropped_imageR = im_arr2[200:700, 800:1500]

noise_img = np.array(255*cropped_imageL, dtype = 'uint8')
noise_img2 = np.array(255*cropped_imageR, dtype = 'uint8')
 
# Apply median filter
median = cv2.medianBlur(noise_img,5)
median2 = cv2.medianBlur(median,11)

median1 = cv2.medianBlur(noise_img2,5)
median12 = cv2.medianBlur(median1,11)
blur = cv2.boxFilter(im_arr,-1,(11,11), normalize = True)
tgb = cv2.bilateralFilter(im_arr,9,160,160)
gray = cv2.cvtColor(median2,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(median12,cv2.COLOR_BGR2GRAY)


thresh = 200
maxValue = 255



	
th, dst = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_TRUNC)
th2, dst2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_TRUNC)


th3l = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
 cv2.THRESH_BINARY,11,2)
th3r = cv2.adaptiveThreshold(gray2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
 cv2.THRESH_BINARY,11,2)


cv2.imwrite("trucL.png", dst)
cv2.imwrite("trucR.png", dst2)
# Display the image
#cv2.imshow('blur',median12)
#cv2.imshow('grey',gray)
#cv2.imshow('blur1',th3l)
#cv2.imshow('blur2',th3r)
cv2.imshow('truc',dst)
#cv2.imshow('blur2',bifilter)
cv2.waitKey(0)
cv2.destroyAllWindows()