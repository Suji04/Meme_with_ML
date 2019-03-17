import numpy as np
import cv2
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

glass = cv2.imread('glasses.png',-1) # png images have alpha channel
img = cv2.imread('1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA) # jpg images don't have alpha channel
img = cv2.resize(img,(600,600))
gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

for (x,y,w,h) in faces:
    glass = cv2.resize(glass,(w,h//4))
    y_= y+h//4
    alpha_s = glass[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    y1, y2 = y_, y_ + glass.shape[0]
    x1, x2 = x, x + glass.shape[1]
    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha_s * glass[:, :, c] +
                                alpha_l * img[y1:y2, x1:x2, c])

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
