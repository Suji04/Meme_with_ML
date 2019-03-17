'''
# for videos
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


cap = cv2.VideoCapture(0)
glass = cv2.imread('glasses.png',-1)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    original = img
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(220,130,50),2)
        glass = cv2.resize(glass,(w,h//4))
        y_= y+h//4
    
        #img[y_:y_+glass.shape[0], x:x+glass.shape[1]] = glass
    
        alpha_s = glass[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        y1, y2 = y_, y_ + glass.shape[0]
        x1, x2 = x, x + glass.shape[1]
        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha_s * glass[:, :, c] +
                                    alpha_l * img[y1:y2, x1:x2, c])
        
    
    #cv2.imshow('img',img)
    cv2.imshow("img",original)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''


# for images


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
    #cv2.rectangle(img,(x,y),(x+w,y+h),(220,130,50),2)
    glass = cv2.resize(glass,(w,h//4))
    y_= y+h//4
    
    #img[y_:y_+glass.shape[0], x:x+glass.shape[1]] = glass
    
    alpha_s = glass[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    y1, y2 = y_, y_ + glass.shape[0]
    x1, x2 = x, x + glass.shape[1]
    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha_s * glass[:, :, c] +
                                alpha_l * img[y1:y2, x1:x2, c])
        
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    print(eyes)
    #for (ex,ey,ew,eh) in eyes:
       #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(30,90,200),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
