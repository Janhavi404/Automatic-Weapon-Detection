''' 
Development of a hybrid model for Automatic Weapon Detection/Anomaly Detection in Surveillance Applications

Made by: Janhavi Jain, Shreyas Gawade, Aviral Singh Halsi, Riya Bhanwar
'''

#functions to name the anomaly and ring the alarm

def mypred(predictions):
    index=0
    max=predictions[0][0]
    for i in range (0,3):
        if predictions[0][i]>max:
            max=predictions[0][i]
            index=i

    if index==0:
        print("Knife")
        playsound('alarm.mp3')
        return 'knife'
    elif index==1:
        print("Long gun")
        playsound('alarm.mp3')
        return 'long gun'
    elif index==2:
        print("Small gun")
        playsound('alarm.mp3')
        return 'small gun'

##############################main code segment ##############################

#importing the libraries
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
x=[]
m,n = 50,50
from playsound import playsound
#playsound('alarm.mp3')
model = load_model('modell.h5')
x=[]

    
  
##########################For video detection##########################  
import cv2 
import numpy as np 
   
# Creating a VideoCapture object to read from input file 
cap = cv2.VideoCapture('gun.mp4') 
   
# Checking if camera has been opened successfully 
if (cap.isOpened()== False):  
  print("Error opening video  file") 
   
# Reading frames until video is over
while(cap.isOpened()): 
      
  # Capturing frame by frame 
  ret, frame = cap.read() 
  if ret == True: 
   
    # Displaying the resulting frame
    if type(frame) is np.ndarray:
        face = cv2.resize(frame, (m, n))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im) 
        img_array = np.expand_dims(img_array, axis=0)
    x=np.array(x)
    predictions = model.predict(x)
    mypred(predictions)
    cv2.imshow('Frame', frame) 
   
    # Press q on keyboard to  exit 
    if cv2.waitKey(25) & 0xFF == ord('q'): 
      break
   
  # Breaking the loop 
  else:  
    break
   
# releasing the video capture object after performing the function 
cap.release() 
   
# Closing all the frames 
cv2.destroyAllWindows()



#################for image detection##################################
import cv2
image=cv2.imread('k2.jpg')

font = cv2.FONT_HERSHEY_SIMPLEX 
  
org = (50, 50) 
  
fontScale = 1
   
color = (255, 0, 0) 
   
thickness = 2
x=[]

imrs = cv2.resize(image,(m,n))
imrs=img_to_array(imrs)/255
imrs=imrs.transpose(2,0,1)
imrs=imrs.reshape(3,m,n)
x.append(imrs)
x=np.array(x)
predictions = model.predict(x)
rr=mypred(predictions)
image = cv2.putText(image, rr, org, font, fontScale, color, thickness, cv2.LINE_AA)
   
# Displaying the image 
cv2.imshow('prediction', image)
cv2.waitKey(0)