import cv2
import time
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
camera = cv2.VideoCapture(0)
#initializin picamera
#camera = PiCamera()
#camera.resolution = (640, 480)
#camera.framerate = 32
#rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)
rec = cv2.face.createLBPHFaceRecognizer();
rec.load("recognizer/traindata.yml")
id=0
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
     ret,im = camera.read();
#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)
#     image = frame.array
     gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
     faces = faceDetect.detectMultiScale(gray,1.3,5);
     for(x,y,w,h) in faces:
          cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,170),2)
          id=rec.predict(gray[y:y+h,x:x+w])
               
          if(id==1):
               id="person"
               cv2.imwrite("detected.jpg",gray[y:y+h,x:x+w])
          elif(id==2):
               id="AWAIS"
               cv2.imwrite("detected.jpg",gray[y:y+h,x:x+w])
          elif(id==3):
               id="person1"
               cv2.imwrite("detected.jpg",gray[y:y+h,x:x+w])
          elif(id==4):
               id="waqas"
               cv2.imwrite("detected.jpg",gray[y:y+h,x:x+w])
          elif(id==5):
               id="Tayyab"
               cv2.imwrite("detected.jpg",gray[y:y+h,x:x+w])
          elif(id==6):
               id="Atif"
          elif(id==7):
               id="Atif"
               cv2.imwrite("detected.jpg",gray[y:y+h,x:x+w])
          cv2.putText(im,str(id),(x,y+h),font,0.55,(0,255,0),2)
     cv2.imshow("Frame",im);
     key = cv2.waitKey(1) & 0xFF
	 #rawCapture.truncate(0)
     if cv2.waitKey(1) == ord("q"):
		break      

camera.release()
cv2.destroyAllWindows()
