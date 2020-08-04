
from imutils import face_utils 
import flirimageextractor
import numpy as np 
import argparse 
import imutils 
import dlib 
import cv2 

src='TD_IR_E_5.jpg'
temps=0
def thermal(image=src):
    global temps
    try:
        flir = flirimageextractor.FlirImageExtractor()
        flir.process_image(image) 
        temps = flir.get_thermal_np()
    except: pass

    detector = dlib.get_frontal_face_detector() 
   


    image = cv2.imread(image) 
    image = imutils.resize(image, width = 500) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    rects = detector(gray, 1) 

    rect=rects[0] # first person
    
  
    (x, y, w, h) = face_utils.rect_to_bb(rect) 
    cv2.rectangle(image, (x+x//2, y), (x + w//3, y + h//6), (0, 0, 255), 2) 
    if len(temps)!=0: f=(np.matrix(temps[y:y+h//6,x:x+w//3]).mean())
    else : f=0
    cv2.putText(image,str(int(f))+" F", (x+x//2, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1) 
    return image

