import cv2
from tensorflow_infer import run_on_video
from social import social
from thermal import thermal
ds_factor=0.6
#fourcc = cv2.VideoWriter_fourcc(*'XVID') 
#video_writer = cv2.VideoWriter("output.avi", fourcc, 20, (680, 480))

class MaskCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        #self.video.set(cv2.CAP_PROP_POS_FRAMES, 50)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        while 1:
            success, image = self.video.read()
            if not success: continue
            image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
            image=run_on_video(self.video)
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
class SocialCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture('videos/video.mp4')
        #self.video.set(cv2.CAP_PROP_POS_FRAMES, 50)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        while 1:
            success, image = self.video.read()
            if not success: continue
            image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
            image=social(image)
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
class ThermalCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture('videos/video.mp4')
        #self.video.set(cv2.CAP_PROP_POS_FRAMES, 50)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        while 1:
            success, image = self.video.read()
            if not success: continue
            image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
            image=thermal()
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
       

