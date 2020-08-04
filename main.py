
from flask import Flask, render_template, Response
from camera import SocialCamera, MaskCamera, ThermalCamera
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mask-scan')
def mask():
    return render_template('mask.html')

@app.route('/social-distance')
def social():
    return render_template('social.html')
@app.route('/thermalscan')
def thermal():
    return render_template('thermal.html')


        
def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None: continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/mask_feed')
def mask_feed():
    return Response(gen(MaskCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/social_feed')
def social_feed():
    return Response(gen(SocialCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/thermal_feed')
def thermal_feed():
    return Response(gen(ThermalCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run( debug=True)
