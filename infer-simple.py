# load config
import json
with open('Roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

    overlap_threshold = config["OVERLAP"]
    confidence_threshold = config["CONFIDENCE"]

import cv2
import base64
import numpy as np
import time
import sys
import requests
import hashlib
from skimage.metrics import structural_similarity as ssim

#ROBOFLOW_URL = "https://detect.roboflow.com/"
#ROBOFLOW_URL = "http://192.168.0.117:9001/"
ROBOFLOW_URL = "http://192.168.43.192:9001/"

# Construct the Roboflow Infer URL
upload_url = "".join([
    ROBOFLOW_URL,
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=image",
    "&stroke=5",
    f'&overlap={overlap_threshold * 100}',
    f'&confidence={confidence_threshold * 100}',
    '&labels=True'
])

# Initialize previous image
prev_img = None
prev_img_time = None
cache = {}

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# Infer via the Roboflow Infer API and return the result
def infer():
    infer_time = time.time()
    global prev_img
    global prev_img_time

    # Get the current image from the webcam
    ret, img = video.read()
    
    # If this is the first frame or the difference with the previous frame is significant
    #if prev_img is None or ssim(img, prev_img, multichannel=True, channel_axis=2) < 0.9:
    if prev_img is None or mse(img, prev_img) > 0.7:
        # Update previous image
        prev_img = img
        prev_img_time = infer_time
     
    else:
        # If the frames are too similar, return the previous inference result
        return cache[prev_img_time]

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))
    
    # Convert the image from BGR to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).raw

    # Parse result image
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    cache[infer_time] = image

    return image


#Could also use gstreamer
# Get webcam interface via opencv-python
video = cv2.VideoCapture(0)

# Check if the camera is opened
if video.isOpened():
    print("Camera is opened.")
else:
    print("Camera is not opened.")
    sys.exit('Camera unavailable.')


# Main loop; infers sequentially until you press "q"
while 1:

    # Capture start time to calculate fps
    start = time.time()

    # Synchronously get a prediction from the Roboflow Infer API
    image = infer()
    # And display the inference results
    cv2.imshow('image', image)

    # Print frames per second
    print((1/(time.time()-start)), " fps")

    # On "q" keypress, exit
    if(cv2.waitKey(1) == ord('q')):
        break


# Release resources when finished
video.release()
cv2.destroyAllWindows()