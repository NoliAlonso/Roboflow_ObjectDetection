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

import asyncio
import cv2
import base64
import numpy as np
import httpx
import time
import sys

# Construct the Roboflow Infer URL
# (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
upload_url = "".join([
    #"https://detect.roboflow.com/",
    "http://192.168.0.117:9001/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=image", # Change to json if you want the prediction boxes, not the visualization
    "&stroke=5",
    f'&overlap={overlap_threshold * 100}',
    f'&confidence={confidence_threshold * 100}',
    '&labels=True'
])


def ExitApp():
    video.release()
    cv2.destroyAllWindows()
    sys.exit('Camera closed.')


#Can also use gstreamer
# Get webcam interface via opencv-python
video = cv2.VideoCapture(0)

# Check if the camera is opened
if video.isOpened():
    print("Camera is opened.")
else:
    ExitApp()


# Infer via the Roboflow Infer API and return the result
# Takes an httpx.AsyncClient as a parameter
async def infer(requests):
    # Get the current image from the webcam
    ret, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_pixel_value = np.average(gray)

    # Check if the image is mostly black or white
    if avg_pixel_value < 50: 
        print("Skipping inference since the image is mostly black")
        return img
    elif avg_pixel_value > 200: 
        print("Skipping inference since the image is mostly white")
        return img
    
# Convert the image from BGR to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = await requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    })

    # Parse result image
    image = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

# Main loop; infers at FRAMERATE frames per second until you press "q"
async def main():
    # Initialize
    last_frame = time.time()

    # Initialize a buffer of images
    futures = []

    async with httpx.AsyncClient() as requests:
        while 1:
            # On "q" keypress, exit
            if(cv2.waitKey(1) == ord('q')):
                break

            # Throttle to FRAMERATE fps and print actual frames per second achieved
            elapsed = time.time() - last_frame
            await asyncio.sleep(max(0, 1/FRAMERATE - elapsed))
            print((1/(time.time()-last_frame)), " fps")
            last_frame = time.time()

            # Enqueue the inference request and safe it to our buffer
            task = asyncio.create_task(infer(requests))
            futures.append(task)

            # Wait until our buffer is big enough before we start displaying results
            if len(futures) < BUFFER * FRAMERATE:
                continue

            # Remove the first image from our buffer
            # wait for it to finish loading (if necessary)
            try:
                image = await futures.pop(0)
                # And display the inference results
                cv2.imshow('image', image)
            except Exception as e:
                print(f"An exception in image await occurred: {e}")

# Run our main loop
try:
    asyncio.run(main())
except Exception as e:
    print(f"An exception occurred: {e}")
    # Handle the exception or continue with other tasks

ExitApp()
