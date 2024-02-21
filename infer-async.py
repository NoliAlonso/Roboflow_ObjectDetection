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
import skimage
from skimage import io, filters
from skimage.metrics import structural_similarity as ssim


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

prev_img = None
prev_result = None

#Can also use gstreamer
# Get webcam interface via opencv-python
video = cv2.VideoCapture(0)

# Infer via the Roboflow Infer API and return the result
# Takes an httpx.AsyncClient as a parameter
async def infer(requests):
    global prev_img, prev_result
    # Get the current image from the webcam
    ret, img = video.read()

    # Check if the image is mostly black
    if np.mean(img) < 30:  # You can adjust this threshold as needed
        print("Skipping inference because the image is mostly black")
        return img

     # Check if the image is similar to the previous one
    if prev_img is not None:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        s = ssim(img_gray, prev_img_gray)

        if s > 0.75:  # You can adjust this threshold as needed
            print("Skipping inference because the image is similar to the previous one")
            return prev_result

    prev_img = img  # Update the previous image

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

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

    prev_result = image  # Store the result

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

# Release resources when finished
video.release()
cv2.destroyAllWindows()
