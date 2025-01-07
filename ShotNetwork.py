# import the necessary packages
import argparse
import imutils
import time
import cv2

import numpy as np
from datetime import datetime as dt

import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from collections import deque

# Record the start time of the script
start = time.time()

# ----------------------------------------------------------------------
# CNN PARAMETERS
# ----------------------------------------------------------------------
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True,
    help="Path to the trained and serialized model.")
ap.add_argument("-l", "--label-bin", required=True,
    help="Path to the label binarizer.")
ap.add_argument("-s", "--size", type=int, default=128,
    help="Size of the queue used for averaging predictions.")
ap.add_argument("-p", "--min-percent", type=float, default=1.0,
    help="Lower boundary for the percentage of motion.")
ap.add_argument("-d", "--max-percent", type=float, default=10.0,
    help="Upper boundary for the percentage of motion.")
ap.add_argument("-w", "--warmup", type=int, default=200,
    help="Number of frames to build a reasonable background model.")
ap.add_argument("-v", "--input", required=True,
    help="Path to the input video.")

args = vars(ap.parse_args())

# ----------------------------------------------------------------------
# Load the model and the label binarizer
# ----------------------------------------------------------------------
print("[INFO] Loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# Mean values for mean subtraction
mean = np.array([123.68, 116.779, 103.939], dtype="float32")

# Queue (deque) for predictions averaging
Q = deque(maxlen=args["size"])

# Initialize the background subtractor (GMG)
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

# Flags and counters
captured = False
total = 0
frames = 0
count = 0
count2 = 0

# Capture the video
vs = cv2.VideoCapture(args["input"])

# Retrieve video properties
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vs.get(cv2.CAP_PROP_FPS)
frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
m = frame_count - int(fps * 2)

print(fps)
print(frame_count)

(W, H) = (None, None)

change_frame = 0

# Lists for time logging, CSV export, and tracking changes
h = []
m_ = []  # Renamed to avoid shadowing the variable 'm'
s = []
ms = []
lista_csv = []
listA = []
cambio = None

# ----------------------------------------------------------------------
# Main loop for reading frames
# ----------------------------------------------------------------------
while True:
    (grabbed, frame) = vs.read()
    change_frame += 1
    
    # If there is no frame, we have reached the end of the video
    if frame is None:
        break
    
    # Make a copy of the original frame
    orig = frame.copy()
    
    # Resize the frame for background subtraction
    frame = imutils.resize(frame, width=600)
    
    # Apply the background subtractor and morphological operations
    mask = fgbg.apply(frame)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Set the width and height if they have not been defined yet
    if W is None or H is None:
        (H, W) = mask.shape[:2]

    # Retrieve the current timestamp in milliseconds from the video
    milliseconds = vs.get(cv2.CAP_PROP_POS_MSEC)

    # Convert the milliseconds to hh:mm:ss (only integer seconds here)
    seconds = int(milliseconds // 1000)
    milliseconds = milliseconds % 1000
    minutes = 0
    hours = 0

    # Convert to hours, minutes, seconds if needed
    if hours >= 60:
        minutes = int(hours // 60)
        hours = int(hours % 60)

    if seconds >= 60:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)

    if minutes >= 60:
        hours = int(minutes // 60)
        minutes = int(minutes % 60)
    
    # Format timestamps as HH:MM:SS
    ore = f"{hours:02}"
    minuti = f"{minutes:02}"
    secondi = f"{seconds:02}"
    timestamp = f"{ore}:{minuti}:{secondi}"

    # Compute the percentage of foreground pixels
    p = (cv2.countNonZero(mask) / float(W * H)) * 100
    
    start_time = time.time()

    # Check if the percentage of motion is below the min threshold
    # and if we have not captured a frame yet, and enough frames have passed
    if p < args['min_percent'] and not captured and frames > int(fps * 2):
        
        # Read one more frame from the video (for consistency with background subtractor state)
        ret, nframe = vs.read()
        
        # Convert and resize the frame before feeding into the CNN
        nframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nframe = cv2.resize(frame, (224, 224)).astype("float32")
        nframe -= mean

        # Make a prediction using the CNN
        preds = model.predict(np.expand_dims(nframe, axis=0))[0]

        # Append the prediction to the queue for averaging
        Q.append(preds)
        results = np.array(Q).mean(axis=0)
        
        # Determine the class label
        i = np.argmax(results)
        label = lb.classes_[i]
        
        # Check if there is a change in the detected label
        if lb.classes_[i] != cambio:
            cambio = lb.classes_[i]
        else:
            label = None
        
        count += 1
        
        # Prepare a string to log scene information
        u = (("Scene_N" + str(count)) + " " + timestamp + " " + str(label))
        print(u)
        
        listA.append(u)
        
        # Mark that we have captured a frame
        captured = True
    
    # Once the percentage of motion goes above the max threshold, we allow capturing again
    elif captured and p >= args["max_percent"]:
        captured = False

    # Check for a keypress; 'm' to break out of the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("m"):
        break
    
    # Increase the frame counter
    frames += 1

# ----------------------------------------------------------------------
# CSV CREATION
# ----------------------------------------------------------------------
vs.release()
lista = pd.DataFrame(listA)
lista.to_csv(r'output/result.csv', mode='a', header=False, index=False)

# ----------------------------------------------------------------------
# END TIME
# ----------------------------------------------------------------------
elapsed = time.time() - start
output = dt.strftime(dt.utcfromtimestamp(elapsed), '%H:%M:%S')
print("PROCESSING TIME:", output)
