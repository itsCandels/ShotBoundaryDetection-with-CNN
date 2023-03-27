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

start=time.time()

#CNN_PARAMETER

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
ap.add_argument("-p", "--min-percent", type=float, default=1.0,
	help="lower boundary of percentage of motion")
ap.add_argument("-d", "--max-percent", type=float, default=10.0,
	help="upper boundary of percentage of motion")
ap.add_argument("-w", "--warmup", type=int, default=200,
	help="# of frames to use to build a reasonable background model")
ap.add_argument("-v", "--input", required=True,
	help="# input video")

args = vars(ap.parse_args())


print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])




fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()



captured = False
total = 0
frames = 0

count = 0
count2=0


vs = cv2.VideoCapture(args["input"])


width= int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = vs.get(cv2.CAP_PROP_FPS)
frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
m=frame_count-int(fps*2)
print(fps)
print(frame_count)
(W, H) = (None, None)

change_frame = 0





h=[]
m=[]
s=[]
ms=[]
lista_csv=[]
listA=[]
cambio=None


while True:
    (grabbed, frame) = vs.read()
    change_frame +=1
    if frame is None:
        break
    orig = frame.copy()
    frame = imutils.resize(frame, width=600)
    mask = fgbg.apply(frame)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    if W is None or H is None:
        (H, W) = mask.shape[:2]

	
    milliseconds = vs.get(cv2.CAP_PROP_POS_MSEC)

    seconds = int(milliseconds//1000)
    milliseconds =milliseconds%1000
    minutes = 0
    hours = 0
    
    if hours >= 60:
        minutes = int(hours//60)
        hours = int(hours % 60)

    if seconds >= 60:

        minutes =int(seconds//60)
        seconds = int(seconds % 60)

    if minutes >= 60:
        hours = int(minutes//60)
        minutes = int(minutes % 60)
        
    ore=(f'{hours:02}')
    minuti=(f'{minutes:02}')
    secondi=(f'{seconds:02}')
        
    timestamp=(f'{hours:02}'+':'+f'{minutes:02}'+':'+f'{seconds:02}')
    p = (cv2.countNonZero(mask) / float(W * H)) * 100
    
    start_time=time.time() 

    if p < args['min_percent']  and not captured and frames > int(fps*2):
        
        ret, nframe=vs.read()
        
        nframe=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        nframe = cv2.resize(frame, (224,224)).astype("float32")
        nframe -= mean
        
        
        preds = model.predict(np.expand_dims(nframe, axis=0))[0]
        
        Q.append(preds)
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = lb.classes_[i]
        if lb.classes_[i] !=cambio:
            cambio=lb.classes_[i]
        else:
            label=None
        
    
        
        count+=1     
        
        u=(('Scena_N'+str(count))+' '+timestamp +' '+str(label))
        print(u)                            
        listA.append(u)
        captured = True       
    elif captured and p >= args["max_percent"]:       

        captured = False
    key = cv2.waitKey(1) & 0xFF
    if key == ord("m"):
        break
    frames += 1
#CSV CREATION

vs.release()
lista = pd.DataFrame(listA)
lista.to_csv(r'output/result.csv',mode='a', header=False, index=False)    

#END TIME
elapsed=time.time()-start
output=dt.strftime(dt.utcfromtimestamp(elapsed), '%H:%M:%S')
print("TEMPOELABORAZIONE:",output)