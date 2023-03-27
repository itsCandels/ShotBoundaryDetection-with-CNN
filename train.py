#PACKAGE
import matplotlib
matplotlib.use("Agg")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC

#VARIABLE
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=25,
	help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())


LABELS = set(["DOCUMENTARI","EVENTI RELIGIOSI","GAMESHOW","TALK_SHOW","TELEVENDITE"])
print("[INFO] loading images...")
ListImage = list(paths.list_images('CNN_DATASET'))
dataNum = []
labels = []


for imagePath in ListImage:

	print(imagePath)
	label = imagePath.split(os.path.sep)[-2]
	if label not in LABELS:
		continue
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	dataNum.append(image)
	labels.append(label)
dataNum = np.array(dataNum)
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

trainX, testX, trainY, testY = train_test_split(dataNum, labels,
	test_size=0.25, stratify=labels, random_state=42)

AugMentationTrain = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
valAug = ImageDataGenerator()
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
AugMentationTrain.mean = mean
valAug.mean = mean

#NETWOTK 

baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224,224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for level in baseModel.layers:
	level.trainable = False
    
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


print("[INFO] HEAD TRAINING")
H = model.fit(
	x=AugMentationTrain.flow(trainX, trainY, batch_size=32),
	steps_per_epoch=len(trainX) // 32,
	validation_data=valAug.flow(testX, testY),
	validation_steps=len(testX) // 32,
	epochs=args["epochs"])


print("[INFO] EVALUATING")
predictions = model.predict(x=testX.astype("float32"), batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

clf=SVC(random_state=42)
clf.fit(trainX,trainY)
plot_confusion_matrix(clf,testX,testY)
plt.show

#PLOT
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('model/plot.png')
print("[INFO] NETWORK VALUE...")
model.save('model/tr.model', save_format="h5")
f = open('model/lb.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()
