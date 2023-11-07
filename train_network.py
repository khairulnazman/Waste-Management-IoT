
#python train_network.py --dataset "C:/Users/King/Documents/Studies/Degree/Sem 6/CSP650/Arduino/WasteSeggregate-vgg/data" --model "C:/Users/King/Documents/Studies/Degree/Sem 6/CSP650/Arduino/WasteSeggregate-vgg"

# set the matplotlib backend so figures can be saved in the background


import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.layers import Input, GlobalAveragePooling2D, Dense
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
from keras.models import Model
import pandas as pd
from keras.callbacks import ReduceLROnPlateau


# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 30
INIT_LR = 1e-3
BS = 64

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images("C:/Users/King/Documents/Studies/Degree/Sem 6/CSP650/Arduino/WasteSeggregate-vgg/data")))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (224, 224))  # Resize images to (224, 224)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "bio" else 0
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
input_tensor = Input(shape=(224, 224, 3))  # Update input shape to (224, 224, 3)
base_model = VGG16(weights="imagenet", include_top=False, input_tensor=input_tensor)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation="relu")(x)
output = Dense(2, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS),
              loss="categorical_crossentropy", metrics=["accuracy"])

# Define the learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1, callbacks=[reduce_lr])

# save the model to disk
print("[INFO] serializing network...")
model.save("C:/Users/King/Documents/Studies/Degree/Sem 6/CSP650/Arduino/WasteSeggregate-vgg/knz.model")

# save the epoch results in a table
epoch_results = pd.DataFrame(H.history)
epoch_results.to_csv("C:/Users/King/Documents/Studies/Degree/Sem 6/CSP650/Arduino/WasteSeggregate-vgg/epoch_results.csv", index=False)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("C:/Users/King/Documents/Studies/Degree/Sem 6/CSP650/Arduino/WasteSeggregate-vgg/plot.png")
