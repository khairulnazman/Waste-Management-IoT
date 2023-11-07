import numpy as np
import math
import cv2
import serial
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import urllib.request

#Image processing
ser = serial.Serial('COM8', 9600)

# Find the index of the external camera (try different values if needed)
external_camera_index = 0

cap = cv2.VideoCapture(external_camera_index)
model = load_model('neural.model')
x = 1
frameRate = cap.get(5) #frame rate
frameId = cap.get(1) #current frame number

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    filename = './test/' +  str(int(x)) + ".jpg"
    x += 1
    cv2.imwrite(filename, frame)
    
    image = cv2.resize(frame, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    # classify the input image
    (nonbio, bio) = model.predict(image)[0]
    
    # build the label
    label = "Non-Biodegradable" if bio > nonbio else "Biodegradable"
    y = label

    print(y)
    if y == "Non-Biodegradable":
        a = "a"
        a = a.encode()
        ser.write(a)
    else:
        b = "b"
        b = b.encode()
        ser.write(b)

    # Put the label text on the image
    label_text = f"Type: {y}"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the image in a window
    cv2.imshow("Captured Image", frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
