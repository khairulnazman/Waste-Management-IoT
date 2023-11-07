import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import serial

# Image processing
ser = serial.Serial('COM8', 9600)

# Find the index of the external camera (try different values if needed)
external_camera_index = 0

cap = cv2.VideoCapture(external_camera_index)

# Load the Keras model from model.h5
model = load_model('model.h5')

x = 1
frameRate = cap.get(5) # frame rate
frameId = cap.get(1) # current frame number

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    filename = './test/' + str(int(x)) + ".jpg"
    x += 1
    cv2.imwrite(filename, frame)

    # Resize the image to match the model's input shape
    image = cv2.resize(frame, (224, 224))

    # Preprocess the image before passing it to the model
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image
    (nonbio, bio) = model.predict(image)[0]

    # build the label
    label = "Biodegradable" if bio > nonbio else "Non-Biodegradable"
    y = label

    print(y)
    if y == "Biodegradable":
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

    # Show the captured image in a window
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(0)  # Wait for a key press
    break  # Break the loop after displaying the captured image

cap.release()
cv2.destroyAllWindows()
