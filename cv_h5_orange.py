import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import serial

# Image processing
ser = serial.Serial('COM8', 9600)

# Try different camera indexes to find the external camera
external_camera_index = None
for index in range(10):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Using Camera Index: {index}")
        external_camera_index = index
        break
    cap.release()

if external_camera_index is None:
    print("Failed to find an external camera.")
    exit(1)

cap = cv2.VideoCapture(external_camera_index)
model = load_model('model.h5')  # Use model.h5 instead of neural.model
x = 1
frameRate = cap.get(5)  # frame rate
frameId = cap.get(1)  # current frame number

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    filename = './test/' + str(int(x)) + ".jpg"
    x += 1
    cv2.imwrite(filename, frame)

    # Resize the frame to the size expected by the model (224x224)
    image = cv2.resize(frame, (224, 224))
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

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
