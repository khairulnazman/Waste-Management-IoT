import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import serial

# Image processing
ser = serial.Serial('COM8', 9600)

# Specify the index of the external camera (change this to the correct index)
external_camera_index = 0  # Replace 1 with the correct index

cap = cv2.VideoCapture(external_camera_index)
if not cap.isOpened():
    print(f"Failed to open the external camera at index {external_camera_index}.")
    exit(1)

model = load_model('model.h5')  # Use model.h5 instead of neural.model
frameRate = cap.get(5)  # frame rate

# Set video resolution (adjust if needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Width (e.g., 640, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Height (e.g., 480, 720)

while True:
    # Capture the frame directly without buffering
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Failed to grab frame")
        break

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
