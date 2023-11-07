import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import serial
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Image processing
ser = serial.Serial('COM8', 9600)

# Find the index of the external camera (try different values if needed)
external_camera_index = 0

cap = cv2.VideoCapture(external_camera_index)

# Load the Keras model from model.h5
model = load_model('model.h5')

def capture_image():
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to grab frame")
        return

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
    cv2.putText(frame, f"Type: {y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Convert the image to RGB format for displaying with tkinter
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    # Update the image on the label in the GUI window
    label_img.config(image=img_tk)
    label_img.image = img_tk

    # Update the label with the classification result
    label_result.config(text=f"Type: {y}")

# Create the GUI window
root = tk.Tk()
root.title("Capture and Process Image")
root.geometry("800x480")

# Create a label to display the captured image
label_img = tk.Label(root)
label_img.pack(side=tk.LEFT)

# Create a frame to hold the processed image and classification result
result_frame = tk.Frame(root, width=400, height=480)
result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Create a label to display the classification result
label_result = tk.Label(result_frame, text="", font=("Helvetica", 16))
label_result.pack()

# Create a "Capture Image" button
capture_btn = tk.Button(result_frame, text="Capture Image", command=capture_image)
capture_btn.pack()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
