import numpy as np
import cv2
import dlib
import numpy as np
import os
import Utils

arr = np.load('0_exp.npy')
print(arr)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

path_to_file = os.path.join(os.path.dirname(
    __file__), "../../Test/disgust.jpg")

image = cv2.imread(path_to_file)
# Convert the image color to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect the face
rects = detector(gray, 1)
# Detect landmarks for each face
for rect in rects:
    # Get the landmark points
    shape = predictor(gray, rect)
# Convert it to the NumPy Array
    shape_np = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)
    shape = shape_np

    # Display the landmarks
    for i, (x, y) in enumerate(shape):
    # Draw the circle to mark the keypoint 
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    
# Display the image
Utils.show_image(image, 'Landmark Detection')