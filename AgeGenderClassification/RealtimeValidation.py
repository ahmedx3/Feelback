import cv2
import os
import numpy as np
import sys
import pickle
import time
import Preprocessing
import FeaturesExtraction
import Utils

# Read the model
face_detector = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), 'Experiments/haarcascade_frontalface_alt2.xml'))
model = pickle.load(open(os.path.join(os.path.dirname(__file__), "Models_Gender/Kaggle_Tra_SVM_LPQ_87_86.model"), 'rb'))

# Capture the video from webcam
video = cv2.VideoCapture(0)
  
while(True):
      
    # Get the video frame
    ret, frame = video.read()

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(frame_grey, scaleFactor=1.3, minNeighbors=5)

    for i in range(len(faces)):
        x, y, w, h = faces[i]
        img = frame_grey[y:y + h, x:x + w]
        img = Preprocessing.preprocess_image(img)

        # Calculate time before processing
        start_time = time.time()

        img_features = FeaturesExtraction.extract_features(img, feature="LPQ")
        predicted = model.predict([img_features])[0]

        # Calculate time after processing in seconds
        end_time = time.time()
        print("[INFO] Time taken is {:.5f} seconds".format(end_time - start_time))

        # Utils.show_image(img, predicted)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 150), 2)
        cv2.putText(frame, f"{predicted}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 200, 0), 2)
  
    # Display the resulting frame
    cv2.imshow('Webcam', frame)
      
    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  

# Release the video and remove window
video.release()
cv2.destroyAllWindows()