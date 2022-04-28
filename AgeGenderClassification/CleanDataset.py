import os
import cv2


# Read the model
face_detector = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), 'Experiments/haarcascade_frontalface_alt2.xml'))

def clean_Gender_Kaggle_dataset(type="Validation"):

    # Check data set type
    path_to_dataset = os.path.join(os.path.dirname(
        __file__), "../Data/Gender_Kaggle/Validation")
    path_to_new_dataset = os.path.join(os.path.dirname(
        __file__), "../Data/new_Gender_Kaggle/Validation")
    if type == "Training":
        path_to_dataset = os.path.join(os.path.dirname(
            __file__), "../Data/Gender_Kaggle/Training")
        path_to_new_dataset = os.path.join(os.path.dirname(
            __file__), "../Data/new_Gender_Kaggle/Training")

    # Initialize Variables
    directoriesNames = os.listdir(path_to_dataset)

    # Loop over directories and get Images
    count = 0
    for directory in directoriesNames:
        for i, fn in enumerate(os.listdir(os.path.join(path_to_dataset, directory))):
            # Increase count
            count += 1

            # Extract path
            path = os.path.join(path_to_dataset, directory, fn)
            img = cv2.imread(path)
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
            faces = face_detector.detectMultiScale(img_grey, scaleFactor=1.05, minNeighbors=5)
            
            # If there are faces copy the image
            if len(faces) > 0:
                x,y,w,d = faces[0]
                img = img[y+10:y+d+30,x:x+w]
                cv2.imwrite(os.path.join(path_to_new_dataset, directory, fn), img)

            # Show progress for debugging purposes
            if count % 500 == 0:
                print(
                    F"Gender Kaggle {type} Dataset: Finished Reading {count}")


if "__main__" :
    clean_Gender_Kaggle_dataset("Training")