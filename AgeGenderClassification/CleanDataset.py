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

def clean_UTK_dataset(age_range=(1,90)):

    # Check data set type
    path_to_dataset = os.path.join(os.path.dirname(
        __file__), "../Data/UTK_AgeGender")
    path_to_new_dataset = os.path.join(os.path.dirname(
        __file__), "../Data/new_UTK_AgeGender")

    # Loop over directories and get Images
    count = 0
    for i, fn in enumerate(os.listdir(path_to_dataset)):

        # Extract label
        image_name_split = fn.split('_')
        age = int(image_name_split[0])

        # Skip if outside of range
        if age < age_range[0] or age > age_range[1]:
            continue

        # Increase count
        count += 1

        # Extract path
        path = os.path.join(path_to_dataset, fn)
        img = cv2.imread(path)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(img_grey, scaleFactor=1.05, minNeighbors=5)
        
        # If there are faces copy the image
        if len(faces) > 0:
            x,y,w,d = faces[0]
            img = img[y:y+d,x:x+w]
            cv2.imwrite(os.path.join(path_to_new_dataset, fn), img)

        # Show progress for debugging purposes
        if count % 500 == 0:
            print(
                F"UTK Dataset: Finished Reading {count}")

def clean_FGNET_dataset():

    # Check data set type
    path_to_dataset = os.path.join(os.path.dirname(
        __file__), "../Data/FGNET_Age")
    path_to_new_dataset = os.path.join(os.path.dirname(
        __file__), "../Data/new_FGNET_Age")

    # Loop over directories and get Images
    count = 0
    for i, fn in enumerate(os.listdir(path_to_dataset)):
        # Increase count
        count += 1

        # Extract path
        path = os.path.join(path_to_dataset, fn)
        img = cv2.imread(path)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(img_grey, scaleFactor=1.05, minNeighbors=5)
        
        # If there are faces copy the image
        if len(faces) > 0:
            x,y,w,d = faces[0]
            img = img[y:y+d,x:x+w]
            cv2.imwrite(os.path.join(path_to_new_dataset, fn), img)

        # Show progress for debugging purposes
        if count % 500 == 0:
            print(
                F"FGNET Dataset: Finished Reading {count}")


if "__main__" :
    # clean_Gender_Kaggle_dataset("Training")
    # clean_UTK_dataset(age_range=(5,80))
    clean_FGNET_dataset()