import os
import cv2
import FeaturesExtraction
import Preprocessing

def load_Gender_Kaggle_dataset(selected_feature="LPQ", type="Validation"):
    """Loads the Kaggle Gender Dataset

    Args:
        selected_feature (str, optional): selected feature for extraction if None feature List is returned empty. Defaults to "LPQ".
        type (str, optional): Load Either Training or Validation Datasets. Defaults to "Validation".

    Returns:
        Tuple(features, labels, image_paths): Features, Labels, and Image Paths of the dataset
    """

    # Check data set type
    path_to_dataset = os.path.join(os.path.dirname(
        __file__), "../Data/Gender_Kaggle/Validation")
    if type == "Training":
        path_to_dataset = os.path.join(os.path.dirname(
            __file__), "../Data/Gender_Kaggle/Training")

    # Initialize Variables
    features = []
    labels = []
    image_paths = []
    directoriesNames = os.listdir(path_to_dataset)

    # Loop over directories and get Images
    count = 0
    for directory in directoriesNames:
        for i, fn in enumerate(os.listdir(os.path.join(path_to_dataset, directory))):
            # Increase count
            count += 1

            # Add Label
            labels.append(directory)

            # Extract path
            path = os.path.join(path_to_dataset, directory, fn)
            image_paths.append(path)

            # Extract Image features
            if selected_feature != None:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = Preprocessing.preprocess_image(img)
                features.append(FeaturesExtraction.extract_features(
                    img, feature=selected_feature))

            # Show progress for debugging purposes
            if count % 500 == 0:
                print(
                    F"Gender Kaggle {type} Dataset: Finished Reading {count}")

    return features, labels, image_paths


def load_UTK_AgeGender_dataset(selected_feature="LPQ", label="gender", age_range=(1, 90)):
    """Loads the UTK AgeGender Dataset

    Args:
        selected_feature (str, optional): selected feature for extraction if None feature List is returned empty. Defaults to "LPQ".
        label (str, optional): label type for the data. Defaults to "gender".
        age_range (tuple, optional): max and min ages. Defaults to (1,90).

    Raises:
        Exception: Wrong label is supplied
        Exception: Wrong age range is supplied

    Returns:
        Tuple(features, labels, image_paths): Features, Labels, and Image Paths of the dataset
    """
    # Args checking
    if label != "gender" and label != "age":
        raise Exception("Wrong Parameter For label argument")
    if type(age_range) != tuple or age_range[0] < 1 or age_range[1] > 90:
        raise Exception("Wrong Parameter For age_range argument")

    # Set path to dataset
    path_to_dataset = os.path.join(os.path.dirname(
        __file__), "../Data/UTK_AgeGender")

    # Initialize Variables
    features = []
    labels = []
    image_paths = []

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

        # Add Label
        if label == "gender":
            gender = "male" if image_name_split[1] == "0" else "female"
            labels.append(gender)
        elif label == "age":
            labels.append(age)

        # Get Image path
        path = os.path.join(path_to_dataset, fn)
        image_paths.append(path)

        # Extract Image features
        if selected_feature != None:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = Preprocessing.preprocess_image(img)
            features.append(
                FeaturesExtraction.extract_features(img, feature=selected_feature))

        # Show progress for debugging purposes
        if count % 500 == 0:
            print(F"UTK Gender_Age Dataset: Finished Reading {count}")

    return features, labels, image_paths
