import os
import cv2
import FeaturesExtraction

def load_Gender_Kaggle_dataset():
    """Loads the Kaggle Gender Dataset

    Returns:
        Tuple(Features, Labels): Features and Labels of the dataset
    """

    # Initialize Variables
    features = []
    labels = []
    path_to_dataset = os.path.join(os.path.dirname(
        __file__), "../Data/Gender_Kaggle/Validation")
    directoriesNames = os.listdir(path_to_dataset)

    # Loop over directories and get Images
    count = 0
    for directory in directoriesNames:
        for i, fn in enumerate(os.listdir(os.path.join(path_to_dataset, directory))):
            # Increase count
            count += 1

            # Add Label
            labels.append(directory)

            # Extract Image features
            path = os.path.join(path_to_dataset, directory, fn)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            features.append(FeaturesExtraction.extractLPQ(img))

            # Show progress for debugging purposes
            if count % 500 == 0:
                print(F"Finished Reading {count}")

    return features, labels
