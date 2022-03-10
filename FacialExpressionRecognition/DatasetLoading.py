import os
import cv2
import FeaturesExtraction

def load_CK_dataset():
    """Loads CK+ Dataset

    Returns:
        Tuple(Features, Labels): Features and Labels of the dataset
    """
    return load_dataset("../Data/CK+48")

def load_FER_dataset():
    """Loads FER Dataset

    Returns:
        Tuple(Features, Labels): Features and Labels of the dataset
    """
    return load_dataset("../Data/FER2013")

def load_dataset(path: str):
    """Loads any Dataset with given path

    Args:
        path (string): relative path to the dataset
    Returns:
        Tuple(Features, Labels): Features and Labels of the dataset
    """

    # Initialize Variables
    features = []
    labels = []
    path_to_dataset = os.path.join(os.path.dirname(
        __file__), path)
    directoriesNames = os.listdir(path_to_dataset)
    featureExtractor = FeaturesExtraction.FeatureExtractor()
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
            features.append(featureExtractor.ExtractHOGFeatures(img))

            # Show progress for debugging purposes
            if count % 500 == 0:
                print(F"Finished Reading {count}")
    print(F"Finished Reading {count}")
    print("Apply Principal component analysis (PCA) on Features")
    D_before = len(features[0])
    features = featureExtractor.TrainPCAonFeatures(features, 900)
    print(F"Finished dimensionality reduction Before: {D_before} => After: {len(features[0])}")
    return features, labels
