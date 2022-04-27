import os
import cv2
import FeaturesExtraction
import numpy as np

def load_CK_dataset():
    """Loads CK+ Dataset

    Returns:
        Tuple(Features, Labels): Features and Labels of the dataset
    """
    # Initialize Variables
    features = []
    labels = []
    path_to_dataset = os.path.join(os.path.dirname(
        __file__), "../Data/CK+_Complete")
    directoriesNames = os.listdir(path_to_dataset)
    featureExtractor = FeaturesExtraction.FeatureExtractor()
    
    # Loop over directories and get Images
    count = 0
    for directory in directoriesNames:
        for i, fn in enumerate(os.listdir(os.path.join(path_to_dataset, directory))):
            # Increase count
            count += 1

            # Extract Image features
            path = os.path.join(path_to_dataset, directory, fn)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # feat = featureExtractor.ExtractHOGFeatures(img,(16,16))
            # feat = featureExtractor.ExtractLandMarks_method1(img)
            feat = featureExtractor.ExtractLandMarks_method2(img)
            # feat = featureExtractor.DWT(img)
            # feat = featureExtractor.GaborFilter(img)

            if feat is None:
                continue

            # Add features and label
            features.append(feat)
            labels.append(directory)

            # Show progress for debugging purposes
            if count % 500 == 0:
                print(F"Finished Reading {count}")

    print(F"Finished Reading {count}")
    
    print("Features Shape: ", len(features[0]))
    # print("Apply Principal component analysis (PCA) on Features")
    # D_before = len(features[0])
    # features = featureExtractor.TrainPCAonFeatures(features, 200)
    # print(F"Finished dimensionality reduction Before: {D_before} => After: {len(features[0])}")
    
    # Standardize Features
    features = featureExtractor.trainNormalizeFeatures(features)
    return features, labels

def load_AffectNet_dataset():
    """Loads AffectNet Dataset

    Returns:
        Tuple(Features, Labels): Features and Labels of the dataset
    """
    # Initialize Variables
    expressions = ['neutral', 'happy', 'sadness', 'surprise', 'fear', 'disgust', 'anger']
    features = []
    labels = []
    path_to_dataset = os.path.join(os.path.dirname(
        __file__), "../Data/val_set")
    featureExtractor = FeaturesExtraction.FeatureExtractor()
    img_directory = 'images'
    label_directory = 'annotations'
    # Loop over directories and get Images
    count = 0
    for i, fn in enumerate(os.listdir(os.path.join(path_to_dataset, img_directory))):
        # Increase count
        count += 1

        # Extract Image features
        path = os.path.join(path_to_dataset, img_directory, fn)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        label_file_name = fn.split('.')[0] + '_exp.npy'
        label_path = os.path.join(path_to_dataset, label_directory, label_file_name)
        label = np.int(np.load(label_path))

        if label > 6: continue

        # feat = featureExtractor.ExtractHOGFeatures(img,(16,16))
        # feat = featureExtractor.ExtractLandMarks_method1(img)
        feat = featureExtractor.ExtractLandMarks_method2(img)
        # feat = featureExtractor.DWT(img)
        # feat = featureExtractor.GaborFilter(img)

        if feat is None:
            continue

        # Add features and label
        features.append(feat)
        labels.append(expressions[label])

        # Show progress for debugging purposes
        if count % 500 == 0:
            print(F"Finished Reading {count}")

    print(F"Finished Reading {count}")
    
    print("Features Shape: ", len(features[0]))
    # print("Apply Principal component analysis (PCA) on Features")
    # D_before = len(features[0])
    # features = featureExtractor.TrainPCAonFeatures(features, 200)
    # print(F"Finished dimensionality reduction Before: {D_before} => After: {len(features[0])}")
    
    # Standardize Features
    features = featureExtractor.trainNormalizeFeatures(features)
    return features, labels


def load_test_data(path: str):
    """Loads test data with given path

    Args:
        path (string): relative path to the data
    Returns:
        Tuple(Features, Labels): Features and Labels of the data
    """

    # Initialize Variables
    features = []
    labels = []
    path_to_dataset = os.path.join(os.path.dirname(
        __file__), path)
    directoriesNames = os.listdir(path_to_dataset)
    featureExtractor = FeaturesExtraction.FeatureExtractor(True)

    # Loop over directories and get Images
    count = 0
    for directory in directoriesNames:
        for i, fn in enumerate(os.listdir(os.path.join(path_to_dataset, directory))):
            # Increase count
            count += 1

            # Extract Image features
            path = os.path.join(path_to_dataset, directory, fn)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # feat = featureExtractor.ExtractHOGFeatures(img,(16,16))
            # feat = featureExtractor.ExtractLandMarks_method1(img)
            feat = featureExtractor.ExtractLandMarks_method2(img)
            # feat = featureExtractor.DWT(img)
            # feat = featureExtractor.GaborFilter(img)

            if feat is None:
                continue
            
            # Add features and label
            features.append(feat)
            labels.append(directory)

            # Show progress for debugging purposes
            if count % 500 == 0:
                print(F"Finished Reading {count}")

    print(F"Finished Reading {count}")

    # print("Apply Principal component analysis (PCA) on Features")
    # D_before = len(features[0])
    # features = featureExtractor.ApplyPCAonFeatures(features)
    # print(F"Finished dimensionality reduction Before: {D_before} => After: {len(features[0])}")
    
    # Standardize Features
    features = featureExtractor.normalizeFeatures(features)
    return features, labels