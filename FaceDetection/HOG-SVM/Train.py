import os
import numpy as np
import cv2
from FeaturesExtraction import ExtractHOGFeatures
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC
import random
import pickle
import time
from sklearn.decomposition import PCA

random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)

used_classifier = "SVM"

classifiers = {
    # SVM with gaussian kernel
    'SVM': svm.SVC(random_state=random_seed, kernel="rbf",cache_size=1000),
    'LinearSVM': LinearSVC(random_state=random_seed),
}

def load_dataset(path_to_dataset):
    features = []
    labels = []
    path_to_dataset = os.path.join(os.getcwd(), path_to_dataset)
    directoriesNames = os.listdir(path_to_dataset)
    print(directoriesNames)
    for directory in directoriesNames:
        print(directory)
        directoryPath = os.path.join(path_to_dataset, directory)
        for root, dirs, files in os.walk(directoryPath):
	        for file in files:
                    #append the file name to the list
                    fileName = os.path.join(root,file)
                    labels.append(directory)
                    #read the image and extract features
                    img = cv2.imread(fileName)
                    features.append(ExtractHOGFeatures(img))
            
    return features, labels

def train_classifier(path_to_dataset):

    # Load dataset with extracted features
    print('Loading dataset. This will take time ...')
    features, labels = load_dataset(path_to_dataset)
    print('Finished loading dataset.')
    D_before = len(features[0])
    pca = PCA(n_components=50)
    pca.fit(features)
    filename = './Models/PCAModel.sav'
    pickle.dump(pca, open(filename, 'wb'))
    features = pca.transform(features)
    D_after = len(features[0])
    print('Reduced the dimension from ', D_before, ' to ', D_after)

    # Since we don't want to know the performance of our classifier on images it has seen before
    # we are going to withhold some images that we will test the classifier on after training
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=random_seed, stratify=labels, shuffle=True)

    print('############## Training ', used_classifier, "##############")
    # Train the model only on the training features
    model = classifiers[used_classifier]
    model.fit(train_features, train_labels)

    # Test the model on images it hasn't seen before
    accuracy = model.score(test_features, test_labels)
    train_accuracy = model.score(train_features, train_labels)

    print(used_classifier, ' Train accuracy:', train_accuracy *
          100, '%', ' Test accuracy:', accuracy*100, '%')

def main():
    train_classifier("Data")
    classifier = classifiers[used_classifier]
    # save the model to disk
    filename = './Models/ModelCBCL-Small-PCA.sav'
    pickle.dump(classifier, open(filename, 'wb'))

if __name__ == "__main__":
    # calculate training time
    start_time = time.time()
    main()
    end_time = time.time()
    print("[INFO] Training time is {:.5f} seconds".format(end_time - start_time))
