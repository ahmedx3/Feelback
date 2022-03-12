"""
Imports
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import random
import pickle
import DatasetLoading

"""
Global Variables
"""

random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)

classifiers = {
    # SVM with gaussian kernel
    'SVM': svm.SVC(random_state=random_seed, kernel="rbf"),
}

"""
Functions
"""
def train_classifier(used_classifier):
    """Trains the model on the selected dataset
    """

    # Load dataset with extracted features
    print('Loading dataset and extract features. This will take time ...')
    features, labels = DatasetLoading.load_Gender_Kaggle_dataset(type="Training")
    print('Finished loading dataset.')

    # Split Dataset to train and test for model fitting 
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.1, random_state=random_seed, stratify=labels, shuffle=True)

    # Training the model
    print('############## Training ', used_classifier, "##############")
    model = classifiers[used_classifier]
    model.fit(train_features, train_labels)

    # Test the model
    accuracy = model.score(test_features, test_labels)
    train_accuracy = model.score(train_features, train_labels)

    # Print Accuracies of train and test
    print(used_classifier, ' Train accuracy:', train_accuracy *
          100, '%', ' Test accuracy:', accuracy*100, '%')
    
    return model

def main():
    classifier = train_classifier("SVM")
    # save the model to disk
    filename = 'Model.sav'
    pickle.dump(classifier, open(filename, 'wb'))

if __name__ == "__main__":
    main()
