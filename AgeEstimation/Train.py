"""
Imports
"""
import numpy as np
from sklearn.metrics import mean_absolute_error
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
    features, labels = DatasetLoading.load_UTK_AgeGender_dataset(label="age", age_range=(10,80))
    print('Finished loading dataset.')

    # Split Dataset to train and test for model fitting 
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=random_seed, stratify=labels, shuffle=True)

    # Training the model
    print('############## Training ', used_classifier, "##############")
    model = classifiers[used_classifier]
    model.fit(train_features, train_labels)

    # Test the model
    mae_train = mean_absolute_error(train_labels, model.predict(train_features))
    mae_test = mean_absolute_error(test_labels, model.predict(test_features))

    # Print Accuracies of train and test
    print(used_classifier, ': Train Mean Absolute Error:', mae_train , ' Test Mean Absolute Error:', mae_test)
    return model

def main():
    classifier = train_classifier("SVM")
    # save the model to disk
    filename = 'Model.sav'
    pickle.dump(classifier, open(filename, 'wb'))

if __name__ == "__main__":
    main()
