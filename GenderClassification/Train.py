"""
Imports
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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


def train_classifier(used_classifier, selected_feature="LPQ"):
    """Trains the model on the selected dataset

    Args:
        used_classifier (string): Name of Sklearn Model to use
        selected_feature (str, optional): Feature to extract. Defaults to "LPQ"._

    Returns:
        Any: Trained SKlearn model selected
    """

    # Load dataset with extracted features
    print('Loading dataset and extract features. This will take time ...')
    features, labels, image_paths = DatasetLoading.load_Gender_Kaggle_dataset(
        selected_feature=selected_feature, type="Training")
    print('Finished loading dataset.')

    # Split Dataset to train and test for model fitting
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=random_seed, stratify=labels, shuffle=True)

    # Training the model
    print('############## Training ', used_classifier, "##############")
    model = classifiers[used_classifier]
    model.fit(train_features, train_labels)

    # Get Model Predictions
    test_predictions = model.predict(test_features)
    train_predictions = model.predict(train_features)

    # Test the model
    accuracy = accuracy_score(test_labels, test_predictions)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    f1 = f1_score(test_labels, test_predictions, pos_label="male")
    train_f1 = f1_score(train_labels, train_predictions, pos_label="male")
    conf_matrix = confusion_matrix(test_labels, test_predictions)
    train_conf_matrix = confusion_matrix(train_labels, train_predictions)


    # Print Accuracies of train and test
    print(used_classifier, ' Train accuracy:', train_accuracy *
          100, '%', ' Test accuracy:', accuracy*100, '%')
    print('Train F1 Score:', train_f1)
    print('Test F1 Score:', f1)
    print('Train Confusion Matrix:', train_conf_matrix)
    print('Train Confusion Matrix:', conf_matrix)


    return model, train_accuracy, accuracy


def main():
    used_classifier = "SVM"
    selected_feature="LPQ"
    classifier, train_accuracy, accuracy = train_classifier(used_classifier, selected_feature=selected_feature)
    # save the model to disk
    name = "Kaggle_Tra"
    filename = "Models/" + name + "_" + used_classifier + "_" + selected_feature + "_" + \
        str(int(train_accuracy * 100)) + "_" + str(int(accuracy * 100)) + ".model"
    pickle.dump(classifier, open(filename, 'wb'))


if __name__ == "__main__":
    main()
