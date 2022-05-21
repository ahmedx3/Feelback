"""
Imports
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import random
import pickle
import DatasetLoading
from datetime import datetime
from sklearn.metrics import mean_absolute_error
import Utils


"""
Global Variables
"""

random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)

classifiers = {
    # SVM with gaussian kernel
    'SVM': svm.SVC(random_state=random_seed, kernel="rbf"),
    # SVM for regression with gaussian kernel
    'SVR': svm.SVR(kernel="rbf", C=100),
}

"""
Functions
"""

def k_fold_validation(used_classifier, selected_feature="LPQ", k_folds=5):
    """Validate the Accuracy and F1 Score through K fold validation

    Args:
        used_classifier (_type_): Name of Sklearn Model to use
        selected_feature (str, optional): Feature to extract. Defaults to "LPQ".
        k_folds (int, optional): Number of folds to split data. Defaults to 5.
    """
    # Load dataset with extracted features
    print('Loading dataset and extract features. This will take time ...')
    features, labels, image_paths = DatasetLoading.load_UTK_AgeGender_dataset(selected_feature=selected_feature,label="age", age_range=(20,50))
    print('Finished loading dataset.')


    # Set the random seed
    np.random.seed(int(datetime.now().timestamp()))

    # Main loop
    train_accuracies, test_accuracies, test_f1_scores, train_f1_scores = [], [], [], []
    for index in range(k_folds):

        # Get a random seed
        random_seed = np.random.randint(0,10000)

        # Split Dataset to train and test for model fitting
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=random_seed, stratify=labels, shuffle=True)

        # Training the model
        print('############## Training ', used_classifier," Fold ", index + 1 , " ##############")

        model = None
        if used_classifier == "SVM":
            model = svm.SVC(random_state=random_seed, kernel="rbf")
        model.fit(train_features, train_labels)

        # Get Model Predictions
        test_predictions = model.predict(test_features)
        train_predictions = model.predict(train_features)

        # Test the model
        accuracy = accuracy_score(test_labels, test_predictions)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        f1 = f1_score(test_labels, test_predictions, pos_label="male")
        train_f1 = f1_score(train_labels, train_predictions, pos_label="male")

        # Print Accuracies of train and test
        print(used_classifier, ' Train accuracy:', train_accuracy *
            100, '%', ' Test accuracy:', accuracy*100, '%')
        print('Train F1 Score:', train_f1, 'Test F1 Score:', f1)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(accuracy)
        train_f1_scores.append(train_f1)
        test_f1_scores.append(f1)
    
    print(used_classifier, ' Average Train accuracy:', np.average(train_accuracies) *
    100, '%', ' Average Test accuracy:', np.average(accuracy)*100, '%')
    print('Average Train F1 Score:', np.average(train_f1), ' Average Test F1 Score:', np.average(f1))



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
    features, labels, image_paths = DatasetLoading.load_UTK_AgeGender_dataset(selected_feature=selected_feature,label="age", age_range=(1,90))
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
    f1 = f1_score(test_labels, test_predictions, average='macro')
    train_f1 = f1_score(train_labels, train_predictions, average='macro')
    conf_matrix = confusion_matrix(test_labels, test_predictions)
    train_conf_matrix = confusion_matrix(train_labels, train_predictions)


    # Print Accuracies of train and test
    print(used_classifier, ' Train accuracy:', train_accuracy *
          100, '%', ' Test accuracy:', accuracy*100, '%')
    print('Train F1 Score:', train_f1)
    print('Test F1 Score:', f1)
    print('Train Confusion Matrix:')
    print(train_conf_matrix)
    print('Test Confusion Matrix:')
    print(conf_matrix)

    return model, train_accuracy, accuracy


def train_classifier_regression(used_classifier, selected_feature="LPQ"):

    # Load dataset with extracted features
    print('Loading dataset and extract features. This will take time ...')
    features, labels, image_paths = DatasetLoading.load_UTK_AgeGender_dataset(selected_feature=selected_feature,label="age_number", age_range=(5,75))
    # features, labels, image_paths = DatasetLoading.load_FGNET_Age_dataset(selected_feature=selected_feature,label="age_number")
    print('Finished loading dataset.')

    # Split Dataset to train and test for model fitting
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=random_seed, shuffle=True)

    # Training the model
    print('############## Training ', used_classifier, "##############")
    model = classifiers[used_classifier]
    model.fit(train_features, train_labels)

    # parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 0.1, 1, 10, 100, 1000]}
    # model = GridSearchCV(svm.SVR(), parameters, scoring='neg_mean_absolute_error', verbose=4)
    # model.fit(train_features, train_labels)
    # print(model.best_params_)
    # print(model.best_score_)


    # Test the model
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    mae_train = round(mean_absolute_error(train_labels, train_predictions), 2)
    mae_test = round(mean_absolute_error(test_labels, test_predictions), 2)

    # Print MAE of train and test
    print(used_classifier, ': Train Mean Absolute Error:', mae_train , ' Test Mean Absolute Error:', mae_test)
 
    # Plot accuracies of different errors
    Utils.plot_regression_accuracies(train_labels, train_predictions)
    Utils.plot_regression_accuracies(test_labels, test_predictions)

    return model, mae_train, mae_test


def main():
    used_classifier = "SVM"
    selected_feature="LPQ"
    classifier, train_accuracy, accuracy = train_classifier_regression(used_classifier, selected_feature=selected_feature)
    # save the model to disk
    name = "UTK"
    filename = "Models_Age/" + name + "_" + used_classifier + "_" + selected_feature + "_" + \
        str(int(train_accuracy * 100)) + "_" + str(int(accuracy * 100)) + ".model"
    pickle.dump(classifier, open(filename, 'wb'))


def main_regression():
    used_classifier = "SVR"
    selected_feature="localLBP"
    classifier, mae_train, mae_test = train_classifier_regression(used_classifier, selected_feature=selected_feature)
    # save the model to disk
    name = "FGNET"
    filename = "Models_Age/" + name + "_" + used_classifier + "_" + selected_feature + "_" + \
        str(mae_train) + "_" + str(mae_test) + ".model"
    pickle.dump(classifier, open(filename, 'wb'))


def validate():
    used_classifier = "SVM"
    selected_feature="LPQ"
    k_fold_validation(used_classifier, selected_feature=selected_feature, k_folds=5)

if __name__ == "__main__":
    main_regression()
    # main()
    # validate()
