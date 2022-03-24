import os
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import SVC
import pickle as pickle
import DatasetLoading
import Utils

# Load Saved model
model: SVC = pickle.load(open(os.path.join(os.path.dirname(
    __file__), "Models/Kaggle_Tra_SVM_LPQ_87_86.model"), 'rb'))

# Load dataset with extracted features
print('Loading dataset and extract features. This will take time ...')
features, labels, image_paths = DatasetLoading.load_UTK_AgeGender_dataset(selected_feature="LPQ",label="gender", age_range=(20,50))
# features, labels, image_paths = DatasetLoading.load_Gender_Kaggle_dataset(selected_feature="LPQ", type="Validation")
print('Finished loading dataset.')

# Get Model Predictions
predictions = model.predict(features)

# Metrics
accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, pos_label="male")
conf_matrix = confusion_matrix(labels, predictions)
print('Accuracy:', accuracy * 100, '%')
print('F1 Score:', f1)
print('Confusion Matrix:', conf_matrix)

# Get missclassified Images
missclassified_images = Utils.get_random_incorrect_labeled(labels, predictions, image_paths, 5)
Utils.show_images(missclassified_images, titles="Incorrect Labeled")
