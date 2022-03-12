import os
from sklearn.svm import SVC
import pickle as pickle
import DatasetLoading


# Load Saved model
model: SVC = pickle.load(open(os.path.join(os.path.dirname(
    __file__), "Model.sav"), 'rb'))

# Load dataset with extracted features
print('Loading dataset and extract features. This will take time ...')
features, labels = DatasetLoading.load_UTK_AgeGender_dataset(label="age", age_range=(10,80))
print('Finished loading dataset.')

# Get Model Accuracy
accuracy = model.score(features, labels)
print(' Accuracy:', accuracy * 100, '%')
