import cv2
import time
import os
from sklearn.svm import SVC
import FeaturesExtraction
import Utils
import pickle as pickle
import DatasetLoading
# Read Image and Model
# path_to_file = os.path.join(os.path.dirname(
#     __file__), "../Test/disgust2.jpg")
# img = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)

# featureExtractor = FeaturesExtraction.FeatureExtractor(True)
model: SVC = pickle.load(open('FacialExpressionRecognition/Models/Model.sav', 'rb'))

start_time = time.time()

# Load data with extracted features
print('Loading data and extract features')
features, labels = DatasetLoading.load_test_data("../Test/TFEID")
print('Finished loading data.')
print("Number of actual used samples: ", len(labels))

# Calculate time before processing
# img_features = featureExtractor.ExtractHOGFeatures(img)
# img_features = featureExtractor.ApplyPCAonFeatures(img_features.reshape(1,-1))[0]
# img_features = featureExtractor.ExtractLandMarks(img)
# img_features = featureExtractor.normalizeFeatures(img_features)

# predicted = model.predict([img_features])[0]
# accuracy = model.score(features, labels)

predicted = model.predict(features)
accuracy = sum(predicted == labels) / len(labels)

# Calculate time after processing in seconds
end_time = time.time()
print("[INFO] Time taken is {:.5f} seconds".format(end_time - start_time))

print("Classification accuracy: ",accuracy*100, "%")
# Utils.show_image(img, predicted)

# f = open("D:\out.txt", "w")
# for i in range(len(labels)):
#     f.write(f"{predicted[i]}  {labels[i]}\n")
# f.close()