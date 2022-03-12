import cv2
import time
import os
from sklearn.svm import SVC
import FeaturesExtraction
import Utils
import pickle as pickle

# Read Image and Model
path_to_file = os.path.join(os.path.dirname(
    __file__), "Examples/Seif.jpg")

img = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)
model: SVC = pickle.load(open(os.path.join(os.path.dirname(
    __file__), "Model.sav"), 'rb'))

# Calculate time before processing
start_time = time.time()

img_features = FeaturesExtraction.extract_features(img, feature="LPQ")
predicted = model.predict([img_features])[0]

# Calculate time after processing in seconds
end_time = time.time()
print("[INFO] Time taken is {:.5f} seconds".format(end_time - start_time))

Utils.show_image(img, predicted)
