# Boilerplate to Enable Relative imports when calling the file directly
if (__name__ == '__main__' and __package__ is None) or __package__ == '':
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    sys.path.append(str(file.parents[3]))
    __package__ = '.'.join(file.parent.parts[len(file.parents[3].parts):])

from operator import mod
from .FeaturesExtraction import *
from .Utils import ConvertToGrayScale, HistogramEqualization
import pickle as pickle
import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix

# load the model from disk
model = pickle.load(open("./Models/Model_v3.sav", 'rb'))
pca = pickle.load(open("./Models/PCA_v3.sav", 'rb'))

path_to_testset = os.path.join(os.getcwd(), "Test")

output = []

for root, dirs, files in os.walk(path_to_testset):
    for file in files:
        fileName = os.path.join(root, file)
        img = cv2.imread(fileName)
        image_features = ExtractHOGFeatures(img,flatten=True)
        image_features = ApplyPCA(image_features,pca)
        predicted_label = model.predict([image_features])
        output.append((fileName, predicted_label))

falsePositive = 0
falseNegative = 0
truePositive = 0
trueNegative = 0

for image in output:
    label, predicted_label = image

    if predicted_label == "Faces" and "Faces" in label:
        truePositive += 1
    elif predicted_label == "Faces" and "Non" in label:
        falsePositive += 1
    elif predicted_label == "Non" and "Faces" in label:
        falseNegative += 1
    else:
        trueNegative += 1
    
precition = truePositive / (truePositive + falsePositive)
recall = truePositive / (truePositive + falseNegative)
accuracy = (truePositive + trueNegative) / (truePositive + falsePositive + trueNegative + falseNegative)
f1 = 2 * (precition * recall) / (precition + recall)

print("Accuracy: ", accuracy)
print("F1: ", f1)