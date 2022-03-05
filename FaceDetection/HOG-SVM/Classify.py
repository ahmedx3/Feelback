# For Drawing Rectangle on Faces
import cv2
import time
from Preprocessing import ConvertToGrayScale, HistogramEqualization
from FeaturesExtraction import ExtractHOGFeatures
from SlidingWindow import sliding_window, pyramid
import pickle as pickle

originalImg = cv2.imread("../HOG-SVM/Examples/Test3.jpg")
# Resize image to half size
# originalImg = cv2.resize(originalImg, (0, 0), fx=0.5, fy=0.5)

copyOriginalImage = originalImg.copy()
(winW, winH) = (16, 16)
model = pickle.load(open('Model.sav', 'rb'))
predicitions = []

# Calculate time before processing
start_time = time.time()

for image in pyramid(originalImg, scale=1.5, minSize=(30, 30)):
    scaleFactor = copyOriginalImage.shape[1] / float(image.shape[1])
    for (x, y, window) in sliding_window(image, stepSize=4, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        x = int(x * scaleFactor)
        y = int(y * scaleFactor)
        w = int(winW * scaleFactor)
        h = int(winH * scaleFactor)

        originalImg = ConvertToGrayScale(window)
        originalImg = HistogramEqualization(originalImg)
        image_features = ExtractHOGFeatures(originalImg)
        predicted_label = model.predict([image_features])

        if predicted_label == "Faces":
            predicitions.append((x, y, x+w,y+h))
            # clone = image.copy()
            # cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow("Window", clone)
            # cv2.waitKey(1)

# Calculate time after processing in seconds
end_time = time.time()
print("[INFO] Time taken is {:.5f} seconds".format(end_time - start_time))

for (x, y, lenX,lenY) in predicitions:
    cv2.rectangle(copyOriginalImage, (x, y), (lenX, lenY), (0, 255, 0), 2)

cv2.imshow("Window", copyOriginalImage)
cv2.waitKey()