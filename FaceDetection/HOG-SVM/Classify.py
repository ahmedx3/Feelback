# For Drawing Rectangle on Faces
import cv2
from matplotlib.pyplot import sca
import numpy as np
import time
from Preprocessing import HistogramEqualization
from FeaturesExtraction import ExtractHOGFeatures
from SlidingWindow import sliding_window, pyramid 
import pickle as pickle
import time
import threading

################################## Hyperparameters ##################################
maxwidth, maxheight = 72*5, 72*5 # max width and height of the image after resizing
(winW, winH) = (10, 10) # window width and height
pyramidScale = 1.5 # Scale factor for the pyramid
stepSize = 2 # Step size for the sliding window
overlappingThreshold = 0.3 # Overlap threshold for non-maximum suppression
#####################################################################################

originalImg = cv2.imread("../HOG-SVM/Examples/Test4.jpg",0)

print("[INFO] Shape of the original image ", originalImg.shape)
shapeBefore = originalImg.shape
copyOriginalImage = originalImg.copy()

# Resize image to certain size to speed up processing
f = min(maxwidth / originalImg.shape[1], maxheight / originalImg.shape[0])
dim = (int(originalImg.shape[1] * f), int(originalImg.shape[0] * f))
originalImg = cv2.resize(originalImg, dim)
print("[INFO] Shape of the image after reshaping", originalImg.shape)

shapeAfter = originalImg.shape

model = pickle.load(open('ModelCBCL.sav', 'rb'))
faces = []

# Calculate time before processing
start_time = time.time()

def getFacesBoundryBoxes(windowList,scaleFactor,winH,winW,model):
    x,y,window = windowList
    if window.shape[0] != winH or window.shape[1] != winW:
        return

    x = int(x * scaleFactor)
    y = int(y * scaleFactor)
    w = int(winW * scaleFactor)
    h = int(winH * scaleFactor)

    originalImg = HistogramEqualization(window)
    image_features = ExtractHOGFeatures(originalImg)
    predicted_label = model.predict([image_features])

    if predicted_label == "Faces":
        faces.append((x, y, x+w,y+h))


for image in pyramid(originalImg, pyramidScale, minSize=(30, 30)):
    scaleFactor = copyOriginalImage.shape[0] / float(image.shape[0])
    windows = sliding_window(image, stepSize, windowSize=(winW, winH))
    print("[INFO] Num of windows in the current image pyramid ",len(windows))

    threads = []
    for i,window in enumerate(windows):
        t = threading.Thread(target=getFacesBoundryBoxes, args=(windows[i],scaleFactor,winH,winW,model))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

    # for (x, y, window) in windows:
    #     if window.shape[0] != winH or window.shape[1] != winW:
    #         continue

    #     x = int(x * scaleFactor)
    #     y = int(y * scaleFactor)
    #     w = int(winW * scaleFactor)
    #     h = int(winH * scaleFactor)

    #     # # start_time = time.time()    
    #     originalImg = HistogramEqualization(window)
    #     image_features = ExtractHOGFeatures(originalImg)
    #     predicted_label = model.predict([image_features])

    #     if predicted_label == "Faces":
    #         faces.append((x, y, x+w,y+h))
        
        # # end_time = time.time()
        # # print("[INFO] Time taken to proecess a frame: {}".format(end_time - start_time))

# Remove overlapping rectangles by using non-maximum suppression
def non_max_suppression(faces, overlapThresh=0.3):
    """ Perform non-maximum suppression on the overlapping rectangles

    Args:
        faces (_type_): array of overlapping boxes (faces)
        overlapThresh (float, optional): threshold of areas of intesecting boxes. Defaults to 0.3.

    Returns:
        _type_: array of non-overlapping boxes
    """
    faces = np.asarray(faces)

    if len(faces) == 0:
        return []

    pickedBoundries = []

    if faces.dtype.kind == "i":
        faces = faces.astype("float")
    
    x1 = faces[:, 0]
    y1 = faces[:, 1]
    x2 = faces[:, 2]
    y2 = faces[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pickedBoundries.append((x1[i], y1[i], x2[i], y2[i]))

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        overlappingWidth = np.maximum(0, xx2 - xx1 + 1)
        overlappingHeight = np.maximum(0, yy2 - yy1 + 1)

        overlapRatio = (overlappingWidth * overlappingHeight) / areas[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlapRatio > overlapThresh)[0])))
    
    return pickedBoundries

faces = non_max_suppression(faces,overlappingThreshold)

# Calculate time after processing in seconds
end_time = time.time()
print("[INFO] Time taken is {:.5f} seconds".format(end_time - start_time))

for (x, y, lenX,lenY) in faces:
    cv2.rectangle(copyOriginalImage, (int(x), int(y)), (int(lenX), int(lenY)), (0, 255, 0), 2)

cv2.imshow("Window", copyOriginalImage)
cv2.waitKey()
