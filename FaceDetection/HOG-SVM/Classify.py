# For Drawing Rectangle on Faces
import cv2
import numpy as np
import time
from FeaturesExtraction import *
from Utils import *
from SlidingWindow import *
import pickle as pickle
import time
import threading

################################## Hyperparameters ##################################
maxwidth, maxheight = 640/2, 360/2 # max width and height of the image after resizing
(winW, winH) = (19, 19) # window width and height
pyramidScale = 2 # Scale factor for the pyramid
stepSize = 2 # Step size for the sliding window
overlappingThreshold = 0.3 # Overlap threshold for non-maximum suppression
skinThreshold = 0.4 # threshold for skin color in the window
#####################################################################################

originalImg = cv2.imread("../HOG-SVM/Examples/Test9.jpg")

print("[INFO] Shape of the original image ", originalImg.shape)
shapeBefore = originalImg.shape
copyOriginalImage = originalImg.copy()

# Resize image to certain size to speed up processing
# f = min(maxwidth / originalImg.shape[1], maxheight / originalImg.shape[0])
# dim = (int(originalImg.shape[1] * f), int(originalImg.shape[0] * f))
# originalImg = cv2.resize(originalImg, dim)
# print("[INFO] Shape of the image after reshaping", originalImg.shape)

originalImg = cv2.resize(originalImg, (int(originalImg.shape[1]/4), int(originalImg.shape[0]/4)))
print("[INFO] Shape of the image after reshaping", originalImg.shape)

modelName = "./Models/ModelCBCL-CV-DataEnhanced6.sav"
model = pickle.load(open(modelName, 'rb'))
faces = []

pca = pickle.load(open("./Models/PCAModel-E6.sav", 'rb'))

print("[INFO] (maxwidth,maxheight) ",maxwidth,maxheight, " (winW,winH) ",winW,winH, " pyramidScale ",pyramidScale, " stepSize ",stepSize, " overlappingThreshold ",overlappingThreshold, " SkinThreshold ",skinThreshold ,"Model ",modelName)
# Calculate time before processing
start_time = time.time()

def getFacesBoundryBoxes(windowList,scaleFactor,winH,winW,model):
    (x,y),window = windowList
    if window.shape[0] != winH or window.shape[1] != winW:
        return

    x = int(x * scaleFactor)
    y = int(y * scaleFactor)
    w = int(winW * scaleFactor)
    h = int(winH * scaleFactor)

    # resize window
    image_features = ExtractHOGFeatures(window)
    predicted_label = model.predict([image_features])

    if predicted_label == "Faces":
        faces.append((x, y, x+w,y+h))


for image in pyramid(originalImg, pyramidScale, minSize=(30, 30)):
    
    scaleFactor = copyOriginalImage.shape[0] / float(image.shape[0])
    mask = DetectSkinColor(image)
    windows = slidingWindow(image, stepSize,(winW, winH),mask,skinThreshold)
    if(len(windows)) == 0:
        break
    print("[INFO] Num of windows in the current image pyramid ",len(windows))

    # threads = []
    # for i,window in enumerate(windows):
    #     t = threading.Thread(target=getFacesBoundryBoxes, args=(windows[i],scaleFactor,winH,winW,model))
    #     threads.append(t)
    #     t.start()
    # for t in threads:
    #     t.join()

    indices, patches = zip(*windows)
    patches_hog = np.array([ApplyPCA(ExtractHOGFeatures(patch),pca) for patch in patches])
    predicted_label = model.predict(patches_hog)
    indices = np.array(indices)
    for i, j in indices[predicted_label == "Faces"]:
        faces.append((int(i * scaleFactor), int(j * scaleFactor), int((i + winW) * scaleFactor), int((j + winH) * scaleFactor)))

    # for ((x, y), window) in windows:
    #     predictTime = 0
    #     if window.shape[0] != winH or window.shape[1] != winW:
    #         continue

    #     indices = mask.astype(np.uint8)  #convert to an unsigned byte
    #     indices *= 255
    #     cv2.imshow("mask", indices[y:y+winH,x:x+winW])
    #     cv2.waitKey(1)
    #     time.sleep(0.025)
    #     print(window.size)
    #     skinRatio = np.sum(mask[y:y+winH,x:x+winW])/ (winW * winH)
    #     if skinRatio < 0.2:
    #         continue

    #     # Debugging
    #     clone = image.copy()
    #     cv2.rectangle(clone, (x, y), (x+winW,y+winH), (0, 255, 0), 1)
    #     cv2.imshow("Window", clone)
    #     cv2.waitKey(1)
    #     time.sleep(0.025)

    #     x = int(x * scaleFactor)
    #     y = int(y * scaleFactor)
    #     w = int(winW * scaleFactor)
    #     h = int(winH * scaleFactor)

    #     image_features = ExtractHOGFeatures(window)
    #     predicted_label = model.predict([image_features])

    #     if predicted_label == "Faces":
    #         faces.append((x, y, x+w,y+h))
        
# Remove overlapping rectangles by using non-maximum suppression
def nonMaxSuppression(faces, overlapThresh=0.3):
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

faces = nonMaxSuppression(faces,overlappingThreshold)

# Calculate time after processing in seconds
end_time = time.time()
print("[INFO] Time taken is {:.5f} seconds".format(end_time - start_time))

for (x, y, lenX,lenY) in faces:
    cv2.rectangle(copyOriginalImage, (int(x), int(y)), (int(lenX), int(lenY)), (0, 255, 0), 2)
# Save the image result
cv2.imwrite("result.jpg", copyOriginalImage)
cv2.imshow("Window", copyOriginalImage)
cv2.waitKey()