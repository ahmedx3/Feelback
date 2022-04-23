# For Drawing Rectangle on Faces
import cv2
import numpy as np
import time
from FeaturesExtraction import *
from Utils import *
from SlidingWindow import *
import pickle as pickle
import time

################################## Hyperparameters ##################################
(winW, winH) = (19, 19) # window width and height
pyramidScale = 1.5 # Scale factor for the pyramid
stepSize = 2 # Step size for the sliding window
overlappingThreshold = 0.3 # Overlap threshold for non-maximum suppression
skinThreshold = 0.4 # threshold for skin color in the window
edgeThreshold = 0.2 # threshold for edge percentage in the window
#####################################################################################

originalImg = cv2.imread("../HOG_SVM/Examples/Test1.jpg")

print("[INFO] Shape of the original image ", originalImg.shape)
shapeBefore = originalImg.shape
copyOriginalImage = originalImg.copy()

originalImg = cv2.resize(originalImg, (int(originalImg.shape[1]/6), int(originalImg.shape[0]/6)))
print("[INFO] Shape of the image after reshaping", originalImg.shape)

modelName = "./Models/ModelCBCL-HOG-TestSliding.sav"
model = pickle.load(open(modelName, 'rb'))
faces = []

pca = pickle.load(open("./Models/PCAModelSliding.sav", 'rb'))

print("[INFO]", " (winW,winH) ",winW,winH, " pyramidScale ",pyramidScale, " stepSize ",stepSize, " overlappingThreshold ",overlappingThreshold, " SkinThreshold ",skinThreshold ,"Model ",modelName)
# Calculate time before processing
start_time = time.time()

for image in pyramid(originalImg, pyramidScale, minSize=(30, 30)):
    
    scaleFactor = copyOriginalImage.shape[0] / float(image.shape[0])
    mask = DetectSkinColor(image)
    edges = EdgeDetection(image)
    windows = slidingWindow(image, stepSize,(winW, winH),mask,edges,skinThreshold,edgeThreshold)
    if(len(windows)) == 0:
        break
    print("[INFO] Num of windows in the current image pyramid ",len(windows))

    indices, patches = zip(*windows)
    st1 = time.time() 
    grayScaledPatches = [HistogramEqualization(patch) for patch in patches]
    hogFeatures = vectorizedHogSlidingWindows(grayScaledPatches)
    patches_hog = []
    for i in range(len(windows)):
        patches_hog.append(ApplyPCA(hogFeatures[:,:,:,:,:,i].flatten(),pca))

    st2 = time.time()
    print("[INFO] HOG a7a taken is {:.5f} seconds".format(st2 - st1))

    predicted_label = model.predict(patches_hog)
    indices = np.array(indices)
    for i, j in indices[predicted_label == "Faces"]:
        faces.append((int(i * scaleFactor), int(j * scaleFactor), int((i + winW) * scaleFactor), int((j + winH) * scaleFactor)))
        
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