import pickle
from Utils import *
from FeaturesExtraction import *
import os
from SlidingWindow import pyramid

# Load Model
modelName = "./Models/ModelCBCL-CV-DataEnhanced6.sav"
model = pickle.load(open(modelName, 'rb'))

pca = pickle.load(open("./Models/PCAModel-E6.sav", 'rb'))

# extract image negative examples from the dataset folder
def load_dataset_negative_images(path_to_dataset):
    negativeImages = []
    path_to_dataset = os.path.join(os.getcwd(), path_to_dataset)
    directoriesNames = os.listdir(path_to_dataset)
    print(directoriesNames)
    for directory in directoriesNames:
        print(directory)
        directoryPath = os.path.join(path_to_dataset, directory)
        for root, dirs, files in os.walk(directoryPath):
	        for file in files:
                    #append the file name to the list
                    fileName = os.path.join(root,file)
                    #read the image
                    img = cv2.imread(fileName)
                    negativeImages.append(img)
            
    return negativeImages

negativeImages = load_dataset_negative_images("./Data/Non/Non-Natural")
################################## Hyperparameters ##################################
(winW, winH) = (19, 19) # window width and height
pyramidScale = 2 # Scale factor for the pyramid
stepSize = 2 # Step size for the sliding window
#####################################################################################

def slidingWindow(img, stepSize, windowSize):
    windowsArr = []
    for y in range(0, img.shape[0] - windowSize[0] + 1, stepSize):
        for x in range(0, img.shape[1] - windowSize[1] + 1, stepSize):
            windowsArr.append(( (x, y), img[y:y + windowSize[1], x:x + windowSize[0]]))
    return windowsArr

uniqueIdentifier = 260
for (countImg,originalImg) in  enumerate(negativeImages):
    # Print image number
    print("[INFO] Image number ", countImg+1)
    for image in pyramid(originalImg, pyramidScale, minSize=(30, 30)):

        scaleFactor = negativeImages[countImg].shape[0] / float(image.shape[0])
        windows = slidingWindow(image, stepSize,(winW, winH))
        if(len(windows)) == 0:
            break
        # print("[INFO] Num of windows in the current image pyramid ",len(windows))

        indices, patches = zip(*windows)
        patches_hog = np.array([ ApplyPCA(ExtractHOGFeatures(patch),pca) for patch in patches])
        predicted_label = model.predict(patches_hog)
        indices = np.array(indices)

        # Save patches where predicted label is 1
        for i in range(len(predicted_label)):
            if predicted_label[i] == "Faces":
                cv2.imwrite("./Data/Non/Mined/"+str(uniqueIdentifier)+".jpg",patches[i])
                uniqueIdentifier += 1