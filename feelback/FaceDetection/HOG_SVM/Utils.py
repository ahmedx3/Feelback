import cv2
import numpy as np
import math
from skimage.measure import block_reduce

np.seterr(divide='ignore', invalid='ignore')

def HistogramEqualization(grayScaleImage):
    """equalize histogram of grayscale image

    Args:
        grayScaleImage (_type_): grayscale image to be equalized

    Returns:
        _type_: equalized grayscale image
    """
    gray = cv2.cvtColor(grayScaleImage, cv2.COLOR_BGR2GRAY)
    equlizedImage = cv2.equalizeHist(gray)
    return equlizedImage

def ConvertToGrayScale(image):
    """convert image to grayscale

    Args:
        image (_type_): image to be converted

    Returns:
        _type_: gray scale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def DetectSkinColor(img):
    """Detect skin color in an image

    Args:
        img (_type_): image

    Returns:
        _type_: a boolean mask of the image skin
    """
    # extract red and green and blue channels
    b, g, r = cv2.split(img)

    # do a piecewise log
    calc1 = cv2.log(r/g)
    calc2 = cv2.log(b/g)

    # mask if test1 between [0.15;1.1] and test2 between [-4;0.3]
    mask = (calc1 >= 0.15) & (calc1 <= 1.1) & (calc2 >= -4) & (calc2 <= 0.3)

    # show image where mask is true
    # indices = mask.astype(np.uint8)  #convert to an unsigned byte
    # indices *= 255
    # cv2.imshow("indices", indices)
    # cv2.waitKey(0)
    
    return mask

def EdgeDetection(img, sigma=0.33):
    """Returns the edges in the image

    Args:
        img (_type_): image to extract features from
        sigma (float, optional): threshold of ranges of values to consider as edges. Defaults to 0.33.

    Returns:
        _type_: Edges in the image
    """
    # Detect edges in the image
    median = np.median(img)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(img, lower, upper) # 50,100 also works well
    return edges

def detectCommonMask(edgeMask, skinMask):
    """Returns the common mask of edge and sking

    Args:
        img1 (_type_): edges of the image
        img2 (_type_): skin mask of the image

    Returns:
        _type_: common mask of two images
    """
    # convert 0 to true and 1 to false of edges
    edges = np.where(edgeMask == 0, False, True)
    # and edges with skin mask
    andedMask = np.logical_and(edges, skinMask)
    return andedMask
    
def hog_slow(img,blockSize=(6,6), cellSize=(3,3), numOfBins=7):
    """ Compute hog features of an image

    Args:
        img (_type_): image to compute hog features
        blockSize (tuple, optional): Size of the block. Defaults to (6,6).
        cellSize (tuple, optional): Size of the cell. Defaults to (3,3).
        numOfBins (int, optional): Number Of bins. Defaults to 7.

    Returns:
        _type_: _description_
    """
    
    img = np.int16(img)
    
    # Calculate the Gradients
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # Calculate the Magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Calculate the Direction
    grad_dir = np.arctan2(grad_y, grad_x)

    stepAngle = 180/numOfBins

    cellsInX = int(img.shape[0]/cellSize[0])
    cellsInY = int(img.shape[1]/cellSize[1])

    # Initialize histogram
    histogram = [x[:] for x in [[0] * cellsInX] * cellsInY]
    
    # Calculate the histogram
    for i in range(cellsInX):
        for j in range(cellsInY):
            # Initialize bin array
            bin = [0] * numOfBins

            # Calculate the bin
            for k in range(cellSize[0]):
                for l in range(cellSize[1]):
                    # Calculate the angle
                    angle = grad_dir[i*cellSize[0]+k, j*cellSize[1]+l]
                    if angle < 0:
                        angle += np.pi

                    # Calculate the left and right bin
                    leftBin = int(angle/stepAngle)
                    rightBin = leftBin + 1 if leftBin < numOfBins - 1 else 0

                    # Calculate the magnitude
                    magnitude = grad_mag[i*cellSize[0]+k, j*cellSize[1]+l]

                    # Calculate the weight
                    weight = (angle - leftBin*stepAngle)/stepAngle

                    # Calculate the bin
                    bin[leftBin] += (1 - weight) * magnitude
                    bin[rightBin] += weight * magnitude

            # Add the bin to the histogram
            histogram[i][j] = bin
    
    numOfBlocksX = int( (img.shape[0] - blockSize[0] + cellSize[0]) / cellSize[0] )
    numOfBlocksY = int( (img.shape[1] - blockSize[1] + cellSize[1]) / cellSize[1] )
    
    numOfCellsInBlockX = int(blockSize[0] / cellSize[0])
    numOfCellsInBlockY = int(blockSize[1] / cellSize[1])

    hogVector = []

    # Normalize the histogram
    for i in range(numOfBlocksX):
        for j in range(numOfBlocksY):
            
            # Calculate the block
            histogram = np.float64(histogram)
            block = histogram[i : i+numOfCellsInBlockX, j : j+numOfCellsInBlockY]
            block = block.flatten()

            # Normalize the block with L1 norm
            block = block / (np.linalg.norm(block) + 0.000001)
            
            # Add the block to the hogVector
            hogVector.append(block)

    hogVector = np.float64(hogVector)
    hogVector = hogVector.flatten()

    return hogVector  

def vectorizedHogSlidingWindows(slidingWindows,blockSize=(6,6), cellSize=(3,3), numOfBins=7, flatten = False):
    """ Extract Histogram of oriented gradient (HOG) features from an image

    Args:
        slidingWindows (_type_): array of windows to extract features from
        blockSize (tuple, optional): Size of the block. Defaults to (6,6).
        cellSize (tuple, optional): Size of the cell. Defaults to (3,3).
        numOfBins (int, optional): Number Of bins. Defaults to 7.

    Returns:
        _type_: array of HOG features
    """

    # Stack the sliding windows into a single image
    stackedWindows = np.stack(slidingWindows,axis=-1)
    
    blockSize = ( int(blockSize[0]) , int(blockSize[1]) )
    cellSize = ( int(cellSize[0]) , int(cellSize[1]) )
    numOfBins = int( numOfBins )
    stepAngle = 180 / numOfBins
    imageShape = stackedWindows.shape
    grad = np.ones(imageShape)
    gradDirection = np.ones(imageShape)
    stackedWindows = np.int64(stackedWindows)

    # Calculate the Gradients in X and Y
    Gx = np.zeros(imageShape)
    Gy = np.zeros(imageShape)
    Gx[:, 1:-1] = stackedWindows[:, 2:] - stackedWindows[:, :-2]
    Gy[1:-1, :] = stackedWindows[2:, :] - stackedWindows[:-2, :]

    # Calculate the Magnitude
    grad = np.sqrt(Gx**2 + Gy**2)

    # Calculate the Direction
    gradDirection = ( ( ( np.arctan2(Gy, Gx) ) / math.pi ) * 180 ) % 180

    # Calculate the left and right bins
    leftBin = np.int16(np.floor(( (gradDirection[:,:] + stepAngle/2) / stepAngle ) - 1) % numOfBins) 
    rightBin = np.int16(np.floor( (gradDirection[:,:] + stepAngle/2) / stepAngle ) % numOfBins)

    # Calculate the weights of the left and right bins
    contributionWeightLeft = np.array(( 1 - ( ( (gradDirection[:, :] + stepAngle/2) / stepAngle ) - ( (gradDirection[:, :] + stepAngle/2) // stepAngle ) ) )).astype(float)
    contributionWeightRight = np.array(( ( (gradDirection[:, :] + stepAngle/2) / stepAngle ) - ( (gradDirection[:, :] + stepAngle/2) // stepAngle ) )).astype(float)

    # Calculate the hog for every pixel
    hogPerPixel = np.zeros((stackedWindows.shape[0],stackedWindows.shape[1],numOfBins,stackedWindows.shape[-1]))
    for bin in range(numOfBins):
        hogPerPixel[:,:,bin] += np.where(np.array(leftBin) == bin,np.array(contributionWeightLeft),0)
        hogPerPixel[:,:,bin] += np.where(np.array(rightBin) == bin,np.array(contributionWeightRight),0)
    grad = np.repeat(grad[:,:],numOfBins,axis=1).reshape((stackedWindows.shape[0],stackedWindows.shape[1],numOfBins,stackedWindows.shape[-1]))
    hogPerPixel = hogPerPixel * grad

    # Calculate the cells
    NumCellsX = int( (stackedWindows.shape[0] / cellSize[0]) )
    NumCellsY = int( (stackedWindows.shape[1] / cellSize[1]) )
    HOGCells = block_reduce(hogPerPixel, block_size=(int(imageShape[0]/NumCellsX), int(imageShape[1]/NumCellsY), 1,1 ), func=np.sum)
        
    numOfBlocksInX = int( (stackedWindows.shape[0] - blockSize[0] + cellSize[0]) / cellSize[0] )
    numOfBlocksInY = int( (stackedWindows.shape[1] - blockSize[1] + cellSize[1]) / cellSize[1] )
    numOfCellsInBlockX = int( blockSize[0] / cellSize[0] )
    numOfCellsInBlockY = int( blockSize[1] / cellSize[1] )

    # Normalize the histogram
    DivisionVector = np.zeros((numOfBlocksInX,numOfBlocksInY,1,stackedWindows.shape[-1]))
    for x in range(numOfBlocksInX):
        for y in range(numOfBlocksInY):
            DivisionVector[x,y] = block_reduce(HOGCells[x : x+numOfCellsInBlockX, y : y+numOfCellsInBlockY], block_size=(numOfBlocksInX, numOfBlocksInY, numOfBins,1), func=np.sum)
    FinalVector = np.zeros((numOfBlocksInX,numOfBlocksInY,numOfCellsInBlockX,numOfCellsInBlockY,numOfBins,stackedWindows.shape[-1]))
    for x in range(numOfBlocksInX):
        for y in range(numOfBlocksInY):          
            FinalVector[x,y] = HOGCells[x : x+numOfCellsInBlockX, y : y+numOfCellsInBlockY] / (DivisionVector[x,y]+ 1e-5)

    if flatten:
        FinalVector = FinalVector.flatten()

    return FinalVector

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