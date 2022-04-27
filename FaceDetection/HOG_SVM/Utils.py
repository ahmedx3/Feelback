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
    mask = (calc1 > 0.15) & (calc1 < 1.1) & (calc2 > -4) & (calc2 < 0.3)
    
    return mask

def getHOG(image, blockSize=(6,6), cellSize=(3,3), numOfBins=7):
    """ Extract Histogram of oriented gradient (HOG) features from an image 

    Args:
        image (_type_): image to extract features from
        blockSize (tuple, optional): Size of block to be normalized onto. Defaults to (6,6).
        cellSize (tuple, optional): Number of pixels in one cell. Defaults to (3,3).
        numOfBins (int, optional): Number of bins in the histogram. Defaults to 7.

    Returns:
        _type_: HOG features
    """

    blockSize = ( int(blockSize[0]) , int(blockSize[1]) )
    cellSize = ( int(cellSize[0]) , int(cellSize[1]) )
    numOfBins = int( numOfBins )

    OrientationStep = 180 / numOfBins

    imageShape = image.shape
    Gradient = np.ones(imageShape)
    GradientOrientation = np.ones(imageShape)

    image = np.int64(image)
    Gx = np.zeros(imageShape)
    Gy = np.zeros(imageShape)
    Gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    Gy[1:-1, :] = image[2:, :] - image[:-2, :]

    # Calculate the Magnitude 
    Gradient = np.sqrt(Gx**2 + Gy**2) 
    # Calculate the Direction (in degrees)
    GradientOrientation = ( ( ( np.arctan2(Gy, Gx) ) / math.pi ) * 180 ) % 180
    
    # TODO : Divide bin
    # Calculate Bins For Every Pixel
    Bin = [ np.floor(( (GradientOrientation[:,:] + OrientationStep/2) / OrientationStep ) - 1) % numOfBins,
                      np.floor( (GradientOrientation[:,:] + OrientationStep/2) / OrientationStep ) % numOfBins]
    Bin = np.int16(Bin)

    # Calculate Contribution Ratios (for how much does the orientation contribute in the range) For Every Pixel
    contributionRatio = [ ( 1 - ( ( (GradientOrientation[:, :] + OrientationStep/2) / OrientationStep ) - ( (GradientOrientation[:, :] + OrientationStep/2) // OrientationStep ) ) ),
             ( ( (GradientOrientation[:, :] + OrientationStep/2) / OrientationStep ) - ( (GradientOrientation[:, :] + OrientationStep/2) // OrientationStep ) )]
    contributionRatio = np.array(contributionRatio).astype(float)

    # Calculate for every pixel all bins by multiplying the Gradient and the contribution Ratio
    hogPerPixel = np.zeros((image.shape[0],image.shape[1],numOfBins))
    for bin in range(numOfBins):
        hogPerPixel[:,:,bin] += np.where(np.array(Bin[0]) == bin,np.array(contributionRatio[0]),0)
        hogPerPixel[:,:,bin] += np.where(np.array(Bin[1]) == bin,np.array(contributionRatio[1]),0)
    Gradient = np.repeat(Gradient[:,:],numOfBins).reshape((image.shape[0],image.shape[1],numOfBins))
    hogPerPixel = hogPerPixel * Gradient
    
    # Calculate HOG For all Cells
    NumCellsX = int( (image.shape[0] / cellSize[0]) )
    NumCellsY = int( (image.shape[1] / cellSize[1]) )
    HOGCells = block_reduce(hogPerPixel, block_size=(int(imageShape[0]/NumCellsX), int(imageShape[1]/NumCellsY), 1), func=np.sum)
            
    # Normalize for blocks
    HOGVector = []
    numOfBlocksInX = int( (image.shape[0] - blockSize[0] + cellSize[0]) / cellSize[0] )
    numOfBlocksInY = int( (image.shape[1] - blockSize[1] + cellSize[1]) / cellSize[1] )
    numOfCellsInBlockX = int( blockSize[0] / cellSize[0] )
    numOfCellsInBlockY = int( blockSize[1] / cellSize[1] )

    for x in range(numOfBlocksInX):
        for y in range(numOfBlocksInY):
            HOGVector.extend(list(np.concatenate( 
                HOGCells[x : x+numOfCellsInBlockX, y : y+numOfCellsInBlockY]/(np.abs(HOGCells[x : x+numOfCellsInBlockX, y : y+numOfCellsInBlockY]).sum()+1e-5)
                ).ravel()))
            
    return HOGVector

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

def vectorizedHogSlidingWindows(slidingWindows,blockSize=(6,6), cellSize=(3,3), numOfBins=7):
    """ Extract Histogram of oriented gradient (HOG) features from an image

    Args:
        slidingWindows (_type_): array of windows to extract features from
        blockSize (tuple, optional): Size of the block. Defaults to (6,6).
        cellSize (tuple, optional): Size of the cell. Defaults to (3,3).
        numOfBins (int, optional): Number Of bins. Defaults to 7.

    Returns:
        _type_: array of HOG features
    """
    stackedWindows = np.stack(slidingWindows,axis=-1)
    
    blockSize = ( int(blockSize[0]) , int(blockSize[1]) )
    cellSize = ( int(cellSize[0]) , int(cellSize[1]) )
    numOfBins = int( numOfBins )

    OrientationStep = 180 / numOfBins

    imageShape = stackedWindows.shape
    Gradient = np.ones(imageShape)
    GradientOrientation = np.ones(imageShape)
    
    stackedWindows = np.int64(stackedWindows)
    Gx = np.zeros(imageShape)
    Gy = np.zeros(imageShape)
    Gx[:, 1:-1] = stackedWindows[:, 2:] - stackedWindows[:, :-2]
    Gy[1:-1, :] = stackedWindows[2:, :] - stackedWindows[:-2, :]

    Gradient = np.sqrt(Gx**2 + Gy**2)

    GradientOrientation = ( ( ( np.arctan2(Gy, Gx) ) / math.pi ) * 180 ) % 180

    Bin = [ np.floor(( (GradientOrientation[:,:] + OrientationStep/2) / OrientationStep ) - 1) % numOfBins,
                      np.floor( (GradientOrientation[:,:] + OrientationStep/2) / OrientationStep ) % numOfBins]
    Bin = np.int16(Bin)
    
    contributionRatio = [ ( 1 - ( ( (GradientOrientation[:, :] + OrientationStep/2) / OrientationStep ) - ( (GradientOrientation[:, :] + OrientationStep/2) // OrientationStep ) ) ),
             ( ( (GradientOrientation[:, :] + OrientationStep/2) / OrientationStep ) - ( (GradientOrientation[:, :] + OrientationStep/2) // OrientationStep ) )]
    contributionRatio = np.array(contributionRatio).astype(float)

    hogPerPixel = np.zeros((stackedWindows.shape[0],stackedWindows.shape[1],numOfBins,stackedWindows.shape[-1]))
    for bin in range(numOfBins):
        hogPerPixel[:,:,bin] += np.where(np.array(Bin[0]) == bin,np.array(contributionRatio[0]),0)
        hogPerPixel[:,:,bin] += np.where(np.array(Bin[1]) == bin,np.array(contributionRatio[1]),0)
    Gradient = np.repeat(Gradient[:,:],numOfBins,axis=1).reshape((stackedWindows.shape[0],stackedWindows.shape[1],numOfBins,stackedWindows.shape[-1]))
    hogPerPixel = hogPerPixel * Gradient


    NumCellsX = int( (stackedWindows.shape[0] / cellSize[0]) )
    NumCellsY = int( (stackedWindows.shape[1] / cellSize[1]) )
    HOGCells = block_reduce(hogPerPixel, block_size=(int(imageShape[0]/NumCellsX), int(imageShape[1]/NumCellsY), 1,1 ), func=np.sum)
        
    numOfBlocksInX = int( (stackedWindows.shape[0] - blockSize[0] + cellSize[0]) / cellSize[0] )
    numOfBlocksInY = int( (stackedWindows.shape[1] - blockSize[1] + cellSize[1]) / cellSize[1] )
    numOfCellsInBlockX = int( blockSize[0] / cellSize[0] )
    numOfCellsInBlockY = int( blockSize[1] / cellSize[1] )

    DivisionVector = np.zeros((numOfBlocksInX,numOfBlocksInY,1,stackedWindows.shape[-1]))

    for x in range(numOfBlocksInX):
        for y in range(numOfBlocksInY):
            DivisionVector[x,y] = block_reduce(HOGCells[x : x+numOfCellsInBlockX, y : y+numOfCellsInBlockY], block_size=(numOfBlocksInX, numOfBlocksInY, numOfBins,1), func=np.sum)

    FinalVector = np.zeros((numOfBlocksInX,numOfBlocksInY,numOfCellsInBlockX,numOfCellsInBlockY,numOfBins,stackedWindows.shape[-1]))

    for x in range(numOfBlocksInX):
        for y in range(numOfBlocksInY):          
            FinalVector[x,y] = HOGCells[x : x+numOfCellsInBlockX, y : y+numOfCellsInBlockY] / (DivisionVector[x,y]+ 1e-5)

    # FinalVector = FinalVector.flatten()

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