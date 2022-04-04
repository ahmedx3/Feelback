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
            HOGVector.append(list(np.concatenate( 
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