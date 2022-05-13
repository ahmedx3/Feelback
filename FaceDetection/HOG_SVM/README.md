### Hyper Parameters Notes

## For laptop camera images (near faces) best use the following:

**resolution will change from (720x1280) -> (90x160)**

pyramidScale = 1.5
stepSize = 2
imageDivision = /8

## For Far faces best use the following:

pyramidScale = 2
stepSize = 2
imageDivision = /2

## Best hyper parameters for fron camera

(winW, winH) = (19, 19) # window width and height
pyramidScale = 1.5 # Scale factor for the pyramid
stepSize = 2 # Step size for the sliding window
overlappingThreshold = 0.3 # Overlap threshold for non-maximum suppression
skinThreshold = 0.4 # threshold for skin color in the window
edgeThreshold = 0.2 # threshold for edge percentage in the window
resizeFactor = 4
