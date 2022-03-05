import cv2

def pyramid(img,scale=1.5, minSize=(30, 30)):
    yield img
    while True:
        w = int(img.shape[1] / scale)
        img = cv2.resize(img, (w, int(img.shape[0] / scale)))
        if img.shape[0] < minSize[1] or img.shape[1] < minSize[0]:
            break
        yield img

def sliding_window(img, stepSize, windowSize):
    for y in range(0, img.shape[0], stepSize):
        for x in range(0, img.shape[1], stepSize):
            yield (x, y, img[y:y + windowSize[1], x:x + windowSize[0]])