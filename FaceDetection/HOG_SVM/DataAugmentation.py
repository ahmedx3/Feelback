# Boilerplate to Enable Relative imports when calling the file directly
if (__name__ == '__main__' and __package__ is None) or __package__ == '':
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    sys.path.append(str(file.parents[3]))
    __package__ = '.'.join(file.parent.parts[len(file.parents[3].parts):])

import os
import cv2


def augmentData(path, mode="flip"):
    path_to_dataset = os.path.join(os.getcwd(), path)
    for root, dirs, files in os.walk(path_to_dataset):
        for file in files:
            # append the file name to the list
            fileName = os.path.join(root, file)
            img = cv2.imread(fileName)
            if mode == "flip":
                filename = os.path.splitext(file)[0]
                fileExtention = os.path.splitext(file)[1]
                flippedImage = cv2.flip(img, 1)
                # Save the flipped image
                cv2.imwrite(f"./Data/Faces/Augmented/{filename}-flipped.jpg", flippedImage)


augmentData("Data/Faces/", 'flip')
