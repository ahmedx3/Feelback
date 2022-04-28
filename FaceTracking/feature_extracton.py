import os
import pickle
import numpy as np
import cv2
from typing import List

__CURRENT_DIR__ = os.path.dirname(os.path.abspath(__file__))
__PARENT_DIR__ = os.path.dirname(__CURRENT_DIR__)


def eigen_faces_features(faces: List[np.ndarray]) -> np.ndarray:
    face_size = (48, 48)
    resized_faces = np.zeros((len(faces), np.prod(face_size)))
    for i, face in enumerate(faces):
        # Resize face to a fixed size then flatten it
        resized_faces[i] = cv2.resize(face, face_size).ravel()

    pca = pickle.load(open(f"{__CURRENT_DIR__}/models/pca_n=50_affectnet.sav", 'rb'))
    eigen_faces = pca.transform(resized_faces)
    return eigen_faces
