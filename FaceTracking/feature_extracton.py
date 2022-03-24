import os
import pickle
import numpy as np
import cv2

__CURRENT_DIR__ = os.path.dirname(os.path.abspath(__file__))
__PARENT_DIR__ = os.path.dirname(__CURRENT_DIR__)


def eigen_faces_features(frame_image: np.ndarray, faces_positions: np.ndarray) -> np.ndarray:
    face_size = (48, 48)
    faces = np.zeros((faces_positions.shape[0], np.prod(face_size)))
    for i, (x1, y1, x2, y2) in enumerate(faces_positions):
        face = frame_image[y1:y2, x1:x2]
        # Resize face to a fixed size then flatten it
        face = cv2.resize(face, face_size).ravel()
        faces[i] = face

    pca = pickle.load(open(f"{__CURRENT_DIR__}/models/pca_n=50_affectnet.sav", 'rb'))
    eigen_faces = pca.transform(faces)
    return eigen_faces
