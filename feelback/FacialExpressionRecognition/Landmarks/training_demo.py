import util
import menpo.io as mio
import menpodetect
import pickle
from cascade_forest import CascadeForestBuilder
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
# import dlib

# read images
ibugin = util.read_images("../train/300w/01_Indoor/", normalise=True)

# face_detector = dlib.get_frontal_face_detector()
face_detector = menpodetect.load_dlib_frontal_face_detector()

train_gt_shapes = util.get_gt_shapes(ibugin)
train_boxes = util.get_bounding_boxes(ibugin, train_gt_shapes, face_detector)

# n_landmarks: number of landmarskd
# n_forests: number of regressor in cascade
# n_trees: number of trees in each regressor
# tree_depth: tree depth
# n_perturbations: number of initializations for each training example
# n_test_split: number of randomly generated candidate split for each node of tree
# n_pixels: number of pixel locations are sampled from the image
# kappa: range of extracted pixel around the current estimated landmarks position
# lr: learning rate
cascade_forest_builder = CascadeForestBuilder(n_landmarks=68,n_forests=10,n_trees=500,
                                tree_depth=5,n_perturbations=20,n_test_split=20,n_pixels=400,kappa=.3,lr=.1)


# training model
model = cascade_forest_builder.build(ibugin, train_gt_shapes, train_boxes)
# save model
pickle.dump(model, open('./model/ert_ibug_training.sav', 'wb'))

# test model
ibug_exam = util.read_images("../train/300w/01_Indoor/indoor_001.*",normalise=True)
ibug_exam_shapes = util.get_gt_shapes(ibug_exam)
ibug_exam_boxes = util.get_bounding_boxes(ibug_exam, ibug_exam_shapes, face_detector)
ibug_exam = ibug_exam[0]
# model = hickle.load(open('./model/ert_ibug_training.hkl', 'rb'))
model = pickle.load(open('./model/ert_ibug_training.sav', 'rb'))

init_shapes, fin_shapes = model.apply(ibug_exam,[ibug_exam_boxes[0]])
try:
    ibug_exam.landmarks.pop('dlib_0')
except:
    pass
print("========================================")
print("========================================")
print("========================================")

print(fin_shapes.points)

print("========================================")
print("========================================")
print("========================================")

ibug_exam_gt = deepcopy(ibug_exam)
ibug_exam_gt.view_landmarks()
cv2.waitKey(0)

print("========================================")

ibug_exam.landmarks['PTS'].points = fin_shapes[0].points
ibug_exam_predict = deepcopy(ibug_exam)
ibug_exam_predict.view_landmarks(marker_face_colour='y',marker_edge_colour='y')
cv2.waitKey(0)