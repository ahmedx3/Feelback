from typing import Tuple
import numpy as np
import numpy.typing as npt

"""
Bounding Box Type is tuple of (x1, y1, x2, y2)
"""
BoundingBox = Tuple[int, int, int, int]


def intersection_over_union(rect_a: BoundingBox, rect_b: npt.NDArray[BoundingBox]) -> npt.NDArray[float]:
    """
    Calculate Intersection Over Union for two bounding boxes A, B.

    This function is vectorized for B, i.e. Calculates the IOU between rect_a, and every rect in rect_b

    Args:
        rect_a (BoundingBox): Bounding Box A, (x1, y1, x2, y2)
        rect_b (np.ndarray[BoundingBox]): Numpy Array of Bounding Boxes B, [(x1, y1, x2, y2)]

    Returns:
        iou: Numpy Array of the ratios of intersection area over union area
    """

    if rect_a is None or len(rect_a) == 0 or rect_b is None or len(rect_b) == 0:
        return np.zeros(1)

    x1, y1, x2, y2 = 0, 1, 2, 3

    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = np.maximum(rect_a[x1], rect_b[:, x1])
    x_b = np.minimum(rect_a[x2], rect_b[:, x2])
    y_a = np.maximum(rect_a[y1], rect_b[:, y1])
    y_b = np.minimum(rect_a[y2], rect_b[:, y2])

    intersection_area = np.maximum(0, x_b - x_a + 1) * np.maximum(0, y_b - y_a + 1)

    area_a = (rect_a[x2] - rect_a[x1] + 1) * (rect_a[y2] - rect_a[y1] + 1)
    area_b = (rect_b[:, x2] - rect_b[:, x1] + 1) * (rect_b[:, y2] - rect_b[:, y1] + 1)

    union_area = area_a + area_b - intersection_area
    iou = intersection_area / union_area

    return iou
