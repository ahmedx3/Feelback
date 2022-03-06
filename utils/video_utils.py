import cv2


def get_duration(video: cv2.VideoCapture) -> float:
    """
    Get Video Duration in Seconds
    :param video: OpenCV VideoCapture
    :return: Video duration in seconds
    """
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_seconds = frame_count / frames_per_second

    return duration_seconds
