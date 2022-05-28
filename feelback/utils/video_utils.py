from typing import Union
import cv2
import math
from . import io


def get_duration(video: Union[cv2.VideoCapture, str], digits: int = None) -> float:
    """
    Get Video Duration in Seconds

    Args:
        video: OpenCV VideoCapture object or video file path
        digits (int): Round duration to n decimal places

    Returns:
        Video duration in seconds
    """

    if isinstance(video, str):
        video = io.read_video(video)

    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_seconds = frame_count / frames_per_second

    return round(duration_seconds, digits)


def get_current_time(video: Union[cv2.VideoCapture, str]) -> float:
    """
    Get Current Video Time in Seconds

    Args:
        video: OpenCV VideoCapture object or video file path

    Returns:
        Current Video Time in seconds
    """

    if isinstance(video, str):
        video = io.read_video(video)

    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    current_time = current_frame / frames_per_second
    return current_time


def get_dimensions(video: Union[cv2.VideoCapture, str]) -> tuple:
    """
    Get Video Dimensions (Width, Height)

    Args:
        video: OpenCV VideoCapture object or video file path

    Returns:
        Video dimensions (width, height)
    """

    if isinstance(video, str):
        video = io.read_video(video)

    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return video_width, video_height


def get_number_of_frames(video: Union[cv2.VideoCapture, str]) -> int:
    """
    Get Video Number of Frames

    Args:
        video: OpenCV VideoCapture object or video file path

    Returns:
        Video number of frames
    """

    if isinstance(video, str):
        video = io.read_video(video)

    video_frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return video_frames_count


def get_fps(video: Union[cv2.VideoCapture, str], digits: int = None) -> int:
    """
    Get Video FPS (Frames Per Second)

    Args:
        video: OpenCV VideoCapture object or video file path
        digits (int): Round fps to n decimal places

    Returns:
        Video FPS (frames per second)
    """

    if isinstance(video, str):
        video = io.read_video(video)

    video_fps = video.get(cv2.CAP_PROP_FPS)
    return round(video_fps, digits)


def create_output_video(original_video: cv2.VideoCapture, output_filename: str, framerate: int):
    """
    Create Writable Output Video with the same dimensions as the original video, and the specified number of frames
    per second, and keep the original video's duration.

    Args:
        original_video: OpenCV VideoCapture object
        output_filename (str): Output video file path
        framerate (int): Number of frames to process each second

    Returns:
        Video Writer object
    """

    if output_filename is None:
        return

    original_video_fps = get_fps(original_video, digits=20)
    original_video_total_frames = get_number_of_frames(original_video)
    output_total_frames = math.ceil(original_video_total_frames / (original_video_fps / framerate))
    accurate_frame_rate = (output_total_frames / original_video_total_frames) * original_video_fps

    dimensions = get_dimensions(original_video)
    output = cv2.VideoWriter(f"{output_filename}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), accurate_frame_rate, dimensions)
    return output
