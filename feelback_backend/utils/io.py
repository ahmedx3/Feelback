from .. import app
from flask import send_from_directory
from werkzeug.security import safe_join


__UPLOAD_FOLDER__ = app.config['UPLOAD_FOLDER']
__ANNOTATED_UPLOAD_FOLDER__ = app.config['ANNOTATED_UPLOAD_FOLDER']
__THUMBNAILS_FOLDER__ = app.config['THUMBNAILS_FOLDER']


def send_annotated_video(video_id):
    """
    Send Annotated Video from Feelback Server
    """

    return send_from_directory(__ANNOTATED_UPLOAD_FOLDER__, f"{video_id}.mp4")


def get_annotated_video_path(video_id):
    """
    Get Annotated Video Path from Feelback Server
    """

    return safe_join(__ANNOTATED_UPLOAD_FOLDER__, f"{video_id}.mp4")


def send_video(video_id):
    """
    Send Video from Feelback Server
    """

    return send_from_directory(__UPLOAD_FOLDER__, f"{video_id}.mp4")


def get_video_path(video_id):
    """
    Get Video Path from Feelback Server
    """

    return safe_join(__UPLOAD_FOLDER__, f"{video_id}.mp4")


def send_video_thumbnail(video_id):
    """
    Send Thumbnail from Feelback Server
    """

    return send_from_directory(__THUMBNAILS_FOLDER__, f"{video_id}.jpg")


def get_video_thumbnail_path(video_id):
    """
    Get Thumbnail Path from Feelback Server
    """

    return safe_join(__THUMBNAILS_FOLDER__, f"{video_id}.jpg")


def send_key_moment_thumbnail(video_id, key_moment_id):
    """
    Send Key Moment Thumbnail from Feelback Server
    """

    return send_from_directory(__THUMBNAILS_FOLDER__, f"{video_id}_key_moment_{key_moment_id}.jpg")


def get_key_moment_thumbnail_path(video_id, key_moment_id):
    """
    Get Key Moment Thumbnail Path from Feelback Server
    """

    return safe_join(__THUMBNAILS_FOLDER__, f"{video_id}_key_moment_{key_moment_id}.jpg")


def send_key_moment_trailer_thumbnail(video_id, key_moment_id):
    """
    Send Key Moment Thumbnail of The Trailer Video from Feelback Server
    """

    return send_from_directory(__THUMBNAILS_FOLDER__, f"{video_id}_key_moment_trailer_{key_moment_id}.jpg")


def get_trailer_key_moment_thumbnail_path(video_id, key_moment_id):
    """
    Get Key Moment Thumbnail Path of The Trailer Video from Feelback Server
    """

    return safe_join(__THUMBNAILS_FOLDER__, f"{video_id}_key_moment_trailer_{key_moment_id}.jpg")
