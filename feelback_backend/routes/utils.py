from flask import jsonify
from functools import wraps
from http import HTTPStatus as Status
from ..models import Video, Attention, Emotion, Person
from .. import db


def require_video_exists(route_function):
    """
    Decorator to check if video exists in database, if not return 404

    Note:
        It requires that the `route_function` has a parameter called `video_id`
    """

    @wraps(route_function)
    def decorator(*args, **kwargs):
        video_id = kwargs['video_id']
        video = db.session.query(Video).filter_by(id=video_id).first()

        if video is None:
            return jsonify({"status": "error", "message": "Video not found"}), Status.NOT_FOUND

        return route_function(*args, **kwargs)

    return decorator


def require_video_processed(route_function):
    """
    Decorator to check if video exists in database, and if it has finished processing
    If didn't exist return 404
    If didn't finish processing return 400

    Note:
        It requires that the `route_function` has a parameter called `video_id`
    """

    @wraps(route_function)
    def decorator(*args, **kwargs):
        video_id = kwargs['video_id']
        video = db.session.query(Video).filter_by(id=video_id).first()

        if video is None:
            return jsonify({"status": "error", "message": "Video not found"}), Status.NOT_FOUND

        if not video.finished_processing:
            return jsonify({"status": "error", "message": "Video not finished processing"}), Status.BAD_REQUEST

        return route_function(*args, **kwargs)

    return decorator
