from .. import db
from flask import request, jsonify
from flask import Blueprint
from http import HTTPStatus as Status
from ..models import Video, Attention, KeyMoment, Emotion, Person
from .utils import require_video_processed
from ..utils import io

"""
Note: <video_id> is in the url_prefix, therefore all the routes in this blueprint will have video_id as a parameter
"""

video_key_moments_routes = Blueprint('key_moments', __name__, url_prefix='/videos/<video_id>/key_moments')


@video_key_moments_routes.get('/<key_moment_id>/thumbnail')
@require_video_processed
def get_thumbnail(video_id, key_moment_id):
    """
    Get Video Key Moment Thumbnail from Feelback Server
    """

    return io.send_key_moment_thumbnail(video_id, key_moment_id)


@video_key_moments_routes.get('/<key_moment_id>')
@require_video_processed
def get_key_moment(video_id, key_moment_id):
    """
    Get Video Key Moment Info from Feelback Server
    """

    thumbnail_base_url = f"{request.host_url}api/v1/videos"
    key_moment = db.session.query(KeyMoment).filter_by(video_id=video_id, id=key_moment_id).first()
    return jsonify({"status": "success", "data": key_moment.to_json(base_url=thumbnail_base_url)}), Status.OK


@video_key_moments_routes.get('/', strict_slashes=False)
@require_video_processed
def get_all_key_moments(video_id):
    """
    Get All Video's Key Moment Info from Feelback Server
    """

    thumbnail_base_url = f"{request.host_url}api/v1/videos"
    key_moments = db.session.query(KeyMoment).filter_by(video_id=video_id).all()
    key_moments = [key_moment.to_json(base_url=thumbnail_base_url) for key_moment in key_moments]
    return jsonify({"status": "success", "data": key_moments}), Status.OK


