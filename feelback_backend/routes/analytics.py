from .. import db
from flask import request, jsonify
from flask import Blueprint
from http import HTTPStatus as Status
from ..models import Video, Attention, Emotion, Person
from .utils import require_video_processed

"""
Note: <video_id> is in the url_prefix, therefore all the routes in this blueprint will have video_id as a parameter
"""

video_analytics_routes = Blueprint('analytics', __name__, url_prefix='/video/<video_id>/analytics')


@video_analytics_routes.get('/')
@video_analytics_routes.get('')
@require_video_processed
def get_all_video_data(video_id):
    """
    Get all data analytics about this video
    """

    video = db.session.query(Video).filter_by(id=video_id).first()
    return jsonify({"status": "success", "data": video.to_json(populate=["persons", "emotions", "attention"])}), Status.OK


@video_analytics_routes.get('/persons')
@require_video_processed
def get_all_persons_data(video_id):
    """
    Get all data analytics about persons in this video
    """

    persons = db.session.query(Person).filter_by(video_id=video_id).all()
    persons = [person.to_json(populate=["emotions", "attention"]) for person in persons]
    return jsonify({"status": "success", "data": persons}), Status.OK


@video_analytics_routes.get('/person/<person_id>')
@require_video_processed
def get_person_data(video_id, person_id):
    """
    Get all data analytics about specific person in this video
    """

    person = db.session.query(Person).filter_by(id=person_id, video_id=video_id).first()
    if person is None:
        return jsonify({"status": "person not found"}), Status.NOT_FOUND

    return jsonify({"status": "success", "data": person.to_json(populate=["emotions", "attention"])}), Status.OK
