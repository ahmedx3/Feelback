from .. import db
from flask import request, jsonify
from flask import Blueprint
from http import HTTPStatus as Status
from ..models import Video, Attention, Emotion, Person
from .utils import require_video_processed, require_person_exists

"""
Note: <video_id> is in the url_prefix, therefore all the routes in this blueprint will have video_id as a parameter
"""

video_analytics_routes = Blueprint('analytics', __name__, url_prefix='/video/<video_id>/analytics')


@video_analytics_routes.get('/', strict_slashes=False)
@require_video_processed
def get_all_video_data(video_id):
    """
    Get all data analytics about this video
    """

    video = db.session.query(Video).filter_by(id=video_id).first()
    return jsonify({
        "status": "success",
        "data": video.to_json(populate=["persons", "emotions", "attention", "positions"])
    }), Status.OK


@video_analytics_routes.get('/persons')
@require_video_processed
def get_all_persons_data(video_id):
    """
    Get all data analytics about persons in this video
    """

    persons = db.session.query(Person).filter_by(video_id=video_id).all()
    persons = [person.to_json(populate=["emotions", "attention", "positions"]) for person in persons]
    return jsonify({"status": "success", "data": persons}), Status.OK


@video_analytics_routes.get('/person/<person_id>')
@require_video_processed
@require_person_exists
def get_person_data(video_id, person_id):
    """
    Get all data analytics about specific person in this video
    """

    person = db.session.query(Person).filter_by(id=person_id, video_id=video_id).first()

    return jsonify({
        "status": "success",
        "data": person.to_json(populate=["emotions", "attention", "positions"])
    }), Status.OK


@video_analytics_routes.get('/person/<person_id>/attention')
@require_video_processed
@require_person_exists
def get_person_attention(video_id, person_id):
    """
    Get specific person's attention in each frame in this video
    """

    person = db.session.query(Person).filter_by(id=person_id, video_id=video_id).first()
    return jsonify({"status": "success", "data": person.to_json(populate=["attention"])}), Status.OK


@video_analytics_routes.get('/person/<person_id>/emotions')
@require_video_processed
@require_person_exists
def get_person_emotions(video_id, person_id):
    """
    Get specific person's emotion in each frame in this video
    """

    person = db.session.query(Person).filter_by(id=person_id, video_id=video_id).first()
    return jsonify({"status": "success", "data": person.to_json(populate=["emotions"])}), Status.OK


@video_analytics_routes.get('/person/<person_id>/position')
@require_video_processed
@require_person_exists
def get_person_face_position(video_id, person_id):
    """
    Get specific person's face position in each frame in this video
    """

    person = db.session.query(Person).filter_by(id=person_id, video_id=video_id).first()
    return jsonify({"status": "success", "data": person.to_json(populate=["positions"])}), Status.OK
