from .. import db
from flask import request, jsonify
from flask import Blueprint
from http import HTTPStatus as Status
from ..models import Video, Attention, Emotion, Person
from .utils import require_video_processed, require_person_exists

"""
Note: <video_id> is in the url_prefix, therefore all the routes in this blueprint will have video_id as a parameter
"""

video_insights_routes = Blueprint('insights', __name__, url_prefix='/video/<video_id>/insights')


@video_insights_routes.get('/', strict_slashes=False)
@require_video_processed
def get_all_insights(video_id):
    """
    Get all insights in this video
    """

    attention_insights = get_attention_insights(video_id)[0].json["data"]
    age_insights = get_age_insights(video_id)[0].json["data"]
    emotions_insights = get_emotions_insights(video_id)[0].json["data"]
    gender_insights = get_gender_insights(video_id)[0].json["data"]
    number_of_persons = db.session.query(Person.id).filter_by(video_id=video_id).count()
    return jsonify({
        "status": "success",
        "data": {
            "number_of_persons": number_of_persons,
            "attention": attention_insights,
            "age": age_insights,
            "gender": gender_insights,
            "emotions": emotions_insights
        }
    }), Status.OK


@video_insights_routes.get('/attention')
@require_video_processed
def get_attention_insights(video_id):
    """
    Get attention (focus) insights in this video
    """

    attention = db.session.query(Attention.attention).filter_by(video_id=video_id, attention=True).count()
    total = db.session.query(Attention.attention).filter_by(video_id=video_id).count()
    return jsonify({"status": "success", "data": attention / total}), Status.OK


@video_insights_routes.get('/gender')
@require_video_processed
def get_gender_insights(video_id):
    """
    Get gender insights in this video
    """

    genders = db.session.query(Person.gender, db.func.count(Person.gender)).filter_by(video_id=video_id).group_by(
        Person.gender).all()
    total = db.session.query(Person).filter_by(video_id=video_id).count()

    genders = {gender.name: count / total for gender, count in genders}
    return jsonify({"status": "success", "data": genders}), Status.OK


@video_insights_routes.get('/age')
@require_video_processed
def get_age_insights(video_id):
    """
    Get age insights in this video
    """

    age = db.session.query(Person.age).filter_by(video_id=video_id).all()
    age = [a[0] for a in age]
    return jsonify({"status": "success", "data": age}), Status.OK


@video_insights_routes.get('/emotions')
@require_video_processed
def get_emotions_insights(video_id):
    """
    Get emotions insights in this video
    """

    emotions = db.session.query(Emotion.emotion, db.func.count(Emotion.emotion)).filter_by(video_id=video_id).group_by(
        Emotion.emotion).all()
    total = db.session.query(Emotion).filter_by(video_id=video_id).count()

    emotions = {emotion.name: count / total for emotion, count in emotions}
    return jsonify({"status": "success", "data": emotions}), Status.OK
