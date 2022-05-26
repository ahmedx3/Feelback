from .. import app, db
from ..utils import video_utils, Feelback
from .. import utils
from .utils import require_video_exists, require_video_processed
from flask import request, jsonify, send_from_directory
from flask import Blueprint
from werkzeug.security import safe_join
from magic import Magic
from http import HTTPStatus as Status
import os
import hashlib
from ..models import Video, Attention, Emotion, Person
from threading import Thread

video_routes = Blueprint('video', __name__)
__UPLOAD_FOLDER__ = app.config['UPLOAD_FOLDER']


def process_video_thread(video_id, video_filename, output_filename, frames_per_second):
    feelback = Feelback(video_filename, frames_per_second, output_filename=output_filename)
    feelback.run()

    store_feelback_data_in_database(video_id, feelback)


def store_feelback_data_in_database(video_id: str, feelback: Feelback):
    try:
        video = db.session.query(Video).filter_by(id=video_id).first()
        if video is None:
            video = Video(video_id, feelback.video_frame_count, feelback.video_duration)
        if video.finished_processing:
            return

        video.finished_processing = True

        for person_id, age, gender in feelback.persons:
            video.persons.append(Person(person_id, age, gender))

        db.session.add(video)

        for person_id, frame_number, emotion, attention in feelback.data:
            db.session.add(Emotion(frame_number, person_id, video_id, emotion))
            db.session.add(Attention(frame_number, person_id, video_id, attention))

        db.session.commit()
    except Exception as e:
        print("Failed to store in Database: ", e)
        db.session.rollback()


@video_routes.put('/<video_id>/process')
@require_video_exists
def process_video(video_id):
    if db.session.query(Video).filter_by(id=video_id).first().finished_processing:
        return jsonify({"status": "finished processing"}), Status.OK

    request_data: dict = request.get_json()
    frames_per_second = int(request_data.get('fps', 5))
    save_debug_video = utils.to_boolean(request_data.get("save_debug_video", False))

    video_filename = safe_join(__UPLOAD_FOLDER__, os.fspath(f"{video_id}.mp4"))
    output_filename = safe_join(__UPLOAD_FOLDER__, f"{video_id}_processed.mp4") if save_debug_video else None

    Thread(target=process_video_thread, args=(video_id, video_filename, output_filename, frames_per_second)).start()

    return jsonify({"status": "started processing"}), Status.ACCEPTED


@video_routes.post('/upload')
def upload_video():
    """
    Upload Video to Feelback Server
    """

    video = request.files['video']
    mime_type = Magic(mime=True).from_buffer(video.stream.read())

    if not mime_type.startswith('video'):
        return jsonify({"status": "error", "message": "Not a video file"}), Status.BAD_REQUEST

    video.seek(0)
    video_id = hashlib.sha1(video.stream.read()).hexdigest()

    if db.session.query(Video).filter_by(id=video_id).first() is not None:
        video = db.session.query(Video).filter_by(id=video_id).first()
    else:
        video.seek(0)
        filepath = safe_join(__UPLOAD_FOLDER__, f"{video_id}.mp4")
        video.save(filepath)

        video = Video(video_id, video_utils.get_number_of_frames(filepath), video_utils.get_duration(filepath, digits=3))
        db.session.add(video)
        db.session.commit()

    return jsonify({"status": "success", "video": video.to_json()}), Status.CREATED


@video_routes.get('/<video_id>')
@require_video_exists
def get_video(video_id):
    """
    Download Video from Feelback Server
    """

    processed = utils.to_boolean(request.args.get("processed", default=False))
    filename = f"{video_id}.mp4" if not processed else f"{video_id}.processed.mp4"

    return send_from_directory(__UPLOAD_FOLDER__, filename)


@video_routes.get('/<video_id>/info')
@require_video_exists
def get_video_info(video_id):
    """
    Get Video Info from Feelback Server
    """

    video = db.session.query(Video).filter_by(id=video_id).first()
    return jsonify({"status": "success", "video": video.to_json()}), Status.OK
