from .. import app, db
from ..utils import video_utils, verbose, Feelback
from .. import utils
from .utils import require_video_exists
from flask import request, jsonify, send_from_directory
from flask import Blueprint
from werkzeug.security import safe_join
# from magic import Magic
from http import HTTPStatus as Status
import os
import hashlib
from ..models import Video, Attention, Emotion, Person, Position
from threading import Thread
import traceback

video_routes = Blueprint('videos', __name__, url_prefix='/videos')
__UPLOAD_FOLDER__ = app.config['UPLOAD_FOLDER']
__ANNOTATED_UPLOAD_FOLDER__ = app.config['ANNOTATED_UPLOAD_FOLDER']


@require_video_exists
def process_video_thread(video_id, video_filename, output_filename, frames_per_second):
    feelback = Feelback(video_filename, frames_per_second, verbose.Level.OFF)

    video = db.session.query(Video).filter_by(id=video_id).first()

    thread = Thread(target=lambda: feelback.run())
    thread.start()

    while thread.is_alive():
        video.progress = feelback.progress()
        db.session.add(video)
        db.session.commit()
        thread.join(timeout=3)

    feelback.save_postprocess_video(output_filename)
    store_feelback_data_in_database(video_id, feelback)


@require_video_exists
def store_feelback_data_in_database(video_id: str, feelback: Feelback):
    try:
        video = db.session.query(Video).filter_by(id=video_id).first()

        if video.finished_processing:
            return

        video.finished_processing = True
        video.progress = 100.0

        for person_id, age, gender in feelback.persons:
            video.persons.append(Person(person_id, age, gender))

        db.session.add(video)

        for person_id, frame_number, emotion, attention, face_position in feelback.data:
            db.session.add(Emotion(frame_number, person_id, video_id, emotion))
            db.session.add(Attention(frame_number, person_id, video_id, attention))
            db.session.add(Position(frame_number, person_id, video_id, *face_position))

        db.session.commit()
    except Exception as e:
        print("[Error] Failed to store in Database: ", e)
        traceback.print_exc()
        db.session.rollback()


@video_routes.put('/<video_id>')
@require_video_exists
def process_video(video_id):
    if db.session.query(Video).filter_by(id=video_id).first().finished_processing:
        return jsonify({"status": "finished processing"}), Status.OK

    request_data: dict = request.get_json()
    frames_per_second = request_data.get('fps', 5)
    save_annotated_video = utils.to_boolean(request_data.get("save_annotated_video", False))

    video_filename = safe_join(__UPLOAD_FOLDER__, os.fspath(f"{video_id}.mp4"))
    output_filename = safe_join(__ANNOTATED_UPLOAD_FOLDER__, f"{video_id}.mp4") if save_annotated_video else None

    Thread(target=process_video_thread, args=(video_id, video_filename, output_filename, frames_per_second)).start()

    return jsonify({"status": "started processing"}), Status.ACCEPTED


@video_routes.post('/', strict_slashes=False)
def upload_video():
    """
    Upload Video to Feelback Server
    """

    video = request.files['video']
    # mime_type = Magic(mime=True).from_buffer(video.stream.read())

    # if not mime_type.startswith('video'):
    #     return jsonify({"status": "error", "message": "Not a video file"}), Status.BAD_REQUEST

    video.seek(0)
    video_id = hashlib.sha1(video.stream.read()).hexdigest()

    print(f"Uploading video {video.filename}")

    if db.session.query(Video).filter_by(id=video_id).first() is not None:
        video = db.session.query(Video).filter_by(id=video_id).first()
    else:
        video.seek(0)
        filepath = safe_join(__UPLOAD_FOLDER__, f"{video_id}.mp4")
        video.save(filepath)

        filename = os.path.splitext(video.filename)[0]
        video = Video(video_id, filename, video_utils.get_number_of_frames(filepath), video_utils.get_duration(filepath, digits=3))
        db.session.add(video)
        db.session.commit()

    return jsonify({"status": "success", "data": video.to_json()}), Status.CREATED


@video_routes.get('/<video_id>/download')
@require_video_exists
def download_video(video_id):
    """
    Download Video from Feelback Server
    """

    annotated = utils.to_boolean(request.args.get("annotated", default=False))
    if annotated:
        return send_from_directory(__ANNOTATED_UPLOAD_FOLDER__, f"{video_id}.mp4")
    return send_from_directory(__UPLOAD_FOLDER__, f"{video_id}.mp4")


@video_routes.get('/<video_id>')
@require_video_exists
def get_video_info(video_id):
    """
    Get Video Info from Feelback Server
    """

    video = db.session.query(Video).filter_by(id=video_id).first()
    return jsonify({"status": "success", "data": video.to_json()}), Status.OK


@video_routes.get('/', strict_slashes=False)
def get_all_videos_info():
    """
    Get Video Info from Feelback Server
    """

    videos = db.session.query(Video).all()
    videos = [video.to_json() for video in videos]
    return jsonify({"status": "success", "data": videos}), Status.OK
