from .. import app, db
from ..utils import video_utils, verbose, Feelback
from .. import utils
from .utils import require_video_exists, require_reaction_video_exists
from flask import request, jsonify, send_from_directory, send_file
from flask import Blueprint
from werkzeug.security import safe_join
from http import HTTPStatus as Status
import os
import hashlib
from ..models import Video, VideoType, Attention, Emotion, KeyMoment, Person, Position
from threading import Thread
import traceback

video_routes = Blueprint('videos', __name__, url_prefix='/videos')
__UPLOAD_FOLDER__ = app.config['UPLOAD_FOLDER']
__ANNOTATED_UPLOAD_FOLDER__ = app.config['ANNOTATED_UPLOAD_FOLDER']
__THUMBNAILS_FOLDER__ = app.config['THUMBNAILS_FOLDER']


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

        for person_id, age, gender in feelback.persons:
            video.persons.append(Person(person_id, age, gender))

        for start, end in feelback.key_moments_seconds:
            video.key_moments.append(KeyMoment(start, end))

        db.session.add(video)
        db.session.flush()  # Flush to database to get key moments autoincrement ids (does not do a transaction commit)

        for key_moment, (start, end) in zip(video.key_moments, feelback.key_moments_frames):
            video_filename = safe_join(__UPLOAD_FOLDER__, f"{video_id}.mp4")
            thumbnail_filename = safe_join(__THUMBNAILS_FOLDER__, f"{video_id}_key_moment_{key_moment.id}.jpg")
            video_utils.generate_thumbnail(video_filename, thumbnail_filename, (start + end) // 2)

        for person_id, frame_number, emotion, attention, face_position in feelback.data:
            db.session.add(Emotion(frame_number, person_id, video_id, emotion))
            db.session.add(Attention(frame_number, person_id, video_id, attention))
            db.session.add(Position(frame_number, person_id, video_id, *face_position))

        video.finished_processing = True
        video.progress = 100.0

        db.session.add(video)
        db.session.commit()

    except Exception as e:
        print("[Error] Failed to store in Database: ", e)
        traceback.print_exc()
        db.session.rollback()


@video_routes.put('/<video_id>')
@require_video_exists
def process_video(video_id):
    video = db.session.query(Video).filter_by(id=video_id).first()

    if video.finished_processing:
        return jsonify({"status": "finished processing"}), Status.OK

    if video.type == VideoType.Trailer:
        return jsonify({"status": "error", "message": "only reaction videos can be processed"}), Status.BAD_REQUEST

    request_data: dict = request.get_json()
    frames_per_second = request_data.get('fps', 5)
    if not str(frames_per_second).isdigit() or (str(frames_per_second).isdigit() and int(frames_per_second) < 1):
        frames_per_second = 'native'

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

    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "video file is required"}), Status.BAD_REQUEST

    if 'type' not in request.form:
        return jsonify({"status": "error", "message": "type parameter is required"}), Status.BAD_REQUEST

    video = request.files['video']
    video_type = VideoType[request.form['type'].capitalize()]

    trailer_id = request.form.get('trailer_id', None)

    if trailer_id is not None and trailer_id.strip() == '':
        trailer_id = None

    video.seek(0)
    video_id = hashlib.sha1(video.stream.read()).hexdigest()

    if db.session.query(Video).filter_by(id=video_id, type=video_type).first() is not None:
        video = db.session.query(Video).filter_by(id=video_id, type=video_type).first()
    else:
        video.seek(0)
        filepath = safe_join(__UPLOAD_FOLDER__, f"{video_id}.mp4")
        video.save(filepath)
        video_utils.generate_thumbnail(filepath, safe_join(__THUMBNAILS_FOLDER__, f"{video_id}.jpg"))

        filename = os.path.splitext(video.filename)[0]
        video = Video(video_id, filename, video_utils.get_number_of_frames(filepath), video_utils.get_duration(filepath, digits=3))

    video.type = video_type

    if video_type == VideoType.Reaction:
        if trailer_id is not None and db.session.query(Video).filter_by(id=trailer_id).first() is None:
            return jsonify({"status": "error", "message": "trailer video not found"}), Status.BAD_REQUEST

        trailer_video = db.session.query(Video).filter_by(id=trailer_id).first()
        if trailer_video is not None:
            video.trailer_id = trailer_id

            if video.duration > trailer_video.duration:
                video.duration = trailer_video.duration
                filepath = safe_join(__UPLOAD_FOLDER__, f"{video.id}.mp4")
                video_utils.trim_video(filepath, video.duration)

    db.session.add(video)
    db.session.commit()

    thumbnail_base_url = f"{request.host_url}api/v1/videos"
    return jsonify({"status": "success", "data": video.to_json(base_url=thumbnail_base_url)}), Status.CREATED


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


@video_routes.get('/<video_id>/trailer/download')
@require_reaction_video_exists(require_trailer=True)
def download_trailer_video(video_id):
    """
    Download The Trailer Video for this Reaction Video from Feelback Server
    This will trim the trailer to match the reaction video duration
    """

    video = db.session.query(Video).filter_by(id=video_id, type=VideoType.Reaction).first()
    filepath = safe_join(__UPLOAD_FOLDER__, f"{video_id}.mp4")
    filepath = video_utils.trim_video(filepath, video.duration, replace=False)
    return send_file(filepath)


@video_routes.get('/<video_id>/thumbnail')
@require_video_exists
def get_thumbnail(video_id):
    """
    Get Video Trailer Thumbnail from Feelback Server
    """

    return send_from_directory(__THUMBNAILS_FOLDER__, f"{video_id}.jpg")


@video_routes.get('/<video_id>/trailer/thumbnail')
@require_reaction_video_exists(require_trailer=True)
def get_trailer_thumbnail(video_id):
    """
    Get The Trailer Video Thumbnail for this Reaction Video from Feelback Server
    """

    trailer_id = db.session.query(Video.trailer_id).filter_by(id=video_id, type=VideoType.Reaction).first()[0]
    return send_from_directory(__THUMBNAILS_FOLDER__, f"{trailer_id}.jpg")


@video_routes.get('/<video_id>')
@require_video_exists
def get_video_info(video_id):
    """
    Get Video Info from Feelback Server
    """

    thumbnail_base_url = f"{request.host_url}api/v1/videos"
    video = db.session.query(Video).filter_by(id=video_id).first()
    return jsonify({"status": "success", "data": video.to_json(base_url=thumbnail_base_url)}), Status.OK


@video_routes.get('/<video_id>/trailer')
@require_reaction_video_exists(require_trailer=True)
def get_trailer_video_info(video_id):
    """
    Get The Trailer Video Info for this Reaction Video from Feelback Server
    """

    thumbnail_base_url = f"{request.host_url}api/v1/videos"
    video = db.session.query(Video).filter_by(id=video_id, type=VideoType.Reaction).first()
    json = video.trailer.to_json(base_url=thumbnail_base_url)
    json['duration'] = video.duration
    return jsonify({"status": "success", "data": json}), Status.OK


@video_routes.get('/', strict_slashes=False)
def get_all_videos_info(type=None):
    """
    Get All Videos Info from Feelback Server
    """

    thumbnail_base_url = f"{request.host_url}api/v1/videos"

    if type is None:
        videos = db.session.query(Video).all()
    else:
        videos = db.session.query(Video).filter_by(type=type).all()

    videos = [video.to_json(base_url=thumbnail_base_url) for video in videos]
    return jsonify({"status": "success", "data": videos}), Status.OK


@video_routes.get('/trailers')
def get_all_trailer_videos_info():
    """
    Get All Trailer Videos Info from Feelback Server
    """

    return get_all_videos_info(type=VideoType.Trailer)


@video_routes.get('/reactions')
def get_all_reaction_videos_info():
    """
    Get All Reaction Videos Info from Feelback Server
    """

    return get_all_videos_info(type=VideoType.Reaction)
