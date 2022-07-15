from .. import db
from ..utils import video_utils, verbose, Feelback, io
from .. import utils
from .utils import require_video_exists, require_reaction_video_exists
from flask import request, jsonify, send_file
from flask import Blueprint
from http import HTTPStatus as Status
import os
import hashlib
from ..models import Video, VideoType, Attention, Emotion, KeyMoment, Person, Position, OverallMood
from threading import Thread
import traceback

video_routes = Blueprint('videos', __name__, url_prefix='/videos')


def generate_key_moment_thumbnail(video_id, key_moment_id, start, end, type=VideoType.Reaction):
    if type == VideoType.Reaction:
        thumbnail_filename = io.get_key_moment_thumbnail_path(video_id, key_moment_id)
        video_filename = io.get_video_path(video_id)
        video_utils.generate_thumbnail(video_filename, thumbnail_filename, (start + end) // 2)

    elif type == VideoType.Trailer:
        trailer = db.session.query(Video).filter_by(id=video_id, type=VideoType.Reaction).first().trailer
        if trailer:
            thumbnail_filename = io.get_trailer_key_moment_thumbnail_path(video_id, key_moment_id)
            video_filename = io.get_video_path(trailer.id)
            video_utils.generate_thumbnail(video_filename, thumbnail_filename, (start + end) // 2)


@require_video_exists
def process_video_thread(video_id, video_filename, output_filename, annotations, frames_per_second):
    feelback = Feelback(video_filename, frames_per_second, verbose.Level.OFF)

    video = db.session.query(Video).filter_by(id=video_id).first()

    thread = Thread(target=lambda: feelback.run())
    thread.start()

    while thread.is_alive():
        video.progress = feelback.progress()
        db.session.add(video)
        db.session.commit()
        thread.join(timeout=3)

    feelback.save_postprocess_video(output_filename, annotations)
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
            generate_key_moment_thumbnail(video_id, key_moment.id, start, end, type=VideoType.Reaction)
            generate_key_moment_thumbnail(video_id, key_moment.id, start, end, type=VideoType.Trailer)

        for time, person_id, emotion, attention in feelback.data_second_by_second:
            db.session.add(Emotion(time, person_id, video_id, emotion))
            db.session.add(Attention(time, person_id, video_id, attention))

        for time, mood in feelback.mood_data:
            video.mood.append(OverallMood(time, mood))

        for frame_number, person_id, face_position in feelback.faces_positions_frame_by_frame:
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
    fps = request_data.get('fps', 5)
    if not str(fps).isdigit() or (str(fps).isdigit() and int(fps) < 1):
        fps = 'native'

    save_annotated_video = utils.to_boolean(request_data.get("save_annotated_video", False))
    annotations = request_data.get('annotations', [])

    video_filename = io.get_video_path(video_id)
    output_filename = io.get_annotated_video_path(video_id) if save_annotated_video else None

    Thread(target=process_video_thread, args=(video_id, video_filename, output_filename, annotations, fps)).start()

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
        filepath = io.get_video_path(video_id)
        video.save(filepath)
        video_utils.generate_thumbnail(filepath, io.get_video_thumbnail_path(video_id))

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
                filepath = io.get_video_path(video_id)
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
        return io.send_annotated_video(video_id)
    return io.send_video(video_id)


@video_routes.get('/<video_id>/trailer/download')
@require_reaction_video_exists(require_trailer=True)
def download_trailer_video(video_id):
    """
    Download The Trailer Video for this Reaction Video from Feelback Server
    This will trim the trailer to match the reaction video duration
    """

    video = db.session.query(Video).filter_by(id=video_id, type=VideoType.Reaction).first()
    filepath = io.get_video_path(video.trailer_id)
    filepath = video_utils.trim_video(filepath, video.duration, replace=False, tolerance=1)
    return send_file(filepath)


@video_routes.get('/<video_id>/thumbnail')
@require_video_exists
def get_thumbnail(video_id):
    """
    Get Video Trailer Thumbnail from Feelback Server
    """

    return io.send_video_thumbnail(video_id)


@video_routes.get('/<video_id>/trailer/thumbnail')
@require_reaction_video_exists(require_trailer=True)
def get_trailer_thumbnail(video_id):
    """
    Get The Trailer Video Thumbnail for this Reaction Video from Feelback Server
    """

    trailer_id = db.session.query(Video.trailer_id).filter_by(id=video_id, type=VideoType.Reaction).first()[0]
    return io.send_video_thumbnail(trailer_id)


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
