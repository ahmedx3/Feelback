from .. import app
from ..utils import video_utils
from .. import utils
from flask import request, jsonify, send_from_directory
from flask import Blueprint
from werkzeug.security import safe_join
from magic import Magic
from http import HTTPStatus as Status
import os
import hashlib


video_routes = Blueprint('video', __name__)


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
    sha1sum = hashlib.sha1(video.stream.read()).hexdigest()

    video.seek(0)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{sha1sum}.mp4")
    video.save(filepath)

    return jsonify({
        "status": "success",
        "id": sha1sum,
        "finished_processing": False,  # TODO: Check if video is finished processing
        "frame_count": video_utils.get_number_of_frames(filepath),
        "duration": video_utils.get_duration(filepath, digits=3)
    }), Status.OK


@video_routes.get('/<video_id>')
def get_video(video_id):
    """
    Download Video from Feelback Server
    """

    processed = utils.to_boolean(request.args.get("processed", default=False))
    filename = f"{video_id}.mp4" if not processed else f"{video_id}.processed.mp4"

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@video_routes.get('/<video_id>/info')
def get_video_info(video_id):
    """
    Get Video Info from Feelback Server
    """
    filepath = safe_join(os.fspath(app.config['UPLOAD_FOLDER']), os.fspath(f"{video_id}.mp4"))

    if filepath is None:
        return jsonify({"status": "error", "message": "Video not found"}), Status.NOT_FOUND

    return jsonify({
        "status": "success",
        "id": video_id,
        "finished_processing": False,  # TODO: Check if video is finished processing
        "frame_count": video_utils.get_number_of_frames(filepath),
        "duration": video_utils.get_duration(filepath, digits=3)
    }), Status.OK
