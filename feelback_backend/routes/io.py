from .. import app
from ..utils import video_utils
from flask import request, jsonify
from magic import Magic
from http import HTTPStatus as Status
import os
import hashlib


@app.post('/upload')
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
        "frame_count": video_utils.get_number_of_frames(filepath),
        "duration": video_utils.get_duration(filepath, digits=3)
    }), Status.OK
