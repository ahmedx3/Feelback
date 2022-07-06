from typing import Iterable
from .. import db
from .BaseModel import BaseModel
import enum


class VideoType(str, enum.Enum):
    Trailer = "Trailer"
    Reaction = "Reaction"


class Video(BaseModel):
    """
    Video Model
    """

    __tablename__ = 'Video'

    id = db.Column(db.String(64), primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    type = db.Column(db.Enum(VideoType), nullable=False, default=VideoType.Reaction)
    trailer_id = db.Column(db.String(64), db.ForeignKey('Video.id'), nullable=True, default=None)
    frame_count = db.Column(db.Integer, nullable=False)
    duration = db.Column(db.Float, nullable=False)
    finished_processing = db.Column(db.Boolean, nullable=False, default=False)
    progress = db.Column(db.Float, nullable=True, default=0.0)

    trailer = db.relationship('Video', backref=db.backref('reaction', uselist=False), remote_side=[id], lazy=True)
    persons = db.relationship('Person', backref='video', lazy=True)

    def __init__(self, id: str, filename: str, frame_count: int, duration: float, progress: float = 0.0,
                 finished_processing: bool = False):
        self.id = str(id)
        self.filename = str(filename)
        self.frame_count = int(frame_count)
        self.duration = float(duration)
        self.progress = float(progress)
        self.finished_processing = bool(finished_processing)

    def to_json(self, populate: Iterable = (), exclude_columns: Iterable = (), include_foreign_keys=False,
                base_url: str = None):
        json = BaseModel.to_json(self, populate, exclude_columns, include_foreign_keys)
        json['trailer_id'] = self.trailer_id
        json['reaction_id'] = self.reaction.id if self.reaction is not None else None

        if base_url is not None:
            json["img_url"] = f"{base_url}/{self.id}/thumbnail"

        return json

    def __repr__(self):
        return f"<Video(id={self.id} name={self.filename} frame_count={self.frame_count} duration={self.duration})>"
