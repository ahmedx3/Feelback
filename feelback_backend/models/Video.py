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
    type = db.Column(db.Enum(VideoType), nullable=False, default=VideoType.Reaction, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    trailer_id = db.Column(db.String(64), db.ForeignKey('Video.id'), nullable=True, default=None)
    frame_count = db.Column(db.Integer, nullable=False)
    duration = db.Column(db.Float, nullable=False)
    finished_processing = db.Column(db.Boolean, nullable=False, default=False)
    progress = db.Column(db.Float, nullable=True, default=0.0)

    persons = db.relationship('Person', backref='video', lazy=True)
    key_moments = db.relationship('KeyMoment', backref='video', lazy=True)

    def __init__(self, id: str, filename: str, frame_count: int, duration: float, progress: float = 0.0,
                 finished_processing: bool = False):
        self.id = str(id)
        self.filename = str(filename)
        self.frame_count = int(frame_count)
        self.duration = float(duration)
        self.progress = float(progress)
        self.finished_processing = bool(finished_processing)

    @property
    def trailer(self):
        return db.object_session(self).query(Video).filter_by(id=self.trailer_id, type=VideoType.Trailer).first()

    @property
    def reactions(self):
        return db.object_session(self).query(Video).filter_by(trailer_id=self.id, type=VideoType.Reaction).all()

    def to_json(self, populate: Iterable = (), exclude_columns: Iterable = (), include_foreign_keys=False,
                base_url: str = None):
        json = BaseModel.to_json(self, populate, exclude_columns, include_foreign_keys)
        json['trailer_id'] = self.trailer_id
        json['reaction_ids'] = [reaction.id for reaction in self.reactions]

        if base_url is not None:
            json["img_url"] = f"{base_url}/{self.id}/thumbnail"

        return json

    def __repr__(self):
        return f"<Video(id={self.id} name={self.filename} frame_count={self.frame_count} duration={self.duration})>"
