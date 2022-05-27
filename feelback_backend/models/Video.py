from .. import db
from .BaseModel import BaseModel


class Video(BaseModel):
    """
    Video Model
    """

    __tablename__ = 'Video'

    id = db.Column(db.String(64), primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    frame_count = db.Column(db.Integer, nullable=False)
    duration = db.Column(db.Float, nullable=False)
    finished_processing = db.Column(db.Boolean, nullable=False, default=False)

    persons = db.relationship('Person', backref='video', lazy=True)

    def __init__(self, id: str, filename: str, frame_count: int, duration: float, finished_processing: bool = False):
        self.id = str(id)
        self.filename = str(filename)
        self.frame_count = int(frame_count)
        self.duration = float(duration)
        self.finished_processing = bool(finished_processing)

    def __repr__(self):
        return f"<Video(id={self.id} name={self.filename} frame_count={self.frame_count} duration={self.duration})>"
