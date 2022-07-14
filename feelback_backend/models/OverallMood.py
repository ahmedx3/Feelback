from .. import db
from .BaseModel import BaseModel
from typing import Iterable


class OverallMood(BaseModel):
    """
    Overall Mood Model
    """

    __tablename__ = 'OverallMood'

    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    video_id = db.Column(db.String(64), db.ForeignKey('Video.id'))

    time = db.Column(db.Float, nullable=False)
    mood = db.Column(db.Float, nullable=False)

    def __init__(self, time: float, mood: float):
        self.time = float(time)
        self.mood = float(mood)

    def to_json(self, populate: Iterable = (), exclude_columns: Iterable = (), include_foreign_keys=False):
        return BaseModel.to_json(self, populate, (*exclude_columns, 'OverallMood.id'), include_foreign_keys)

    def __repr__(self):
        return f"<OverallMood(video_id={self.video_id} time={self.time} mood={self.mood})>"
