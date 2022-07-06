from .. import db
from .BaseModel import BaseModel
from typing import Iterable


class KeyMoment(BaseModel):
    """
    Key Moment Model
    """

    __tablename__ = 'KeyMoment'

    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    video_id = db.Column(db.String(64), db.ForeignKey('Video.id'))

    start = db.Column(db.Float, nullable=False)
    end = db.Column(db.Float, nullable=False)

    def __init__(self, start: float, end: float):
        self.start = float(start)
        self.end = float(end)

    def to_json(self, populate: Iterable = (), exclude_columns: Iterable = (), include_foreign_keys=False,
                base_url: str = None):
        json = BaseModel.to_json(self, populate, exclude_columns, include_foreign_keys)

        if base_url is not None:
            json["img_url"] = f"{base_url}/{self.video_id}/key_moments/{self.id}/thumbnail"

        return json

    def __repr__(self):
        return f"<KeyMoment(video_id={self.video_id} start={self.start} end={self.end})>"
