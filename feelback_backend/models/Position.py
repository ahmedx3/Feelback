from .. import db
from .Person import Person
from .BaseModel import BaseModel


class Position(BaseModel):
    """
    Face Position Model in a frame
    """

    __tablename__ = 'Position'

    frame_number = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(64), primary_key=True)
    x1 = db.Column(db.Integer, nullable=False)
    y1 = db.Column(db.Integer, nullable=False)
    x2 = db.Column(db.Integer, nullable=False)
    y2 = db.Column(db.Integer, nullable=False)

    __table_args__ = (db.ForeignKeyConstraint((person_id, video_id), (Person.id, Person.video_id)), {})

    def __init__(self, frame_number: int, person_id: int, video_id: str, x1: int, y1: int, x2: int, y2: int):
        self.frame_number = int(frame_number)
        self.person_id = int(person_id)
        self.video_id = str(video_id)
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)

    def __repr__(self):
        return f"<Position(x1: {self.x1} y1: {self.y1} x2: {self.x2} y2: {self.y2})>"
