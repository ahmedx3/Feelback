from .. import db
from .Person import Person
from .BaseModel import BaseModel


class Attention(BaseModel):
    """
    Attention Model
    """

    __tablename__ = 'Attention'

    frame_number = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(64), primary_key=True)
    attention = db.Column(db.Boolean, nullable=False)

    __table_args__ = (db.ForeignKeyConstraint((person_id, video_id), (Person.id, Person.video_id)), {})

    def __init__(self, frame_number: int, person_id: int, video_id: str, attention: bool):
        self.frame_number = int(frame_number)
        self.person_id = int(person_id)
        self.video_id = str(video_id)
        self.attention = attention

    def __repr__(self):
        return f"<Attention({self.attention})>"
