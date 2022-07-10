from .. import db
from .Person import Person
from .BaseModel import BaseModel
import enum


class EmotionType(str, enum.Enum):
    Happy = "Happy"
    Sadness = "Sad"
    Angry = "Angry"
    Surprise = "Surprise"
    Disgust = "Disgust"
    Afraid = "Afraid"
    Neutral = "Neutral"


class Emotion(BaseModel):
    """
    Emotion Model
    """

    __tablename__ = 'Emotion'

    time = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(64), primary_key=True)
    emotion = db.Column(db.Enum(EmotionType), nullable=False)

    __table_args__ = (db.ForeignKeyConstraint((person_id, video_id), (Person.id, Person.video_id)), {})

    def __init__(self, time: int, person_id: int, video_id: str, emotion: EmotionType):
        self.time = int(time)
        self.person_id = int(person_id)
        self.video_id = str(video_id)
        self.emotion = EmotionType[emotion.capitalize()]

    def __repr__(self):
        return f"<Emotion({self.emotion})>"
