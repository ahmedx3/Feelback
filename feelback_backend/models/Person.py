from .. import db
from .BaseModel import BaseModel
import enum


class GenderType(str, enum.Enum):
    Male = "Male"
    Female = "Female"


class Person(BaseModel):
    """
    Person Model
    """

    __tablename__ = 'Person'

    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Enum(GenderType), nullable=False)

    video_id = db.Column(db.String(64), db.ForeignKey('Video.id'), primary_key=True)
    emotions = db.relationship('Emotion', backref='person', lazy=True)
    attention = db.relationship('Attention', backref='person', lazy=True)

    def __init__(self, id: int, age: int, gender: GenderType):
        self.id = int(id)
        self.age = int(age)
        self.gender = GenderType[gender.capitalize()]

    def __repr__(self):
        return f"<Person(id={self.id} video_id={self.video_id} age={self.age} gender={self.gender})>"
