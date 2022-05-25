from .. import db
import enum


class GenderType(enum.Enum):
    Male = 0
    Female = 1


class Person(db.Model):
    """
    Person Model
    """

    __tablename__ = 'persons'

    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Enum(GenderType), nullable=False)

    video_id = db.Column(db.String(64), db.ForeignKey('videos.id'), primary_key=True)
    emotions = db.relationship('Emotion', backref='person', lazy=True)
    attention = db.relationship('Attention', backref='person', lazy=True)

    def __init__(self, id: int, age: int, gender: GenderType):
        self.id = id
        self.age = age
        self.gender = gender

    def __repr__(self):
        return f"<Person {self.id} video_id: {self.video_id} age: {self.age} gender: {self.gender}>"
