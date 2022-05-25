from .. import db


class Attention(db.Model):
    """
    Attention Model
    """

    __tablename__ = 'attentions'

    frame_number = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('persons.id'), primary_key=True)
    video_id = db.Column(db.String(64), db.ForeignKey('videos.id'), primary_key=True)

    attention = db.Column(db.Boolean, nullable=False)

    def __init__(self, frame_number: int, person_id: int, video_id: str, attention: bool):
        self.frame_number = frame_number
        self.person_id = person_id
        self.video_id = video_id
        self.attention = attention

    def __repr__(self):
        return f"<Attention {self.attention}>"
