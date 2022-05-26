from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy_utils import database_exists, create_database
from .config import DevelopmentConfig

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)
db = SQLAlchemy(app)

if not database_exists(db.engine.url):
    create_database(db.engine.url)


from .models import *

db.create_all()

from .routes import *

app.register_blueprint(video_routes, url_prefix='/video')
