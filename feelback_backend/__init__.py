from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from .config import DevelopmentConfig

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)
db = SQLAlchemy(app)

from .models import *

db.create_all()

from .routes.io import *
