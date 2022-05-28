from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy_utils import database_exists, create_database, drop_database
from .config import DevelopmentConfig

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)
db = SQLAlchemy(app)

if app.config['DROP_DATABASE_ON_STARTUP'] and database_exists(db.engine.url):
    drop_database(db.engine.url)
    print(f"[WARNING] Dropped {db.engine.url.drivername} database '{db.engine.url.database}'")

if not database_exists(db.engine.url):
    create_database(db.engine.url)


from .models import *

db.create_all()

from .utils.error_handlers import *
from .routes import *

app.register_blueprint(video_routes, url_prefix='/videos')
app.register_blueprint(video_analytics_routes, url_prefix='/videos/<video_id>/analytics')
