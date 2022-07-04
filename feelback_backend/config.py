import dotenv
import os
from . import utils


class Config:
    """
    Flask Base configuration
    """

    dotenv.load_dotenv(verbose=True)

    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    JSON_SORT_KEYS = utils.to_boolean(os.environ.get("JSON_SORT_KEYS", False))
    SQLALCHEMY_TRACK_MODIFICATIONS = utils.to_boolean(os.environ.get("SQLALCHEMY_TRACK_MODIFICATIONS", False))
    SQLALCHEMY_ECHO = utils.to_boolean(os.environ.get("SQLALCHEMY_ECHO", False))
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")
    DATABASE_URL = os.environ.get("DATABASE_URL")
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
    ANNOTATED_UPLOAD_FOLDER = os.environ.get("ANNOTATED_UPLOAD_FOLDER", "annotated_uploads")
    FLASK_RUN_PORT = os.environ.get("FLASK_RUN_PORT", 5000)
    FLASK_RUN_HOST = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    if not os.path.exists(ANNOTATED_UPLOAD_FOLDER):
        os.makedirs(ANNOTATED_UPLOAD_FOLDER)


class ProductionConfig(Config):
    """
    Flask Production configuration
    """

    ENV = "production"
    DEVELOPMENT = utils.to_boolean(os.environ.get("DEVELOPMENT", False))
    DEBUG = utils.to_boolean(os.environ.get("DEBUG", False))
    SQLALCHEMY_TRACK_MODIFICATIONS = utils.to_boolean(os.environ.get("SQLALCHEMY_TRACK_MODIFICATIONS", False))
    DROP_DATABASE_ON_STARTUP = False


class DevelopmentConfig(Config):
    """
    Flask Development configuration
    """

    ENV = "development"
    DEVELOPMENT = utils.to_boolean(os.environ.get("DEVELOPMENT", True))
    DEBUG = utils.to_boolean(os.environ.get("DEBUG", True))
    SQLALCHEMY_TRACK_MODIFICATIONS = utils.to_boolean(os.environ.get("SQLALCHEMY_TRACK_MODIFICATIONS", True))
    DROP_DATABASE_ON_STARTUP = utils.to_boolean(os.environ.get("DROP_DATABASE_ON_STARTUP", False))
