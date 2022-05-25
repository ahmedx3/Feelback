from flask import Flask
from .config import DevelopmentConfig

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)


@app.get('/')
def hello_world():
    return 'Hello World!'
