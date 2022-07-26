from feelback_backend import app

if __name__ == '__main__':
    app.run(host=app.config['FLASK_RUN_HOST'], port=app.config['FLASK_RUN_PORT'])

