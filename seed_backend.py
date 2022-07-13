"""
This script is used to load feelback data into the database without processing the video.
It takes an already processed dump of feelback object and the video, and loads them into the database.
All processing such as video analytics, and key moments are skipped except for thumbnail generation.
"""

from feelback_backend import app
from http import HTTPStatus as Status
from feelback_backend.routes.video import store_feelback_data_in_database
import pickle
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Load feelback data into the database without processing the video.')
    parser.add_argument('video_filename', type=str, help='Path to the video file')
    parser.add_argument('--load-feelback', '-l', type=str, required=True, help='Path to feelback dumped pickle file')
    return parser.parse_args()


def main(video_filename, feelback):
    app.testing = True
    client = app.test_client()

    response = client.post('/api/v1/videos', data={
        'video': open(video_filename, 'rb'),
        'type': 'Reaction',
    })

    assert response.status_code == Status.CREATED

    video_id = response.json['data']['id']
    store_feelback_data_in_database(video_id, feelback)


if __name__ == '__main__':
    args = get_args()
    feelback = pickle.load(open(args.load_feelback, 'rb'))
    main(args.video_filename, feelback)


