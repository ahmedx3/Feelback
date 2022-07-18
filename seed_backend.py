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
import time


def get_args():
    parser = argparse.ArgumentParser(description='Load feelback data into the database without processing the video.')
    parser.add_argument('video_filename', type=str, help='Path to the video file')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--process', '-p', action='store_true', default=False, help='Process the Video')
    group.add_argument('--load-feelback', '-l', type=str, help='Path to feelback dumped pickle file')

    parser.add_argument('--trailer', '-t', type=str, required=False, help='Path to Trailer Video File to Add')
    return parser.parse_args()


def main(video_filename, trailer, feelback, process=False):
    app.testing = True
    client = app.test_client()

    trailer_id = ''
    if trailer:
        response = client.post('/api/v1/videos', data={
            'video': open(trailer, 'rb'),
            'type': 'Trailer',
        })
        assert response.status_code == Status.CREATED
        trailer_id = response.json['data']['id']

    response = client.post('/api/v1/videos', data={
        'video': open(video_filename, 'rb'),
        'type': 'Reaction',
        'trailer_id': trailer_id,
    })

    assert response.status_code == Status.CREATED

    video_id = response.json['data']['id']

    if process:
        client.put(f'/api/v1/videos/{video_id}', json={
            "fps": "native",
            "save_annotated_video": True,
            "annotations": []
        })

        response = client.get(f'/api/v1/videos/{video_id}')
        while not response.json['data']['finished_processing']:
            response = client.get(f'/api/v1/videos/{video_id}')
            print(f'Processing... {response.json["data"]["progress"]}%')
            time.sleep(3)

        print(f'Processing Complete!')
    else:
        store_feelback_data_in_database(video_id, feelback)


if __name__ == '__main__':
    args = get_args()
    feelback = pickle.load(open(args.load_feelback, 'rb')) if args.load_feelback else None
    main(args.video_filename, args.trailer, feelback, args.process)


