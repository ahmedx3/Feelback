from http import HTTPStatus as Status
from flask import jsonify
import traceback
from .. import app


@app.errorhandler(400)
def bad_request(error):
    return jsonify({"status": "error", "message": "Bad Request"}), Status.BAD_REQUEST


@app.errorhandler(401)
def unauthorized(error):
    return jsonify({"status": "error", "message": "Unauthorized"}), Status.UNAUTHORIZED


@app.errorhandler(403)
def forbidden(error):
    return jsonify({"status": "error", "message": "Forbidden"}), Status.FORBIDDEN


@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Not Found"}), Status.NOT_FOUND


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"status": "error", "message": "Method Not Allowed"}), Status.METHOD_NOT_ALLOWED


@app.errorhandler(429)
def too_many_requests(error):
    return jsonify({"status": "error", "message": "Too Many Requests"}), Status.TOO_MANY_REQUESTS


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"status": "error", "message": "Internal Server Error"}), Status.INTERNAL_SERVER_ERROR


@app.errorhandler(Exception)
def unhandled_exception(error):
    print(f"[ERROR] Unhandled Exception: {error}")
    traceback.print_exc()
    return jsonify({"status": "error", "message": "Internal Server Error"}), Status.INTERNAL_SERVER_ERROR
