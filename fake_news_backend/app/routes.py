from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from .image_analysis import analyze_image
from .video_analysis import analyze_video
import os

api_bp = Blueprint("api", __name__)

@api_bp.route("/analyze/image", methods=["POST"])
def image_analysis():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join("uploads", secure_filename(file.filename))
    file.save(filepath)

    result = analyze_image(filepath)
    os.remove(filepath)
    return jsonify(result)


@api_bp.route("/analyze/video", methods=["POST"])
def video_analysis():
    if "file" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join("uploads", secure_filename(file.filename))
    file.save(filepath)

    result = analyze_video(filepath)
    os.remove(filepath)
    return jsonify(result)
