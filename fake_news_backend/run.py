from flask import Flask
from app.routes import api_bp
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

app.register_blueprint(api_bp, url_prefix="/api")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
