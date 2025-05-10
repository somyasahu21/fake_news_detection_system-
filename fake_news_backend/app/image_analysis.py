import os
import requests
from PIL import Image
from PIL.ExifTags import TAGS
from PIL.TiffImagePlugin import IFDRational

# === API Configuration ===
API_URL = "https://ping.arya.ai/api/v1/document-tampering-detection"
API_TOKEN = "c975f8caf0333997f12ae2b24cd1ad4c"  # Replace with your valid token
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


# === Helper: Convert EXIF to JSON-safe format ===
def safe_convert(value):
    if isinstance(value, IFDRational):
        return float(value)
    elif isinstance(value, bytes):
        return value.decode(errors="ignore")
    elif isinstance(value, (list, tuple)):
        return [safe_convert(v) for v in value]
    elif isinstance(value, dict):
        return {k: safe_convert(v) for k, v in value.items()}
    else:
        return value


# === Extract Image Metadata ===
def extract_metadata(image_path):
    metadata = {}
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if exif:
            for tag_id, val in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                metadata[tag] = safe_convert(val)
    except Exception as e:
        metadata["error"] = str(e)
    return metadata


# === Main Image Analyzer ===
def analyze_image(image_path):
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(API_URL, headers=HEADERS, files=files)

        if response.status_code == 200:
            result_json = response.json()
            tampered = result_json.get("tampered")
            if tampered is None:
                status = "UNKNOWN"
            else:
                status = "FAKE (Tampered)" if tampered else "REAL (Untampered)"
        else:
            status = f"API Error: {response.status_code}"
            result_json = {"error": response.text}

        metadata = extract_metadata(image_path)

        return {
            "status": status,
            "api_response": result_json,
            "metadata": metadata
        }

    except Exception as e:
        return {"error": str(e)}


# === Example Usage ===
if __name__ == "__main__":
    image_path = "example.jpg"  # Replace with your local image file
    result = analyze_image(image_path)

    if "error" in result:
        print(f"[❌] Error: {result['error']}")
    else:
        print(f"\n[🔍] Analysis Result for: {image_path}")
        print(f"[✅] Image Status: {result['status']}")
        print(f"\n[📊] API Raw Response: {result['api_response']}")
        print(f"\n[🧾] Extracted Metadata:")
        for key, value in result["metadata"].items():
            print(f"  - {key}: {value}")
