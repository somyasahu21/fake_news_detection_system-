import cv2
import os
from .image_analysis import analyze_image

def analyze_video(video_path, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    results = []

    temp_frame_path = "temp_frame.jpg"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip == 0:
            cv2.imwrite(temp_frame_path, frame)
            result = analyze_image(temp_frame_path)
            results.append({"frame": frame_id, "result": result})

        frame_id += 1

    cap.release()
    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)

    tampered = [r for r in results if r["result"]["tamper_result"].get("tampered")]
    return {
        "frames_analyzed": len(results),
        "tampered_frames": len(tampered),
        "tampered_details": tampered
    }
