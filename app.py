
from ultralytics import YOLO
import os
import base64
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template, request
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# --------------------------------------------------
# LOAD YOUR TRAINED CNN MODEL
# --------------------------------------------------
MODEL_PATH = os.path.join("models", "traffic_sign_model.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Check your models folder."
    )

print("Loading trained traffic sign model...", flush=True)
model = tf.keras.models.load_model(MODEL_PATH)
print("CNN model loaded successfully!", flush=True)

# --------------------------------------------------
# LOAD YOLO11 NANO MODEL
# --------------------------------------------------
print("Loading YOLO11n model...", flush=True)
detector = YOLO("yolo11n.pt")
print("YOLO11n loaded successfully!", flush=True)

# --------------------------------------------------
# GTSRB CLASS NAMES: 0 TO 42
# --------------------------------------------------
CLASS_NAMES = [
    "Speed limit (20 km/h)",
    "Speed limit (30 km/h)",
    "Speed limit (50 km/h)",
    "Speed limit (60 km/h)",
    "Speed limit (70 km/h)",
    "Speed limit (80 km/h)",
    "End of speed limit (80 km/h)",
    "Speed limit (100 km/h)",
    "Speed limit (120 km/h)",
    "No passing",
    "No passing for heavy vehicles",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tonnes prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice or snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing for heavy vehicles"
]

# --------------------------------------------------
# CNN PREDICTION
# --------------------------------------------------
def predict_traffic_sign(rgb_image):
    """Classify an RGB traffic-sign crop using the trained CNN."""
    resized = cv2.resize(rgb_image, (32, 32))

    image_array = np.asarray(resized, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]

    class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[class_id]) * 100

    return class_id, confidence


# --------------------------------------------------
# YOLO DETECTION HELPER
# --------------------------------------------------
def detect_and_classify(frame):
    """
    Detect objects using YOLO11n.
    Classify a detected crop using the CNN.

    NOTE:
    Standard yolo11n.pt is not trained specifically to detect
    traffic signs. It may return no detection for a traffic sign.
    """
    results = detector.predict(
        source=frame,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return frame, None, None

    height, width = frame.shape[:2]

    # Choose the largest detected object
    best_box = max(
        result.boxes,
        key=lambda box: (
            float(box.xyxy[0][2] - box.xyxy[0][0])
            * float(box.xyxy[0][3] - box.xyxy[0][1])
        )
    )

    x1, y1, x2, y2 = (
        best_box.xyxy[0].cpu().numpy().astype(int)
    )

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return frame, None, None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return frame, None, None

    # Convert the cropped object to RGB for the CNN
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    class_id, cnn_confidence = predict_traffic_sign(crop_rgb)
    yolo_confidence = float(best_box.conf[0])

    # Draw the detected object boundary
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Warn when the CNN is not confident enough
    if cnn_confidence >= 60:
        label = CLASS_NAMES[class_id]
        display_text = f"{label}: {cnn_confidence:.1f}%"
    else:
        display_text = "Object detected - uncertain sign"

    cv2.putText(
        frame,
        display_text[:55],
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"YOLO confidence: {yolo_confidence * 100:.1f}%",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame, class_id, cnn_confidence


# --------------------------------------------------
# IMAGE UPLOAD PAGE
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    image_data = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or not file.filename:
            error = "Please select an image first."
        else:
            try:
                img = Image.open(file.stream).convert("RGB")
                original = img.copy()

                rgb_image = np.asarray(original)
                class_id, confidence = predict_traffic_sign(
                    rgb_image
                )
                prediction = CLASS_NAMES[class_id]

                original.thumbnail((500, 500))
                buffer = BytesIO()
                original.save(buffer, format="PNG")

                image_data = base64.b64encode(
                    buffer.getvalue()
                ).decode("utf-8")

            except Exception as exc:
                print("Image prediction error:", exc, flush=True)
                error = (
                    "Could not process this image. "
                    "Please try a valid JPG or PNG."
                )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_data=image_data,
        error=error
    )


# --------------------------------------------------
# LIVE CAMERA PAGE
# --------------------------------------------------
@app.route("/live")
def live_camera():
    return render_template("live.html")


# --------------------------------------------------
# LIVE CAMERA FRAME GENERATOR
# --------------------------------------------------
def generate_frames():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        print("ERROR: Could not open webcam.", flush=True)
        return

    try:
        while True:
            success, frame = camera.read()

            if not success:
                break

            try:
                frame, class_id, confidence = detect_and_classify(
                    frame
                )

                if class_id is None:
                    cv2.putText(
                        frame,
                        "No object detected by YOLO11n",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA
                    )

                cv2.rectangle(
                    frame,
                    (10, 55),
                    (min(frame.shape[1] - 10, 650), 95),
                    (15, 25, 40),
                    -1
                )

                cv2.putText(
                    frame,
                    "TRAFFICVISION AI - YOLO11n + CNN",
                    (20, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 200, 50),
                    2,
                    cv2.LINE_AA
                )

            except Exception as exc:
                print("Live prediction error:", exc, flush=True)

                cv2.putText(
                    frame,
                    "Prediction error",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            success, buffer = cv2.imencode(".jpg", frame)

            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    finally:
        camera.release()
        print("Webcam released.", flush=True)


# --------------------------------------------------
# VIDEO STREAM ROUTE
# --------------------------------------------------
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --------------------------------------------------
# FILE SIZE ERROR
# --------------------------------------------------
@app.errorhandler(413)
def file_too_large(error):
    return (
        "Image too large. Please upload an image smaller than 5 MB.",
        413
    )


# --------------------------------------------------
# START FLASK SERVER
# --------------------------------------------------
if __name__ == "__main__":
    print("Starting TrafficVision AI Studio...", flush=True)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
