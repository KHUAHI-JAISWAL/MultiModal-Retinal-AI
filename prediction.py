
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = "retinal_fusion_model.keras"
CONFIG_PATH = "model_config.json"

model = load_model(MODEL_PATH)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

selected = config["findings"]
report_columns = config["report_columns"]
IMAGE_SIZE = tuple(config["image_size"])
THRESHOLD = config["threshold"]

def check_finding(report, keyword):
    report = report.lower()

    for neg in negations:
        pattern = rf"\b{re.escape(neg)}\s+(?:[\w-]+\s+){{0,3}}{re.escape(keyword)}\b"
        if re.search(pattern, report):
            return 0

    return int(keyword in report)


def predict_findings(image_path, report_text):

    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)

    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    report_array = np.array([
        check_finding(report_text.lower(), word)
        for word in report_columns
    ], dtype=np.float32).reshape(1, 6)

    probabilities = model.predict(
        [img_array, report_array], verbose=0
    )[0]

    results = {}

    for name, probability in zip(selected, probabilities):
        results[name] = {
            "prediction": (
                "DETECTED" if probability >= THRESHOLD
                else "NOT DETECTED"
            ),
            "probability": round(float(probability), 4)
        }

    return results
