from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__, static_folder='static')

@app.route('/', methods=['GET'])
def index():
    return app.send_static_file("index.html")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/dashboard', methods=['GET'])
def dashboard():
    return app.send_static_file("dashboard.html")


# ---------------- Classification Model (Iris) ----------------
try:
    with open('model.pkl', 'rb') as f:
        clf_model = pickle.load(f)
    print("Classification model loaded successfully.")
except Exception as e:
    print("Error loading classification model:", e)
    clf_model = None

expected_features_class = 4  # Iris dataset has 4 features

def validate_classification_features(features):
    if not isinstance(features, list):
        return False, "Input should be a list."
    if len(features) == 0:
        return False, "Input list is empty."
    if all(isinstance(x, (int, float)) for x in features):
        if len(features) != expected_features_class:
            return False, f"Input sample must have exactly {expected_features_class} features."
        return True, "single"
    elif isinstance(features[0], list):
        for sample in features:
            if not isinstance(sample, list) or len(sample) != expected_features_class:
                return False, f"Each sample must have exactly {expected_features_class} features."
            if not all(isinstance(x, (int, float)) for x in sample):
                return False, "Each feature must be a number."
        return True, "multiple"
    else:
        return False, "Invalid input format."

@app.route('/predict/classification', methods=['GET', 'POST'])
def predict_classification():
    """
    GET  -> Returns sample predictions for classification (multiple inputs) without confidences.
    POST -> Performs classification prediction on user-provided data.
           - For a single input, returns prediction and confidence.
           - For multiple inputs, returns only predictions.
    """
    if clf_model is None:
        return jsonify({"error": "Classification model not available"}), 500

    if request.method == 'GET':
        sample_features = [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 3.4, 5.4, 2.3]
        ]
        predictions = []
        for features in sample_features:
            sample = np.array(features).reshape(1, -1)
            pred = clf_model.predict(sample)[0]
            predictions.append(int(pred))
        return jsonify({
            "message": "GET request: Sample predictions for classification (multiple) inputs",
            "sample_features": sample_features,
            "predictions": predictions
        })

    data = request.get_json(force=True)
    if 'features' not in data:
        return jsonify({"error": "Missing 'features' key."}), 400

    features = data['features']
    valid, mode = validate_classification_features(features)
    if not valid:
        return jsonify({"error": mode}), 400

    if mode == "single":
        try:
            features = [float(x) for x in features]
        except ValueError:
            return jsonify({"error": "All feature values must be valid floats."}), 400
        sample = np.array(features).reshape(1, -1)
        pred = clf_model.predict(sample)[0]
        probas = clf_model.predict_proba(sample)[0]
        confidence = max(probas)
        return jsonify({
            "prediction": int(pred),
            "confidence": round(float(confidence), 2)
        })
    else:
        predictions = []
        for s in features:
            try:
                s = [float(x) for x in s]
            except ValueError:
                return jsonify({"error": "All feature values must be valid floats."}), 400
            sample = np.array(s).reshape(1, -1)
            pred = clf_model.predict(sample)[0]
            predictions.append(int(pred))
        return jsonify({"predictions": predictions})


# ---------------- Regression Model (Housing) ----------------
try:
    with open('reg_model.pkl', 'rb') as f:
        reg_model = pickle.load(f)
    print("Regression model loaded successfully.")
except Exception as e:
    print("Error loading regression model:", e)
    reg_model = None

expected_features_reg = 20  # Expected number of features for regression

def validate_regression_features(features):
    if not isinstance(features, list):
        return False, "Input should be a list."
    if len(features) == 0:
        return False, "Input list is empty."
    if all(isinstance(x, (int, float)) for x in features):
        if len(features) != expected_features_reg:
            return False, f"Input sample must have exactly {expected_features_reg} features."
        return True, "single"
    elif isinstance(features[0], list):
        for sample in features:
            if not isinstance(sample, list) or len(sample) != expected_features_reg:
                return False, f"Each sample must have exactly {expected_features_reg} features."
            if not all(isinstance(x, (int, float)) for x in sample):
                return False, "Each feature must be a number."
        return True, "multiple"
    else:
        return False, "Invalid input format."

def get_regression_prediction_and_confidence(sample):
    if hasattr(reg_model, 'estimators_'):
        tree_preds = np.array([est.predict(sample)[0] for est in reg_model.estimators_])
        prediction = np.mean(tree_preds)
        std = np.std(tree_preds)
        confidence = 1 / (1 + std) if std > 0 else 1.0
    else:
        prediction = reg_model.predict(sample)[0]
        confidence = 1.0
    return prediction, confidence

@app.route('/predict/regression', methods=['GET', 'POST'])
def predict_regression():
    """
    GET  -> Returns sample predictions for regression with confidences.
    POST -> Performs regression prediction on user-provided data.
           - For a single input, returns prediction and confidence.
           - For multiple inputs, returns predictions and confidences.
    """
    if reg_model is None:
        return jsonify({"error": "Regression model not available"}), 500

    if request.method == 'GET':
        sample_features = [
            [7420, 4, 2, 3, 2, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0],
            [1500, 2, 1, 20, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
        ]
        predictions = []
        confidences = []
        for features in sample_features:
            sample = np.array(features).reshape(1, -1)
            pred, conf = get_regression_prediction_and_confidence(sample)
            predictions.append(float(pred))
            confidences.append(round(float(conf), 2))
        return jsonify({
            "message": "GET request: Sample predictions for regression",
            "sample_features": sample_features,
            "predictions": predictions,
            "confidences": confidences
        })

    data = request.get_json(force=True)
    if 'features' not in data:
        return jsonify({"error": "Missing 'features' key."}), 400
    features = data['features']
    valid, mode = validate_regression_features(features)
    if not valid:
        return jsonify({"error": mode}), 400

    if mode == "single":
        sample = np.array(features).reshape(1, -1)
        pred, conf = get_regression_prediction_and_confidence(sample)
        return jsonify({
            "prediction": float(pred),
            "confidence": round(float(conf), 2)
        })
    else:
        predictions = []
        confidences = []
        for s in features:
            sample = np.array(s).reshape(1, -1)
            pred, conf = get_regression_prediction_and_confidence(sample)
            predictions.append(float(pred))
            confidences.append(round(float(conf), 2))
        return jsonify({
            "predictions": predictions,
            "confidences": confidences
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
