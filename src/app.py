from flask import Flask, request, jsonify
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

print("Loading models...")

# -----------------------------
# LOAD MODELS
# -----------------------------
severity_model = pickle.load(open("models/severity_model.pkl", "rb"))
cwe_model = pickle.load(open("models/cwe_model.pkl", "rb"))
cvss_model = pickle.load(open("models/cvss_model.pkl", "rb"))
encoder = pickle.load(open("models/feature_encoder.pkl", "rb"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Models loaded successfully!")

# -----------------------------
# DEFAULT STRUCTURED FEATURES
# -----------------------------
def get_default_structured():
    return [[
        "NETWORK",
        "LOW",
        "NONE",
        "NONE"
    ]]

# -----------------------------
# ROUTE
# -----------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    description = data.get("description", "")

    if not description:
        return jsonify({"error": "Description is required"}), 400

    # -----------------------------
    # EMBEDDING
    # -----------------------------
    embedding = embedder.encode([description])

    # -----------------------------
    # STRUCTURED FEATURES
    # -----------------------------
    structured_input = get_default_structured()
    structured_features = encoder.transform(structured_input)

    # -----------------------------
    # COMBINE FEATURES
    # -----------------------------
    final_input = np.hstack([embedding, structured_features])

    # -----------------------------
    # PREDICTIONS
    # -----------------------------
    severity = severity_model.predict(final_input)[0]
    cwe = cwe_model.predict(final_input)[0]
    cvss = float(cvss_model.predict(final_input)[0])

    # -----------------------------
    # RESPONSE
    # -----------------------------
    return jsonify({
        "severity": severity,
        "cwe": cwe,
        "cvss_score": round(cvss, 2)
    })

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)