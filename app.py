from flask import Flask, render_template, request, jsonify
from main import SentimentAnalyzer
import os
from training_data import get_sample_data
import threading
import pandas as pd
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Dictionary to hold multiple SentimentAnalyzer instances
models = {}
current_model_id = None
training_in_progress = False

def load_model(model_path):
    global models, current_model_id
    model_id = str(uuid.uuid4())
    analyzer = SentimentAnalyzer(model_type='logistic')
    analyzer.load_model(model_path)
    models[model_id] = {"analyzer": analyzer, "name": os.path.basename(model_path)}
    current_model_id = model_id
    return model_id

# Load default model if exists
default_model_path = os.path.join(app.config['MODEL_FOLDER'], 'sentiment_model.pkl')
if os.path.exists(default_model_path):
    load_model(default_model_path)

def retrain_model_thread(texts, labels, model_name=None):
    global training_in_progress, current_model_id, models
    training_in_progress = True
    analyzer = SentimentAnalyzer(model_type='logistic')
    analyzer.train(texts, labels, preprocess_method='spacy')
    model_filename = model_name or f"model_{uuid.uuid4().hex[:8]}.pkl"
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
    analyzer.save_model(model_path)
    # Register new model
    model_id = str(uuid.uuid4())
    models[model_id] = {"analyzer": analyzer, "name": model_filename}
    current_model_id = model_id
    training_in_progress = False

@app.route("/")
def home():
    return render_template("index.html", models=models, current_model_id=current_model_id)

@app.route("/analyze", methods=["POST"])
def analyze():
    global current_model_id
    if training_in_progress:
        return jsonify({"error": "Model training in progress. Please wait..."}), 400

    if not current_model_id:
        return jsonify({"error": "No model trained yet. Please train a new model first."}), 400

    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Please enter some text"}), 400

    analyzer = models[current_model_id]["analyzer"]
    sentiment, confidence = analyzer.predict(text)
    doc = analyzer.nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(float(confidence), 4),
        "entities": entities
    })

@app.route("/retrain", methods=["POST"])
def retrain():
    global training_in_progress
    if training_in_progress:
        return jsonify({"status": "Training already in progress"}), 400

    texts, labels = get_sample_data()
    thread = threading.Thread(target=retrain_model_thread, args=(texts, labels))
    thread.start()
    return jsonify({"status": "Training started on default sample data", "refresh": True})

# @app.route("/upload", methods=["POST"])
# def upload():
#     global training_in_progress
#     if training_in_progress:
#         return jsonify({"status": "Training already in progress"}), 400

#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     if not file.filename.endswith('.csv'):
#         return jsonify({"error": "Only CSV files are allowed"}), 400

#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(filepath)

#     try:
#         df = pd.read_csv(filepath)
#         if 'text' not in df.columns or 'label' not in df.columns:
#             return jsonify({"error": "CSV must have 'text' and 'label' columns"}), 400

#         texts = df['text'].tolist()
#         labels = df['label'].tolist()
#         thread = threading.Thread(target=retrain_model_thread, args=(texts, labels, file.filename))
#         thread.start()

#         return jsonify({"status": f"Training started on uploaded dataset ({len(texts)} samples)", "refresh": True})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route("/switch_model", methods=["POST"])
def switch_model():
    global current_model_id
    data = request.get_json()
    model_id = data.get("model_id")
    if model_id in models:
        current_model_id = model_id
        return jsonify({"status": f"Switched to model: {models[model_id]['name']}"}), 200
    else:
        return jsonify({"error": "Invalid model ID"}), 400

if __name__ == "__main__":
    app.run(debug=True)
