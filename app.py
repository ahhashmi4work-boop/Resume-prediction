from flask import Flask, render_template, request, jsonify
import pandas as pd
import traceback
import os

# Optional ML / similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)

CSV_PATH = "read.csv"
MODEL_PATH = "model_svc.pkl"
TARGET_COLUMN = "Resume"   # the column to produce

# -----------------------
# Load CSV
# -----------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found. Place your CSV in the project root.")

df = pd.read_csv(CSV_PATH, dtype=str).fillna("")   # read all as strings for simplicity

# If the target column isn't present, stop early
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"CSV must contain a '{TARGET_COLUMN}' column.")

# -----------------------
# Build FEATURES config for frontend automatically
# (All columns except the target become inputs)
# -----------------------
FEATURES = []
for col in df.columns:
    if col == TARGET_COLUMN:
        continue

    # treat numeric-looking columns or 'Experience' specially
    unique_vals = df[col].dropna().unique().tolist()
    num_unique = len(unique_vals)

    # if short text with limited unique values -> dropdown
    if num_unique <= 25 and num_unique > 0:
        options = sorted(unique_vals)
        FEATURES.append({"name": col, "label": col, "type": "dropdown", "options": options})
    else:
        # for longer free-form text use textarea
        FEATURES.append({"name": col, "label": col, "type": "textarea"})

# -----------------------
# Try to load optional model
# -----------------------
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
        print("Loaded model.pkl")
    except Exception as e:
        print("Could not load model.pkl:", e)
        model = None

# Precompute TF-IDF on resumes for retrieval fallback
resume_texts = df[TARGET_COLUMN].astype(str).tolist()
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2)).fit(resume_texts)
resume_tfidf = tfidf.transform(resume_texts)


# -----------------------
# Routes
# -----------------------
@app.route("/")
def home():
    # pass FEATURES to template to render inputs dynamically
    return render_template("index.html", FEATURES=FEATURES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.json or {}
        # build a single-row dataframe of inputs (matching CSV column names)
        input_row = {}
        for f in FEATURES:
            name = f["name"]
            val = payload.get(name, "")
            # Keep everything stringified
            input_row[name] = str(val) if val is not None else ""

        # If a model exists, try to call it. We try a few sensible interfaces:
        if model is not None:
            try:
                # If the model is a pipeline expecting a single merged text input:
                # join all features into one string
                merged = " ".join([v for v in input_row.values()])
                try:
                    # some models expect a dataframe row or list; try both
                    pred = model.predict(pd.DataFrame([input_row]))
                except Exception:
                    pred = model.predict([merged])

                # Ensure we get a result
                if isinstance(pred, (list, tuple, pd.Series)):
                    pred_text = str(pred[0])
                else:
                    pred_text = str(pred)

                return jsonify({
                    "prediction_text": pred_text,
                    "explain": "Predicted using provided model.pkl"
                })

            except Exception as me:
                # model failed — log and fall back to retrieval
                print("Model call failed:", me)
                traceback.print_exc()

        # ---------------------
        # Retrieval fallback (no model or model failed)
        # ---------------------
        # 1) Exact / contains filtering on non-empty fields
        candidates = df.copy()
        for k, v in input_row.items():
            if v and len(v.strip()) > 0:
                # case-insensitive contains check
                candidates = candidates[candidates[k].str.contains(str(v), case=False, na=False)]

        if len(candidates) > 0:
            # return the top candidate's Resume (first match)
            top_resume = candidates.iloc[0][TARGET_COLUMN]
            return jsonify({
                "prediction_text": top_resume,
                "explain": f"Returned first matched resume from CSV (exact/contains filter). Matches found: {len(candidates)}"
            })

        # 2) No contains match — fall back to TF-IDF similarity by comparing the merged input to resumes
        merged_query = " ".join([v for v in input_row.values() if v]).strip()
        if merged_query == "":
            # no inputs provided: show a top example resume
            sample_resume = resume_texts[0] if resume_texts else ""
            return jsonify({
                "prediction_text": sample_resume,
                "explain": "No inputs given. Showing a sample resume from CSV."
            })

        q_tfidf = tfidf.transform([merged_query])
        sims = cosine_similarity(q_tfidf, resume_tfidf).flatten()
        top_idx = sims.argmax()
        top_score = float(sims[top_idx])
        top_resume = resume_texts[top_idx]

        return jsonify({
            "prediction_text": top_resume,
            "explain": f"Retrieved by TF-IDF similarity (score={top_score:.3f})."
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Prediction error", "details": str(e)}), 500


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
