# Smart Crop Recommendation System

Machine Learning powered crop recommendation web app built with Flask. Given environmental and soil parameters (Temperature, Humidity, Rainfall, pH, Nitrogen, Phosphorus, Potassium), it predicts the most suitable crop and provides confidence. Includes model comparison analytics.

## Features
- Real-time crop prediction with confidence score
- Trains and evaluates multiple ML models; deploys Random Forest
- Interactive accuracy chart (Chart.js) from saved metrics
- Clean, responsive UI (HTML/CSS/JS)
- JSON API for programmatic access

## Tech Stack
- Backend: Python 3, Flask
- ML: scikit-learn, NumPy, pandas, joblib
- Frontend: HTML, CSS, JavaScript, Chart.js

## Project Structure
```
/workspaces/cropml/
├─ backend/
│  ├─ app.py               # Flask app (routes/templates/static)
│  ├─ predict.py           # Loads model artifacts and predicts
│  └─ train_model.py       # Training, evaluation, metrics export
├─ dataset/
│  └─ crop_data.csv        # Training dataset
├─ model/                  # Saved artifacts (created by training)
│  ├─ random_forest_model.pkl
│  ├─ label_encoder.pkl
│  └─ scaler.pkl
├─ static/
│  ├─ css/styles.css       # Styles
│  ├─ js/main.js           # Client-side scripts (optional)
│  ├─ data/metrics.json    # Saved metrics for charts (created by training)
│  └─ images/accuracy_chart.png
├─ templates/              # HTML templates
│  ├─ index.html           # Landing page
│  ├─ recommend.html       # Input form
│  ├─ result.html          # Prediction result
│  ├─ comparison.html      # Model accuracy chart
│  └─ about.html           # About/stack info
├─ requirements.txt
└─ README.md
```

## Prerequisites
- Python 3.9+ (tested with 3.10/3.11)
- pip

## Setup and Run (Step-by-step)

1) Create and activate a virtual environment
- macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
- Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) Train the model and export artifacts/metrics
This creates model/random_forest_model.pkl, model/label_encoder.pkl, model/scaler.pkl, and static/data/metrics.json (also saves a PNG chart in static/images/).
```bash
python backend/train_model.py
```

4) Run the web app
```bash
python backend/app.py
# App listens on http://localhost:5000
```
Open your browser to http://localhost:5000

## JSON API
Endpoint: POST /api/predict

Request body (JSON):
```json
{
  "temperature": 25.5,
  "humidity": 65.2,
  "rainfall": 150.0,
  "ph": 6.5,
  "nitrogen": 45.0,
  "phosphorus": 35.0,
  "potassium": 40.0
}
```

cURL example:
```bash
curl -sS -X POST http://localhost:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "temperature": 25.5,
    "humidity": 65.2,
    "rainfall": 150.0,
    "ph": 6.5,
    "nitrogen": 45.0,
    "phosphorus": 35.0,
    "potassium": 40.0
  }'
```
Response:
```json
{"crop": "rice", "confidence": 0.992}
```

## Re-training and Analytics
- To re-train, edit hyperparameters in backend/train_model.py and run it again.
- Metrics are written to static/data/metrics.json and rendered on the Comparison page (Chart.js).

## Troubleshooting
- Error: Model artifacts not found → Run: `python backend/train_model.py`
- Port already in use → Stop other apps on 5000 or run: `FLASK_RUN_PORT=5001 python backend/app.py`
- Verify Python version and virtual environment are active if imports fail.

## License
Provide or update a LICENSE file if needed for your use case.
