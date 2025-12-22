# Crop Recommendation System

This is a final-year project: a Machine Learning based Crop Recommendation System.

Overview
- Backend: Python + Flask
- ML: Scikit-learn (Random Forest final model)
- Frontend: HTML, CSS, JavaScript (Chart.js)

Folder structure

CropRecommendationSystem/
├── dataset/
│   └── crop_data.csv
├── model/
│   └── (random_forest_model.pkl will be created after training)
├── backend/
│   ├── train_model.py
│   ├── predict.py
│   └── app.py
├── static/
│   ├── css/
│   ├── js/
│   └── data/
└── templates/

Quick start (recommended)

1. Create a Python virtual environment and activate it (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib joblib flask
```

3. Train models and save artifacts (this creates `model/random_forest_model.pkl`, `model/label_encoder.pkl`, `model/scaler.pkl` and `static/data/metrics.json`):

```powershell
python backend/train_model.py
```

4. Run the Flask app:

```powershell
python backend/app.py
# or set FLASK_APP env and run flask run
```

5. Visit `http://localhost:5000` in your browser.

Notes for demo
- The training script evaluates multiple classifiers and saves a Random Forest model.
- The Model Comparison page reads `static/data/metrics.json` and renders the accuracy chart.
- If you want to reproduce experiments, edit `backend/train_model.py` to change hyperparameters or add cross-validation.

Academic/Documentation Tips
- Commented code is provided to help understand each step.
- For the viva, show the Model Comparison page, then demonstrate live predictions using the Recommend page.
