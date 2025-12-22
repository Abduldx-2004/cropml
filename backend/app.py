"""
Flask web application for the Crop Recommendation System.

Routes:
  /            -> Home page
  /recommend   -> Input form
  /result      -> Shows prediction result (POST)
  /api/predict -> JSON API for programmatic predictions
  /comparison  -> Model comparison page (uses metrics JSON)
  /about       -> About page

Run:
  set FLASK_APP=backend/app.py; flask run
  (or) python backend/app.py
"""
import os
from flask import Flask, render_template, request, jsonify, redirect, url_for

from predict import predict_crop

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend')
def recommend():
    return render_template('recommend.html')


@app.route('/result', methods=['POST'])
def result():
    # Read form inputs, convert to float and predict
    try:
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        rainfall = float(request.form['rainfall'])
        ph = float(request.form['ph'])
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
    except Exception as e:
        return f"Invalid input: {e}", 400

    features = [temp, humidity, rainfall, ph, nitrogen, phosphorus, potassium]
    try:
        crop, confidence = predict_crop(features)
    except Exception as e:
        return f"Prediction error: {e}", 500

    confidence_pct = round(confidence * 100, 2)
    return render_template('result.html', crop=crop, confidence=confidence_pct)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    required = ['temperature', 'humidity', 'rainfall', 'ph', 'nitrogen', 'phosphorus', 'potassium']
    if not data:
        return jsonify({'error': 'No JSON provided'}), 400
    try:
        features = [float(data[k]) for k in required]
    except Exception as e:
        return jsonify({'error': f'Invalid/missing fields: {e}'}), 400

    try:
        crop, confidence = predict_crop(features)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'crop': crop, 'confidence': float(confidence)})


@app.route('/comparison')
def comparison():
    return render_template('comparison.html')


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    # Run Flask app for local testing
    app.run(host='0.0.0.0', port=5000, debug=True)
