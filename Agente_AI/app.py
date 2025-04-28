# app.py - Flask backend for the AI Medical Assistant

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
import tensorflow as tf

# Initialize Flask application
app = Flask(__name__)

# Load models (in a real implementation, these would be trained models)
def load_models():
    models = {
        'metabolic': joblib.load('models/metabolic_model.pkl') if os.path.exists('models/metabolic_model.pkl') else None,
        'inflammatory': joblib.load('models/inflammatory_model.pkl') if os.path.exists('models/inflammatory_model.pkl') else None,
        'cardiovascular': joblib.load('models/cardiovascular_model.pkl') if os.path.exists('models/cardiovascular_model.pkl') else None,
        'aging_clock': joblib.load('models/aging_clock_model.pkl') if os.path.exists('models/aging_clock_model.pkl') else None,
    }
    return models

# Initialize NLP components for the conversational assistant
def init_nlp():
    # Download NLTK resources if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    # Load Hugging Face transformer for question answering
    qa_pipeline = pipeline("question-answering")
    
    return {
        'qa_pipeline': qa_pipeline,
        'stop_words': set(stopwords.words('english'))
    }

# Load reference ranges for biomarkers
def load_reference_ranges():
    # In production, this would load from a database
    return {
        'glucose': {'min': 70, 'max': 99, 'unit': 'mg/dL'},
        'hba1c': {'min': 4.0, 'max': 5.6, 'unit': '%'},
        'hscrp': {'min': 0, 'max': 1.0, 'unit': 'mg/L'},
        'il6': {'min': 0, 'max': 1.8, 'unit': 'pg/mL'},
        'vitamin_d': {'min': 30, 'max': 50, 'unit': 'ng/mL'},
        'homocysteine': {'min': 0, 'max': 10, 'unit': 'μmol/L'},
        'hdl': {'min': 40, 'max': 60, 'unit': 'mg/dL'},
        'ldl': {'min': 0, 'max': 100, 'unit': 'mg/dL'},
        'triglycerides': {'min': 0, 'max': 150, 'unit': 'mg/dL'},
        'insulin': {'min': 2, 'max': 20, 'unit': 'μIU/mL'},
    }

# Analyze biomarker trends over time
def analyze_biomarker_trends(patient_id, biomarker_category=None, time_period=None):
    # Define time range
    end_date = datetime.now()
    if time_period == '3m':
        start_date = end_date - timedelta(days=90)
    elif time_period == '1y':
        start_date = end_date - timedelta(days=365)
    else:  # Default to 6 months
        start_date = end_date - timedelta(days=180)
    
    # Generate dates
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Simulated data for Isabel Romero
    if patient_id == '458912':
        data = {
            'date': date_range,
            'glucose': [118, 115, 112, 110, 108, 105],
            'hba1c': [5.9, 5.9, 5.8, 5.7, 5.7, 5.6],
            'hscrp': [3.8, 3.6, 3.2, 2.8, 2.4, 2.1],
            'il6': [4.2, 4.0, 3.6, 3.2, 3.0, 2.8],
            'vitamin_d': [24, 26, 28, 30, 31, 32],
            'homocysteine': [14.1, 13.8, 13.5, 13.2, 12.8, 12.4],
            'hdl': [45, 47, 48, 50, 51, 52],
            'ldl': [128, 125, 122, 120, 118, 115],
            'triglycerides': [165, 155, 145, 140, 135, 132],
            'insulin': [14, 13, 12, 12, 11, 11]
        }
        
        df = pd.DataFrame(data)
        
        # Filter by category if specified
        if biomarker_category == 'metabolic':
            columns_to_keep = ['date', 'glucose', 'hba1c', 'insulin']
            df = df[columns_to_keep]
        elif biomarker_category == 'inflammatory':
            columns_to_keep = ['date', 'hscrp', 'il6']
            df = df[columns_to_keep]
        elif biomarker_category == 'cardiovascular':
            columns_to_keep = ['date', 'hdl', 'ldl', 'triglycerides', 'homocysteine']
            df = df[columns_to_keep]
        
        # Calculate trends
        trends = {}
        for column in df.columns:
            if column != 'date':
                first_value = df[column].iloc[0]
                last_value = df[column].iloc[-1]
                if first_value > 0:  # Avoid division by zero
                    percent_change = ((last_value - first_value) / first_value) * 100
                    trends[column] = percent_change
        
        return {
            'biomarker_data': df.to_dict(orient='records'),
            'trends': trends
        }
    else:
        return {'error': 'Patient not found'}

# Generate health insights based on biomarker data
def generate_health_insights(patient_id):
    insights = []
    biomarker_data = analyze_biomarker_trends(patient_id)
    if 'error' in biomarker_data:
        return {'error': 'Could not generate insights: ' + biomarker_data['error']}
    
    latest_data = biomarker_data['biomarker_data'][-1]
    trends = biomarker_data['trends']
    reference_ranges = load_reference_ranges()
    
    # Inflammation status
    if latest_data['hscrp'] > reference_ranges['hscrp']['max']:
        if trends['hscrp'] < 0:
            insights.append({
                'type': 'inflammation',
                'title': 'Improving Inflammatory Status',
                'content': f"Inflammatory markers are showing significant improvement. hsCRP has decreased by {abs(trends['hscrp']):.1f}% and IL-6 has decreased by {abs(trends['il6']):.1f}%.",
                'recommendation': "Continue current anti-inflammatory protocol."
            })
        else:
            insights.append({
                'type': 'inflammation',
                'title': 'Elevated Inflammatory Markers',
                'content': f"Inflammatory markers remain elevated with hsCRP at {latest_data['hscrp']} mg/L.",
                'recommendation': "Evaluate potential inflammatory triggers."
            })
    
    # Metabolic health
    if latest_data['glucose'] > reference_ranges['glucose']['max']:
        if trends['glucose'] < 0:
            insights.append({
                'type': 'metabolic',
                'title': 'Improving Glucose Metabolism',
                'content': f"Fasting glucose has decreased to {latest_data['glucose']} mg/dL, showing a {abs(trends['glucose']):.1f}% improvement.",
                'recommendation': "Increase Magnesio Quelado to 400mg daily."
            })
        else:
            insights.append({
                'type': 'metabolic',
                'title': 'Metabolic Health Concerns',
                'content': f"Glucose remains elevated at {latest_data['glucose']} mg/dL.",
                'recommendation': "Implement time-restricted eating."
            })
    
    # Cardiovascular health
    cv_risk_factors = sum([
        latest_data['ldl'] > reference_ranges['ldl']['max'],
        latest_data['hdl'] < reference_ranges['hdl']['min'],
        latest_data['triglycerides'] > reference_ranges['triglycerides']['max'],
        latest_data['homocysteine'] > reference_ranges['homocysteine']['max']
    ])
    
    if cv_risk_factors >= 2:
        insights.append({
            'type': 'cardiovascular',
            'title': 'Cardiovascular Risk Factors',
            'content': f"Multiple cardiovascular risk factors present including elevated homocysteine ({latest_data['homocysteine']} μmol/L).",
            'recommendation': "Add Methylfolate and B12."
        })
    
    # Vitamin D status
    if latest_data['vitamin_d'] < reference_ranges['vitamin_d']['min']:
        insights.append({
            'type': 'nutrient',
            'title': 'Vitamin D Deficiency',
            'content': f"Vitamin D level is {latest_data['vitamin_d']} ng/mL.",
            'recommendation': "Supplement with 5000 IU Vitamin D3 daily."
        })
    elif trends['vitamin_d'] > 20:
        insights.append({
            'type': 'nutrient',
            'title': 'Improving Vitamin D Status',
            'content': f"Vitamin D has improved by {trends['vitamin_d']:.1f}% to {latest_data['vitamin_d']} ng/mL.",
            'recommendation': "Maintain current supplementation."
        })
    
    return {
        'patient_id': patient_id,
        'generation_date': datetime.now().isoformat(),
        'insights': insights
    }

# Generate treatment recommendations based on biomarker analysis
def generate_treatment_recommendations(patient_id):
    recommendations = []
    biomarker_data = analyze_biomarker_trends(patient_id)
    if 'error' in biomarker_data:
        return {'error': 'Could not generate recommendations: ' + biomarker_data['error']}
    
    latest_data = biomarker_data['biomarker_data'][-1]
    
    if latest_data['glucose'] > 100:
        recommendations.append({
            'id': 'magnesio_quelado',
            'title': 'Magnesio Quelado Optimization',
            'recommended_dosage': '400mg daily',
            'target_biomarkers': ['Glucose']
        })
    
    if latest_data['homocysteine'] > 10:
        recommendations.append({
            'id': 'methylation_support',
            'title': 'Methylation Support Protocol',
            'recommended_protocol': 'Methylfolate (1000mcg) + B12 (1000mcg) daily',
            'target_biomarkers': ['Homocysteine']
        })
    
    return {
        'patient_id': patient_id,
        'generation_date': datetime.now().isoformat(),
        'recommendations': recommendations
    }

# Process natural language questions from the doctor
def process_doctor_query(patient_id, query):
    query_lower = query.lower()
    biomarker_data = analyze_biomarker_trends(patient_id)
    if 'error' in biomarker_data:
        return {'error': 'Could not process query: ' + biomarker_data['error']}
    
    latest_data = biomarker_data['biomarker_data'][-1]
    trends = biomarker_data['trends']
    
    if 'metabolic' in query_lower or 'glucose' in query_lower:
        response = {
            'type': 'metabolic_analysis',
            'content': f"Glucose: {latest_data['glucose']} mg/dL, improved by {abs(trends['glucose']):.1f}%."
        }
    elif 'inflammation' in query_lower or 'crp' in query_lower:
        response = {
            'type': 'inflammation_analysis',
            'content': f"hsCRP reduced by {abs(trends['hscrp']):.1f}% to {latest_data['hscrp']} mg/L."
        }
    else:
        response = {
            'type': 'general_health_status',
            'content': f"Key improvements: hsCRP down {abs(trends['hscrp']):.1f}%, glucose at {latest_data['glucose']} mg/dL."
        }
    
    return {
        'patient_id': patient_id,
        'query': query,
        'response': response,
        'timestamp': datetime.now().isoformat()
    }

# Load TensorFlow models (assuming pre-trained models exist)
bio_age_model = tf.keras.models.load_model('biological_age_model.h5') if os.path.exists('biological_age_model.h5') else None
glucose_model = tf.keras.models.load_model('glucose_prediction_model.h5') if os.path.exists('glucose_prediction_model.h5') else None

# Define API endpoints
@app.route('/api/biomarkers', methods=['GET'])
def get_biomarkers():
    patient_id = request.args.get('patient_id')
    category = request.args.get('category')
    time_period = request.args.get('period')
    
    if not patient_id:
        return jsonify({'error': 'Patient ID is required'}), 400
    
    result = analyze_biomarker_trends(patient_id, category, time_period)
    return jsonify(result)

@app.route('/api/insights', methods=['GET'])
def get_insights():
    patient_id = request.args.get('patient_id')
    
    if not patient_id:
        return jsonify({'error': 'Patient ID is required'}), 400
    
    result = generate_health_insights(patient_id)
    return jsonify(result)

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    patient_id = request.args.get('patient_id')
    
    if not patient_id:
        return jsonify({'error': 'Patient ID is required'}), 400
    
    result = generate_treatment_recommendations(patient_id)
    return jsonify(result)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    
    if not data or 'patient_id' not in data or 'query' not in data:
        return jsonify({'error': 'Patient ID and query are required'}), 400
    
    result = process_doctor_query(data['patient_id'], data['query'])
    return jsonify(result)

# New API endpoint for calculating biological age
@app.route('/api/biological_age', methods=['POST'])
def calculate_biological_age():
    if not bio_age_model:
        return jsonify({'error': 'Biological age model not loaded'}), 500
    
    data = request.json
    if not all(key in data for key in ['glucose', 'hba1c', 'hscrp']):
        return jsonify({'error': 'Missing required biomarkers'}), 400
    
    biomarkers = np.array([data['glucose'], data['hba1c'], data['hscrp']]).reshape(1, -1)
    age = bio_age_model.predict(biomarkers, verbose=0)[0][0]
    return jsonify({'biological_age': float(age)})

# New API endpoint for predicting biomarkers
@app.route('/api/predict_biomarkers', methods=['POST'])
def predict_biomarkers():
    if not glucose_model:
        return jsonify({'error': 'Glucose prediction model not loaded'}), 500
    
    data = request.json
    if 'historical_glucose' not in data or len(data['historical_glucose']) < 3:
        return jsonify({'error': 'At least 3 historical glucose values required'}), 400
    
    historical_data = np.array(data['historical_glucose']).reshape(1, 3, 1)  # 3 time steps
    prediction = glucose_model.predict(historical_data, verbose=0)[0][0]
    return jsonify({'predicted_glucose': float(prediction)})

# New API endpoint for simulating treatment effects
@app.route('/api/simulate_treatment', methods=['POST'])
def simulate_treatment():
    data = request.json
    if 'treatment' not in data or 'current_glucose' not in data:
        return jsonify({'error': 'Treatment and current glucose required'}), 400
    
    treatment = data['treatment']
    current_glucose = data['current_glucose']
    
    if treatment == 'Magnesio Quelado':
        simulated_glucose = current_glucose * 0.9  # 10% reduction
    else:
        simulated_glucose = current_glucose  # No change for other treatments
    
    return jsonify({'simulated_glucose': float(simulated_glucose)})

# Main entry point
if __name__ == '__main__':
    # Initialize models and NLP components
    models = load_models()
    nlp_components = init_nlp()
    
    # Start the Flask application
    app.run(debug=True, port=5000)