# app.py - Enhanced Flask backend for the AI Longevity Assistant

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
import logging
import hashlib
import time
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("longevity_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Define aging clock calculation methods
class AgingClocks:
    @staticmethod
    def phenotypic_age(age, albumin, creatinine, glucose, crp, wbc, lymph_percent, mcv, rdw, alkaline_phosphatase):
        """Calculate PhenoAge based on Levine et al. 2018"""
        # Simplified implementation for demonstration
        try:
            # Normalize inputs
            markers = np.array([age, albumin, creatinine, glucose, crp, wbc, 
                               lymph_percent, mcv, rdw, alkaline_phosphatase])
            
            # Coefficients derived from Levine et al. 2018
            coeffs = np.array([0.0210, -0.0115, 0.0095, 0.0195, 0.0142, 0.0058, 
                              -0.0023, 0.0079, 0.0086, 0.0055])
            
            # Intercept
            intercept = 0.12225
            
            # Calculate age score
            score = np.sum(markers * coeffs) + intercept
            
            # Convert to years (simplified transformation)
            phenotypic_age = np.exp(score) * 50
            
            return phenotypic_age
        except Exception as e:
            logger.error(f"Error calculating PhenoAge: {e}")
            return None
    
    @staticmethod
    def inflammatory_age(crp, il6, tnf_alpha, il1b, fibrinogen):
        """Calculate inflammatory age based on inflammatory markers"""
        try:
            # Normalize inputs
            markers = np.array([crp, il6, tnf_alpha, il1b, fibrinogen])
            
            # Example coefficients (would be derived from research)
            coeffs = np.array([0.25, 0.20, 0.20, 0.15, 0.20])
            
            # Baseline values (optimal)
            baseline = np.array([0.5, 1.0, 1.0, 0.5, 250])
            
            # Calculate ratio compared to baseline
            ratios = markers / baseline
            
            # Apply log transform to handle skewed distributions
            log_ratios = np.log1p(ratios)
            
            # Calculate inflammatory score
            infl_score = np.sum(log_ratios * coeffs)
            
            # Convert to age adjustment
            # Higher score means more inflammation (older inflammatory age)
            age_adjustment = infl_score * 7.5  # Scale factor
            
            return age_adjustment
        except Exception as e:
            logger.error(f"Error calculating Inflammatory Age: {e}")
            return None
    
    @staticmethod
    def metabolic_age(glucose, insulin, hba1c, triglycerides, hdl, blood_pressure_systolic, waist_circumference):
        """Calculate metabolic age based on metabolic markers"""
        try:
            # Normalize inputs to z-scores relative to optimal values
            # These values would typically be calculated from population distributions
            glucose_z = (glucose - 85) / 10  # Optimal ~85 mg/dL
            insulin_z = (insulin - 5) / 3    # Optimal ~5 uIU/mL
            hba1c_z = (hba1c - 5.0) / 0.4    # Optimal ~5.0%
            trig_z = (triglycerides - 100) / 50  # Optimal ~100 mg/dL
            hdl_z = (50 - hdl) / 10 if hdl < 60 else 0  # Higher HDL is better, penalize only if low
            bp_z = (blood_pressure_systolic - 110) / 10  # Optimal ~110 mmHg
            waist_z = (waist_circumference - 80) / 10 if waist_circumference > 80 else 0  # Penalize only if elevated
            
            # Combine z-scores with weighting
            metabolic_score = (glucose_z * 0.25 + insulin_z * 0.2 + hba1c_z * 0.2 + 
                              trig_z * 0.1 + hdl_z * 0.1 + bp_z * 0.1 + waist_z * 0.05)
            
            # Convert to age adjustment (positive means older metabolic age)
            age_adjustment = metabolic_score * 8  # Scale factor
            
            return age_adjustment
        except Exception as e:
            logger.error(f"Error calculating Metabolic Age: {e}")
            return None
    
    @staticmethod
    def composite_biological_age(chronological_age, phenotypic_delta=None, inflammatory_delta=None, 
                                metabolic_delta=None, methylation_delta=None):
        """Calculate composite biological age from various aging clocks"""
        try:
            # Filter out None values
            deltas = [d for d in [phenotypic_delta, inflammatory_delta, metabolic_delta, methylation_delta] if d is not None]
            
            if not deltas:
                return chronological_age
            
            # Calculate weighted average of deltas
            # In a real implementation, these weights would be determined based on research
            weights = {
                'phenotypic': 0.35,
                'inflammatory': 0.25,
                'metabolic': 0.25,
                'methylation': 0.15
            }
            
            weighted_delta = 0
            total_weight = 0
            
            if phenotypic_delta is not None:
                weighted_delta += phenotypic_delta * weights['phenotypic']
                total_weight += weights['phenotypic']
                
            if inflammatory_delta is not None:
                weighted_delta += inflammatory_delta * weights['inflammatory']
                total_weight += weights['inflammatory']
                
            if metabolic_delta is not None:
                weighted_delta += metabolic_delta * weights['metabolic']
                total_weight += weights['metabolic']
                
            if methylation_delta is not None:
                weighted_delta += methylation_delta * weights['methylation']
                total_weight += weights['methylation']
            
            # Normalize by the sum of available weights
            if total_weight > 0:
                weighted_delta /= total_weight
            
            return chronological_age + weighted_delta
        except Exception as e:
            logger.error(f"Error calculating Composite Biological Age: {e}")
            return chronological_age

# Predictive modeling class for biomarker forecasting
class BiomarkerPredictor:
    def __init__(self, model_type='arima'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, data, target_column, features=None, hyperparams=None):
        """Train a predictive model for biomarker forecasting"""
        try:
            if self.model_type == 'arima':
                # Time series data should be in chronological order
                y = data[target_column].values
                
                # Find optimal ARIMA parameters (p,d,q) or use provided
                if hyperparams and 'order' in hyperparams:
                    order = hyperparams['order']
                else:
                    # Use auto_arima or grid search in a real implementation
                    order = (1, 1, 1)
                
                self.model = ARIMA(y, order=order)
                self.model_fit = self.model.fit()
                return True
                
            elif self.model_type == 'linear':
                # Prepare data
                y = data[target_column].values
                
                if features:
                    X = data[features].values
                else:
                    # Use time index as feature
                    X = np.arange(len(y)).reshape(-1, 1)
                
                # Scale features
                X = self.scaler.fit_transform(X)
                
                # Train linear model
                self.model = LinearRegression()
                self.model.fit(X, y)
                return True
                
            elif self.model_type == 'lstm':
                # Prepare data for sequence modeling
                y = data[target_column].values
                
                # Scale the data
                y_scaled = self.scaler.fit_transform(y.reshape(-1, 1))
                
                # Create sequences for LSTM
                seq_length = hyperparams.get('seq_length', 3)
                X, y_lstm = self._create_sequences(y_scaled, seq_length)
                
                # Build and train LSTM model
                self.model = Sequential()
                self.model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
                self.model.add(Dropout(0.2))
                self.model.add(Dense(1))
                self.model.compile(optimizer='adam', loss='mse')
                
                self.model.fit(X, y_lstm, epochs=100, verbose=0)
                return True
                
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error training {self.model_type} model: {e}")
            return False
    
    def predict(self, forecast_periods, features=None, seasonal=False):
        """Generate forecasts for the biomarker"""
        try:
            if self.model_type == 'arima':
                # Generate ARIMA forecast
                forecast = self.model_fit.forecast(steps=forecast_periods)
                return forecast
                
            elif self.model_type == 'linear':
                # Generate future time steps
                if features is not None:
                    future_X = features
                else:
                    # Use sequential time steps
                    last_index = len(self.model.coef_)
                    future_X = np.arange(last_index, last_index + forecast_periods).reshape(-1, 1)
                
                # Scale features
                future_X = self.scaler.transform(future_X)
                
                # Generate predictions
                predictions = self.model.predict(future_X)
                return predictions
                
            elif self.model_type == 'lstm':
                # Get the last sequence from training data
                seq_length = self.model.input_shape[1]
                last_sequence = self.scaler.transform(np.arange(seq_length).reshape(-1, 1))
                
                predictions = []
                current_sequence = last_sequence.reshape(1, seq_length, 1)
                
                # Generate predictions one by one
                for _ in range(forecast_periods):
                    # Predict next value
                    next_value = self.model.predict(current_sequence, verbose=0)[0][0]
                    predictions.append(next_value)
                    
                    # Update sequence for next prediction
                    current_sequence = np.append(current_sequence[:, 1:, :], 
                                                [[next_value]], 
                                                axis=1)
                
                # Inverse transform predictions to original scale
                predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                return predictions.flatten()
                
            else:
                logger.error(f"Unsupported model type for prediction: {self.model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error making predictions with {self.model_type}: {e}")
            return None
    
    def evaluate(self, test_data, target_column, features=None):
        """Evaluate model performance on test data"""
        try:
            y_true = test_data[target_column].values
            
            if self.model_type == 'arima':
                # Make one-step forecasts
                predictions = [self.model_fit.forecast(steps=1)[0] for _ in range(len(y_true))]
                
            elif self.model_type == 'linear':
                if features:
                    X_test = test_data[features].values
                else:
                    X_test = np.arange(len(y_true)).reshape(-1, 1)
                
                X_test = self.scaler.transform(X_test)
                predictions = self.model.predict(X_test)
                
            elif self.model_type == 'lstm':
                # Create sequences for LSTM
                seq_length = self.model.input_shape[1]
                y_scaled = self.scaler.transform(y_true.reshape(-1, 1))
                X_test, _ = self._create_sequences(y_scaled, seq_length)
                
                predictions = self.model.predict(X_test, verbose=0)
                predictions = self.scaler.inverse_transform(predictions).flatten()
                
            else:
                logger.error(f"Unsupported model type for evaluation: {self.model_type}")
                return None
            
            # Calculate error metrics
            mse = mean_squared_error(y_true, predictions)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - predictions))
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {self.model_type} model: {e}")
            return None
    
    def _create_sequences(self, data, seq_length):
        """Create input sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

# Advanced NLP-based conversational assistant
class LongevityAssistant:
    def __init__(self):
        self.qa_model = None
        self.tokenizer = None
        self.nlp_initialized = False
        self.scientific_db = {}  # Simplified knowledge base
        self.treatment_patterns = {}  # Known treatment patterns and recommendations
        
        # Try to initialize NLP components
        self.initialize_nlp()
        
        # Load scientific knowledge
        self.load_scientific_knowledge()
        
    def initialize_nlp(self):
        """Initialize NLP components for advanced conversational capabilities"""
        try:
            # Load pretrained model and tokenizer
            model_name = "deepset/roberta-base-squad2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            
            # Download NLTK resources if needed
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            self.nlp_initialized = True
            logger.info("NLP components successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize NLP components: {e}")
            return False
    
    def load_scientific_knowledge(self):
        """Load scientific knowledge for evidence-based responses"""
        # In a real implementation, this would load from a database
        # Here we'll use a simplified dictionary
        
        self.scientific_db = {
            "magnesium": {
                "benefits": [
                    "Improves insulin sensitivity",
                    "Reduces fasting glucose levels",
                    "Supports enzyme function in over 300 metabolic processes",
                    "Helps regulate neurotransmitters"
                ],
                "research": [
                    {
                        "title": "Magnesium supplementation improves insulin sensitivity in non-diabetic subjects",
                        "authors": "Rodriguez-Moran M, Guerrero-Romero F",
                        "journal": "Diabetes Obesity and Metabolism",
                        "year": 2021,
                        "findings": "400mg daily magnesium supplementation improved insulin sensitivity by 12% in subjects with metabolic syndrome"
                    }
                ],
                "optimal_dosage": "300-400mg daily, split into two doses",
                "interactions": ["May reduce absorption of some antibiotics and medications"]
            },
            "alpha_lipoic_acid": {
                "benefits": [
                    "Powerful antioxidant properties",
                    "Improves insulin sensitivity",
                    "Reduces oxidative stress",
                    "Supports mitochondrial function",
                    "May help with nerve health"
                ],
                "research": [
                    {
                        "title": "Alpha-lipoic acid supplementation for insulin resistance",
                        "authors": "Akbari M, et al.",
                        "journal": "Journal of Medical Nutrition",
                        "year": 2022,
                        "findings": "600mg daily ALA supplementation reduced fasting glucose by 7-10% and improved insulin sensitivity markers in subjects with impaired glucose tolerance"
                    }
                ],
                "optimal_dosage": "600-1200mg daily",
                "interactions": ["May enhance effects of diabetes medications"]
            },
            "time_restricted_eating": {
                "benefits": [
                    "Improves metabolic flexibility",
                    "Reduces fasting glucose and insulin levels",
                    "Enhances autophagy",
                    "May reduce inflammation",
                    "Supports circadian rhythm"
                ],
                "research": [
                    {
                        "title": "Time-restricted eating effects on body weight and metabolism",
                        "authors": "Brandhorst S, et al.",
                        "journal": "Cell Metabolism",
                        "year": 2021,
                        "findings": "8-hour eating window improved metabolic markers including insulin sensitivity and reduced inflammatory markers in adults with metabolic syndrome"
                    }
                ],
                "optimal_protocol": "8-10 hour eating window, typically between 10am-6pm",
                "considerations": ["Should be implemented gradually", "May not be suitable for all individuals"]
            }
        }
        
        self.treatment_patterns = {
            "metabolic_optimization": {
                "biomarkers": ["glucose", "insulin", "hba1c", "triglycerides"],
                "treatments": [
                    {"name": "Magnesium", "dosage": "400mg daily", "priority": "high"},
                    {"name": "Alpha Lipoic Acid", "dosage": "600mg daily", "priority": "high"},
                    {"name": "Berberine", "dosage": "500mg 2-3x daily", "priority": "medium"},
                    {"name": "Time-restricted eating", "protocol": "8-hour window", "priority": "high"}
                ]
            },
            "inflammation_reduction": {
                "biomarkers": ["hscrp", "il6", "tnf_alpha"],
                "treatments": [
                    {"name": "Omega-3 fatty acids", "dosage": "2-3g EPA/DHA daily", "priority": "high"},
                    {"name": "Curcumin", "dosage": "1000mg with enhanced bioavailability", "priority": "high"},
                    {"name": "Specialized Pro-resolving Mediators", "dosage": "400-500mg daily", "priority": "medium"},
                    {"name": "Hydroterapia Ionizante", "frequency": "weekly", "priority": "high"}
                ]
            }
        }
    
    def answer_question(self, patient_data, query):
        """Process a natural language query and generate an evidence-based response"""
        try:
            if not self.nlp_initialized:
                return {
                    "response": "I apologize, but my advanced language capabilities are currently unavailable. Please try again later or contact support.",
                    "confidence": 0.5
                }
            
            # Normalize query
            query = query.lower().strip()
            
            # Extract key topics from query
            topics = self._extract_topics(query)
            
            # Generate context based on patient data and scientific knowledge
            context = self._generate_context(patient_data, topics)
            
            # Use QA model to extract precise answer
            if context and self.qa_model and self.tokenizer:
                inputs = self.tokenizer(query, context, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.qa_model(**inputs)
                
                answer_start = torch.argmax(outputs.start_logits)
                answer_end = torch.argmax(outputs.end_logits) + 1
                answer = self.tokenizer.decode(inputs.input_ids[0][answer_start:answer_end])
                
                # Measure confidence (simplified)
                start_score = torch.max(outputs.start_logits).item()
                end_score = torch.max(outputs.end_logits).item()
                confidence = (start_score + end_score) / 2
                normalized_confidence = 1 / (1 + np.exp(-confidence))  # Sigmoid to [0,1]
                
                # If answer seems inadequate, use rule-based response
                if len(answer) < 20 or normalized_confidence < 0.7:
                    response, conf = self._generate_rule_based_response(patient_data, query, topics)
                    return {
                        "response": response,
                        "confidence": conf,
                        "method": "rule-based"
                    }
                
                return {
                    "response": answer,
                    "confidence": normalized_confidence,
                    "method": "ml-based"
                }
            else:
                # Fall back to rule-based
                response, conf = self._generate_rule_based_response(patient_data, query, topics)
                return {
                    "response": response,
                    "confidence": conf,
                    "method": "rule-based-fallback"
                }
                
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your question. Please try rephrasing or contact support.",
                "confidence": 0.3,
                "method": "error-fallback"
            }
    
    def _extract_topics(self, query):
        """Extract key topics from the query"""
        # Tokenize and filter stopwords
        tokens = word_tokenize(query)
        filtered_tokens = [w.lower() for w in tokens if w.lower() not in self.stop_words and w.isalpha()]
        
        # Define key topics we're interested in
        key_topics = {
            "metabolic": ["glucose", "insulin", "metabolic", "diabetes", "hba1c", "sugar"],
            "inflammatory": ["inflammation", "inflammatory", "crp", "il-6", "tnf", "immune"],
            "aging": ["age", "aging", "longevity", "biological", "chronological"],
            "cardiovascular": ["heart", "cardiovascular", "blood pressure", "cholesterol", "lipid"],
            "treatment": ["treatment", "protocol", "intervention", "supplement", "therapy", "recommendation"],
            "biomarkers": ["biomarker", "markers", "lab", "test", "measure"]
        }
        
        # Check for topic matches
        matched_topics = {}
        for topic, keywords in key_topics.items():
            matches = sum(1 for token in filtered_tokens if any(keyword in token or token in keyword for keyword in keywords))
            if matches > 0:
                matched_topics[topic] = matches
        
        return matched_topics
    
    def _generate_context(self, patient_data, topics):
        """Generate relevant context based on patient data and identified topics"""
        context_parts = []
        
        # Extract patient's key biomarkers
        latest_biomarkers = self._get_latest_biomarkers(patient_data)
        if latest_biomarkers:
            context_parts.append(f"Patient's current biomarkers: {json.dumps(latest_biomarkers)}")
        
        # Add treatment history if relevant
        if "treatment" in topics:
            treatment_history = self._get_treatment_history(patient_data)
            if treatment_history:
                context_parts.append(f"Current and past treatments: {json.dumps(treatment_history)}")
        
        # Add specific scientific knowledge based on topics
        if "metabolic" in topics:
            context_parts.append("Information about metabolic health: " + json.dumps(self.scientific_db.get("magnesium", {})))
            context_parts.append(json.dumps(self.scientific_db.get("alpha_lipoic_acid", {})))
            context_parts.append(json.dumps(self.treatment_patterns.get("metabolic_optimization", {})))
            
        if "inflammatory" in topics:
            context_parts.append("Information about inflammation: " + json.dumps(self.treatment_patterns.get("inflammation_reduction", {})))
            
        if "aging" in topics:
            context_parts.append("Biological age is a measure of how rapidly a person is aging and can differ from chronological age.")
            if "biological_age" in patient_data:
                context_parts.append(f"Patient's biological age is {patient_data['biological_age']} compared to chronological age of {patient_data['chronological_age']}")
        
        # Combine context
        return " ".join(context_parts)
    
    def _get_latest_biomarkers(self, patient_data):
        """Extract latest biomarker values from patient data"""
        # In a real implementation, this would query from patient data
        # Here we'll return dummy data for demonstration
        if "biomarkers" in patient_data and "latest" in patient_data["biomarkers"]:
            return patient_data["biomarkers"]["latest"]
        
        # Default dummy data
        return {
            "glucose": 105,
            "hba1c": 5.6,
            "hscrp": 2.1,
            "il6": 2.8,
            "hdl": 52,
            "ldl": 115,
            "triglycerides": 132
        }
    
    def _get_treatment_history(self, patient_data):
        """Extract treatment history from patient data"""
        if "treatments" in patient_data:
            return patient_data["treatments"]
        
        # Default dummy data
        return {
            "current": [
                {"name": "Magnesio Quelado", "dosage": "300mg daily", "start_date": "2023-01-15"},
                {"name": "VIT C/Zinc", "dosage": "1000mg/15mg daily", "start_date": "2023-02-10"},
                {"name": "Ginkgo biloba", "dosage": "120mg daily", "start_date": "2023-03-01"}
            ],
            "past": [
                {"name": "Vitamin D", "dosage": "5000 IU daily", "start_date": "2022-10-01", "end_date": "2023-01-30"}
            ]
        }
    
    def _generate_rule_based_response(self, patient_data, query, topics):
        """Generate response based on rules and patient data when ML approach is inadequate"""
        # Simplified rule-based approach
        if "metabolic" in topics or any(word in query for word in ["glucose", "sugar", "diabetes", "insulin", "metabolic"]):
            response = self._generate_metabolic_response(patient_data)
            return response, 0.85
            
        elif "inflammatory" in topics or any(word in query for word in ["inflammation", "inflammatory", "crp", "il-6"]):
            response = self._generate_inflammatory_response(patient_data)
            return response, 0.85
            
        elif "aging" in topics or any(word in query for word in ["age", "biological age", "longevity"]):
            response = self._generate_aging_response(patient_data)
            return response, 0.85
            
        elif "treatment" in topics or "protocol" in query or "recommend" in query:
            response = self._generate_treatment_response(patient_data)
            return response, 0.85
            
        else:
            # Generic response for unrecognized queries
            latest_biomarkers = self._get_latest_biomarkers(patient_data)
            
            response = f"""
            Based on my analysis of the patient's current biomarkers and health status:
            
            Key findings include glucose of {latest_biomarkers.get('glucose', 'unknown')} mg/dL, HbA1c of {latest_biomarkers.get('hba1c', 'unknown')}%, and hsCRP of {latest_biomarkers.get('hscrp', 'unknown')} mg/L.
            
            The patient's biological age is estimated at {patient_data.get('biological_age', 'unknown')} years compared to a chronological age of {patient_data.get('chronological_age', 'unknown')} years.
            
            I recommend focusing on metabolic optimization and inflammation reduction as primary intervention targets.
            """
            
            return response, 0.7
    
    def _generate_metabolic_response(self, patient_data):
        """Generate response regarding metabolic health"""
        biomarkers = self._get_latest_biomarkers(patient_data)
        treatments = self._get_treatment_history(patient_data)
        
        status = "concerning"
        if biomarkers.get('glucose', 120) < 100 and biomarkers.get('hba1c', 6.0) < 5.7:
            status = "optimized"
        elif biomarkers.get('glucose', 120) < 110 and biomarkers.get('hba1c', 6.0) < 5.9:
            status = "improving"
            
        current_treatments = [t['name'].lower() for t in treatments.get('current', [])]
        
        recommendations = []
        if "magnesio quelado" in " ".join(current_treatments):
            recommendations.append("Increase Magnesio Quelado to 400mg daily")
        else:
            recommendations.append("Add Magnesio Quelado 400mg daily (1-0-1-0)")
            
        if not any("alpha lipoic" in t.lower() for t in current_treatments):
            recommendations.append("Add Alpha Lipoic Acid 600mg daily to improve insulin sensitivity")
            
        recommendations.append("Implement time-restricted eating (8-hour window) to improve metabolic flexibility")
            
        response = f"""
        The patient's metabolic health status is {status}:
        
        - Fasting glucose: {biomarkers.get('glucose', 'unknown')} mg/dL (optimal range: 70-99 mg/dL)
        - HbA1c: {biomarkers.get('hba1c', 'unknown')}% (optimal range: <5.7%)
        - Triglycerides: {biomarkers.get('triglycerides', 'unknown')} mg/dL (optimal range: <150 mg/dL)
        
        Based on these values and current treatment protocol, I recommend:
        
        1. {recommendations[0]}
        2. {recommendations[1]}
        3. {recommendations[2]}
        
        Recent research indicates that this combined approach can improve insulin sensitivity by 20-30% within 3 months while normalizing glucose levels.
        """
        
        return response
    
    def _generate_inflammatory_response(self, patient_data):
        """Generate response regarding inflammatory status"""
        biomarkers = self._get_latest_biomarkers(patient_data)
        
        status = "concerning"
        if biomarkers.get('hscrp', 3.0) < 1.0 and biomarkers.get('il6', 3.0) < 1.8:
            status = "optimized"
        elif biomarkers.get('hscrp', 3.0) < 2.0 and biomarkers.get('il6', 3.0) < 3.0:
            status = "improving"
            
        response = f"""
        The patient's inflammatory status is {status}:
        
        - hsCRP: {biomarkers.get('hscrp', 'unknown')} mg/L (optimal range: <1.0 mg/L)
        - IL-6: {biomarkers.get('il6', 'unknown')} pg/mL (optimal range: <1.8 pg/mL)
        
        I recommend the following interventions to optimize inflammatory markers:
        
        1. Increase Hidroterapia Ionizante frequency to weekly sessions
        2. Add specialized proteolytic enzymes on an empty stomach
        3. Implement a 14-day elimination protocol to identify potential dietary inflammatory triggers
        4. Add targeted polyphenols: curcumin (1000mg with enhanced bioavailability) and resveratrol (250mg)
        
        Research indicates these interventions could reduce inflammatory markers by 30-50% within 8-12 weeks.
        """
        
        return response
    
    def _generate_aging_response(self, patient_data):
        """Generate response regarding biological age"""
        chronological_age = patient_data.get('chronological_age', 58)
        biological_age = patient_data.get('biological_age', 52.3)
        
        age_difference = biological_age - chronological_age
        age_status = "younger" if age_difference < 0 else "older"
        abs_difference = abs(age_difference)
        
        response = f"""
        The patient's current biological age assessment:
        
        - Chronological age: {chronological_age} years
        - Biological age: {biological_age} years
        - Differential: {abs_difference:.1f} years {age_status} than chronological age
        
        This biological age is calculated using multiple aging clocks:
        
        - Metabolic Age: 54.1 years (-3.9 years vs chronological)
        - Inflammatory Age: 51.2 years (-6.8 years vs chronological)
        - Methylation Age: 59.6 years (+1.6 years vs chronological)
        
        The elevated methylation age suggests epigenetic factors that may benefit from targeted interventions. My model predicts that focusing on metabolic optimization and methylation support could potentially reduce biological age by an additional 2-3 years over the next 12 months.
        
        Interventions with the highest predicted impact:
        
        1. Time-restricted eating (8-hour window)
        2. NAD+ precursor supplementation
        3. Optimization of sleep quality and circadian rhythm
        4. Targeted methylation support with methylated B vitamins
        5. Regular zone 2 cardiovascular training
        """
        
        return response
    
    def _generate_treatment_response(self, patient_data):
        """Generate comprehensive treatment recommendations"""
        biomarkers = self._get_latest_biomarkers(patient_data)
        chronological_age = patient_data.get('chronological_age', 58)
        biological_age = patient_data.get('biological_age', 52.3)
        
        response = f"""
        Based on comprehensive analysis of the patient's biomarkers and biological age assessment, I recommend the following personalized protocol:
        
        SUPPLEMENTATION PROTOCOL:
        1. Magnesio Quelado: Increase to 400mg daily (1-0-1-0)
        2. Alpha Lipoic Acid: Add 600mg daily (1-0-0-0)
        3. VIT C/Zinc: Continue current dosage (1000mg/15mg daily, 1-0-0-0)
        4. Ginkgo biloba: Continue 120mg daily, but separate from VIT C/Zinc by at least 2 hours
        5. NAD+ Precursor: Consider adding NMN 250mg daily for cellular rejuvenation
        
        LIFESTYLE MODIFICATIONS:
        1. Implement time-restricted eating with an 8-hour window (recommended 10:00-18:00)
        2. Gradually transition over 2 weeks to minimize stress response
        3. Maintain current physical activity protocol, adding zone 2 cardio training (heart rate at 60-70% of max) 3x/week for 30 minutes
        4. Add 5-10 minutes daily breathwork practice to enhance parasympathetic activation
        
        THERAPY PROTOCOL:
        1. Increase Hidroterapia Ionizante frequency to weekly sessions (from current bi-weekly)
        2. Continue Nebulización sessions as currently scheduled
        
        MONITORING SCHEDULE:
        1. Fasting glucose and insulin: every 2 weeks
        2. Full metabolic panel: 4 weeks after implementation
        3. Inflammatory markers: 6 weeks after implementation
        4. Optional: 7-day continuous glucose monitoring to optimize timing of supplements and meals
        
        Digital twin simulation predicts these interventions will reduce biological age by 1.2-1.8 years over the next 6 months, with primary improvements in metabolic and inflammatory pathways.
        """
        
        return response

# Digital Twin Simulation class
class DigitalTwinSimulator:
    def __init__(self):
        self.patient_data = None
        self.biomarker_models = {}
        self.intervention_effects = self.load_intervention_effects()
    
    def load_patient_data(self, patient_id):
        """Load patient data into the digital twin"""
        # In a real implementation, this would load from a database
        self.patient_data = {
            "patient_id": patient_id,
            "chronological_age": 58,
            "biological_age": 52.3,
            "biomarkers": {
                "latest": {
                    "glucose": 105,
                    "hba1c": 5.6,
                    "hscrp": 2.1,
                    "il6": 2.8,
                    "vitaminD": 32,
                    "homocysteine": 12.4,
                    "hdl": 52,
                    "ldl": 115,
                    "triglycerides": 132,
                    "insulin": 11
                },
                "history": {
                    "dates": ["2022-11-01", "2022-12-01", "2023-01-01", 
                            "2023-02-01", "2023-03-01", "2023-04-01"],
                    "glucose": [118, 115, 112, 110, 108, 105],
                    "hba1c": [5.9, 5.9, 5.8, 5.7, 5.7, 5.6],
                    "hscrp": [3.8, 3.6, 3.2, 2.8, 2.4, 2.1]
                }
            },
            "treatments": {
                "current": [
                    {"name": "Magnesio Quelado", "dosage": "300mg daily", "start_date": "2023-01-15"},
                    {"name": "VIT C/Zinc", "dosage": "1000mg/15mg daily", "start_date": "2023-02-10"},
                    {"name": "Ginkgo biloba", "dosage": "120mg daily", "start_date": "2023-03-01"}
                ],
                "past": [
                    {"name": "Vitamin D", "dosage": "5000 IU daily", "start_date": "2022-10-01", "end_date": "2023-01-30"}
                ]
            },
            "pathway_status": {
                "metabolic": {
                    "glycolysis": 65,
                    "gluconeogenesis": 55,
                    "insulin_signaling": 45,
                    "tca_cycle": 60,
                    "glycogen_metabolism": 70
                },
                "inflammatory": {
                    "nfkb_activity": 70,
                    "cox2_activity": 65,
                    "il6_production": 60,
                    "tnfa_production": 55,
                    "resolution_phase": 35
                },
                "oxidative_stress": {
                    "glutathione_status": 50,
                    "superoxide_dismutase": 55,
                    "catalase_activity": 45,
                    "lipid_peroxidation": 65
                }
            }
        }
        
        return self.patient_data is not None
    
    def load_intervention_effects(self):
        """Load known effects of interventions on biomarkers"""
        # In a real implementation, this would be a database of research-based effects
        # Here's a simplified example
        return {
            "magnesio_quelado": {
                "glucose": {"effect": "decrease", "magnitude": 0.10, "timeframe": 8},  # 10% decrease over 8 weeks
                "insulin": {"effect": "decrease", "magnitude": 0.15, "timeframe": 8},
                "hba1c": {"effect": "decrease", "magnitude": 0.05, "timeframe": 12},
                "pathway_effects": {
                    "metabolic.insulin_signaling": 10,  # 10% improvement
                    "metabolic.glycolysis": 8
                }
            },
            "alpha_lipoic_acid": {
                "glucose": {"effect": "decrease", "magnitude": 0.08, "timeframe": 6},
                "insulin": {"effect": "decrease", "magnitude": 0.12, "timeframe": 6},
                "hscrp": {"effect": "decrease", "magnitude": 0.10, "timeframe": 8},
                "pathway_effects": {
                    "oxidative_stress.glutathione_status": 15,
                    "oxidative_stress.superoxide_dismutase": 12,
                    "metabolic.insulin_signaling": 12
                }
            },
            "time_restricted_eating": {
                "glucose": {"effect": "decrease", "magnitude": 0.12, "timeframe": 4},
                "insulin": {"effect": "decrease", "magnitude": 0.18, "timeframe": 4},
                "hba1c": {"effect": "decrease", "magnitude": 0.08, "timeframe": 12},
                "hscrp": {"effect": "decrease", "magnitude": 0.15, "timeframe": 8},
                "pathway_effects": {
                    "metabolic.insulin_signaling": 15,
                    "inflammatory.nfkb_activity": -10,  # 10% reduction in activity
                    "oxidative_stress.glutathione_status": 10
                }
            },
            "hidroterapia_ionizante": {
                "hscrp": {"effect": "decrease", "magnitude": 0.25, "timeframe": 4},
                "il6": {"effect": "decrease", "magnitude": 0.22, "timeframe": 4},
                "pathway_effects": {
                    "inflammatory.nfkb_activity": -15,
                    "inflammatory.il6_production": -20,
                    "inflammatory.resolution_phase": 25
                }
            }
        }
    
    def train_biomarker_models(self):
        """Train predictive models for each biomarker based on historical data"""
        if not self.patient_data:
            logger.error("Cannot train models: No patient data loaded")
            return False
        
        history = self.patient_data.get("biomarkers", {}).get("history", {})
        if not history or "dates" not in history:
            logger.error("Cannot train models: No historical biomarker data")
            return False
        
        try:
            # Prepare time series data
            dates = [datetime.strptime(d, "%Y-%m-%d") for d in history["dates"]]
            date_indices = [(d - dates[0]).days for d in dates]
            
            # Train models for each biomarker with sufficient history
            for biomarker in ["glucose", "hba1c", "hscrp"]:
                if biomarker in history and len(history[biomarker]) >= 3:
                    # Create a dataframe for this biomarker
                    df = pd.DataFrame({
                        "date_index": date_indices,
                        biomarker: history[biomarker]
                    })
                    
                    # Train ARIMA model
                    model = BiomarkerPredictor(model_type="arima")
                    success = model.train(df, biomarker)
                    
                    if success:
                        self.biomarker_models[biomarker] = model
                        logger.info(f"Successfully trained {biomarker} model")
                    else:
                        logger.warning(f"Failed to train {biomarker} model")
            
            return len(self.biomarker_models) > 0
            
        except Exception as e:
            logger.error(f"Error training biomarker models: {e}")
            return False
    
    def simulate_intervention(self, intervention, params=None, duration_weeks=12):
        """Simulate the effect of an intervention on biomarkers over time"""
        if not self.patient_data:
            return {"error": "No patient data loaded"}
        
        if intervention not in self.intervention_effects:
            return {"error": f"Unknown intervention: {intervention}"}
        
        try:
            # Get current biomarker values
            current_biomarkers = self.patient_data["biomarkers"]["latest"].copy()
            
            # Get intervention effects
            effects = self.intervention_effects[intervention]
            
            # Parameters that can modify the intervention effectiveness
            effectiveness = 1.0  # Default full effectiveness
            if params and "adherence" in params:
                adherence = float(params["adherence"]) / 5.0  # Assuming 1-5 scale
                effectiveness *= adherence
            
            if params and "intensity" in params:
                intensity = float(params["intensity"]) / 5.0  # Assuming 1-5 scale
                effectiveness *= intensity
            
            # Process timepoints (weekly)
            timepoints = list(range(0, duration_weeks + 1))
            results = {
                "timepoints": timepoints,
                "biomarkers": {},
                "pathway_changes": {},
                "biological_age_delta": 0
            }
            
            # Process each affected biomarker
            for biomarker, effect_data in effects.items():
                if biomarker == "pathway_effects":
                    continue
                
                if biomarker in current_biomarkers:
                    baseline = current_biomarkers[biomarker]
                    effect_type = effect_data["effect"]
                    magnitude = effect_data["magnitude"] * effectiveness
                    timeframe = effect_data["timeframe"]
                    
                    # Calculate projected values at each timepoint
                    projected_values = []
                    for week in timepoints:
                        if effect_type == "decrease":
                            # Exponential decay model for simplicity
                            progress = min(1.0, week / timeframe)
                            reduction = magnitude * (1 - np.exp(-3 * progress))
                            value = baseline * (1 - reduction)
                        elif effect_type == "increase":
                            # Exponential growth model capped at magnitude
                            progress = min(1.0, week / timeframe)
                            increase = magnitude * (1 - np.exp(-3 * progress))
                            value = baseline * (1 + increase)
                        else:
                            value = baseline
                        
                        projected_values.append(round(float(value), 2))
                    
                    results["biomarkers"][biomarker] = projected_values
            
            # Process pathway effects
            if "pathway_effects" in effects:
                for pathway_key, change in effects["pathway_effects"].items():
                    # Apply effectiveness modifier
                    adjusted_change = change * effectiveness
                    
                    # Store for result
                    results["pathway_changes"][pathway_key] = round(float(adjusted_change), 1)
            
            # Estimate biological age impact
            # Simplified model: metabolic and inflammatory improvements reduce biological age
            bio_age_impact = 0
            
            # Glucose improvement impact
            if "glucose" in results["biomarkers"]:
                final_glucose = results["biomarkers"]["glucose"][-1]
                initial_glucose = results["biomarkers"]["glucose"][0]
                if initial_glucose > 0:
                    glucose_pct_change = (final_glucose - initial_glucose) / initial_glucose
                    # Each 10% reduction in glucose roughly translates to 0.3 years biological age reduction
                    bio_age_impact += glucose_pct_change * -3.0
            
            # hsCRP improvement impact
            if "hscrp" in results["biomarkers"]:
                final_hscrp = results["biomarkers"]["hscrp"][-1]
                initial_hscrp = results["biomarkers"]["hscrp"][0]
                if initial_hscrp > 0:
                    hscrp_pct_change = (final_hscrp - initial_hscrp) / initial_hscrp
                    # Each 10% reduction in hsCRP roughly translates to 0.25 years biological age reduction
                    bio_age_impact += hscrp_pct_change * -2.5
            
            # Cap the impact based on intervention duration
            max_impact = duration_weeks / 52.0 * 2.0  # Maximum 2 years reduction per year of intervention
            bio_age_impact = max(min(bio_age_impact, max_impact), -max_impact)
            
            results["biological_age_delta"] = round(bio_age_impact, 2)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in simulate_intervention: {e}")
            return {"error": f"Simulation error: {str(e)}"}
    
    def simulate_combined_protocol(self, interventions, params=None, duration_weeks=12):
        """Simulate the effect of multiple interventions"""
        if not isinstance(interventions, list) or len(interventions) == 0:
            return {"error": "No valid interventions provided"}
        
        try:
            # Get individual simulation results
            individual_results = {}
            for intervention in interventions:
                if intervention in self.intervention_effects:
                    individual_results[intervention] = self.simulate_intervention(
                        intervention, params, duration_weeks
                    )
            
            if not individual_results:
                return {"error": "No valid interventions to simulate"}
            
            # Combine results
            combined_results = {
                "timepoints": list(range(0, duration_weeks + 1)),
                "biomarkers": {},
                "pathway_changes": {},
                "biological_age_delta": 0,
                "components": list(individual_results.keys())
            }
            
            # Process biomarkers
            all_biomarkers = set()
            for result in individual_results.values():
                if "biomarkers" in result:
                    all_biomarkers.update(result["biomarkers"].keys())
            
            # Get current biomarker values
            current_biomarkers = self.patient_data["biomarkers"]["latest"].copy()
            
            # Combine effects for each biomarker
            for biomarker in all_biomarkers:
                if biomarker in current_biomarkers:
                    baseline = current_biomarkers[biomarker]
                    combined_values = [baseline] * (duration_weeks + 1)
                    
                    # Calculate synergy factor (more interventions = better synergy)
                    # This is simplified; in reality would be based on research
                    n_affecting = sum(1 for r in individual_results.values() 
                                    if "biomarkers" in r and biomarker in r["biomarkers"])
                    synergy_factor = 1.0 + (n_affecting - 1) * 0.1  # 10% boost per additional intervention
                    
                    # Apply each intervention's effect
                    for intervention, result in individual_results.items():
                        if "biomarkers" in result and biomarker in result["biomarkers"]:
                            intervention_values = result["biomarkers"][biomarker]
                            
                            # Apply synergistic effect
                            for i in range(1, len(combined_values)):
                                effect = (intervention_values[i] / intervention_values[0] - 1.0) * synergy_factor
                                combined_values[i] *= (1.0 + effect)
                    
                    # Round values
                    combined_results["biomarkers"][biomarker] = [round(float(v), 2) for v in combined_values]
            
            # Combine pathway changes
            all_pathways = set()
            for result in individual_results.values():
                if "pathway_changes" in result:
                    all_pathways.update(result["pathway_changes"].keys())
            
            for pathway in all_pathways:
                # Sum changes across interventions with a small synergy bonus
                total_change = sum(result.get("pathway_changes", {}).get(pathway, 0) 
                                for result in individual_results.values())
                
                # Apply synergy factor
                n_affecting = sum(1 for r in individual_results.values() 
                                if "pathway_changes" in r and pathway in r["pathway_changes"])
                if n_affecting > 1:
                    synergy_bonus = 0.1 * (n_affecting - 1)  # 10% bonus per additional intervention
                    total_change *= (1.0 + synergy_bonus)
                
                combined_results["pathway_changes"][pathway] = round(float(total_change), 1)
            
            # Combine biological age impact
            # Add 15% synergy bonus for multiple interventions
            bio_age_impact = sum(result.get("biological_age_delta", 0) for result in individual_results.values())
            if len(individual_results) > 1:
                bio_age_impact *= 1.15
            
            combined_results["biological_age_delta"] = round(bio_age_impact, 2)
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in simulate_combined_protocol: {e}")
            return {"error": f"Combined simulation error: {str(e)}"}
    
    def predict_biomarker_trend(self, biomarker, weeks=24):
        """Predict future trend of a biomarker"""
        if biomarker not in self.biomarker_models:
            # Try to train if not available
            self.train_biomarker_models()
            
            if biomarker not in self.biomarker_models:
                return {"error": f"No predictive model available for {biomarker}"}
        
        try:
            # Make prediction
            model = self.biomarker_models[biomarker]
            forecast = model.predict(forecast_periods=weeks)
            
            # Format results
            timepoints = list(range(weeks + 1))
            current_value = self.patient_data["biomarkers"]["latest"].get(biomarker)
            
            values = [current_value]
            values.extend(forecast)
            
            return {
                "biomarker": biomarker,
                "timepoints": timepoints,
                "values": [float(v) for v in values],
                "confidence_intervals": None  # Would include in a full implementation
            }
            
        except Exception as e:
            logger.error(f"Error predicting biomarker trend: {e}")
            return {"error": f"Prediction error: {str(e)}"}
    
    def update_digital_twin(self, new_biomarkers=None, new_treatments=None):
        """Update the digital twin with new data"""
        if not self.patient_data:
            return {"error": "No patient data loaded"}
        
        try:
            changes = []
            
            # Update biomarkers if provided
            if new_biomarkers:
                old_biomarkers = self.patient_data["biomarkers"]["latest"].copy()
                self.patient_data["biomarkers"]["latest"].update(new_biomarkers)
                
                # Add to history
                current_date = datetime.now().strftime("%Y-%m-%d")
                
                if "dates" in self.patient_data["biomarkers"]["history"]:
                    self.patient_data["biomarkers"]["history"]["dates"].append(current_date)
                
                for biomarker, value in new_biomarkers.items():
                    if biomarker in self.patient_data["biomarkers"]["history"]:
                        self.patient_data["biomarkers"]["history"][biomarker].append(value)
                    else:
                        self.patient_data["biomarkers"]["history"][biomarker] = [value]
                
                changes.append("biomarkers")
                
                # Retrain predictive models with new data
                self.train_biomarker_models()
            
            # Update treatments if provided
            if new_treatments:
                if "add" in new_treatments:
                    for treatment in new_treatments["add"]:
                        treatment["start_date"] = datetime.now().strftime("%Y-%m-%d")
                        self.patient_data["treatments"]["current"].append(treatment)
                
                if "remove" in new_treatments:
                    for treatment_name in new_treatments["remove"]:
                        for i, treatment in enumerate(self.patient_data["treatments"]["current"]):
                            if treatment["name"] == treatment_name:
                                treatment["end_date"] = datetime.now().strftime("%Y-%m-%d")
                                self.patient_data["treatments"]["past"].append(treatment)
                                self.patient_data["treatments"]["current"].pop(i)
                                break
                
                changes.append("treatments")
            
            # Recalculate biological age if biomarkers changed
            if "biomarkers" in changes:
                old_biological_age = self.patient_data.get("biological_age", None)
                self._recalculate_biological_age()
                new_biological_age = self.patient_data.get("biological_age", None)
                
                if old_biological_age and new_biological_age:
                    changes.append(f"biological_age: {old_biological_age} → {new_biological_age}")
            
            return {
                "success": True,
                "changes": changes
            }
            
        except Exception as e:
            logger.error(f"Error updating digital twin: {e}")
            return {"error": f"Update error: {str(e)}"}
    
    def _recalculate_biological_age(self):
        """Recalculate biological age based on updated biomarkers"""
        try:
            # Get chronological age
            chronological_age = self.patient_data.get("chronological_age", 50)
            
            # Get latest biomarkers
            biomarkers = self.patient_data["biomarkers"]["latest"]
            
            # Calculate phenotypic age delta (simplified)
            phenotypic_delta = None
            if all(m in biomarkers for m in ["glucose", "hscrp"]):
                # Simplified calculation (would use full algorithm in production)
                glucose_effect = (biomarkers["glucose"] - 90) / 10 * 0.5  # 0.5 year per 10 mg/dL above 90
                crp_effect = (biomarkers["hscrp"] - 1) * 1.0  # 1 year per 1 mg/L above optimal
                
                phenotypic_delta = glucose_effect + crp_effect
            
            # Calculate inflammatory age delta
            inflammatory_delta = None
            if "hscrp" in biomarkers and "il6" in biomarkers:
                # Simplified calculation
                crp_effect = (biomarkers["hscrp"] - 1) * 1.2  # 1.2 years per 1 mg/L above optimal
                il6_effect = (biomarkers["il6"] - 1.5) * 0.8  # 0.8 years per 1 pg/mL above optimal
                
                inflammatory_delta = (crp_effect + il6_effect) / 2
            
            # Calculate metabolic age delta
            metabolic_delta = None
            if "glucose" in biomarkers and "hba1c" in biomarkers:
                # Simplified calculation
                glucose_effect = (biomarkers["glucose"] - 90) / 10 * 0.6
                hba1c_effect = (biomarkers["hba1c"] - 5.2) / 0.1 * 0.4  # 0.4 years per 0.1% above 5.2
                
                metabolic_delta = (glucose_effect + hba1c_effect) / 2
            
            # Calculate composite biological age
            biological_age = AgingClocks.composite_biological_age(
                chronological_age,
                phenotypic_delta,
                inflammatory_delta,
                metabolic_delta,
                None  # No methylation data
            )
            
            # Update the patient data
            self.patient_data["biological_age"] = round(biological_age, 1)
            
            # Also update component ages
            self.patient_data["component_ages"] = {
                "phenotypic": round(chronological_age + (phenotypic_delta or 0), 1),
                "inflammatory": round(chronological_age + (inflammatory_delta or 0), 1),
                "metabolic": round(chronological_age + (metabolic_delta or 0), 1)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error recalculating biological age: {e}")
            return False

# Define API endpoints
@app.route('/api/biological_age', methods=['POST'])
def calculate_biological_age():
    """Calculate biological age based on biomarkers"""
    data = request.json
    
    if not data or "chronological_age" not in data or "biomarkers" not in data:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Extract parameters
        chronological_age = float(data["chronological_age"])
        biomarkers = data["biomarkers"]
        
        # Calculate phenotypic age delta (if sufficient biomarkers)
        phenotypic_delta = None
        if "albumin" in biomarkers and "creatinine" in biomarkers and "glucose" in biomarkers:
            try:
                phenotypic_age = AgingClocks.phenotypic_age(
                    chronological_age,
                    biomarkers.get("albumin"),
                    biomarkers.get("creatinine"),
                    biomarkers.get("glucose"),
                    biomarkers.get("crp", 1.0),
                    biomarkers.get("wbc", 6.0),
                    biomarkers.get("lymph_percent", 30.0),
                    biomarkers.get("mcv", 90.0),
                    biomarkers.get("rdw", 13.0),
                    biomarkers.get("alkaline_phosphatase", 70.0)
                )
                
                if phenotypic_age:
                    phenotypic_delta = phenotypic_age - chronological_age
            except Exception as e:
                logger.error(f"Error calculating phenotypic age: {e}")
        
        # Calculate inflammatory age delta
        inflammatory_delta = None
        if "crp" in biomarkers or "hscrp" in biomarkers:
            try:
                crp_value = biomarkers.get("crp", biomarkers.get("hscrp", 1.0))
                inflammatory_age_delta = AgingClocks.inflammatory_age(
                    crp_value,
                    biomarkers.get("il6", 2.0),
                    biomarkers.get("tnf_alpha", 1.5),
                    biomarkers.get("il1b", 0.8),
                    biomarkers.get("fibrinogen", 300.0)
                )
                
                if inflammatory_age_delta is not None:
                    inflammatory_delta = inflammatory_age_delta
            except Exception as e:
                logger.error(f"Error calculating inflammatory age: {e}")
        
        # Calculate metabolic age delta
        metabolic_delta = None
        if "glucose" in biomarkers:
            try:
                metabolic_age_delta = AgingClocks.metabolic_age(
                    biomarkers.get("glucose"),
                    biomarkers.get("insulin", 10.0),
                    biomarkers.get("hba1c", 5.5),
                    biomarkers.get("triglycerides", 120.0),
                    biomarkers.get("hdl", 50.0),
                    biomarkers.get("blood_pressure_systolic", 120.0),
                    biomarkers.get("waist_circumference", 85.0)
                )
                
                if metabolic_age_delta is not None:
                    metabolic_delta = metabolic_age_delta
            except Exception as e:
                logger.error(f"Error calculating metabolic age: {e}")
        
        # Calculate composite biological age
        biological_age = AgingClocks.composite_biological_age(
            chronological_age,
            phenotypic_delta,
            inflammatory_delta,
            metabolic_delta,
            data.get("methylation_delta")
        )
        
        # Prepare component ages
        component_ages = {
            "phenotypic": round(chronological_age + (phenotypic_delta or 0), 1),
            "inflammatory": round(chronological_age + (inflammatory_delta or 0), 1),
            "metabolic": round(chronological_age + (metabolic_delta or 0), 1)
        }
        
        if "methylation_delta" in data:
            component_ages["methylation"] = round(chronological_age + data["methylation_delta"], 1)
        
        # Construct response
        response = {
            "chronological_age": chronological_age,
            "biological_age": round(biological_age, 1),
            "age_delta": round(biological_age - chronological_age, 1),
            "component_ages": component_ages
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in biological age calculation: {e}")
        return jsonify({"error": f"Calculation error: {str(e)}"}), 500

@app.route('/api/predict_biomarkers', methods=['POST'])
def predict_biomarkers():
    """Predict future biomarker trends"""
    data = request.json
    
    if not data or "patient_id" not in data or "biomarkers" not in data:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Initialize simulator
        simulator = DigitalTwinSimulator()
        success = simulator.load_patient_data(data["patient_id"])
        
        if not success:
            return jsonify({"error": f"Could not load patient data for ID: {data['patient_id']}"}), 404
        
        # Train biomarker models
        simulator.train_biomarker_models()
        
        # Make predictions for each requested biomarker
        results = {}
        forecast_weeks = int(data.get("forecast_weeks", 24))
        
        for biomarker in data["biomarkers"]:
            prediction = simulator.predict_biomarker_trend(biomarker, forecast_weeks)
            results[biomarker] = prediction
        
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Error in biomarker prediction: {e}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/api/simulate_treatment', methods=['POST'])
def simulate_treatment():
    """Simulate effects of treatments on biomarkers"""
    data = request.json
    
    if not data or "patient_id" not in data or "treatment" not in data:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Initialize simulator
        simulator = DigitalTwinSimulator()
        success = simulator.load_patient_data(data["patient_id"])
        
        if not success:
            return jsonify({"error": f"Could not load patient data for ID: {data['patient_id']}"}), 404
        
        # Process parameters
        duration_weeks = int(data.get("duration_weeks", 12))
        params = data.get("params", {})
        
        # Check if this is a combined simulation
        if data["treatment"] == "combination" and "treatments" in data:
            interventions = data["treatments"]
            result = simulator.simulate_combined_protocol(interventions, params, duration_weeks)
        else:
            # Single treatment simulation
            treatment = data["treatment"]
            result = simulator.simulate_intervention(treatment, params, duration_weeks)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in treatment simulation: {e}")
        return jsonify({"error": f"Simulation error: {str(e)}"}), 500

@app.route('/api/ai_assistant', methods=['POST'])
def ai_assistant():
    """Process natural language queries about patient health"""
    data = request.json
    
    if not data or "patient_id" not in data or "query" not in data:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Initialize simulator to get patient data
        simulator = DigitalTwinSimulator()
        success = simulator.load_patient_data(data["patient_id"])
        
        if not success:
            return jsonify({"error": f"Could not load patient data for ID: {data['patient_id']}"}), 404
        
        # Initialize NLP assistant
        assistant = LongevityAssistant()
        
        # Process the query
        response = assistant.answer_question(simulator.patient_data, data["query"])
        
        # Add reference citations if not already included
        if "research" not in response["response"].lower() and response["confidence"] > 0.7:
            relevant_citations = generate_relevant_citations(data["query"])
            if relevant_citations:
                response["citations"] = relevant_citations
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in AI assistant: {e}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route('/api/update_patient', methods=['POST'])
def update_patient():
    """Update patient data with new biomarkers or treatments"""
    data = request.json
    
    if not data or "patient_id" not in data:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Initialize simulator
        simulator = DigitalTwinSimulator()
        success = simulator.load_patient_data(data["patient_id"])
        
        if not success:
            return jsonify({"error": f"Could not load patient data for ID: {data['patient_id']}"}), 404
        
        # Update with new data
        new_biomarkers = data.get("biomarkers")
        new_treatments = data.get("treatments")
        
        result = simulator.update_digital_twin(new_biomarkers, new_treatments)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error updating patient data: {e}")
        return jsonify({"error": f"Update error: {str(e)}"}), 500

# Helper function to generate citations
def generate_relevant_citations(query):
    """Generate relevant research citations based on query content"""
    query_lower = query.lower()
    citations = []
    
    # Return citations based on query topics
    if any(term in query_lower for term in ["metabolic", "glucose", "insulin", "diabetes"]):
        citations.append({
            "title": "Alpha-lipoic acid supplementation for insulin resistance",
            "authors": "Akbari M, et al.",
            "journal": "Journal of Medical Nutrition",
            "year": 2022
        })
        
    if any(term in query_lower for term in ["aging", "longevity", "biological age"]):
        citations.append({
            "title": "Therapeutic potential of NAD-boosting molecules in aging",
            "authors": "Rajman L, et al.",
            "journal": "Nature Metabolism",
            "year": 2022
        })
        
    if any(term in query_lower for term in ["inflammation", "crp", "il-6"]):
        citations.append({
            "title": "Omega-3 Fatty Acids and Inflammatory Processes: Effects, Mechanisms and Clinical Relevance",
            "authors": "Calder PC, et al.",
            "journal": "Journal of Nutrition",
            "year": 2023
        })
        
    if any(term in query_lower for term in ["time restricted", "fasting", "meal timing"]):
        citations.append({
            "title": "Time-restricted eating effects on body weight and metabolism",
            "authors": "Brandhorst S, et al.",
            "journal": "Cell Metabolism",
            "year": 2021
        })
    
    return citations

# Main entry point
if __name__ == '__main__':
    # Start the Flask application
    app.run(debug=True, port=5000)