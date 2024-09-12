import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf
import logging
import warnings

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Global variables
logistic_model = None
random_forest_model = None
mlp_model = None
scaler = None
le_dict = None
column_names = None

def load_models():
    global logistic_model, random_forest_model, mlp_model, scaler, le_dict, column_names
    
    try:
        logistic_model = joblib.load('./Model/Logistic_Model.joblib')
        random_forest_model = joblib.load('./Model/Random_Forest_Model.joblib')
        mlp_model = tf.keras.models.load_model('./Model/MLP_Model.keras')
        scaler = joblib.load('./Model/scaler.joblib')
        le_dict = joblib.load('./Model/label_encoders.joblib')
        with open('./Model/column_names.json', 'r') as f:
            column_names = json.load(f)
        app.logger.info("All models and preprocessing components loaded successfully.")
    except Exception as e:
        app.logger.error(f"Error loading models: {str(e)}")
        raise

# Call load_models() when the app starts
load_models()

def preprocess_input(input_data):
    global scaler, le_dict, column_names
    
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    for col in df.select_dtypes(include=['object']).columns:
        if col in le_dict:
            # Get the most frequent category from training data
            most_frequent = le_dict[col].classes_[0]
            # Replace unseen categories with the most frequent category
            df[col] = df[col].map(lambda x: x if x in le_dict[col].classes_ else most_frequent)
            # Transform
            df[col] = le_dict[col].transform(df[col])
    
    # Ensure all columns are present and in the correct order
    for col in column_names:
        if col not in df.columns:
            df[col] = 0  # or some appropriate default value
    df = df[column_names]
    
    # Scale numerical features
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    
    return df_scaled

def assess_risk(probability):
    if probability < 0.3:
        return "High Risk"
    elif probability < 0.7:
        return "Medium Risk"
    else:
        return "Low Risk"

def predict_logistic(processed_input):
    prediction = logistic_model.predict(processed_input)
    prediction_proba = logistic_model.predict_proba(processed_input)
    
    loan_status = le_dict['Loan_Status'].inverse_transform(prediction)[0]
    approval_probability = prediction_proba[0][1]
    risk_assessment = assess_risk(approval_probability)
    
    return {
        "model": "Logistic Regression",
        "loan_approved": loan_status == 'Y',
        "approval_probability": float(approval_probability),
        "risk_assessment": risk_assessment
    }

def predict_random_forest(processed_input):
    prediction = random_forest_model.predict(processed_input)
    prediction_proba = random_forest_model.predict_proba(processed_input)
    
    loan_status = le_dict['Loan_Status'].inverse_transform(prediction)[0]
    approval_probability = prediction_proba[0][1]
    risk_assessment = assess_risk(approval_probability)
    
    return {
        "model": "Random Forest",
        "loan_approved": loan_status == 'Y',
        "approval_probability": float(approval_probability),
        "risk_assessment": risk_assessment
    }

def predict_mlp(processed_input):
    prediction = mlp_model.predict(processed_input)
    
    approval_probability = float(prediction[0][0])
    risk_assessment = assess_risk(approval_probability)
    
    return {
        "model": "Multilayer Perceptron",
        "loan_approved": approval_probability > 0.5,
        "approval_probability": approval_probability,
        "risk_assessment": risk_assessment
    }

@app.route('/predict/all', methods=['POST'])
def predict_all():
    try:
        input_data = request.json
        app.logger.info(f"Received input for all models: {input_data}")
        processed_input = preprocess_input(input_data)
        
        logistic_result = predict_logistic(processed_input)
        random_forest_result = predict_random_forest(processed_input)
        mlp_result = predict_mlp(processed_input)
        
        all_results = [logistic_result, random_forest_result, mlp_result]
        
        # Determine the best model based on the highest approval probability
        best_model = max(all_results, key=lambda x: x['approval_probability'])
        
        response = {
            "individual_predictions": all_results,
            "best_model": best_model['model'],
            "overall_recommendation": "Approved" if best_model['loan_approved'] else "Rejected",
            "confidence": best_model['approval_probability'],
            "risk_assessment": best_model['risk_assessment']
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        app.logger.error(f"Error in predicting with all models: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/predict/logistic', methods=['POST'])
def logistic_endpoint():
    try:
        input_data = request.json
        processed_input = preprocess_input(input_data)
        result = predict_logistic(processed_input)
        return jsonify(result), 200
    except Exception as e:
        app.logger.error(f"Error in logistic prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/predict/random_forest', methods=['POST'])
def random_forest_endpoint():
    try:
        input_data = request.json
        processed_input = preprocess_input(input_data)
        result = predict_random_forest(processed_input)
        return jsonify(result), 200
    except Exception as e:
        app.logger.error(f"Error in random forest prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/predict/mlp', methods=['POST'])
def mlp_endpoint():
    try:
        input_data = request.json
        processed_input = preprocess_input(input_data)
        result = predict_mlp(processed_input)
        return jsonify(result), 200
    except Exception as e:
        app.logger.error(f"Error in MLP prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/echo', methods=['POST'])
def echo():
    return jsonify(request.json), 200

if __name__ == '__main__':
    app.run(debug=False)  # Set to False for production