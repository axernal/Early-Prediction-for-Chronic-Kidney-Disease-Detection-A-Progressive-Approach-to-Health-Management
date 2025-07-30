from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the Pickle model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and convert form inputs
        blood_urea = float(request.form['blood_urea'])
        blood_glucose_random = float(request.form['blood_glucose_random'])

        anemia = 1 if request.form['anemia'] == 'yes' else 0
        cad = 1 if request.form['cad'] == 'yes' else 0
        pus_cell = 1 if request.form['pus_cell'] == 'yes' else 0
        red_blood_cell = 1 if request.form['rbc'] == 'abnormal' else 0
        diabetes = 1 if request.form['diabetes'] == 'yes' else 0
        pedal_edema = 1 if request.form['pedal_edema'] == 'yes' else 0

        # Create input DataFrame with correct feature names (must match training exactly)
        input_dict = {
            'red_blood_cells': [red_blood_cell],
            'pus_cell': [pus_cell],
            'blood glucose random': [blood_glucose_random],
            'blood_urea': [blood_urea],
            'pedal_edema': [pedal_edema],
            'anemia': [anemia],
            'diabetesmellitus': [diabetes],
            'coronary_artery_disease': [cad]
        }

        input_df = pd.DataFrame(input_dict)

        # Predict using the model
        prediction = model.predict(input_df)[0]
        prediction_prob = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None

        # Determine result
        result = 'CKD Detected' if prediction == 1 else 'No CKD Detected'
        is_ckd = prediction == 1

        return render_template('result.html', prediction_text=result, is_ckd=is_ckd, prob=prediction_prob)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
