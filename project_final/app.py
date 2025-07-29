from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the Keras model
model = load_model('CKDD.h5')

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

        # Arrange inputs
        input_features = np.array([[anemia, cad, blood_glucose_random,
                                    blood_urea, pus_cell,
                                    red_blood_cell, diabetes, pedal_edema]])

        # Predict using model
        prediction = model.predict(input_features)[0][0]

        # Determine result
        result = 'CKD Detected' if prediction >= 0.5 else 'No CKD Detected'
        is_ckd = prediction >= 0.5  # Boolean flag for template

        return render_template('result.html', prediction_text=result, is_ckd=is_ckd)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
