from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        rbc = 1 if request.form['red_blood_cells'] == 'abnormal' else 0
        pc = 1 if request.form['pus_cell'] == 'abnormal' else 0
        bgr = float(request.form['blood_glucose_random'])
        bu = float(request.form['blood_urea'])
        pe = 1 if request.form['pedal_edema'] == 'yes' else 0
        ane = 1 if request.form['anemia'] == 'yes' else 0
        dm = 1 if request.form['diabetesmellitus'] == 'yes' else 0
        cad = 1 if request.form['coronary_artery_disease'] == 'yes' else 0

        features = np.array([[rbc, pc, bgr, bu, pe, ane, dm, cad]])
        prediction = model.predict(features)[0]

        result = "CKD Detected" if prediction == 0 else "No CKD Detected"
        return render_template('result.html', prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)