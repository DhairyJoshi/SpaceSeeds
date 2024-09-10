import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.ensemble import RandomForestClassifier
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
PATH = "templates/plant_recommendation.csv"
df = pd.read_csv(PATH)

# Load the pre-trained RandomForest model
with open("RandomForest.pkl", "rb") as RF_Model_pkl:
    RF_model = pickle.load(RF_Model_pkl)

# Flask app initialization
app = Flask(__name__)

@app.route("/")
def home():
    """
    Render the home page with default values when user visits.
    """
    return render_template("index.html", 
                           top_two_crops=None, 
                           hybrid_prediction=None, 
                           hybrid_crop_df=None, 
                           selected_planet="Earth")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the top two crops and hybrid performance based on the user input.
    """
    try:
        # Get input values from the form
        int_features, error_message = get_input_features(request)
        if error_message:
            return render_template("index.html", message=error_message)
        
        data = np.array([int_features], dtype=float)
        selected_planet = "Earth"

        # Adjust for Mars if the user selects it
        if request.form.get('planet'):
            data = adjust_for_mars(data)
            selected_planet = change_planet(selected_planet)

        logging.info(f"Data used for prediction: {data}")

        # Predict the probabilities for each crop
        proba = RF_model.predict_proba(data)
        top_two_crops, error_message = get_top_two_crops(proba, RF_model)
        if error_message:
            return render_template('index.html', message=error_message, selected_planet=selected_planet)

        # Perform hybridization based on top two crops
        hybrid_crop_df, hybrid_prediction = perform_hybridization(df, top_two_crops)

        logging.info(f"Top two crops: {top_two_crops}, Hybrid prediction: {hybrid_prediction}")

        # Render the results
        return render_template('index.html', 
                               top_two_crops=top_two_crops,
                               hybrid_prediction=hybrid_prediction,
                               hybrid_crop_df=hybrid_crop_df.to_html(classes='custom-table'),
                               selected_planet=selected_planet)

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return render_template('index.html', message="An error occurred during prediction.")


def get_input_features(request):
    """
    Extract and validate input features from the request form.
    """
    int_features = []
    keys = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Soil PH', 'Rainfall']
        
    for key in keys:
        try:
            int_features.append(float(request.form[key]))
        except ValueError as e:
            logging.error(f"Error parsing input for {key}: {str(e)}")
            return None, f"Invalid input for {key}"

    return int_features, None


def adjust_for_mars(data):
    """
    Adjust input data for Mars by modifying specific features.
    """
    data[0][3] -= 30  # Adjust temperature
    data[0][4] *= 0.1 #Adjust humidity
    data[0][6] *= 0.1  # Adjust rainfall
    logging.info(f"Data adjusted for Mars: {data}")
    return data


def change_planet(selected_planet):
    """
    Change the selected planet to Mars.
    """
    return "Mars"


def get_top_two_crops(proba, model):
    """
    Get the top two crops based on prediction probabilities.
    """
    if proba is not None and len(proba[0]) >= 2:
        top_two_indices = np.argsort(proba[0])[-2:][::-1]
        top_two_crops = model.classes_[top_two_indices].tolist()
        return top_two_crops, None
    else:
        return None, "Not enough crops predicted."


def perform_hybridization(df, top_two_crops):
    """
    Perform hybridization between the top two crops and predict the performance of the hybrid crop.
    """
    # Extract traits for the top two crops
    crop1_traits = df[df["label"] == top_two_crops[0]].iloc[0][["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    crop2_traits = df[df["label"] == top_two_crops[1]].iloc[0][["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]

    crop1_df = pd.DataFrame([crop1_traits])
    crop2_df = pd.DataFrame([crop2_traits])

    # Hybridize crops using the average method
    hybrid_crop_df = hybridize_crops(crop1_df, crop2_df, method="average")

    # Predict hybrid crop performance
    hybrid_prediction = RF_model.predict(hybrid_crop_df)[0]

    return hybrid_crop_df, hybrid_prediction


def hybridize_crops(crop1, crop2, method="average"):
    """
    Hybridize two crops based on the chosen method (average, random, weighted).
    """
    hybrid = {}
    if method == "average":
        for trait in crop1.columns:
            hybrid[trait] = (crop1[trait].values[0] + crop2[trait].values[0]) / 2
    elif method == "random":
        for trait in crop1.columns:
            hybrid[trait] = np.random.choice([crop1[trait].values[0], crop2[trait].values[0]])
    elif method == "weighted":
        weight1, weight2 = 0.6, 0.4
        for trait in crop1.columns:
            hybrid[trait] = weight1 * crop1[trait].values[0] + weight2 * crop2[trait].values[0]

    return pd.DataFrame([hybrid])


if __name__ == '__main__':
    # Start Flask app
    app.run(debug=True)
