import logging
import pandas as pd
import numpy as np
import pickle
from django.shortcuts import render, redirect
from django.contrib import messages
from sklearn.ensemble import RandomForestClassifier

PATH = "templates/plant_recommendation.csv"
df = pd.read_csv(PATH)

try:
    with open("RandomForest.pkl", "rb") as RF_Model_pkl:
        RF_model = pickle.load(RF_Model_pkl)
except FileNotFoundError:
    logging.error("Model file not found")
except pickle.UnpicklingError:
    logging.error("Error unpickling the model file")

# Create your views here.

def index(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST':
        int_features = []
        keys = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Soil PH', 'Rainfall']
            
        try:
            for key in keys:
                int_features.append(float(request.POST[key]))
        except KeyError as e:
            logging.error(f"Missing key in POST data: {e}")
            return render(request, 'index.html', {'error': 'Missing input data'})


        data = np.array([int_features], dtype=float)
        selected_planet = "Earth"

        if request.POST.get('planet'):
            data = adjust_for_mars(data)
            selected_planet = change_planet(selected_planet)

        logging.info(f"Data used for prediction: {data}")

        proba = RF_model.predict_proba(data)
        top_two_crops, error_message = get_top_two_crops(proba, RF_model)
        logging.info(error_message)

        # Perform hybridization based on top two crops
        hybrid_crop_df, hybrid_prediction = perform_hybridization(df, top_two_crops)

        logging.info(f"Top two crops: {top_two_crops}, Hybrid prediction: {hybrid_prediction}")

        messages.info(request, f"Top 2 Crops: {top_two_crops[0]}, {top_two_crops[1]}")
        messages.info(request, f"Predicted Hybrid Crop: {hybrid_prediction} [Dominant]")
        messages.info(request, f"Selected Planet: {selected_planet}")
        messages.info(request, hybrid_crop_df.to_html(classes='custom-table'))

        return redirect('index')

        
    return redirect('index')


def adjust_for_mars(data):
    """
    Adjust input data for Mars by modifying specific features.
    """
    data[0][3] -= 30  # Adjust temperature
    data[0][4] *= 0.1 #Adjust humidity
    data[0][5] *= 0.1 #Adjust Soil PH
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