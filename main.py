import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
PATH = "templates\plant_recommendation.csv"
df = pd.read_csv(PATH)

# Load the pre-trained RandomForest model
with open("RandomForest.pkl", "rb") as RF_Model_pkl:
    RF_model = pickle.load(RF_Model_pkl)


app = Flask(__name__)

@app.route("/")
def home():
    # Render the page with default (empty) values when the user first visits
    return render_template("index.html", 
                           top_two_crops=None, 
                           hybrid_prediction=None, 
                           hybrid_crop_df=None, 
                           selected_planet="earth")



@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from the form
    int_features = []
    keys = ['Nitrogen', 'phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Soil PH', 'Rainfall']
    planet = request.form.get('planet')

    for i in keys:
        try:
            int_features.append(float(request.form[i]))
        except ValueError as e:
            print(f"Error parsing input for {i}: {str(e)}")
            return render_template("index.html", message=f"Invalid input for {i}")

    data = [np.array(int_features)]
    selected_planet = "Earth"

    # Adjust features based on the planet selected
    if planet:
        data = adjust_for_mars(data)
        selected_planet = change_planet(selected_planet)

    # Print adjusted data for debugging
    print("Adjusted data:", data)

    # Predict the top two crops
    try:
        proba = RF_model.predict_proba(data)
        print("Prediction probabilities:", proba)
    except Exception as e:
        print("Error during prediction:", str(e))
        return render_template('index.html', message="Error during prediction", 
                               top_two_crops=None, hybrid_prediction=None, 
                               hybrid_crop_df=None, selected_planet=selected_planet)

    # Ensure the prediction has at least two crops
    if proba is not None and len(proba[0]) >= 2:
        top_two_indices = np.argsort(proba[0])[-2:][::-1]
        top_two_crops = RF_model.classes_[top_two_indices].tolist()
        print("Top two crops:", top_two_crops)
    else:
        return render_template('index.html', message="Not enough crops predicted",
                               top_two_crops=None, hybrid_prediction=None,
                               hybrid_crop_df=None, selected_planet=selected_planet)

    # Get crop traits for hybridization
    crop1_traits = df[df["label"] == top_two_crops[0]].iloc[0][
        ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    ]
    crop2_traits = df[df["label"] == top_two_crops[1]].iloc[0][
        ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    ]

    crop1_df = pd.DataFrame([crop1_traits])
    crop2_df = pd.DataFrame([crop2_traits])

    # Hybridize crops
    hybrid_crop_df = hybridize_crops(crop1_df, crop2_df, method="average")

    # Predict hybrid crop's performance
    hybrid_prediction = RF_model.predict(hybrid_crop_df)[0]

    print("Hybrid crop prediction:", hybrid_prediction)

    # Render the index.html template with the results
    return render_template('index.html', top_two_crops=top_two_crops,
                           hybrid_prediction=hybrid_prediction,
                           hybrid_crop_df=hybrid_crop_df.to_html(classes='custom-table'),  # Safely pass the DataFrame as HTML
                           selected_planet=selected_planet)



    '''   
        prediction_text=f"Top 2 Crops: {top_two_crops[0]}, {top_two_crops[1]}",
        hybrid_text=f"Predicted Hybrid Crop: {hybrid_prediction} Dominant",
        hybrid_traits=f"Traits: {hybrid_crop_df}",
    '''


# Function to adjust data for Mars
def adjust_for_mars(data):
    # Simple example of adjusting temperature and rainfall for Mars
    data = np.array(data, dtype=float)  # Make sure data is a NumPy array of floats
    print("Before adjustment:", data)   # Debugging: Check data before adjustment
    data[0][3] = data[0][3] - 30        # Adjust temperature
    data[0][6] = data[0][6] * 0.1       # Adjust rainfall
    print("After adjustment:", data)    # Debugging: Check data after adjustment
    return data



def change_planet(selected_planet):
    selected_planet = "Mars"
    return selected_planet 


# Hybridization function
def hybridize_crops(crop1, crop2, method="average"):
    hybrid = {}
    if method == "average":
        for trait in crop1.columns:
            hybrid[trait] = (crop1[trait].values[0] + crop2[trait].values[0]) / 2
    elif method == "random":
        for trait in crop1.columns:
            hybrid[trait] = np.random.choice(
                [crop1[trait].values[0], crop2[trait].values[0]]
            )
    elif method == "weighted":
        weight1 = 0.6
        weight2 = 0.4
        for trait in crop1.columns:
            hybrid[trait] = (
                weight1 * crop1[trait].values[0] + weight2 * crop2[trait].values[0]
            )
    return pd.DataFrame([hybrid])


if __name__ == '__main__':  
      app.run()