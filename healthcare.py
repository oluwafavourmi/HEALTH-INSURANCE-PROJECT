import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.preprocessing import StandardScaler


gender_dict = {
    'female' : [1, 0],
    'male' : [0, 1]
}

smokers_dict = {
    'no' : [1, 0],
    'yes' : [0, 1]
}

locations_dict = {
    'north_east' : [1, 0, 0, 0],
    'north_west' : [0, 1, 0, 0],
    'south_east' : [0, 0, 1, 0],
    'south_west' : [0, 0, 0, 1]
}


app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        Age = request.args.get['client-id']
        BMI = request.args.get['bmi-id']
        children = request.args.get['children-id']

        gender = request.args.get['gender-id']
        gender_selected = []  
        if gender=="female":
            gender_selected.append(gender_dict['female'])
        else:
            gender_selected.append(gender_dict['male'])
    

        smoker = request.args.get['smoker-id']
        smoker_selected = []
        if smoker == "no":
            smoker_selected.append(smokers_dict['no'])
        else:
            smoker_selected.append(smokers_dict['yes'])


        location = request.args.get['location-id']
        location_selected = []
        if location == "north_east":
            location_selected.append(locations_dict['north_east'])
        elif location == "north_west":
            location_selected.append(locations_dict['north_west'])
        elif location == "south_east":
            location_selected.append(locations_dict['south_east'])
        elif location == "south_west":
            location_selected.append(locations_dict['south_west'])
    
    #CONVERT THE INDEPENDENT FEATURES TO ARRAYS
    input_array = np.asarray([[Age, BMI, children, gender_selected, smoker_selected, location_selected]])

    #RESHAPE THE INDEPENDENT FEATURES
    scaler = StandardScaler()
    input_reshape = input_array.reshape(1, -1)
    feat_scaled = scaler.fit_transform(input_reshape)
    model = joblib.load('healthcare.pkl')
    predictions = model.predict(feat_scaled)

    return render_template('after.html', message=f"This Client's likely bill is{predictions}")

if __name__ == '__main__':
    app.run(debug=True, port=8080)