import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

#-----------------------------------------------------------------------------------------------------

scaler = StandardScaler()

model = open('C://Users//TeeFaith//Desktop//ML PROJECTS//HEALTH CARE INSURANCE PROJECT//healthcare.pkl', 'rb')
healthcare_model = joblib.load(model)

def predict_function(age, bmi, children, sex_array, smoker_array, location_array):
    full_array = np.array([age, bmi, children, sex_array, smoker_array, location_array])
    reshaped_array = full_array.reshape(1, -1)
    reshaped_array_int = reshaped_array.astype(int)
    scaled_array = scaler.transform(reshaped_array_int)
    
    predicted_value = healthcare_model.predict(scaled_array)
    return predicted_value
#---------------------------------------------------------------------------------------------

st.title("Health Insurance Cost Prediction")

age = st.text_input('Enter Age:')

bmi = st.text_input('Enter BMI:')

children = st.text_input('Enter Number of children:')

sex = st.selectbox( 'Select Client Sex:',
              ['Female', 'Male'], index=0)

smoker = st.selectbox('Smoker?:',
                      ['No', 'Yes'], index=0)

location = st.selectbox('Client Location:',
                        ['NorthEast', 'NorthWest', 'SouthEast', 'SouthWest'], index=0)


sex_mode = 1
 
if sex== 'Female':
    sex_mode = 1
if sex == 'Male':
    sex_mode = 2
    
sex_array = ()
if sex_mode==1:
    sex_array = (1, 0)
else:
    sex_array = (0, 1)
    
smoker_mode = 1

if smoker ==  'No':
    smoker_mode = 1
if smoker == 'Yes':
    smoker_mode = 2

smoker_array = ()
if smoker_mode==1:
    smoker_array = (1,0)
else:
    smoker_array = (0, 1)
    


location_mode = 1

if location == 'NorthEast':
    location_mode = 1
if location == 'NorthWest':
    location_mode = 2
if location == 'SouthEast':
    location_mode = 3
if location == 'SouthWest':
    location_mode = 4

location_array = ()    
if location_mode == 1:
    location_array = (1, 0, 0, 0)
elif location_mode == 2:
    location_array = (0, 1, 0, 0)
elif location_mode == 3:
    location_array = (0, 0, 1, 0)
elif location_mode == 4:
    location_array = (0, 0, 0, 1)
    
    
prediction = ''
if st.button('Predict'):
    prediction = predict_function(age, bmi, children, sex_array, smoker_array, location_array)
    st.success('The predicted Health insurance cost is : {}'.format(prediction))