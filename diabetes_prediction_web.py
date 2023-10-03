# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 17:26:30 2023

@author: pachi
"""

import numpy as np
import pickle
import streamlit as st

#loading the model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def diabetes_prediction(input_data):
    
    input_data_numpy = np.asarray(input_data)
    input_data_reshape = input_data_numpy.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)

    if (prediction[0]==0):
        return 'The person is NOT DIABETIC'
    else:
        return 'The person is DIABETIC'
    
    

def main():
    
    
    st.title('DIABETES PREDICTION')
     
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('BloodPressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age of the Person')
     
    diagnosis = ''
     
    if st.button('Diabetes Test Result'):
         diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
     
    st.success(diagnosis) 
    

     
     
     
if __name__== '__main__':
    main()
    