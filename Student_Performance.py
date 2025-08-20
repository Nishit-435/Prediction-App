import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://nishitdb:Cnd_Cdr_423#152@cluster0.7opr5pi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['Student']
collection = db["Student-Prediction"]

def load_model():
    with open("DataScience\Maths\Student-Performance-App\Student_lr_final_model.pkl", "rb") as file:
        model,scaler,le = pickle.load(file)
    return model,scaler,le

def preprocessing_input_data(data,scaler,le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model, scaler, le = load_model()
    preprocessed_data = preprocessing_input_data(data, scaler, le)
    prediction = model.predict(preprocessed_data)
    return prediction

def main():
    st.title("Student Performance Prediction")
    st.write("Enter Your Data To Get Prediction Of Your Performance")  

    hours_studied = st.number_input("Hours Studied", min_value=1, max_value=10, value=4)
    previous_scores = st.number_input("Previous Scores", min_value=40, max_value=100, value=89)
    extra_curr_activity = st.selectbox("Extracurricular Activities", options=['Yes', 'No'])
    sleeping_hours = st.number_input("Sleep Hours", min_value=4, max_value=10, value=7)
    number_Of_paper_solved = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=10, value=5)


    if st.button("Predict Your Score"):
        user_data = {
            "Hours Studied" : hours_studied,
            "Previous Scores" : previous_scores,
            "Extracurricular Activities" : extra_curr_activity,
            "Sleep Hours" : sleeping_hours,
            "Sample Question Papers Practiced" :number_Of_paper_solved
        }
        prediction = predict_data(user_data)
        st.success(f"Your Prediction Result Is {prediction}")
        user_data["Prediction"] = round(float(prediction[0]), 2)
        user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value for key, value in user_data.items()}
        collection.insert_one(user_data)

if __name__ == "__main__":
    main()


