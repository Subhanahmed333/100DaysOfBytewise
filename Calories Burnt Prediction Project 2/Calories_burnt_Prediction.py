import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Calories Burned Prediction", layout="wide")

st.write("## Calories Burned Prediction")
st.write("In this WebApp, you can predict the calories burned based on parameters such as `Age`, `Gender`, `BMI`, etc. Just input your details, and you'll see the predicted kilocalories burned.")

st.sidebar.header("User Input Parameters:")

def user_input_features():
    age = st.sidebar.slider("Age:", 10, 100, 30, help="Enter your age in years.")
    bmi = st.sidebar.slider("BMI:", 15, 40, 20, help="Enter your Body Mass Index (BMI).")
    duration = st.sidebar.slider("Duration (min):", 0, 35, 15, help="Enter the duration of exercise in minutes.")
    heart_rate = st.sidebar.slider("Heart Rate:", 60, 130, 80, help="Enter your heart rate during exercise.")
    body_temp = st.sidebar.slider("Body Temperature (C):", 36, 42, 38, help="Enter your body temperature during exercise.")
    gender = st.sidebar.radio("Gender:", ("Male", "Female"), help="Select your gender.")

    data = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": 1 if gender == "Male" else 0
    }

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.write("---")
st.header("Your Parameters:")
st.write(df)

@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    return calories, exercise

calories, exercise = load_data()

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
exercise_df["BMI"] = round(exercise_df["BMI"], 2)

exercise_df['Gender_male'] = (exercise_df['Gender'] == 'male').astype(int)
exercise_df = exercise_df.drop('Gender', axis=1)

features = ["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Gender_male", "Calories"]
exercise_df = exercise_df[features]

X = exercise_df.drop("Calories", axis=1)
y = exercise_df["Calories"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

@st.cache_resource
def train_model():
    random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    random_reg.fit(X_train, y_train)
    return random_reg

random_reg = train_model()
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction:")
with st.spinner("Calculating..."):
    time.sleep(1)
st.write(f"**{round(prediction[0], 2)} kilocalories**")

st.write("---")
st.header("Similar Results:")
with st.spinner("Finding similar cases..."):
    time.sleep(1)
range_ = [prediction[0] - 10, prediction[0] + 10]
ds = exercise_df[(exercise_df["Calories"] >= range_[0]) & (exercise_df["Calories"] <= range_[-1])]
st.write(ds.sample(5))

st.write("---")
st.header("General Information:")
st.write("### How do you compare with others?")

boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.write(f"You are older than **{round(sum(boolean_age) / len(boolean_age) * 100, 2)}%** of other people.")
st.write(f"Your exercise duration is longer than **{round(sum(boolean_duration) / len(boolean_duration) * 100, 2)}%** of other people.")
st.write(f"Your heart rate during exercise is higher than **{round(sum(boolean_heart_rate) / len(boolean_heart_rate) * 100, 2)}%** of other people.")
st.write(f"Your body temperature during exercise is higher than **{round(sum(boolean_body_temp) / len(boolean_body_temp) * 100, 2)}%** of other people.")

st.write("### Visual Comparisons")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Age Distribution")
    fig = px.histogram(exercise_df, x="Age", nbins=20, title="Age Distribution")
    fig.add_vline(x=df["Age"].values[0], line_width=3, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("BMI Distribution")
    fig = px.histogram(exercise_df, x="BMI", nbins=20, title="BMI Distribution")
    fig.add_vline(x=df["BMI"].values[0], line_width=3, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

st.write("---")
st.header("Model Evaluation:")

y_pred = random_reg.predict(X_test)
st.write(f"**Mean Absolute Error (MAE):** {metrics.mean_absolute_error(y_test, y_pred):.2f}")
st.write(f"**Mean Squared Error (MSE):** {metrics.mean_squared_error(y_test, y_pred):.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.2f}")
st.write(f"**R² Score:** {metrics.r2_score(y_test, y_pred):.2f}")

st.write("---")
st.header("Want to Improve Your Calorie Burn?")
st.write("Here are some tips based on your input parameters:")
if df["BMI"].values[0] < 25:
    st.write("- Increasing your BMI (within a healthy range) by gaining muscle mass could help you burn more calories.")
if df["Duration"].values[0] < 30:
    st.write("- Increasing your exercise duration by a few more minutes could significantly increase the calories burned.")
if df["Body_Temp"].values[0] < 38:
    st.write("- A slight increase in body temperature (due to more intense exercise) can lead to higher calorie burn.")
