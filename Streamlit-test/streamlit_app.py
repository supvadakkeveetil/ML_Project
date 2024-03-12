import streamlit as st
import pickle
import numpy as np
import pandas as pd
#import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier
#import tensorflow as tf

# Function to load models
def load_models():
  model1 = pickle.load(open("https://github.com/supvadakkeveetil/ML_Project/raw/main/Streamlit-test/model.pkl", "rb"))
  return model1
spotify_df = pd.read_csv('https://raw.githubusercontent.com/supvadakkeveetil/ML_Project/main/Cleaned_Data/processed_data_cleaned_12March.csv')

spotify_df.drop(columns = ['track_id', 'track_name', 'track_artist', 'track_popularity', 'stream_count'], inplace=True)
# Load models
model = load_models()

# Form variables
variables = ['danceability', 'energy', 'key', 'loudness', 'mode',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'duration_sec', 'months',
        'playlist_genre', 'playlist_subgenre']

# Define bins
bins = [0, 50000000, 100000000, 150000000, 200000000, 500000000, 1000000000]

def predict(model, user_input):
  prediction = model.predict(user_input)
  # Convert the prediction to a pandas Series for binning
  prediction_series = pd.Series(prediction)

  # Bin the predicted value into the specified bins
  binned_prediction = pd.cut(prediction_series, bins=bins, labels=[f'Bin {i+1}' for i in range(len(bins)-1)])
  st.write(f"Predicted Number of Streams Bin: {binned_prediction}")
  return prediction

st.title("Regression Model Dashboard")

#Dashboard layout
col1 = st.columns(1)  # Correctly create a Streamlit column layout

with col1[0]:  # Now using the layout with context manager
    # Display charts or text for model 1
    st.subheader("XGB Model")
    # Add charts or text based on model 1

column_groups = {
  "Genre": spotify_df["playlist_genre"].unique(),
  "Subgenre": spotify_df["playlist_subgenre"].unique()  
}
int_vars = ['danceability', 'energy', 'key', 'loudness', 'mode',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'duration_sec', 'months']

# User input form
st.subheader("Make a Prediction")

user_input = {}
# Define min and max values for each feature explicitly
min_max_values = {
    'danceability': (0.0, 1.0),
    'energy': (0.0, 1.0),
    'key': (-1, 11),
    'loudness': (-60.0, 1.275 ),
    'mode': (0, 1),
    'speechiness': (0.0, 1.0),
    'acousticness': (0.0, 1.0),
    'instrumentalness': (0.0, 1.0),
    'liveness': (0.0, 1.0),
    'valence': (0.0, 1.0),
    'tempo': (0.0, 240.0),
    'duration_sec': (4.0, 600.00),
    'months': (0, 240)
}

# Define step_val for different sliders
step_val_dance = .05
step_val_energy = .05
step_val_key = 1
step_val_loudness = .5
step_val_mode = 1
step_val_speech = .01
step_val_acoustic = .05
step_val_instrument = .00001
step_val_live = .05
step_val_valence = .1
step_val_tempo = 1.0
step_val_sec = 1.0
step_val_month = 1

user_input["playlist_genre"] = st.selectbox("Genre", column_groups["Genre"])



# # Use the specified min and max values in the slider
# for var in int_vars:
#     if var in min_max_values:
#         min_val, max_val = min_max_values[var]
#         step_val = {
#             'track_popularity': 1,
#             'danceability': 0.05,
#             'energy': 0.05,
#             'key': 1,
#             'loudness': 0.5,
#             'mode': 1,
#             'speechiness': 0.01,
#             'acousticness': 0.05,
#             'instrumentalness': 0.00001,
#             'liveness': 0.05,
#             'valence': 0.1,
#             'tempo': 1,
#             'duration_sec': 1,
#             'months': 1
#         }[var]
#         default_value = (min_val + max_val) // 2
#         min_val = int(min_val)
#         max_val = int(max_val)
#         user_input[var] = st.slider(var, min_value=min_val, max_value=max_val, value=default_value, step=step_val)



# Submit button and prediction display
if st.button("Predict"):
  # Make prediction using the chosen model
  prediction = predict(model, user_input)
  st.write(f"Number of predicted streams: {prediction}")
