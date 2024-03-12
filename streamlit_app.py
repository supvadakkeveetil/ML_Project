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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier
#import tensorflow as tf

# Function to load models
def load_models():
  with open("pickled_model/best_xgboost_model.pkl", "rb") as model_file:
    model1 = pickle.load(model_file)
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

  st.write(f"Predicted Number of Streams: {bins[prediction[0]]}-{bins[prediction[0]+1]}")
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
    'danceability': [0.0, 1.0, .5, .05],
    'energy': [0.0, 1.0, .5, .05],
    'key': [-1, 11, 6, 1],
    'loudness': [-60.0, 1.275, -30.0, .5],
    'mode': [0, 1, 0, 1],
    'speechiness': [0.0, 1.0, .5, .01],
    'acousticness': [0.0, 1.0, .5, .05],
    'instrumentalness': [0.0, 1.0, .5, .00001],
    'liveness': [0.0, 1.0, .5, .05],
    'valence': [0.0, 1.0, .5, .1],
    'tempo': [0, 240, 120, 1],
    'duration_sec': [4.0, 600.00, 180.0, 1.0],
    'months': [0, 240, 120, 1]
}

user_input["playlist_genre"] = st.selectbox("Genre", column_groups["Genre"])
user_input["playlist_subgenre"] = st.selectbox("Subgenre", column_groups["Subgenre"])



# Use the specified min and max values in the slider
for var in int_vars:
    if var in min_max_values:
      
      user_input[var] = st.slider(var, min_max_values[var][0], min_max_values[var][1], min_max_values[var][2], min_max_values[var][3])

user_input = pd.DataFrame(user_input, index = [0])
combined_df = pd.concat([spotify_df, user_input], axis=0)
combined_df = pd.get_dummies(combined_df, columns=['playlist_genre', 'playlist_subgenre'], drop_first=True, dtype=int)
user_input = combined_df.iloc[-1,:]
user_input = pd.DataFrame(user_input.values.reshape(1,-1), columns=combined_df.columns)
# Submit button and prediction display
if st.button("Predict"):
  # Make prediction using the chosen model
  prediction = predict(model, user_input)

