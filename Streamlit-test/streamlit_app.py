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
  model1 = pickle.load(open("model.pkl", "rb"))
  return model1

# Load models
model = load_models()

# Form variables
variables = ['track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'duration_sec', 'months',
       'playlist_genre_latin', 'playlist_genre_pop', 'playlist_genre_r&b',
       'playlist_genre_rap', 'playlist_genre_rock',
       'playlist_subgenre_big room', 'playlist_subgenre_classic rock',
       'playlist_subgenre_dance pop', 'playlist_subgenre_electro house',
       'playlist_subgenre_electropop', 'playlist_subgenre_gangster rap',
       'playlist_subgenre_hard rock', 'playlist_subgenre_hip hop',
       'playlist_subgenre_hip pop', 'playlist_subgenre_indie poptimism',
       'playlist_subgenre_latin hip hop', 'playlist_subgenre_latin pop',
       'playlist_subgenre_neo soul', 'playlist_subgenre_new jack swing',
       'playlist_subgenre_permanent wave', 'playlist_subgenre_pop edm',
       'playlist_subgenre_post-teen pop',
       'playlist_subgenre_progressive electro house',
       'playlist_subgenre_reggaeton', 'playlist_subgenre_southern hip hop',
       'playlist_subgenre_trap', 'playlist_subgenre_tropical',
       'playlist_subgenre_urban contemporary']

def predict(model, user_input):
  prediction = model.predict(user_input)
  return prediction

st.title("Regression Model Dashboard")

#Dashboard layout
col1 = st.columns(1)  # Correctly create a Streamlit column layout

with col1:  # Now using the layout with context manager
    # Display charts or text for model 1
    st.subheader("XGB Model")
    # Add charts or text based on model 1

column_groups = {
  "Genre": ['playlist_genre_latin', 'playlist_genre_pop', 'playlist_genre_r&b',
       'playlist_genre_rap', 'playlist_genre_rock',],
  "Subgenre": ['playlist_subgenre_big room', 'playlist_subgenre_classic rock',
       'playlist_subgenre_dance pop', 'playlist_subgenre_electro house',
       'playlist_subgenre_electropop', 'playlist_subgenre_gangster rap',
       'playlist_subgenre_hard rock', 'playlist_subgenre_hip hop',
       'playlist_subgenre_hip pop', 'playlist_subgenre_indie poptimism',
       'playlist_subgenre_latin hip hop', 'playlist_subgenre_latin pop',
       'playlist_subgenre_neo soul', 'playlist_subgenre_new jack swing',
       'playlist_subgenre_permanent wave', 'playlist_subgenre_pop edm',
       'playlist_subgenre_post-teen pop',
       'playlist_subgenre_progressive electro house',
       'playlist_subgenre_reggaeton', 'playlist_subgenre_southern hip hop',
       'playlist_subgenre_trap', 'playlist_subgenre_tropical',
       'playlist_subgenre_urban contemporary'],    
}
int_vars = ['track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'duration_sec', 'months']

# User input form
st.subheader("Make a Prediction")

user_input = {}
for var in variables:
  # Removed unnecessary with statement
  if var in column_groups:
    # Use radio buttons or checkboxes based on your requirement
    selected_option = st.radio(var, column_groups[var])
    user_input[var] = selected_option
  else:
    user_input[var] = st.slider(var, min_value=..., max_value=..., value=...)

# Submit button and prediction display
if st.button("Predict"):
  # Make prediction using the chosen model
  prediction = predict(model, user_input)
  st.write(f"Number of predicted streams: {prediction}")
