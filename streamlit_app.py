import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier
#import tensorflow as tf

df=pd.read_csv('processed_data_cleaned.csv')
df.drop(columns=['track_id', 'track_name', 'track_artist','track_popularity','stream_count'], inplace=True)
# Function to load models

def load_models():
  model1 = pickle.load(open("model.pkl", "rb"))
  return model1

# Load models
model = load_models()

# Form variables
variables = ['danceability', 'energy', 'key', 'loudness', 'mode',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'duration_sec', 'months','playlist_genre','playlist_subgenre']

st.title("Regression Model Dashboard")

#Dashboard layout
col1 = st.columns(1)

column_groups = {
  "playlist_genre": df['playlist_genre'].unique(),
  "playlist_subgenre": df['playlist_subgenre'].unique()    
}
int_vars = ['danceability', 'energy', 'key', 'loudness', 'mode',
              'speechiness', 'acousticness', 'instrumentalness', 'liveness',
              'valence', 'tempo', 'duration_sec', 'months']


# Define min/max values using data
int_var_ranges = {var: (df[var].min(), df[var].max()) for var in int_vars}

# User input form
st.subheader("Make a Prediction")

user_input = {}
for var in variables:
    if var in column_groups:  # Use selectbox for genres
        selected_option = st.selectbox(var, column_groups[var])
        user_input[var] = selected_option
    elif var in int_vars:  # Use slider for numerical variables
        min_value, max_value = int_var_ranges[var]
        user_input[var] = st.slider(var, min_value=min_value, max_value=max_value)

#concat the inputs
user_input_df=pd.DataFrame(user_input,index=[0])
df = pd.concat([df, user_input_df], axis=0)

#make dummies
df = pd.get_dummies(df, columns=['playlist_genre', 'playlist_subgenre'], drop_first=True, dtype=int)

#remove last row to make prediction
user_input_df=df.iloc[-1,:]
user_input_df=pd.DataFrame(user_input_df.values.reshape(1,-1),columns=df.columns)

# Submit button and prediction display
if st.button("Predict"):
    # Make prediction using the chosen model
    prediction = model.predict(user_input_df)
    st.write(f"Number of predicted streams: {prediction}")
