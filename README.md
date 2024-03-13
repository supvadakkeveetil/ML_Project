# ML_Project

## Spotify Tracks Machine Learning Model
A Data-Driven Approach to Predicting Music Streaming Popularity

### Objective

The primary aim of this project is to create an interactive app that can be used to make predictions on number of streams. Utilizing a regression analysis of the Spotify 30000 song dataset, we are looking to predict this score of a given song based on different variables.

### Tools
The following tools were used to build this project

•	Machine Learning Models - Regression Models - Elastic Net, Random Forest , K Fold and XGBoost

•	Streamlit

•	SQLite Database

### Process
#### 1. Data Collection or discovery
The data for Spotify song tracks was collected from Kaggle and Spotify API

#### 2. Data Cleaning and Storage
1. Extracting Stream count data from API and concatenating with Kaggle dataset
2. Clean data to remove null values, drop unnecessary columns and checking stucture to ensure that the datatypes are correctly set.(Using Pandas)
3. Uploading the cleaned data from CSV to the database to use for our models. Stored data into SQLlite database file and extracted the data into dataframe for use in the models.

#### 3. Machine Learning Models
-	Elastic Net
- Random Forest
- K Fold
-	XGBoost 

#### 4. StreamLit App
- https://app-test-zrg86zbek2ds55tzjdle58.streamlit.app/

#### 5. Analysis 
Scores of different models
1. Elastic Net - R^2: 0.157
2. Random Forest - R^2 : 0.779
3. K-fold - Average Score : 0.5144
4. XG Boost - Accuracy : 0.7937

The XG Boost model seems to be a good Machine Learning model that we can use for our predictions.We build our app on Streamlit based on the pickel file saved from this model.

#### Files on Github Folder:
1. Cleaned_Data - Data Cleaning using Pandas (Jupyter notebook)
2. SQL_Setup – Loading data to SQLite Database file and conversion to df for use
3. Models – Code for the Machine Learning Models 
4. Graph (feature_importance) - Features ranked based on importance
5. streamlit_app - Streamlit App code
6. MachineLearning_Project_Collated - Presentation 

#### References
1. Kaggle Dataset : [(https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs )](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs )
2. Spotify API Documentation - https://developer.spotify.com/documentation/web-api/reference/get-audio-features
3. XGBoost Documentation - https://xgboost.readthedocs.io/en/stable/
4. Streamlit Documentation - https://docs.streamlit.io/

##### Acknowledgements:
- Leonard Paul-Kamara and Natural Chan

##### Team:
Danny Kuffel, Jesse Reeves, Supriya Vadakkeveetil
