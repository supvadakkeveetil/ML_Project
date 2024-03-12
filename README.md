# ML_Project

## Spotify Tracks Machine Learning Model

### Objective

The primary aim of this project is to create an interactive app that can be used to predict Song popularity. Utilizing a regression analysis of the Spotify 30000 song dataset, we are looking to predict the “popularity” score of a given song based on different variables.

### Tools
The following tools were used to build this project

•	Machine Learning Models - Regression Models , Random Forest and XGBoost

•	Streamlit

•	SQLite Database and SQLAlchemy

### Process
#### 1. Data Collection or discovery
The data for Spotify song tracks was collected from Kaggle and Spotify API

#### 2. Data Cleaning and Storage
1. Extracting Stream count data from API and concatenating with Kaggle dataset
2. Clean data to remove null values, drop unnecessary columns and checking stucture to ensure that the datatypes are correctly set.(Using Pandas)
3. Uploading the cleaned data from CSV to the database to use for our models. Stored data into SQLlite database file and extracted the data into dataframe for use in the models.

#### 3. Machine Learning Models
-	Regression Analysis and Random Forest
-	XGBoost 

#### 4. StreamLit App

#### 5. Analysis 

#### Code Files:
1.   - Data Cleaning using Pandas
2.	 – Loading data to SQLite Database file and conversion to df for use
3.	 – Regression Model and Random Forest Code files
4.	 – XG Boost Code
5.	 - Streamlit App

#### References
1. Kaggle Dataset : [
](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs )
2. Spotify API Documentation
3. XGBoost Documentation
4. Streamlit Documentation

##### Acknowledgements:
- Leonard Paul-Kamara and Natural Chan

##### Team:
Danny Kuffel, Jesse Reeves, Supriya Vadakkeveetil
