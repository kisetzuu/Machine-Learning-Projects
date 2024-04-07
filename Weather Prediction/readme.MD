# Weather Temperature Prediction Project

## Overview
This project leverages historical weather data to predict the mean temperature in London. By employing machine learning models such as Decision Trees and Random Forests, the aim is to achieve accurate predictions of temperature based on variables like cloud cover and sunshine. The project encapsulates several data science practices including data cleaning, exploratory data analysis, feature selection, data preprocessing, model training and evaluation, and logging with MLFlow.

## Dataset
The primary dataset, `london_weather.csv`, comprises various weather metrics recorded over time. Key features include date, cloud cover, and sunshine, which have been cleaned, preprocessed, and used to predict mean temperature.

## Dependencies
The project is built using Python and several libraries essential for data manipulation, visualization, model building, and evaluation:

- Python 3.x
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
- MLFlow

## Installation
To set up the project environment:

1. Ensure Python 3.x is installed.
2. Install the necessary libraries using:

## bash
pip install pandas numpy matplotlib seaborn scikit-learn mlflow

## Usage
1. Data Cleaning: The date column is converted to datetime format, and month and year are extracted as separate features.
2. Exploratory Data Analysis (EDA): Utilizes line plots and heatmaps to uncover patterns and correlations in the data.
3. Feature Selection: Identifies cloud_cover, sunshine, and mean_temp as the primary features for temperature prediction.
4. Data Splitting: Divides the data into training and testing sets, ensuring a representative sample for both training the models and evaluating their performance.
5. Preprocessing: Includes imputing missing values and scaling features to improve model performance.
6. Model Training and Evaluation: Engages Decision Tree and Random Forest models at varying depths, with performance evaluated using the Root Mean Squared Error (RMSE) metric.
7. Logging with MLFlow: Tracks experiment details including parameters, metrics, and models to facilitate comparison and selection of the best model.

## Results
Model performances were logged and compared using MLFlow, with RMSE serving as the primary metric for evaluation. This approach allowed for the identification of the most effective model configurations based on their prediction accuracy.

## Future Work
Exploration of additional features and advanced model tuning techniques to enhance prediction accuracy.
Investigation into other machine learning models and ensemble methods.
Development of a deployment strategy for real-time temperature prediction applications.
Contribiting
Contributions to the project are welcome! Please follow these steps:

## Fork the repository.
Create a new branch for your feature or bug fix.
Submit a pull request for review.