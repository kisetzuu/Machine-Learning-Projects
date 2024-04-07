# Run this cell to import the modules you require
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Read in the data
weather = pd.read_csv("london_weather.csv")
weather.info()

# Data cleaning
weather['date'] = pd.to_datetime(weather['date'], format="%Y-%m-%d %H:%M:%S")

weather['month'] = weather['date'].dt.month
weather['year'] = weather['date'].dt.year

#Exploratory data analysis
sns.lineplot(data = weather, x = "year", y = "mean_temp")
plt.show()
corr_matrix = weather.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f', linewidths = .5, cbar_kws = {"shrink": .5})
plt.show()

#Feature selection
feature_selection = ["cloud_cover", "sunshine", "mean_temp"]
weather = weather.dropna(subset = feature_selection)
feature_set = weather[["cloud_cover", "sunshine"]]
target = weather["mean_temp"]

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(feature_set, target, test_size = 0.2, random_state = 42)

#Preprocessing data
imputer = SimpleImputer(strategy = 'mean')

#Imputing Methods
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

#StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

mlflow.set_experiment("Weather_Prediction_Model")

#Machine learning training and evaluation
depths = [1, 5, 10, 20]

for depth in depths:
    
    dt_run_name = f"Decision_Tree_Depth_{depth}"
    
    #Starting the MLFlow run
    with mlflow.start_run(run_name = dt_run_name):
        
        #Decision Tree Model
        dt_model = DecisionTreeRegressor(max_depth=depth)
        dt_model.fit(X_train_scaled, y_train)
        dt_pred = dt_model.predict(X_test_scaled)
        dt_rmse = mean_squared_error(y_test, dt_pred, squared=False)

        #Logging for Decision Tree
        mlflow.log_param("model", "Decision Tree")
        mlflow.log_metric("rmse", dt_rmse)
        mlflow.sklearn.log_model(dt_model, f"decision_tree_depth_{depth}")
    
    rf_run_name = f"Random_Forest_Depth_{depth}"    
    with mlflow.start_run(run_name = rf_run_name): 
        
        #Random Forest Model
        rf_model = RandomForestRegressor(max_depth=depth)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)

        #Logging for Random Forest
        mlflow.log_param("model", "Random Forest")
        mlflow.log_metric("rmse", rf_rmse)
        mlflow.sklearn.log_model(rf_model, f"random_forest_depth_{depth}")

# Searching runs
experiment_id = mlflow.get_experiment_by_name("Weather_Prediction_Model").experiment_id
runs_df = mlflow.search_runs(experiment_ids=[experiment_id])

# Creating 'experiment_results' as a copy of 'runs_df' for demonstration
experiment_results = runs_df[['run_id', 'start_time', 'end_time', 'params.model', 'params.max_depth', 'metrics.rmse']].copy()

print(experiment_results)

filtered_runs_df = runs_df[runs_df['metrics.rmse'] < 0.5]
print(filtered_runs_df[['run_id', 'params.model', 'params.max_depth', 'metrics.rmse']])

# Filter DataFrame for Decision Tree runs
dt_runs = runs_df[runs_df['params.model'] == 'Decision Tree']

# Plotting Decision Tree Model Performance
sns.lineplot(data=dt_runs, x='params.max_depth', y='metrics.rmse', marker='o')
plt.title('Decision Tree Model: RMSE vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('RMSE')
plt.show()
     