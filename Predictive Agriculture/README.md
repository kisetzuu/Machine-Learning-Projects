# Crop Recommendation System

## Overview
This project combines the power of machine learning with agricultural science to assist in making data-driven decisions about crop selection. By analyzing key soil metrics—Nitrogen (N), Phosphorous (P), Potassium (K) levels, and pH value—I aim to maximize agricultural yield through tailored crop recommendations. This solution addresses the challenge of expensive soil testing, providing an efficient and scalable approach for farmers worldwide.

## Dataset
The foundation of this analysis is the `soil_measures.csv` dataset, which includes the following features for each field:
- `N`: Nitrogen content ratio, essential for plant growth.
- `P`: Phosphorous content ratio, vital for root development.
- `K`: Potassium content ratio, crucial for water regulation.
- `pH`: Soil pH level, affecting nutrient solubility and availability.
- `crop`: The optimal crop for the given soil conditions, serving as the target variable for our model.

## Objective
The project is guided by two main objectives:
1. Develop a predictive model to recommend the best crop based on soil conditions.
2. Identify the most impactful soil metric on the crop recommendation model's predictive accuracy.

## Methodology
The approach taken in this project is methodical and iterative:
1. **Data Preprocessing**: I begin by cleaning the data, addressing missing values, and preparing categorical variables for analysis.
2. **Exploratory Data Analysis (EDA)**: An in-depth EDA provides insights into the relationships between soil metrics and crop suitability.
3. **Model Building**: Various multi-class classification algorithms are tested, including Random Forest, Gradient Boosting, and SVM, to identify the model with the highest accuracy and F1 score.
4. **Feature Importance Analysis**: Leveraging the best-performing model, I analyze the importance of each soil metric to understand their impact on crop selection.
5. **Evaluation**: The model's effectiveness is gauged using a combination of accuracy, precision, recall, and F1 scores.

## Tools and Technologies
- **Python**: The primary language for development and analysis.
- **Pandas & NumPy**: For efficient data manipulation and numerical computation.
- **Scikit-learn**: To apply and evaluate machine learning models.
- **Matplotlib & Seaborn**: For comprehensive data visualization.
- **Jupyter Notebook**: For documenting the project workflow and analysis.

## How to Run the Project
1. Install Python on your machine.
2. Clone the repository: `git clone <repository-url>`.
3. Navigate to the project directory and install required dependencies: `pip install -r requirements.txt`.
4. Launch Jupyter Notebook: `jupyter notebook crop_recommendation.ipynb`, and execute the cells in sequence.

## Results and Discussion
This section provides a detailed analysis of the model's performance, including key metrics such as accuracy, precision, recall, and the F1 score. The significance of each soil metric in determining the optimal crop is also discussed, offering valuable insights for effective soil management.

## Conclusion
This independent project highlights the potential of integrating machine learning with agricultural practices to enhance decision-making and productivity. Future work may explore incorporating additional environmental factors, such as climate conditions, to further refine crop recommendations.

