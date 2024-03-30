import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score, balanced_accuracy_score

# Read the CSV
df = pd.read_csv('soil_measures.csv')

# Check for missing values
df = df.dropna()

# Checking for unique crop types
print("Unique Crop Types: ", df['crop'].unique())

# Splitting the dataset
X = df.drop(columns=['crop'])  # Features
y = df['crop']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

features_dict = {}
feature_performance = {}
best_predictive_feature = {}

for feature in ["N", "P", "K", "ph"]:
    # Select the current feature for training and testing
    X_train_feature = X_train[[feature]]
    X_test_feature = X_test[[feature]]
    
    # Create the Logistic Regression model
    log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    
    # Fit the model on the training set for the current feature
    log_reg.fit(X_train_feature, y_train)
    
    # Predict on the test set using the trained model
    y_pred = log_reg.predict(X_test_feature)
    
    # Calculate F1 score (weighted) and balanced accuracy score
    f1 = f1_score(y_test, y_pred, average='weighted')
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    feature_performance[feature] = {
        'F1 Score': f1,
        'Balanced Accuracy Score': balanced_accuracy
    }
    
    # Print out the performance for the current feature
    print(f"F1-score for {feature}: {f1}")
    print(f"Balanced Accuracy Score for {feature}: {balanced_accuracy}")
    
    # Calculate and print accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy for feature '{feature}': {accuracy}")
    
    # Store the model, its predictions, and accuracy in the dictionary
    features_dict[feature] = {
        'model': log_reg,
        'predictions': y_pred,
        'accuracy': accuracy
    }

# Determine the best predictive feature based on F1 score
best_score = -1
best_feature = ""

for feature, metrics in feature_performance.items():
    if metrics['F1 Score'] > best_score:  # Ensure correct key and case
        best_score = metrics['F1 Score']
        best_feature = feature
        
best_predictive_feature[best_feature] = best_score

# Printing the best predictive feature and its F1 score
print("Best Predictive Feature and Score:", best_predictive_feature)
