# RandomForestClass.py
# This program implements a version of Random Forest Classification on machinery data of a mill for predictions of
# machine failure and no machine failure. It separates the dataset into testing data and training data, then tunes
# the training data for hyperparameters, this causes the code to take a long time, and then fits the data on Random
# Forest model. Then the test data is input into the trained model to make predictions. Then the prediction results
# are compared to the actual results for a classification report.
#

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# Load the dataset
data = pd.read_csv('ai4i2020-edit-1.csv')

# Split the dataset into features (X) and labels (y)
X = data.drop('machine_failure', axis=1)
y = data['machine_failure']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV for hyperparameter tuning with a focus on accuracy
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print('Best Hyperparameters:', best_params)

# Train the model with the best hyperparameters on the entire training set
best_random_forest = grid_search.best_estimator_
best_random_forest.fit(X_train, y_train)

# Make predictions on the test data
y_pred = best_random_forest.predict(X_test)


# Generate a classification report
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)

# Generate ROC-AUC curve
y_prob = best_random_forest.predict_proba(X_test)[:, 1]  # Probability estimates of the positive class
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC-AUC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
