import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset and check for any missing values (week 1)
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv('car.data', names=columns)

print("Missing values per column:")
print(df.isnull().sum())

# Ordinal Encoding for categorical features (e.g., mapping safety 'low' < 'med' < 'high')
mappings = {
    'buying': {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4},
    'maint': {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4},
    'doors': {'2': 2, '3': 3, '4': 4, '5more': 5},
    'persons': {'2': 2, '4': 4, 'more': 6},
    'lug_boot': {'small': 1, 'med': 2, 'big': 3},
    'safety': {'low': 1, 'med': 2, 'high': 3},
    'class': {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
}

#Apply the mappings to the dataframe (week 1)
for column, mapping in mappings.items():
    df[column] = df[column].map(mapping)

# Model Training(week 2)
# Split the dataset into features and target variable
X = df.drop('class', axis=1)
y = df['class']

# Use stratify=y to handle the 70% 'unacc' class imbalance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning using GridSearchCV
# 'gini': measures impurity by calculating the probability of a random car being misclassified. It is computationally faster and focuses on making the groups pure.
# 'entropy': measures disorder or uncertainty in the data. It uses information gain to ask: "Which attribute (like safety) gives us the most info about the car's class?"
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'criterion': ['gini', 'entropy']
}

print("Searching for the best model settings...")
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train, y_train)

# Output results
best_rf = grid_search.best_estimator_
print(f"Best Parameters Found: {grid_search.best_params_}")

# Final performance report
y_pred = best_rf.predict(X_test)
print("\n" + "="*60)
print("Final Performance Report".center(60))
print("="*60)

target_names = ['unacc', 'acc', 'good', 'vgood']
print(classification_report(y_test, y_pred, target_names=target_names))
print("="*60)

#Save the preprocessed dataset to a new CSV file
df.to_csv('car_encoded.csv', index=False)

# --- Visual Chart 1: Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix: Predicted vs Actual Car Acceptability')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.show()

# --- Week 3: Extract feature importance scores and conduct "sensitivity analysis" ---

print("\n" + "="*60)
print("Week 3: Feature Importance Scores".center(60))
print("="*60)
# Extract and sort feature importances from the best Random Forest model
importances = best_rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df.to_string(index=False))

# --- Visual Chart 2: Feature Importance Bar Chart ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('What drives the decision?(Feature Importance)')
plt.xlabel('Importance Score')
plt.ylabel('Car Attributes')
plt.show()

print("\n" + "="*60)
print("Week 3: Sensitivity Analysis (Edge Cases)".center(60))
print("="*60)

# Define edge cases based on "deal-breakers" mentioned in the project logic
# Feature order: ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
edge_cases = pd.DataFrame([
    # Case 1: The "Perfect" car (Low price/maint, max doors/persons/boot/safety)
    [1, 1, 5, 6, 3, 3], 
    # Case 2: The "Safety Deal-breaker" (Perfect car, but LOW safety)
    [1, 1, 5, 6, 3, 1], 
    # Case 3: The "Cost Deal-breaker" (Very high price/maint, but max safety and others)
    [4, 4, 5, 6, 3, 3],
    # Case 4: The "Space Deal-breaker" (Perfect car, but holds only 2 persons)
    [1, 1, 5, 2, 3, 3]
], columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

case_descriptions = [
    "1. Perfect Car (Cheap, Spacious, High Safety)",
    "2. Safety Deal-breaker (Perfect, but LOW Safety)",
    "3. Cost Deal-breaker (Perfect, but VERY HIGH Cost)",
    "4. Space Deal-breaker (Perfect, but ONLY 2 Persons)"
]

# Predict classes for edge cases
predictions = best_rf.predict(edge_cases)

# Map numeric predictions back to class labels for readability
inverse_class_mapping = {0: 'unacc', 1: 'acc', 2: 'good', 3: 'vgood'}
predicted_labels = [inverse_class_mapping[p] for p in predictions]

for desc, label in zip(case_descriptions, predicted_labels):
    print(f"{desc}")
    print(f"   -> Predicted Class: {label}\n")

print("="*60)