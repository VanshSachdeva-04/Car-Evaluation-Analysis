import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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