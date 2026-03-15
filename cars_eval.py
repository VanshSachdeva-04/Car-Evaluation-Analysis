import pandas as pd
import numpy as np

#1. Load the dataset and check for any missing values
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv('car.data', names=columns)

print("Missing values per column:")
print(df.isnull().sum())

#2. Ordinal Encoding for categorical features (e.g., mapping safety 'low' < 'med' < 'high')
mappings = {
    'buying': {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4},
    'maint': {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4},
    'doors': {'2': 2, '3': 3, '4': 4, '5more': 5},
    'persons': {'2': 2, '4': 4, 'more': 6},
    'lug_boot': {'small': 1, 'med': 2, 'big': 3},
    'safety': {'low': 1, 'med': 2, 'high': 3},
    'class': {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
}

#Apply the mappings to the dataframe
for column, mapping in mappings.items():
    df[column] = df[column].map(mapping)

#3. Verify the encoding
print("\n First 5 rows of the encoded dataset:")
print(df.head())

#Save the preprocessed dataset to a new CSV file
df.to_csv('car_encoded.csv', index=False)