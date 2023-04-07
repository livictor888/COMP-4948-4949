import pandas as pd

# Read the dataset
df = pd.read_csv('dataset.csv', delimiter=';')

print(df.head())