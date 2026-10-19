import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Datasetprojpowerbi.csv")

print("Dataset loaded successfully!")

print("\nNumber of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())