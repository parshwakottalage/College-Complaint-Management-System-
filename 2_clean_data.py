import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/Datasetprojpowerbi.csv")

print("Original dataset:")
print(df.shape)

# Show missing values
print("\nMissing values:")
print(df.isnull().sum())

# Show duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset after removing duplicates:")
print(df.shape)

# Save cleaned dataset
df.to_csv("dataset/cleaned_complaints.csv", index=False)

print("\nCleaned dataset saved successfully!")