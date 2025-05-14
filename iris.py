#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

#Load dataset and convert to csv format
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    # Save to CSV
    df.to_csv("iris_dataset.csv", index=False)
    print("Dataset saved to 'iris_dataset.csv'")
except Exception as e:
    print(f"Error loading or saving Iris dataset: {e}")

#Load dataset using pandas
try:
    df = pd.read_csv("iris_dataset.csv")
    print("Dataset loaded successfully from CSV.\n")
except FileNotFoundError:
    print("CSV file not found.")
    exit()

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check data types and structure
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Clean dataset
df.dropna(inplace=True)

# Compute statistics
print("\nDescriptive statistics:")
print(df.describe())

# Group by species and compute the mean of each numerical column
print("\nMean values grouped by species:")
grouped = df.groupby('species').mean()
print(grouped)

# Identify any patterns or findings (in comments)
# Example: Setosa has smaller petal length and width compared to others.

sns.set(style="whitegrid")

# Line chart: Sepal length trend by index
plt.figure(figsize=(8, 5))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.plot(subset.index, subset['sepal length (cm)'], label=species)
plt.title("Sepal Length Trend by Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart: Average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x='species', y='petal length (cm)', data=df, ci=None)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram: Distribution of sepal width
plt.figure(figsize=(6, 4))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True)
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.tight_layout()
plt.show()

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(6, 4))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show() 
