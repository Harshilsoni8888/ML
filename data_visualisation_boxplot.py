
# Load Pandas Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Creating DataFrame and Loading csv file
df = pd.read_csv('/content/sample_data/Iris.csv')
df

# Data Visualization - Histogram
#1. Histogram for Iris Dataset

x = df["SepalLengthCm"]
plt.hist(x, bins = 20, color = "blue")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")

# Data Visualization - Boxplot
#2. Boxplot for Iris Dataset

df1 = df[["SepalLengthCm",'SepalWidthCm',	'PetalLengthCm',	'PetalWidthCm']]
print(df1.describe())
df1.boxplot()

#IQR = Q3-Q1
#Outliers are the data points below and above the lower and upper limit.
# The lower and upper limit is calculated as – 

#Lower Limit = Q1 - 1.5*IQR
#Upper Limit = Q3 + 1.5*IQR