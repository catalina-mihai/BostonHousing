import matplotlib.pyplot as plt
import pandas as pd

#load the csv and store it in a variable
data_frame = pd.read_csv("BostonHousing.csv")

#get an overview of the data
print(data_frame.head())

#Summary Statistics
print(data_frame.describe())
print(data_frame.mean())
print(data_frame.median())
print(data_frame.std())
print(data_frame.isnull().sum())

Q1 = df['medv'].quantile(.25)
Q3 = df['medv'].quantile(.75)
IQR = Q3 - Q1
lower_bound=Q1 - 1.5*IQR
upper_bound=Q3 + 1.5*IQR
print(df[df['medv']>upper_bound])
print(df[df['medv']<lower_bound])

