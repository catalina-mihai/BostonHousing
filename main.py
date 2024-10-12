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
