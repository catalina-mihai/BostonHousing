import matplotlib.pyplot as plt
import pandas as pd

#load the csv and store it in a variable
#df - data frame
df = pd.read_csv("BostonHousing.csv")
print("")
print('Here are the first 5 rows of our data')
print("")
#get an overview of the data
print(df.head())

#Summary Statistics
print("")
print('Here we are describing the data')
print("")
print(df.describe())
print("")
print('These are the means of our features')
print("")
print(df.mean())
print("")
print('These are the medians of our features')
print("")
print(df.median())
print("")
print('These are the standard deviation of our features')
print("")
print(df.std())
print("")
print('There are these missing values in each column')
print("")
print(df.isnull().sum())
#df.boxplot()
#plt.show()
print("")
print('There is the histogram for MEDV')
print("")
df['medv'].hist(bins=30)
plt.xlabel('Median_Value_of_Homes')
plt.ylabel('Frequency')
plt.show()