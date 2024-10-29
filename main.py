import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

for feature in df.columns:
  sns.boxplot(df[feature])
  plt.title(f"Box Plot of {feature}")
  plt.show()
  
 def remove_outlier(df,feature):
  Q3= df[feature].quantile(0.75)
  Q1= df[feature].quantile(0.75)
  IQR= Q3-Q1
  lower_bound = Q1- (IQR* 1.5)
  upper_bound =Q3 + (IQR* 1.5)
  #now filtering the points which are not outliers
  filtered_df=df[(df[feature]>=lower_bound) & (df[feature]<=upper_bound)]
  return filtered_df


outlier_features =['crim','zn','rm','dis','ptratio','lstat','medv']
for feature in outlier_features:
  filtered_df= remove_outlier(df,feature)

for feature in outlier_features:
  sns.boxplot(filtered_df[feature])
  plt.title(f"Box Plot of {feature}")
  plt.show()

df['medv_log']=np.log(df['medv']+1)
sns.boxplot(df['medv_log'])
plt.show()

df['lstat_log']= np.log(df['lstat']+1)
sns.boxplot(df['lstat_log'])
plt.show()

df['crim_log']=np.log(df['crim']+1)
sns.boxplot(df['crim_log'])
plt.show()
