import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler
df=pd.read_csv('BostonHousing.csv')
df.head()
df.describe()
df.isnull().sum()
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.show()
df.dtypes

df['chas'].unique()
df['rad'].unique()

(df[['zn','rm','age']]<0).sum()

df.duplicated().sum()

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
corr_matrix = filtered_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm')
plt.show()

plt.scatter(filtered_df['medv'],filtered_df['crim'])
plt.xlabel('medv')
plt.ylabel('crim')
plt.show()

plt.scatter(filtered_df['medv'],filtered_df['rm'])
plt.xlabel('medv')
plt.ylabel('rm')
plt.show()
plt.scatter(filtered_df['medv'],filtered_df['lstat'])
plt.xlabel('medv')
plt.ylabel('lstat')
plt.show()

plt.scatter(filtered_df['medv'],filtered_df['chas'])
plt.xlabel('medv')
plt.ylabel('chas')
plt.show()

#print(filtered_df['chas'].value_counts())
filtered_df['chas'].unique()

df['medv_log']=np.log(df['medv']+1)
df['medv_log'].head()
sns.boxplot(df['medv_log'])
plt.show()
df['lstat_log']= np.log(df['lstat']+1)
df['lstat_log'].head()
sns.boxplot(df['lstat_log'])
plt.show()
df['crim_log']=np.log(df['crim']+1)
df['crim_log'].head()
sns.boxplot(df['crim_log'])
plt.show()
plt.scatter(df['rm'],df['medv_log'])
plt.xlabel('rm')
plt.ylabel('medv_log')

plt.show()
plt.scatter(df['lstat'],df['medv_log'])
plt.xlabel('lstat_log')
plt.ylabel('medv_log')
plt.show()
plt.scatter(df['crim_log'],df['medv_log'])
plt.xlabel('crim_log')
plt.ylabel('medv_log')
plt.show()
df_after_log= df.drop(['crim','lstat','medv'],axis=1)
corr_matrix_log= df_after_log.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix_log,annot=True,cmap='coolwarm')
plt.show()
# Set the correlation threshold
threshold = 0.5

# Get the correlation values with 'medv_log'
corr_with_target = corr_matrix_log['medv_log'].abs()

# Select features that have correlation > threshold with 'medv_log'
relevant_features = corr_with_target[corr_with_target > threshold].index
relevant_features=relevant_features.drop('medv_log')
print("Selected Features: ", relevant_features)
X=df_after_log.drop('medv_log',axis=1)
Y=df_after_log['medv_log']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled =scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled,Y_train)
y_predict = model.predict(X_test_scaled)
mse_linearReg= mean_squared_error(Y_test,y_predict)
r2_linearReg= r2_score(Y_test,y_predict)
print("Mean Squared Error:",mse_linearReg)
print("R2- R Squared:",r2_linearReg)
plt.scatter(Y_test,y_predict, color='blue', label='Predicted Vs Actual')
# Adding a line that represents a perfect prediction (y = x line)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', lw=2, label='Perfect Fit')

# Labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.legend()
plt.grid()
plt.show()

#------ Ridge Regression Model--------#

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled,Y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
mse_ridge = mean_squared_error(Y_test,y_pred_ridge)
r2_ridge = r2_score(Y_test,y_pred_ridge)
print(f"Ridge Regression MSE: {mse_ridge}")
print(f"Ridge Regression R-squared: {r2_ridge}")

plt.scatter(Y_test,y_pred_ridge,color='blue', label='Predicted Vs Actual')
# Adding a line that represents a perfect prediction (y = x line)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', lw=2, label='Perfect Fit')

# Labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.legend()
plt.grid()
plt.show()

#--------- Lasso Regression -------------#

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled,Y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)
mse_lasso = mean_squared_error(Y_test,y_pred_lasso)
r2_lasso = r2_score(Y_test,y_pred_lasso)
print(f"Lasso Regression MSE : {mse_lasso}")
print(f"Lasso Regression R-Squared : {r2_lasso}")

plt.scatter(Y_test,y_pred_lasso,color='blue', label='Predicted Vs Actual')
# Adding a line that represents a perfect prediction (y = x line)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', lw=2, label='Perfect Fit')

# Labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
#plt.legend()
plt.grid()
plt.show()

#--------------- DecisiontreeRegressor ------------------#
dec_tree_model = DecisionTreeRegressor(random_state=42)
dec_tree_model.fit(X_train_scaled,Y_train)
y_pred_decTree = dec_tree_model.predict(X_test_scaled)
mse_decTree = mean_squared_error(Y_test,y_pred_decTree)
r2_decTree = r2_score(Y_test,y_pred_decTree)
print(f"Decision Tree Regressor - MSE : {mse_decTree}")
print(f"Decision Tree Regressor - R-Squared : {r2_decTree}")

plt.scatter(Y_test,y_pred_decTree,color='blue', label='Predicted Vs Actual')
# Adding a line that represents a perfect prediction (y = x line)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', lw=2, label='Perfect Fit')

# Labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
#plt.legend()
plt.grid()
plt.show()

#---------- RandomForestRegressor -----------------#

rf_model= RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled,Y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
mse_rf = mean_squared_error(Y_test,y_pred_rf)
r2_rf = r2_score(Y_test,y_pred_rf)
print(f"RandomForestRegressor - MSE : {mse_rf}")
print(f"RandomForestRegressor - R-Squared :{r2_rf} ")

plt.scatter(Y_test,y_pred_rf,color='blue', label='Predicted Vs Actual')
# Labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
#plt.legend()
plt.grid()
plt.show()

#---------------Comparison of MSE -----------#

mse_values = {
    'LinearRegression': mse_linearReg,
    'Ridge Regression': mse_ridge,
    'Lasso Regression': mse_lasso,
    'DecisionTreeRegressor': mse_decTree,
    'RandomForestRegressor': mse_rf
}

model_names_mse= list(mse_values.keys())
model_values_mse = list(mse_values.values())


plt.figure(figsize=(10,6))
plt.bar(model_names_mse,model_values_mse,color=['blue','green','red','purple','orange'])
plt.xlabel('Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE Comparison for Different Models')
plt.xticks(rotation=45)  # Rotate the model names if needed
plt.tight_layout()

# Show the plot
plt.show()

#---------------Comparison of R-Squared -----------#

r2_values = {
    'LinearRegression': r2_linearReg,
    'Ridge Regression': r2_ridge,
    'Lasso Regression': r2_lasso,
    'DecisionTreeRegressor': r2_decTree,
    'RandomForestRegressor': r2_rf
}

model_names_r2= list(r2_values.keys())
model_values_r2 = list(r2_values.values())


plt.figure(figsize=(10,6))
plt.bar(model_names_r2,model_values_r2,color=['blue','green','red','purple','orange'])
plt.xlabel('Models')
plt.ylabel('R-Squared')
plt.title('R-Squared Comparison for Different Models')
#plt.xticks(rotation=45)  # Rotate the model names if needed
plt.tight_layout()

# Show the plot
plt.show()

