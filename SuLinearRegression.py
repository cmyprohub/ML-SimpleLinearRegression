# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Splitting the data to training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#fitting the SL regression model to the training set (X_train and y_train)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test results (Y_pred)
y_pred = regressor.predict(X_test)

#visualizing the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

#visualizing the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test,regressor.predict(X_test), color = 'blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()
