# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the required packages.
Read the data set.
Apply label encoder to the non-numerical column inoreder to convert into numerical values.
Determine training and test data set.
Apply decision tree regression on to the dataframe and get the values of Mean square error, r2 and data prediction.

## Program:
~~~
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SYED MUHAMMED ZAHI
RegisterNumber:  212221230114
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]]) 
~~~



## Output:
![image](https://user-images.githubusercontent.com/94187572/204819177-dae88ec1-db79-4fc8-aae5-da08323902d0.png)
![image](https://user-images.githubusercontent.com/94187572/204819242-770cfc1d-c4b1-486f-9c0c-4da44a513f2f.png)
![image](https://user-images.githubusercontent.com/94187572/204819376-60b26af9-dc51-4479-bb0f-4394525d7235.png)
![image](https://user-images.githubusercontent.com/94187572/204819571-123ebb56-37b4-4c03-8ec4-41fc670500d9.png)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
