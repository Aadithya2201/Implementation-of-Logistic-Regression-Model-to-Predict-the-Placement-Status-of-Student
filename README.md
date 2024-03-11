# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: AADITHYA.R
RegisterNumber:212223240001
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![4,1](https://github.com/Aadithya2201/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145917810/0665b469-1e69-4421-a1e8-dba572d58b2b)
![4,2](https://github.com/Aadithya2201/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145917810/fde9e2ab-5be2-46f0-b0e8-0006335b9e4d)
![4,3](https://github.com/Aadithya2201/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145917810/2357d6a7-6f13-4888-9c7e-9c16ca4c7722)
![4,4](https://github.com/Aadithya2201/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145917810/206d1849-2c2e-44f1-8903-596754f64852)
![4,5](https://github.com/Aadithya2201/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145917810/c66abcec-9b6c-4e75-9739-66bcde8e6638)
![4,6](https://github.com/Aadithya2201/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145917810/e7971780-d90c-4d7b-827f-f16e41ff33aa)
![4,7](https://github.com/Aadithya2201/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145917810/405db13e-4903-404e-a524-f8ada78964c0)
![4,8](https://github.com/Aadithya2201/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145917810/377de725-52d1-4491-acf7-8f6d2fd5ca85)![4,11](https://github.com/Aadithya2201/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145917810/b367cc87-8571-4cb8-9c94-59fc14b980a3)
![4,9](https://github.com/Aadithya2201/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145917810/2da54a70-9e02-416f-8f29-e1c6542b911c)
![4,10](https://github.com/Aadithya2201/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145917810/2c80459f-46d3-46e9-b8ed-94c070b7da2b)
![4,11](https://github.com/Aadithya2201/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145917810/cbd1ba05-6acd-484c-a7a9-025912df0384)
![4,12](https://github.com/Aadithya2201/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145917810/b28564ba-31f9-455a-8d91-3969a3dd1502)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
