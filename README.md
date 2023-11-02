# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas to read the csv files.
2. Display the head and tail of the dataset.
3. Import LabelEncoder() from sklearn.preprocessing.
4. Label the data which are not in the integer type.
5. Assign the X and Y from the dataset.
6. Split the dataset using train_test_split from sklearn.model_selection.
7. Import DecisionTreeClassifier from sklearn.tree
8. Fit the Training set in a variable.
9. Import metrics from sklearn to find the accuracy.
10. Predict the result for the given values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Thirugnanamoorthi G
RegisterNumber: 212221230117
*/
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation",
"number_project","average_montly_hours","time_spend_company",
"Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### 1. data.head()
![1s](https://github.com/souvik798/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94752764/babc371b-75c8-4bc2-9501-b767ae2dc431)

### 2. data.info()

![2s](https://github.com/souvik798/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94752764/b875dc3d-db60-44b7-bfb3-b6c9c019f900)

### 3. isnull() and sum()
![3s](https://github.com/souvik798/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94752764/f959b1b1-c5b6-488b-8744-b6cc8a2391de)


### 4. data value counts()
![4s](https://github.com/souvik798/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94752764/80401c3b-b5c7-4961-919c-f2a5724947de)


### 5. data.head() for salary
![5s](https://github.com/souvik798/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94752764/2f76aeab-a242-4c74-9b90-3e1090266c05)


### 6. x.head()
![6s](https://github.com/souvik798/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94752764/716cd0bd-1482-4e3f-9222-a245eadd9918)


### 7. accuracy value
![7s](https://github.com/souvik798/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94752764/415e3fc9-d74b-42cf-96a8-e0fe36333877)


### 8. data prediction
![8s](https://github.com/souvik798/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94752764/455baab0-865d-424c-9e47-5df051857bc9)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
