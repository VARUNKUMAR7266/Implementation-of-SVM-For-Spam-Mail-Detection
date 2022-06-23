# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the necessary packages using import statement.
2. Read the given csv file and print the number of contents to be displayed. 
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy. 5.Display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VARUNKUMAR K
RegisterNumber:  212219040173
*/
```
import pandas as pd

data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["EmailText"].values

y=data["Label"].values

from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()

x_train=cv.fit_transform(x_train)

x_test=cv.transform(x_test)

from sklearn.svm import SVC

svc=SVC()

svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)

y_pred

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy

## Output:
Data.head():

![image](https://user-images.githubusercontent.com/88264052/175312275-34361067-2710-4901-b663-e497d7dc4743.png)

Data.info():

![image](https://user-images.githubusercontent.com/88264052/175312350-11054d7d-9d7d-4a13-86fe-2d6e417d4f21.png)

Data.isnull().sum():

![image](https://user-images.githubusercontent.com/88264052/175312422-c84a3fd8-49f9-4276-9c09-76ba43867627.png)

Y_Pred:

![image](https://user-images.githubusercontent.com/88264052/175312490-c9dd1c38-a575-4a68-8d71-f255831edcde.png)

Accuracy:

![image](https://user-images.githubusercontent.com/88264052/175312554-e2c2fb57-4b76-4c53-b3a5-d4db7861ba61.png)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
