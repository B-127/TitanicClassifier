# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:09:39 2020

@author: Tony Jesuthasan
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns


train_data=pd.read_csv(r"C:\Users\Asus\Machine Learning Data\Titanic\train.csv")
test_data=pd.read_csv(r"C:\Users\Asus\Machine Learning Data\Titanic\test.csv")

#Concatenating both the training and testing data using Pandas.
td=pd.concat([train_data,test_data], ignore_index=True, sort=False)

#Finding the number of cells which contain a null value
print(td.isnull().sum)

sns.heatmap(td.isnull(),cbar=False).set_title("Missing Values Heat Map")
#plt.show()

#Finding all the unique data in the concatenatied dataset
print(td.nunique())

#Finding the total number of family members
td['family']= td.Parch+td.SibSp

#if a person was alone, survival chances were much higher. 
td['is_alone']=td['family']==0

td['Fare_Category'] = pd.cut(td['Fare'], bins=[0,7.90,14.45,31.28,120], labels=['Low','Mid','High_Mid','High'])

td['Fare_Category'].hist()
#plt.show()

#Data Imputation - the practice of replacing missing data with some substituted values. 
#Since ‘Embarked’ only had two missing values and the largest number of commuters embarked from Southampton, the probability of boarding from Southampton is higher. So, we fill the missing values with Southampton.
td.Embarked.fillna(td.Embarked.mode()[0],inplace=True)

#As the column ‘Cabin’ had a lot of missing data. I decided to categorize all the missing data as a different class. I named it NA. I assigned all the missing values with this value.
td.Cabin=td.Cabin.fillna('NA')

#As the column ‘Cabin’ had a lot of missing data. I decided to categorize all the missing data as a different class. I named it NA. I assigned all the missing values with this value.
td['Salutation']= td.Name.apply(lambda name:name.split(',')[1].split('.')[0].strip())

#Then grouping the class with Sex and PClass
grp = td.groupby(['Sex', 'Pclass'])

#The median of the group was then substituted in the missing rows.
grp.Age.apply(lambda x: x.fillna(x.median()))
td.Age.fillna(td.Age.median, inplace = True)

#Converting the non-numeric sex coloumn into numeric values
td['Sex']=LabelEncoder().fit_transform(td['Sex'])

#It adds columns corresponding to all the possible values. So, if there could be three embarkment values — Q, C, S, the get_dummies method would create three different columns and assign values 0 or 1 depending on the embarking point.
pd.get_dummies(td.Embarked, prefix="Emb", drop_first = True)

#Removing coloumns not required for the prediction.
td.drop(['Pclass', 'Fare','Cabin', 'Fare_Category','Name','Salutation', 'Ticket','Embarked', 'SibSp', 'Parch', 'Age'], axis=1, inplace=True)

#Data to be predicted
X_to_be_predicted= td[td.Survived.isnull()]
X_to_be_predicted= X_to_be_predicted.drop(['Survived'],axis=1)

training_data=td
training_data=training_data.dropna()
feature_train= training_data['Survived']                  #For X_train feature matrix
label_train=training_data.drop(['Survived'],axis=1)       #For y_train labels

#Classification
#classifier=RandomForestClassifier(criterion="entropy", n_estimators=700, min_samples_split=10, min_samples_leaf=1, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
classifier=SVC(kernel='linear',C=0.025,random_state=101)
#80% of label_train and feature_train for training and the rest for testing
X_train, X_test, y_train, y_test = train_test_split(label_train, feature_train, test_size=0.2)


classifier.fit(X_train,y_train)
print("RF Accuracy: "+repr(round(classifier.score(X_test, y_test) * 100, 2)) + "%")

result_rf=cross_val_score(classifier,X_train,y_train,cv=10,scoring='accuracy')
print(result_rf)
print('The cross validated score for Random forest is:',round(result_rf.mean()*100,2))

y_predict=cross_val_predict(classifier,X_train,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_predict),annot=True,fmt='3.0f',cmap="summer")
plt.title("Confusion Matrix for RF",y=1.05, size=15)

result = classifier.predict(X_to_be_predicted)
print(result)

submission = pd.DataFrame({'PassengerId':X_to_be_predicted.PassengerId,'Survived':result})
submission.Survived = submission.Survived.astype(int)
print(submission.shape)
filename = 'Titanic Predictions.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)

