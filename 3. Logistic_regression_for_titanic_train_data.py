#Titanic Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import random
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv('titanic_train.csv')
df = pd.DataFrame(df)
sb.countplot(x = 'Survived', hue='Sex',data=df)
plt.show()
sb.countplot(x = 'Survived',hue='Pclass',data = df)
plt.show()
sb.distplot(df['Pclass'],bins=50,kde=False)
plt.show()
print(df['Cabin'].isnull())

#Cleaning the data
df.drop('PassengerId',axis=1,inplace=True)
df.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
print(df.head(5))
def fill_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            a = random.randrange(35,41)
            return a
        elif Pclass == 2:
            b = random.randrange(25,33)
            return b
        else:
            c = random.randrange(17,23)
            return c
    else:
        return Age

df['Age'] = df[['Age','Pclass']].apply(fill_age,axis=1)
df.dropna(inplace=True)
print(df.isnull)
Sex = pd.get_dummies(df['Sex'],drop_first=True)
Embark = pd.get_dummies(df['Embarked'],drop_first=True)
Class = pd.get_dummies(df['Pclass'],drop_first=True)
df.drop(['Sex','Embarked','Pclass'],axis=1,inplace=True)
df = pd.concat([df,Sex,Embark,Class],axis=1)
print(df.head())
log = LogisticRegression()
X = df.drop('Survived',axis = 1)
y = df['Survived']
X_train,X_test,y_train,y_test = train_test_split(X,y)
log.fit(X_train,y_train)
predict = log.predict(X_test)
print(classification_report(y_test,predict))
print(confusion_matrix(y_test,predict))

