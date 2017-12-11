import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import random
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv('advertising.csv')
df = pd.DataFrame(df)
print(df.head(5),df.info(),df.describe())
sb.distplot(df['Age'],bins=30,kde=False)
plt.show()
sb.jointplot(x = 'Age',y='Area Income',data=df)
plt.show()
sb.jointplot(x = 'Age',y='Daily Time Spent on Site',data=df,kind='kde',color='red')
plt.show()
sb.jointplot(x = 'Daily Internet Usage',y='Daily Time Spent on Site',data=df,kind='hex')
plt.show()
sb.pairplot(df,hue='Clicked on Ad')
plt.show()
df.drop(['Ad Topic Line','City','Country','Timestamp'],axis = 1,inplace=True)
X = df[['Age','Area Income','Daily Internet Usage','Daily Time Spent on Site','Clicked on Ad']]
y = df['Male']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
log = LogisticRegression()
log.fit(X_train,y_train)
pre = log.predict(X_test)
print(classification_report(y_test,pre))

