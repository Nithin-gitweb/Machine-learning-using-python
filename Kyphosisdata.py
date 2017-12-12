import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('kyphosis.csv')
df = pd.DataFrame(df)
print(df.head(),df.info())
sb.pairplot(df,hue='Kyphosis',palette='coolwarm')
plt.show()
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred = dtree.predict(X_test)
print(classification_report(y_test,pred))
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
pred2 = rf.predict(X_test)
print(classification_report(pred2,y_test))
