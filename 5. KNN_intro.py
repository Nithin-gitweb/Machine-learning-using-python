import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('KNN_Project.csv')
df = pd.DataFrame(df)
print(df.head())
sb.pairplot(df,hue='TARGET CLASS')
plt.show()
scalar = StandardScaler()
scalar.fit(df.drop('TARGET CLASS',axis=1))
scalar = scalar.transform(df.drop('TARGET CLASS',axis=1))
scalar = pd.DataFrame(scalar,columns=df.columns[:-1])
print(scalar.head())
X = scalar
y = df['TARGET CLASS']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
arr = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)
    arr.append(np.mean(pred != y_test))
plt.figure(figsize=(12,5))
plt.plot(range(1,40),arr,color = 'red',linestyle = 'dashed',marker = 'o',markerfacecolor = 'blue')
plt.show()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(classification_report(y_test,pred))








