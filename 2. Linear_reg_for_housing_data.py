import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

df = pd.read_csv('USA_Housing.csv')
df = pd.DataFrame(df)
sb.pairplot(df)
plt.show()
sb.jointplot(x = 'Avg. Area Income',y = 'Avg. Area Number of Bedrooms',data=df)
plt.show()
X = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms']]
y = df['Price']
X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.3,random_state=101)
lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
plt.scatter(y_test,pred)
plt.show()