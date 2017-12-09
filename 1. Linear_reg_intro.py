import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

var = pd.read_csv('Ecommerce.csv')
df = pd.DataFrame(var)
print(df.head(5),df.info(),df.describe())
sb.jointplot(x = 'Time on Website',y='Yearly Amount Spent',data=df)
plt.show()
sb.jointplot(x = 'Time on App',y='Yearly Amount Spent',data=df)
plt.show()
sb.jointplot(x = 'Time on App',y='Length of Membership',data=df, kind='hex')
plt.show()
sb.pairplot(df)
plt.show()
sb.lmplot(x ='Length of Membership',y= 'Yearly Amount Spent',data=df)
plt.show()
X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = df['Yearly Amount Spent']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.coef_)
predict = lm.predict(X_test)
plt.scatter(y_test,predict)
plt.show()
print(metrics.mean_absolute_error(y_test,predict))
print(metrics.mean_squared_error(y_test,predict))
print(np.sqrt(metrics.mean_squared_error(y_test,predict)))
