import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

iris = sb.load_dataset('iris')
iris = pd.DataFrame(iris)
sb.pairplot(iris)
plt.show()
sb.jointplot(x='sepal_width',y='sepal_length',data=iris,kind='kde',color='red')
plt.show()
X = iris.drop('species',axis=1)
y = iris['species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
plot1 = SVC()
plot1.fit(X_train,y_train)
pred1 = plot1.predict(X_test)
print(classification_report(y_test,pred1))
dic = {'C':[0.1,1,10,100,1000],'gamma':[1000,100,10,1,0.1]}
plot2 = GridSearchCV(SVC(),dic,verbose=3)
plot2.fit(X_train,y_train)
pred2 = plot2.predict(X_test)
print(classification_report(y_test,pred2))
