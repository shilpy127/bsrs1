import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import PowerTransformer
df=pd.read_csv(r"/content/Naive-Bayes-Classification-Data.csv")
df.head(5)
df.info()
df.describe()
sns.boxplot(df['glucose'])
sns.heatmap(df.corr(),annot=True)
from sklearn.preprocessing import PowerTransformer
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x
y
pt=PowerTransformer()
X=pt.fit_transform(x)
sns.distplot(X)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
nb.score(x_train,y_train)
y_pred=nb.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("Accuracy of GaussianNB:" , accuracy_score(y_test,y_pred)*100)
print("CM of GaussianNB:")
print(confusion_matrix(y_test,y_pred))
print("Report of GaussianNB:")
print( classification_report(y_test,y_pred))
t=nb.predict([[35,63]])
print(t)
