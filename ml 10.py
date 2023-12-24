import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import plotly.express as px
import seaborn as sns
df = pd.read_csv("/content/Mall_Customers.csv")
df.head()
df.columns = ['customer_ID','gender','age','annual_income','spending_score']
df.head()
df.shape
df.duplicated().any()
df.isnull().any()
df = df.set_index(['customer_ID'])
df.head()
X = df.drop(['gender'], axis=1)
X.head()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
clusters = []
ss = []
#Calculate all the sum of within-cluster variance for n_clusters from 2 to 14
for i in range(2,15):
    km = KMeans(n_clusters = i)
    km.fit(X)
    clusters.append(km.inertia_)
    ss.append(silhouette_score(X, km.labels_, metric='euclidean'))
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(2, 15)), y=clusters, ax=ax)
ax.set_title('Searching for Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')
# Annotate arrow
ax.annotate('Possible Elbow Point', xy=(5, 80000), xytext=(5, 150000), xycoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))
plt.show()
km5 = KMeans(n_clusters=5).fit(X)
X['Labels'] = km5.labels_
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(121)
sns.scatterplot(x=X['annual_income'], y=X['spending_score'], hue=X['Labels'], palette='viridis')
ax.set_title('KMeans with 5 Clusters')
ax.legend(loc='center right')
plt.show()
from sklearn.cluster import AgglomerativeClustering
agglo = AgglomerativeClustering(n_clusters=5, linkage='complete').fit(X)
labels = agglo.labels_
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(121)
sns.scatterplot(x=X['annual_income'],y=X['spending_score'], hue=labels, palette='viridis')
ax.set_title('Agglomerative 5 clusters with complete linkage')
ax.legend(loc='center right')
plt.show()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dbscan = DBSCAN(eps = 0.7)
clusters = dbscan.fit_predict(X_scaled)
length = len(np.unique(clusters))
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(121)
sns.scatterplot(x=X['annual_income'], y=X['spending_score'], hue=clusters, palette='viridis')
ax.set_title('DBSCAN with 5 Clusters')
ax.legend(loc='center right')
plt.show()
