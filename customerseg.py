import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data={
    'customer':[1,2,3,4,5,6,7,8,9,10],
    'annualIncome':[15,16,17,60,62,63,90,91,92,93],
    'spentammount':[39,81,6,77,40,5,88,20,10,96]
}
df=pd.DataFrame(data)
x=df[['annualIncome','spentammount']]
Kmeans=KMeans(n_clusters=3,random_state=42)
df['cluster']=Kmeans.fit_predict(x)
print(df)
plt.figure(figsize=(8,6))
plt.scatter(df['annualIncome'],df['spentammount'],
    c=df['cluster'],s=100)
plt.scatter(Kmeans.cluster_centers_[:,0],
    Kmeans.cluster_centers_[:,1],c='red',s=200,marker="X")
plt.legend()
plt.grid(True)
plt.show()