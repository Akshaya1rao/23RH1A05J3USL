from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
hashtags=[
"Morning workout #gym #fitness",
"Delicious chocolate cake recipe #foodie #dessert",
"Leg day at the gym! #fitness #workout",
"Travelling to the Mountains #travel #nature",
"Chocolate lava cake is my weakness #foodie #chocolate",
"Exploring a beautiful waterfall #nature # travel",
"Deadlifts and squats today #gym #fitness",
"Easy pasta recipe for dinner #foodie #cooking",
"Sunrise hike was worth it! #hiking #mountain #nature",
"Meal prep ideas for the fat loss #fitness #foodie #mealprep"]
#clean dataset and convert hashtag into matrix
vectorizer=TfidfVectorizer(stop_words='english')
x=vectorizer.fit_transform(hashtags)
print(x)
#food nature fitness
kmeans=KMeans(n_clusters=3,random_state=42)
kmeans.fit(x)
labels=kmeans.labels_
cluster_names_map={0:"nature",1:"foodie",2:"gym"}
df=pd.DataFrame({'caption':hashtags,'cluster':labels})
df['category']=df['cluster'].map(cluster_names_map)
print(df.sort_values('cluster'))

#plotting
#visival using PCA
x_array=x.toarray()
pca=PCA()
x_reduced=pca.fit_transform(x_array)
#plotting
plt.figure(figsize=(8,5))
scatter=plt.scatter(x_reduced[:,0],x_reduced[:,1],c=labels,s=100)
for i,row in df.iterrows():
    plt.annotate(row['category'],(x_reduced[i,0]+00.5,x_reduced[i,1]+0.05),fontsize=8)
plt.show()