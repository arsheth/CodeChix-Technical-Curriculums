import pandas as pd
import os
import numpy as np
import math

data = pd.read_csv(os.path.join("/Users/arpita/Downloads","yelp_data.csv"))
df = data[1:10001]

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist,pdist
feature_cols = ['latitude','longitude','business_avg_stars']
cluster_data = df[feature_cols]

vals = cluster_data.values
krange = range(1,15)
kmeans = [KMeans(n_clusters = k).fit(vals) for k in krange]
centroids = [k.cluster_centers_ for k in kmeans]
euclid_dist = [cdist(vals,centroid,'euclidean') for centroid in centroids]
distance = [np.min(ke,axis=1) for ke in euclid_dist]

#sum of squares within cluster
wcs = [sum(d**2) for d in distance]
#total sum of squares
tcs = sum(pdist(vals)**2)/vals.shape[0]

#sum of squares between clusters
bss = (tcs - wcs).tolist()

import matplotlib.pyplot as plt
plt.plot(krange, bss , color='blue')
plt.xticks((krange))
plt.yticks((bss))
plt.xlabel('K')
plt.ylabel('Variance')
plt.show()

k=3
kmeans = KMeans(n_clusters=k)
kmeans.fit(vals)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

for i in range(k):
	data = vals[np.where(labels==i)]
	plt.plot(data[:,0],data[:,1],'o')
	lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
plt.show()
