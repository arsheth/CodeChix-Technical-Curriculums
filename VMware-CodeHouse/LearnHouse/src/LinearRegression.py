import pandas as pd
import os
import numpy as np
import math

data = pd.read_csv(os.path.join("/Users/arpita/Downloads","yelp_data.csv"))
df = data[1:10001]

#linear regression
# create X and y
#feature_cols = ['latitude','longitude','business_']
feature_cols = ['business_avg_stars','user_avg_stars','user_review_count', 'business_review_count']
X = df[feature_cols]
y = df.stars

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
print lm.intercept_
print lm.coef_

test_data = data[10002:10007]

X_test = test_data[feature_cols]
print lm.predict(X_test)
print test_data['stars']
