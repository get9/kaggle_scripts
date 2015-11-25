import sys

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


if len(sys.argv) < 2:
    print('Usage:')
    print('    {} [traincsv]'.format(sys.argv[0]))
    sys.exit(1)

# Read input
train_file_name = sys.argv[1]
df = pd.read_csv(train_file_name)

# Split into data/labels
X = df.filter(regex='pixel').astype(np.float).values
Y = df.label.values

# Normalize X values
print('Scaling training data between [0, 1]')
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Perform PCA for dimensionality reduction
print('Performing PCA')
pca = PCA(n_components=0.75)
X_reduce = pca.fit_transform(X)

# Then split into training/testing data
X_train, X_test, Y_train, Y_test = train_test_split(X_reduce, Y, test_size=0.20)

# Start up a KMeans classifier
clf = KMeans(n_clusters=15, n_init=100, max_iter=600, \
        precompute_distances=True, copy_x=False)
print('Fitting KMeans classifier')
clf.fit(X_train, Y_train)
Y_predict = clf.predict(X_test)
print('accuracy: ', np.count_nonzero(Y_predict - Y_test) / len(Y_test))
