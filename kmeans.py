import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

n_digits = 10

if len(sys.argv) < 2:
    print('Usage:')
    print('    {} [traincsv] [testcsv]'.format(sys.argv[0]))
    sys.exit(1)

# Read input
print('Reading input')
train_file_name, test_file_name = sys.argv[1:]
df_train = pd.read_csv(train_file_name)
df_test = pd.read_csv(test_file_name)

# Split into data/labels
X_train = df_train.filter(regex='pixel').astype(np.float).values
Y_train = df_train.label.values
X_test = df_test.filter(regex='pixel').astype(np.float).values

# Build classifier
n_clusters = 300
scaler = MinMaxScaler()
pca = PCA(n_components=0.7, whiten=False)
kmeans = KMeans(n_clusters=n_clusters,
                init='k-means++',
                n_init=50,
                precompute_distances='auto',
                n_jobs=-1,
                random_state=42,
                verbose=0,
                copy_x=True)
clf = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('kmeans', kmeans)])

print('Fitting input data')
clf.fit(X_train)

print('Determining cluster labels')
counts = np.zeros([n_clusters, n_digits])
labels = clf.named_steps['kmeans'].labels_
for p in np.dstack([labels, Y_train])[0]:
    counts[p[0], p[1]] += 1
cluster_labels = np.argmax(counts, axis=1)

print('Testing')
Y_predict = cluster_labels[clf.predict(X_test)]

# Save results
sample_counts = np.arange(1, len(Y_predict) + 1)
output = np.dstack([sample_counts, Y_predict])[0]
np.savetxt('digits_predict3.csv', output, fmt='%d,%d', header='ImageId,Label')
