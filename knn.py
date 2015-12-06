import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

if len(sys.argv) < 2:
    print('Usage:')
    print('    {} [traincsv] [testcsv]'.format(sys.argv[0]))
    sys.exit(1)

# Read input
train_file_name, test_file_name = sys.argv[1:]
df_train = pd.read_csv(train_file_name)
df_test = pd.read_csv(test_file_name)

# Split into data/labels
X_train = df_train.filter(regex='pixel').astype(np.float).values
Y_train = df_train.label.values
X_test = df_test.filter(regex='pixel').astype(np.float).values

# Build classifier
scaler = MinMaxScaler()
pca = PCA(n_components=0.80)
knn = KNeighborsClassifier(n_neighbors=10,
                           weights='distance',
                           algorithm='auto',
                           n_jobs=-1)
clf = Pipeline(steps=[
    ('unit_scale', scaler),
    ('pca', pca),
    ('knn', knn)
])

# Start up a KMeans classifier
print('Fitting classifier')
clf.fit(X_train, Y_train)
Y_predict = clf.predict(X_test)

# Save results
sample_counts = np.arange(1, len(Y_predict) + 1)
output = np.dstack([sample_counts, Y_predict])[0]
np.savetxt('digits_predict2.csv', output, fmt='%d,%d', header='ImageId,Label')
