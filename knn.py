import sys

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.neighbors import KNeighborsClassifier


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

# Normalize X values
print('Scaling training data between [0, 1]')
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Then split into training/testing data
#X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.10, random_state=42)

# Perform PCA for dimensionality reduction
print('Performing PCA')
pca = PCA(n_components=0.80).fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# Start up a KMeans classifier
clf = KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='auto', n_jobs=-1)
print('Fitting KNN classifier')
clf.fit(X_train, Y_train)
Y_predict = clf.predict(X_test)
#print('accuracy: ', 1 - np.count_nonzero(Y_predict - Y_test) / len(Y_test))

# Save results
sample_counts = np.arange(1, len(Y_predict)+1)
output = np.dstack([sample_counts, Y_predict]).reshape(len(sample_counts), 2)
np.savetxt('digits_predict2.csv', output, fmt='%d,%d', header='ImageId,Label')
