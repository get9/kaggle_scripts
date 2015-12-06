import sys

import pandas as pd
import numpy as np
from scipy.ndimage import convolve
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


def nudge_dataset(X, Y):
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]
    ]

    shift = lambda x, w: convolve(x.reshape((28, 28)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector)
                              for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


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

# Augment data
print('Augmenting data')
X_train, Y_train = nudge_dataset(X_train, Y_train)

# Set up pipeline
scaler = MinMaxScaler()
pca = PCA(n_components=0.90)
rf = RandomForestClassifier(n_jobs=-1,
                            n_estimators=1000,
                            random_state=42,
                            criterion='gini',
                            bootstrap=False,)

# Make pipeline
clf = Pipeline(steps=[
    ('scaler', scaler),
    ('rf', rf),
])

# Fit classifier
print('Fitting classifier')
clf.fit(X_train, Y_train)

# Dump for later use
joblib.dump(clf, 'rf_pca_augmented.clf')

# Run Classifier
print('Predicting')
Y_predict = clf.predict(X_test)
print('accuracy:', 1 - (np.count_nonzero(Y_predict - Y_test) / len(Y_predict)))
