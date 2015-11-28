import sys

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.linear_model import LogisticRegression


if len(sys.argv) < 2:
    print('Usage:')
    print('    {} [traincsv] [testcsv]'.format(sys.argv[0]))
    sys.exit(1)

# Read input
print('Reading input')
train_file_name, test_file_name = sys.argv[1:]
df_train = pd.read_csv(train_file_name)
#df_test = pd.read_csv(test_file_name)

# Split into data/labels
X_train = df_train.filter(regex='pixel').astype(np.float).values
Y_train = df_train.label.values
#X_test = df_test.filter(regex='pixel').astype(np.float).values

# Normalize X values
print('Scaling training data between [0, 1]')
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)

# Perform PCA for dimensionality reduction
print('Performing PCA')
pca = PCA(n_components=0.80).fit(X_train)
X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)

# Then split into training/testing data
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.10, random_state=42)

# Start up SVC classifier (default params)
print('Fitting Logit model')
logit = LogisticRegression(penalty='l2', random_state=42, C=1.0, \
                           fit_intercept=True, solver='newton-cg', n_jobs=-1)
logit.fit(X_train, Y_train)

print('accuracy:', logit.score(X_test, Y_test))
