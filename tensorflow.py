import sys

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split


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

# Then split into training/testing data
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.10, random_state=42)

# Set up softmax
print('Setting up TF graph')
x = tf.placeholder('float', [None, X_train.shape[1]])
W = tf.Variable(tf.zeros([X_train.shape[1], 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Cross entropy loss
yprime = tf.placeholder('float', [None, 10])
cross_entropy = -tf.reduce_sum(yprime * tf.log(y))

# Model
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Initialize all vars
init = tf.initialize_all_variables()

# Train all the things!
for i in xrange(0, len(X_train), 100):
    batch_xs, batch_ys = X_train[i:i+100], Y_train[i:i+100]
