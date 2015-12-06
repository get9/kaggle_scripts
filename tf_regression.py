import sys

import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.ndimage import convolve
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler

def one_hot_encode(X):
    a = np.zeros([X.size, X.max() + 1])
    a[np.arange(X.size), X] = 1
    return a

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
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
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((28, 28)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
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
#df_test = pd.read_csv(test_file_name)

# Split into data/labels (data is scaled between [0, 1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(df_train.filter(regex='pixel').astype(np.float).values)
Y_train = one_hot_encode(df_train.label.values)
#print('Augmenting dataset')
#X_train, Y_train = nudge_dataset(X_train, Y_train)
#X_test = df_test.filter(regex='pixel').astype(np.float).values

# Then split into training/testing data
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.10, random_state=42)

# Set up softmax
print('Setting up TF graph')
x = tf.placeholder('float', [None, len(X_train[0])])
y_ = tf.placeholder('float', [None, len(Y_train[0])])
W = tf.Variable(tf.zeros([len(X_train[0]), len(Y_train[0])]))
b = tf.Variable(tf.zeros([len(Y_train[0])]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Cross entropy loss
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# Model
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Initialize all vars
init = tf.initialize_all_variables()

# Train all the things!
sess = tf.Session()
sess.run(init)
for i in xrange(1000):
    batch_xs, batch_ys = X_train[i:i+100], Y_train[i:i+100]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print 'accuracy:', sess.run(accuracy, feed_dict={x: X_test, y_: Y_test})

# Prediction
#prediction = tf.argmax(y, 1)
#pred = prediction.eval(session=sess, feed_dict={x: X_test})
#for y in pred:
#    print y

#print sess.run(accuracy, feed_dict={x: X_test, y_: Y_test})
