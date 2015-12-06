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

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

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

# Augment dataset
print('Augmenting dataset')
X_train, Y_train = nudge_dataset(X_train, Y_train)
#X_test = df_test.filter(regex='pixel').astype(np.float).values

# Then split into training/testing data
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.10, random_state=42)

# Set up softmax
print('Setting up TF graph')
x = tf.placeholder('float', shape=[None, len(X_train[0])])
y_ = tf.placeholder('float', shape=[None, len(Y_train[0])])

# First conv layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second conv layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Initialize training target
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# Train all the things!
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in xrange(0, len(X_train), 100):
    batch_xs, batch_ys = X_train[i:i+100], Y_train[i:i+100]
    if i % 1000 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0}, session=sess)
        print('step:', i / 1000, 'training accuracy:', train_accuracy)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5}, session=sess)

# Accuracy
print 'test accuracy:', accuracy.eval(feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0}, session=sess)

# Prediction
#prediction = tf.argmax(y, 1)
#pred = prediction.eval(session=sess, feed_dict={x: X_test})
#for y in pred:
#    print y

#print sess.run(accuracy, feed_dict={x: X_test, y_: Y_test})
