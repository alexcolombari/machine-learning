#------------------------------------------------------------------------------#
#                            By: Alex Colombari                                #
#                     http://www.github.com/alexcolombari                      #
#    Following this: https://www.guru99.com/autoencoder-deep-learning.html     #
#------------------------------------------------------------------------------#
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

# Import the data:
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

# Convert the data to black and white format:
def grayscale(im):
    return im.reshape(im.shape[0], 3, 32, 32).mean(1).reshape(im.shape[0], -1)

# Append all the batches:
# Load the data into memory
data, labels = [], []
## Loop ober the b:
for i in range(1, 6):
    filename = './cifar-10-batches-py/data_batch_' + str(i)
    open_data = unpickle(filename)
    if len(data) > 0:
        data = np.vstack((data, open_data['data']))
        labels = np.hstack((labels, open_data['labels']))
    else:
        data = open_data['data']
        labels = open_data['labels']

data = grayscale(data)
x = np.matrix(data)
y = np.array(labels)

# Construct the training dataset:
horse_i = np.where(y == 7)[0]
horse_x = x[horse_i]

# Construct an image visualizer:
## Plot the images
def plot_image(image, shape = [32, 32], cmap = "Greys_r"):
    plt.imshow(image.reshape(shape), cmap = cmap, interpolation = "nearest")    
    plt.axis("off")
    plt.show()


## Parameters:
n_inputs = 32 * 32
BATCH_SIZE = 1
batch_size = tf.placeholder(tf.int64)

# Using placeholder:
x = tf.placeholder(tf.float32, shape = [None, n_inputs])
## Dataset:
dataset = tf.data.Dataset.from_tensor_slices(x).repeat().batch(batch_size)
iter = dataset.make_initializable_iterator()
features = iter.get_next()


# ----------------------- NEURAL NETWORK ----------------------------

# Parameters:
from functools import partial

## Encoder
n_hidden_1 = 300
n_hidden_2 = 150    # codings

## Decoder
n_hidden_3 = n_hidden_1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

## Xavier Initialization
xav_init = tf.contrib.layers.xavier_initializer()
## L2 Regularizer
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)

# LAYERS
# Define the layers
## Create the dense layer
dense_layer = partial(tf.layers.dense,
                        activation = tf.nn.elu,
                        kernel_initializer = xav_init,
                        kernel_regularizer = l2_regularizer)

# Model architecture
hidden_1 = dense_layer(features, n_hidden_1)
hidden_2 = dense_layer(hidden_1, n_hidden_2)
hidden_3 = dense_layer(hidden_2, n_hidden_3)
outputs = dense_layer(hidden_3, n_outputs, activation = None)

# Define the optimization
loss = tf.reduce_mean(tf.square(outputs - features))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)


# -------------------------- RUN MODEL --------------------------

## Set params
BATCH_SIZE = 150
### Number of batches :  length dataset / batch size
n_batches = horse_x.shape[0] // BATCH_SIZE
n_epochs = 100

## Call Saver to save the model and re-use it later during evaluation
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # initialise iterator with train data
    sess.run(iter.initializer, feed_dict = {x: horse_x,
                                            batch_size: BATCH_SIZE})
    print('Training...')
    print(sess.run(features).shape)
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            sess.run(train)
        if epoch % 10 == 0:
            loss_train = loss.eval()
            print("{}".format(epoch), "Train MSE:", loss_train)

    save_path = saver.save(sess, "./model.ckpt")    
    print("Model saved in path: %s" % save_path)

# Evaluate the model
test_data = unpickle('./cifar-10-batches-py/test_batch')
test_x = grayscale(test_data['data'])

# ----------------------- RECONSTRUCT IMAGE ------------------------

def reconstruct_image(df, image_number = 1):
    ## Part 1: Reshape the image to the correct dimension i.e 1, 1024
    x_test = df[image_number]
    x_test_1 = x_test.reshape((1, 32 * 32))

    ## Part 2: Feed the model with the unseen image, encode/decode the image
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        sess.run(iter.initializer, feed_dict={x: x_test_1,
                                      batch_size: 1})

        ## Part 3:  Print the real and reconstructed image
        # Restore variables from disk.
        saver.restore(sess, "./model.ckpt")
        print("Model Restored.")
        # Reconstruct image
        outputs_val = outputs.eval()
        print(outputs_val.shape)
        fig = plt.figure()
        # Plot real
        ax1 = fig.add_subplot(121)
        plot_image(x_test_1, shape=[32, 32], cmap = "Greys_r")
        # Plot estimated
        ax2 = fig.add_subplot(122)
        plot_image(outputs_val, shape=[32, 32], cmap = "Greys_r")
        plt.tight_layout()
        fig = plt.gcf()
        fig.show()

reconstruct_image(df = test_x, image_number = 13)
