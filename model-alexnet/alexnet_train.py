#!/usr/bin/python
import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

############################
#### GENERAL PARAMETERS ####
############################
mode = 'Train' # 'Train' or Test' or 'Val'
loadH5 = False

############################
#### Dataset Parameters ####
############################
batch_size = 256
load_size = 224 # original image size
fine_size = 224 # cropped image size
color_channels = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

############################
### Training Parameters ####
############################
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
beta = 0 # L2 regularization
epochs = 100 # training steps
step_display = 100 # how often to show training/validation loss
step_save = 200 # how often to save model
path_save = 'saved_models/alexnet'
start_from = ''

######################
#### DATA LOADING ####
######################
# Construct dataloader
opt_data_train = {
    'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }
opt_data_test = {
    'data_h5': 'miniplaces_256_test.h5',
    'data_root': '../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../data/test.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }


### Load Data ###
if not loadH5:
    loader_train = DataLoaderDisk(**opt_data_train)
    loader_val = DataLoaderDisk(**opt_data_val)
    loader_test = DataLoaderDisk(**opt_data_test)
else:
    loader_train = DataLoaderH5(**opt_data_train)
    loader_val = DataLoaderH5(**opt_data_val)
    loader_test = DataLoaderH5(**opt_data_test)

######################### 
### DEFINE NEURAL NET ###
######################### 
def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)

def alexnet(x, keep_dropout, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=np.sqrt(2./(11*11*3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
        'wf6': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
    }
    biases = {
        'bo': tf.Variable(tf.ones(100))
    }
    # Conv + ReLU + Pool, 224->56->28
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Conv + ReLU  + Pool, 28-> 14
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Conv + ReLU, 14-> 14
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)
    # Conv + ReLU, 14-> 14
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)
    # Conv + ReLU + Pool, 14->7
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.matmul(fc6, weights['wf6'])
    fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)
    # FC + ReLU + Dropout
    fc7 = tf.matmul(fc6, weights['wf7'])
    fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)
    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])
    #### Regularizer
    regularizer = tf.nn.l2_loss(weights['wc1'])+tf.nn.l2_loss(weights['wc2'])
    regularizer += tf.nn.l2_loss(weights['wc3'])+tf.nn.l2_loss(weights['wc4'])
    regularizer += tf.nn.l2_loss(weights['wc5'])+tf.nn.l2_loss(weights['wf6'])
    regularizer += tf.nn.l2_loss(weights['wf7'])+tf.nn.l2_loss(weights['wo'])
    return out, regularizer

###############################
### DEFINE MODEL EVAL TOOLS ###
###############################
# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, color_channels])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
logits, regularizer = alexnet(x, keep_dropout, train_phase)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)+beta*regularizer)
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))
prediction = tf.nn.top_k(logits,5,sorted=True)

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

#############
### TRAIN ###
#############
num_batches = loader_train.size()/batch_size
# Launch the graph
with tf.Session() as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
        myprint("Model restored.")
    else:
        sess.run(init)
    
    step = 0

    while step < num_batches*epochs and mode=='Train':
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)
        if step % step_display == 0:
            myprint('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Calculate batch loss and accuracy on training set
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False}) 
            myprint("-Iter " + str(step) + ", Training Loss= " + \
            "{:.6f}".format(l) + ", Accuracy Top1 = " + \
            "{:.4f}".format(acc1) + ", Top5 = " + \
            "{:.4f}".format(acc5))

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
            myprint("-Iter " + str(step) + ", Validation Loss= " + \
            "{:.6f}".format(l) + ", Accuracy Top1 = " + \
            "{:.4f}".format(acc1) + ", Top5 = " + \
            "{:.4f}".format(acc5))
        
        # Run optimization op (backprop)
        sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})
        step += 1
        # Save model
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            myprint("Model saved at Iter %d !" %(step))

    if mode=='Val':
        ##################
        ### EVALUATION ###
        ##################
        # Evaluate on the whole validation set
        myprint('Evaluation on the whole validation set...')
        num_batch = loader_val.size()/batch_size
        acc1_total = 0.
        acc5_total = 0.
        loader_val.reset()
        for i in range(num_batch):
            myprint('Batch '+str(i+1)+' of '+str(num_batch+1))
            images_batch, labels_batch = loader_val.next_batch(batch_size)
            acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
            acc1_total += acc1
            acc5_total += acc5
            myprint("Validation Accuracy Top1 = " + \
                "{:.4f}".format(acc1) + ", Top5 = " + \
                "{:.4f}".format(acc5))
        acc1_total /= num_batch
        acc5_total /= num_batch
        myprint('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))
    elif mode=='Test':
        ######################
        ### WRITE TEST SET ###
        ######################
        # Evaluate on the whole test set
        myprint('Evaluation on the test set...')
        num_batch = loader_test.size()//batch_size
        loader_test.reset()
        y_pred = np.zeros((10000,5)) # initialize array of results
        for ii in range(num_batch+1):
            myprint('Batch '+str(ii+1)+' of '+str(num_batch+1))
            images_batch, labels_batch = loader_test.next_batch(batch_size)
            y_local = sess.run([prediction],feed_dict={x: images_batch, keep_dropout:1, train_phase: False})
            y_batch = y_local[0].indices # current batch predictions
            idcs = range(ii*batch_size,min(ii*batch_size+y_batch.shape[0],len(y_pred)))
            y_pred[idcs,:] = y_batch[:(idcs[-1]-idcs[0]+1),:] # assign to global results vector
        # write results file
        f0 = opt_data_test['data_list']
        with open(f0) as f:
            lines = f.readlines()
        f1 = 'test_output.txt'
        f1w = open(f1,'w')
        for ii,l in enumerate(lines):
            im = l.split(' ')[0]
            lwrite = im +' '+' '.join(str(int(lab)) for lab in y_pred[ii])+'\n'
            f1w.write(lwrite)
        f1w.close()

