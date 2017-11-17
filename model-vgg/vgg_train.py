#!/usr/bin/python
import os, datetime
import numpy as np
import tensorflow as tf
from DataLoader import *
import layers as L

############################
#### GENERAL PARAMETERS ####
############################
mode = 'Train' # 'Train' or Test'
loadH5 = False

############################
#### Dataset Parameters ####
############################
batch_size = 32
load_size = 224 # original image size
fine_size = 224 # cropped image size
color_channels = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

############################
### Training Parameters ####
############################
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
epochs = 100 # training steps
step_display = 1 # how often to show training/validation loss
step_save = 200 # how often to save model
path_save = 'saved_models/vgg'
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

def vgg(input_tensor, keep_prob, train_phase):
    # assuming 224x224x3 input_tensor
    net = input_tensor
    # block 1 -- outputs 112x112x64
    net = L.conv(net, name="conv1_1", kh=3, kw=3, n_out=32, train_phase=train_phase)
    net = L.conv(net, name="conv1_2", kh=3, kw=3, n_out=32, train_phase=train_phase)
    net = L.pool(net, name="pool1", kh=2, kw=2, dw=2, dh=2)
    # block 2 -- outputs 56x56x128
    net = L.conv(net, name="conv2_1", kh=3, kw=3, n_out=64, train_phase=train_phase)
    net = L.conv(net, name="conv2_2", kh=3, kw=3, n_out=64, train_phase=train_phase)
    net = L.pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)
    # # block 3 -- outputs 28x28x256
    net = L.conv(net, name="conv3_1", kh=3, kw=3, n_out=128, train_phase=train_phase)
    net = L.conv(net, name="conv3_2", kh=3, kw=3, n_out=128, train_phase=train_phase)
    net = L.conv(net, name="conv3_3", kh=3, kw=3, n_out=128, train_phase=train_phase)
    net = L.conv(net, name="conv3_4", kh=3, kw=3, n_out=128, train_phase=train_phase)
    net = L.pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)
    # block 4 -- outputs 14x14x512
    net = L.conv(net, name="conv4_1", kh=3, kw=3, n_out=256, train_phase=train_phase)
    net = L.conv(net, name="conv4_2", kh=3, kw=3, n_out=256, train_phase=train_phase)
    net = L.conv(net, name="conv4_3", kh=3, kw=3, n_out=256, train_phase=train_phase)
    net = L.pool(net, name="pool4", kh=2, kw=2, dh=2, dw=2)
    # flatten
    flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
    net = tf.reshape(net, [-1, flattened_shape], name="flatten")
    # fully connected
    net = L.fully_connected(net, name="fc6", n_out=1024)
    net = tf.nn.dropout(net, keep_prob)
    net = L.fully_connected(net, name="fc7", n_out=1024)
    net = tf.nn.dropout(net, keep_prob)
    net = L.fully_connected(net, name="fc8", n_out=100)
    return net


###############################
### DEFINE MODEL EVAL TOOLS ###
###############################
# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, color_channels])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
logits = vgg(x, keep_dropout, train_phase)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
acc1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
acc5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))
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
            l, a1, a5 = sess.run([loss, acc1, acc5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False}) 
            myprint("-Iter " + str(step) + ", Training Loss= " + \
            "{:.6f}".format(l) + ", Accuracy Top1 = " + \
            "{:.4f}".format(a1) + ", Top5 = " + \
            "{:.4f}".format(a5))
            print(images_batch.mean())
            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, a1, a5 = sess.run([loss, acc1, acc5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
            myprint("-Iter " + str(step) + ", Validation Loss= " + \
            "{:.6f}".format(l) + ", Accuracy Top1 = " + \
            "{:.4f}".format(a1) + ", Top5 = " + \
            "{:.4f}".format(a5))
        
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
        acc_total = 0.
        loader_val.reset()
        for i in range(num_batch):
            myprint('Batch '+str(i+1)+' of '+str(num_batch+1))
            images_batch, labels_batch = loader_val.next_batch(batch_size)
             # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)
            l, acc = sess.run([loss, acc5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
            myprint("-Iter " + str(step) + ", Validation Loss= " + \
            "{:.6f}".format(l) + ", Accuracy Top5 = " + \
            "{:.4f}".format(acc))
            acc_total += acc
        acc_total /= num_batch
        myprint('Evaluation Finished! Accuracy Top5 = ' + "{:.4f}".format(acc_total))
    elif mode=='Test':
        ######################
        ### WRITE TEST SET ###
        ######################
        # Evaluate on the whole test set
        myprint('Evaluation on the test set...')
        num_batch = loader_test.size()/batch_size
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

