import tensorflow as tf
import pickle

#loading the dataset and storing the features and the labels of the training,validation and testing features
training_file = '/home/varun/deep_learning/tensorflow_project/traffic-signs-data/train.p'
validation_file = '/home/varun/deep_learning/tensorflow_project/traffic-signs-data/valid.p'
testing_file = '/home/varun/deep_learning/tensorflow_project/traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, Y_train = train['features'], train['labels']
X_valid, Y_valid = valid['features'], valid['labels']
X_test, Y_test = test['features'], test['labels']

###############################################################

import numpy as np
### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
#print(len(y_train))
n_train = len(y_train)

# TODO: Number of validation examples
n_validation = len(y_valid)

# TODO: Number of testing examples.
n_test = len(y_test)

# TODO: What's the shape of an traffic sign image?
ans = np.shape(X_train[0])
image_shape = ans

# TODO: How many unique classes/labels there are in the dataset.
#print(np.size(y_train[0]))
n_classes = 43

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

##############################################################
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random
# Visualizations will be shown in the notebook.
%matplotlib inline
index = random.randint(0,len(X_valid))
image = X_valid[index].squeeze()
plt.figure(figsize=(1,1))
plt.imshow(image)
print(y_valid[index])
print(len(X_train))

#########################################################################
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
#from sklearn.model_selection import train_test_split
'''
X_train = (X_train-128)/128
X_valid = (X_valid-128)/128
X_test = (X_valid-128)/128'''
#############################################################################


### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
#X_train, y_train = shuffle(X_train, y_train)

epoch = 10
batch_size = 35
learning_rate = 0.0008
keep_probability = tf.placeholder(tf.float32)

def lenet(x):
    w_1 = tf.Variable(tf.truncated_normal(shape=(5,5,3,6),mean=0,stddev=0.1))
    b_1 = tf.Variable(tf.zeros(6))
    one_1 = tf.nn.conv2d(x,w_1,strides=[1,1,1,1],padding='VALID') + b_1
    one_2 = tf.nn.relu(one_1)
    #one_2 = tf.nn.dropout(one_2,keep_probability)
    #####28*28*6
    one = tf.nn.max_pool(one_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    ####14*14*6
    w_2 = tf.Variable(tf.truncated_normal(shape=(5,5,6,16),mean=0,stddev=0.1))
    b_2 = tf.Variable(tf.zeros(16))
    two_1 = tf.nn.conv2d(one,w_2,strides=[1,1,1,1],padding='VALID') + b_2
    two_2 = tf.nn.relu(two_1)
    #two_2 = tf.nn.dropout(two_2,keep_probability)
    #####10*10*16
    two = tf.nn.max_pool(two_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    #####5*5*16
    #flatenning
    ans = flatten(two)
    #400
    w_3 = tf.Variable(tf.truncated_normal(shape=(400,120),mean=0,stddev=0.1))
    b_3 = tf.Variable(tf.zeros(120))
    three_1 = tf.matmul(ans,w_3) + b_3
    three = tf.nn.relu(three_1)
    #three = tf.nn.dropout(three,keep_probability)

    w_4 = tf.Variable(tf.truncated_normal(shape=(120,84),mean=0,stddev=0.1))
    b_4 = tf.Variable(tf.zeros(84))
    four_1 = tf.matmul(three,w_4) + b_4
    four = tf.nn.relu(four_1)
    #four = tf.nn.dropout(four,keep_probability)

    w_5 = tf.Variable(tf.truncated_normal(shape=(84,43),mean=0,stddev=0.1))
    b_5 = tf.Variable(tf.zeros(43))
    five = tf.matmul(four,w_5) + b_5
    return five

#######################################################################
features = tf.placeholder(tf.float32,(None,32,32,3))
label = tf.placeholder(tf.int32,(None))
labels_one_hot_encoded = tf.one_hot(label,43)

logits = lenet(features)

cost_temp = tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = labels_one_hot_encoded)
cost = tf.reduce_mean(cost_temp)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(cost)

##################################################################################
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_one_hot_encoded, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data,y_data):
    total = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0,total,batch_size):
        end = offset+batch_size
        batch_x,batch_y = X_data[offset:end],y_data[offset:end]
        accuracy_temp = sess.run(accuracy,feed_dict={features:batch_x,label:batch_y,keep_probability:1})
        total_accuracy = total_accuracy + (accuracy_temp*len(batch_x))
        #total_acuracy = total_accuracy + (accuracy_temp)
    return total_accuracy/total

##################################################################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_examples = len(X_train)
    for i in range(epoch):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0,total_examples,batch_size):
            end = offset+batch_size
            batch_x,batch_y = X_train[offset:end],y_train[offset:end]
            training = sess.run(training_operation,feed_dict={features:batch_x,label:batch_y,keep_probability:0.5})
            #accuracy_1 = sess.run(accuracy,feed_dict={features:batch_x,label:batch_y})
            #print(accuracy_1)
        validation_accuracy = evaluate(X_valid,y_valid)
        print('Epoch:{} '.format(i+1))
        print('Validation Accuracy:{:.3f}'.format(validation_accuracy))
    test_accuracy = sess.run(accuracy,feed_dict={features:X_test,label:y_test,keep_probability:1})
    print('Test Accuracy:{}'.format(test_accuracy))
    try:
        saver
    except NameError:
        saver = tf.train.Saver()
    saver.save(sess,'lenet')
    print("model saved")

########################################################################################
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import cv2

image_1 = cv2.imread('10.ppm')
image_2 = cv2.imread('11.ppm')
image_3 = cv2.imread('12.ppm')
image_4 = cv2.imread('13.ppm')
image_5 = cv2.imread('14.ppm')

label_1 = 10
label_2 = 11
label_3 = 13
label_4 = 14
label_5 = 15

print(image_1.shape)
plt.figure(figsize=(1,1))
plt.title('label 10')
plt.imshow(image_1)
plt.show()
print(image_2.shape)
plt.figure(figsize=(1,1))
plt.title('label 11')
plt.imshow(image_2)
plt.show()
print(image_3.shape)
plt.figure(figsize=(1,1))
plt.title('label 12')
plt.imshow(image_3)
plt.show()
print(image_4.shape)
plt.figure(figsize=(1,1))
plt.title('label 13')
plt.imshow(image_4)
plt.show()
print(image_5.shape)
plt.figure(figsize=(1,1))
plt.title('label 14')
plt.imshow(image_5)
plt.show()

#############################################################################
image = np.array([image_1,image_2,image_3,image_4,image_5])
image = image.astype('uint8')

label = np.array([10,11,12,13,14])
label = label.astype('uint8')

def evaluate_1(X_data,y_data):
    total = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0,total,5):
        end = offset+5
        batch_x,batch_y = X_data[offset:end],y_data[offset:end]
        print(batch_x)
        print("hello")
        print(batch_y)
        accuracy_temp = sess.run(accuracy,feed_dict={features:batch_x,label:batch_y,keep_probability:1})
        total_accuracy = total_accuracy + (accuracy_temp*len(batch_x))
        #total_acuracy = total_accuracy + (accuracy_temp)
    return total_accuracy/total

with tf.Session() as sess:
    saver.restore(sess,'lenet')
    image_accuracy = evaluate_1(image,label)
    #image_accuracy = evaluate_1(X_valid,y_valid)
    print('image_accuracy:{}'.format(image_accuracy))
