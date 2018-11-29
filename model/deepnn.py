import os
import json
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


from sklearn.model_selection import train_test_split


min_max_scaler = MinMaxScaler()

max_path = os.path.join(os.getcwd(),'../features/means.txt')
min_path = os.path.join(os.getcwd(),'../features/stds.txt')
out_dir  = os.path.join(os.getcwd(),'../output/')

file_path = os.path.join(os.getcwd(),'../features/features.json')

write_path = os.path.join(os.getcwd(),'../features/features_new.json')

# result = []
# with open(file_path) as f:
#     result = json.load(f)
#
# final_result = []
# for reading in result:
#     if(reading['x_mag'] != 0 ):
#         final_result.append(reading)
#
#
# with open(write_path,'w') as f:
#     json.dump(final_result, f)



data = pd.read_json(write_path)
# print(data.head())

data = data.drop(['beacon_avg_dist', 'beacon_avg_rssi' ,'beacon_dev_dist', 'beacon_dev_rssi', 'beacon_max_dist', 'beacon_max_rssi', 'beacon_min_dist', 'beacon_min_rssi'],axis=1)

X    = data.drop(['particles'], axis=1)

# X_max = X.apply(lambda  x: x.max())
# X_min = X.apply(lambda  x : x.min())
#
# print(X_min)



Y    = data[['particles']]




Y = Y.particles.apply(pd.Series).merge(Y, left_index = True, right_index = True).drop(["particles"], axis = 1)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

mean_x  = X_train.mean()
std_x   = X_train.std()

mean_y  = Y.mean()
std_y   = Y.std()

X_train  = (X_train-mean_x)/(std_x)
y_train  = (y_train-mean_y)/std_y

X_train  = X_train.values
y_train  = y_train.values







def batchGenerator(dim, batchsize, batches):
    arr = np.arange(dim)
    np.random.shuffle(arr)
    indexes = []
    start = 0
    for i in range(batches):
        indexes.append(arr[start:start+batchsize])
        start += batchsize
    return indexes




X_in = tf.placeholder('float',[None,12],name='input')
Y_in = tf.placeholder('float', [None,6])

#Neural Network Model

def buildModel(X):


    layer1 = tf.contrib.layers.fully_connected(X,500)

    relu1   = tf.nn.leaky_relu(layer1)

    # keep_prob = tf.placeholder(tf.float32)
    drop_1 = tf.nn.dropout(relu1, 0.8)

    layer2 = tf.contrib.layers.fully_connected(drop_1,500)

    relu2  = tf.nn.leaky_relu(layer2)

    # keep_prob_2 = tf.placeholder(tf.float32)
    drop_2 = tf.nn.dropout(relu2, 0.8)

    layer3 = tf.contrib.layers.fully_connected(drop_2,6)

    output  = tf.nn.leaky_relu(layer3)
    # drop_3  = tf.nn.dropout(relu3, 0.8)
    #
    # layer4 = tf.contrib.layers.fully_connected(drop_3,6)
    #
    #
    #
    # output = tf.nn.leaky_relu(layer4,name='output')


    #loss = tf.reduce_mean(tf.squared_difference(output, Y_in))

    return output



epochs = 50000
batch_size = 1000
learning_rate = 0.09



output    = buildModel(X_in)
loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, Y_in)),
                         axis=1)),axis=0,name="loss",)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(epochs):
        tf.train.write_graph(sess.graph_def, out_dir, 'model.pbtxt')

        avg_cost = 0
        total_batch = int(X_train.shape[0] / batch_size)

        indexes_list = batchGenerator(X_train.shape[0], batch_size, total_batch)
        for index,indexes in enumerate(indexes_list):
            x_batch = X_train[indexes]
            y_batch = y_train[indexes]
            _,cost  = sess.run([optimizer,loss],feed_dict = {X_in:x_batch, Y_in:y_batch})
            avg_cost += cost / total_batch
            #print("Epoch:", (epoch + 1), "batchnum=",str(index),"cost =", "{:.5f}".format(avg_cost))
        saver.save(sess, out_dir + 'model.ckpt',global_step=epoch+1)
        print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))

    print("\nTraining complete!")

#saver.restore(sess,model_path)








