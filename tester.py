# import tensorflow as tf
# import pandas as pd
# import os
#
# data = {"wifi_min_rssi": -96, "beacon_min_dist": 5.886423916356039, "beacon_max_rssi": -98, "yz_mag": 20.047068355712216, "z_mag": -1.9996740818023682, "wifi_rssi_dev": 4.812459175266786, "wifi_total_rssi": -1165, "xy_mag": 71.96098223360576, "particles": [[1475255.2742203989, 6095169.663052855, 1123463.2412700655], [1475255.2742203989, 1123463.2412700655, 6095169.663052855]], "beacon_min_rssi": -98, "beacon_avg_dist": 5.886423916356039, "beacon_max_dist": 5.886423916356039, "wifi_avg_rssi": -89.61538461538461, "beacon_dev_dist": 0.0, "xz_mag": 69.17004703795993, "y_mag": 19.947086334228516, "magnitude": 71.98876065372119, "beacon_dev_rssi": 0.0, "x_mag": 69.1411361694336, "wifi_max_rssi": -82, "beacon_avg_rssi": -98}
#
# X = [[(data['magnitude'] -39.367269)/17.922281 , (data['wifi_avg_rssi'] + 86.054071)/(4.012481) , (data['wifi_max_rssi'] + 68.133809)/(9.499131) , (data['wifi_min_rssi'] + 93.677243)/(3.655876) , (data['wifi_rssi_dev'] - 7.581577)/(2.298151) , (data['wifi_total_rssi'] + 1592.885259)/(766.736693), (data['x_mag'] - 4.230324)/(20.549991) , (data['xy_mag'] - 37.507939)/(16.919943) , (data['xz_mag'] - 18.845000)/(16.214845), (data['y_mag'] + 3.893108)/(35.182298), (data['yz_mag'] - 33.083384)/(18.338149), (data['z_mag'] + 6.2274423)/(11.793235) ]]
#
#
# Y    = []
#
# for data in data['particles']:
#     for val in data:
#         Y.append(val)
#
# tmp   = Y[-1]
# Y[-1] = Y[-2]
# Y[-2] = tmp
# print(Y)
#
# out_dir  = os.path.join(os.getcwd(),'output/model.ckpt-50000')
#
#
# X_in = tf.placeholder('float',[None,12],name='input')
# Y_in = tf.placeholder('float', [None,6])
#
# #Neural Network Model
#
# def buildModel(X):
#
#
#     layer1 = tf.contrib.layers.fully_connected(X,500)
#
#     relu1   = tf.nn.leaky_relu(layer1)
#
#     # keep_prob = tf.placeholder(tf.float32)
#     drop_1 = tf.nn.dropout(relu1, 0.8)
#
#     layer2 = tf.contrib.layers.fully_connected(drop_1,500)
#
#     relu2  = tf.nn.leaky_relu(layer2)
#
#     # keep_prob_2 = tf.placeholder(tf.float32)
#     drop_2 = tf.nn.dropout(relu2, 0.8)
#
#     layer3 = tf.contrib.layers.fully_connected(drop_2,500)
#
#     relu3  = tf.nn.leaky_relu(layer3)
#     drop_3  = tf.nn.dropout(relu3, 0.8)
#
#     layer4 = tf.contrib.layers.fully_connected(drop_3,6)
#
#
#
#     output = tf.nn.leaky_relu(layer4,name='output')
#
#
#     #loss = tf.reduce_mean(tf.squared_difference(output, Y_in))
#
#     return output
#
#
# output    = buildModel(X_in)
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess, out_dir)
#     output_pred = sess.run(output,feed_dict={X_in:X})
#     print(output_pred)

import tensorflow as tf

Y1 = tf.placeholder('float',[None,2])
Y2 = tf.placeholder('float',[None,2])
loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(Y1, Y2)),
                         name="loss",axis=1)),axis=0)
# 1 1
# 0 25

with tf.Session() as sess:
    print(sess.run(loss, feed_dict={Y1:[[3.0,4.0],[2.0,1.0]], Y2:[[4.0,5.0],[2.0,6.0]]}))