import tensorflow as tf
import pandas as pd
import os

data = {"wifi_min_rssi": -96, "beacon_min_dist": 5.886423916356039, "beacon_max_rssi": -98, "yz_mag": 20.047068355712216, "z_mag": -1.9996740818023682, "wifi_rssi_dev": 4.812459175266786, "wifi_total_rssi": -1165, "xy_mag": 71.96098223360576, "particles": [[1475255.2742203989, 6095169.663052855, 1123463.2412700655], [1475255.2742203989, 1123463.2412700655, 6095169.663052855]], "beacon_min_rssi": -98, "beacon_avg_dist": 5.886423916356039, "beacon_max_dist": 5.886423916356039, "wifi_avg_rssi": -89.61538461538461, "beacon_dev_dist": 0.0, "xz_mag": 69.17004703795993, "y_mag": 19.947086334228516, "magnitude": 71.98876065372119, "beacon_dev_rssi": 0.0, "x_mag": 69.1411361694336, "wifi_max_rssi": -82, "beacon_avg_rssi": -98}

X = [[(data['magnitude'] - 91.351989)/(91.351989 - 0.131245) , (data['wifi_avg_rssi'] + 94.260870)/(94.260870 - 79.000000) , (data['wifi_max_rssi'] + 87.000000)/(87.000000 - 56.000000) , (data['wifi_min_rssi'] + 102.000000)/(102.000000 - 87.000000) , (data['wifi_rssi_dev'] - 3.347626)/(9.646185 - 3.347626) , (data['wifi_total_rssi'] + 3672.000000)/(3672.000000 -700.000000), (data['x_mag'] + 31.453394)/(31.453394 + 89.016609) , (data['xy_mag'] - 0.032130)/(89.136086 - 0.032130) , (data['xz_mag'] - 0.034120)/(91.235414 - 0.034120), (data['y_mag'] + 58.436069)/(58.436069 + 59.541859), (data['yz_mag'] - 0.026175)/(77.196297 - 0.026175), (data['z_mag'] + 52.527943)/(52.527943 + 25.490009) ]]


Y    = []

for data in data['particles']:
    for val in data:
        Y.append(val)

tmp   = Y[-1]
Y[-1] = Y[-2]
Y[-2] = tmp
print(Y)

out_dir  = os.path.join(os.getcwd(),'output/model.ckpt-1000')


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

    output = tf.nn.leaky_relu(layer3,name='output')


    #loss = tf.reduce_mean(tf.squared_difference(output, Y_in))

    return output


output    = buildModel(X_in)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, out_dir)
    output_pred = sess.run(output,feed_dict={X_in:X})
    print(output_pred * 6371000)