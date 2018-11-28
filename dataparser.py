import os
import json
import math
import statistics

data_dir = os.getcwd()+"/data/"

files_list = os.listdir(data_dir);
readings = []


for file in files_list:
    full_path = data_dir+file

    with open(full_path) as f:
        inside_json = json.load(f)
        readings.extend(inside_json)



features = []
for reading in readings:
    single_feature = {};

    beacon_rssi, beacon_dist, wifi_rssi = [], [], []
    if(len(reading['beacons']) > 0):
        for i in reading['beacons']:
            beacon_rssi.append(i['rssi'])
            beacon_dist.append(i['distance'])
        for i in reading['wifi']:
            wifi_rssi.append(i['rssi'])

        # MAGNETIC
        x, y, z = reading['magnetic'][0], reading['magnetic'][1], reading['magnetic'][2]
        x_sqr, y_sqr, z_sqr = math.pow(x, 2), math.pow(y, 2), math.pow(z, 2)

        magnitude = math.sqrt(x_sqr + y_sqr + z_sqr)
        xy_plane = math.sqrt(x_sqr + y_sqr)
        yz_plane = math.sqrt(y_sqr + z_sqr)
        xz_plane = math.sqrt(x_sqr + z_sqr)

        single_feature['magnitude'] = magnitude
        single_feature['xy_mag']    = xy_plane
        single_feature['yz_mag']    = yz_plane
        single_feature['xz_mag']    = xz_plane

        single_feature['x_mag'] = x
        single_feature['y_mag'] = y
        single_feature['z_mag'] = z

        # BEACON
        beacon_avg_rssi, beacon_avg_dist = statistics.mean(beacon_rssi), statistics.mean(beacon_dist)
        beacon_rssi_dev, beacon_dist_dev = statistics.pstdev(beacon_rssi), statistics.pstdev(beacon_dist)
        beacon_max_rssi, beacon_max_dist = max(beacon_rssi), max(beacon_dist)
        beacon_min_rssi, beacon_min_dist = min(beacon_rssi), min(beacon_dist)

        single_feature['beacon_avg_rssi']        = beacon_avg_rssi
        single_feature['beacon_dev_rssi']        = beacon_rssi_dev
        single_feature['beacon_min_rssi'] = beacon_min_rssi
        single_feature['beacon_max_rssi'] = beacon_max_rssi
        single_feature['beacon_min_dist'] = beacon_min_dist
        single_feature['beacon_max_dist'] = beacon_max_dist
        single_feature['beacon_dev_dist'] = beacon_dist_dev
        single_feature['beacon_avg_dist'] = beacon_avg_dist


        # WIFI
        wifi_avg_rssi = statistics.mean(wifi_rssi)
        wifi_rssi_dev = statistics.pstdev(wifi_rssi)
        wifi_max_rssi = max(wifi_rssi)
        wifi_min_rssi = min(wifi_rssi)
        wifi_total_rssi = sum(wifi_rssi)

        single_feature['wifi_avg_rssi'] = wifi_avg_rssi
        single_feature['wifi_rssi_dev'] = wifi_rssi_dev
        single_feature['wifi_max_rssi'] = wifi_max_rssi
        single_feature['wifi_min_rssi'] = wifi_min_rssi
        single_feature['wifi_total_rssi'] = wifi_total_rssi

        landmarks = []
        ####
        for index,particle in enumerate(reading['particles']) :
            if(index == 0):
                landmarks.extend([particle['x'], particle['y'], particle['z']])
            else:
                landmarks.extend([particle['x'], particle['z'] , particle['y']])


        single_feature['particles'] = landmarks
        features.append(single_feature)


with open('./features/features.json', 'w') as f:
    json.dump(features,f)





