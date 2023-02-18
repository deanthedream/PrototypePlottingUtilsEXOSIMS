#from PIL import Image
#im = Image.open('./shipdensity_global.tif.ovr')
#im.show()

# from osgeo import gdal
# import matplotlib.pyplot as plt

# Image = gdal.Open('./shipdensity_global.tif.ovr')


# nBands = Image.RasterCount
# Band = Image.GetRasterBand(1)
# #nBands = Image.RasterCount      # how many bands, to help you loop
# nRows  = Image.RasterYSize      # how many rows
# nCols  = Image.RasterXSize      # how many columns
# dType  = Band.DataType          # the datatype for this band

# RowRange = range(nRows)
# for ThisRow in RowRange:
#     # read a single line from this band
#     ThisLine = Band.ReadRaster(0,ThisRow,nCols,1,nCols,1,dType)

#     if ThisRow % 100 == 0: # report every 100 lines
#         print("Scanning %d of %d" % (ThisRow,nRows))

#     for Val in ThisLine: # some simple test on the values
#         if Val == 65535:
#             print('Bad value')


from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

#number (''), trip_count, prev_port, next_port, lat, lon, frequency
routes = genfromtxt('routes.csv', delimiter=',')
routes_num = routes[1:,0].astype(int)
routes_trip_count = routes[1:,1].astype(int)
routes_prev_port = routes[1:,2].astype(int)
routes_next_port = routes[1:,3].astype(int)
routes_lat = routes[1:,4]
routes_lon = routes[1:,5]
routes_freq = routes[1:,6]
del routes

routes = dict()
#prev_trip_count = None
# for j in tqdm(range(10)):
#     print(j)
for i in tqdm(range(len(routes_trip_count))):
    #print(j)
    #i = j+1
    #print(str(not (routes_trip_count[i] in routes.keys())) + ' not (routes_trip_count[i] in routes.keys())')
    if not (routes_trip_count[i] in routes.keys()):
        #print('Not In Keys')
        route = {'prev_port':routes_prev_port[i],'next_port':routes_next_port[i],'frequency':routes_freq[i]}
        route['lat_lon'] = [[routes_lat[i],routes_lon[i]]]
        routes[routes_trip_count[i]] = route
    else:
        routes[routes_trip_count[i]]['lat_lon'].append([routes_lat[i],routes_lon[i]])

    #     routes[routes_trip_count[i]] = route
    # current_trip_count = routes[i,1]
    # if prev_trip_count == current_trip_count:

    # tmp_route = {}
del routes_num, routes_prev_port, routes_next_port, routes_lat, routes_lon, routes_freq

#Number of Waypoints per route histogram
num_waypoints = list()
for i in list(routes.keys()):
    num_waypoints.append(len(routes[i]['lat_lon']))
plt.figure(2)
plt.hist(num_waypoints,bins=50)
plt.show(block=False)





#######TODO FIX THIS!!!!
#Compute Angular Distance
pts_per_sumtheta = list()
for i in list(routes.keys()):
    ll = routes[i]['lat_lon']
    xs = np.zeros(len(ll))
    xs[0] = np.cos(ll[0][1]*np.pi/180.)
    ys = np.zeros(len(ll))
    ys[0] = np.sin(ll[0][0]*np.pi/180.)
    thetas = np.zeros(len(ll)-1)
    for j in np.arange(len(ll)-1)+1:
        xs[j] = np.cos(ll[j][1]*np.pi/180.)
        ys[j] = np.sin(ll[j][0]*np.pi/180.)
        thetas[j-1] = np.arctan2(ys[j]-ys[j-1],xs[j]-xs[j-1])
    routes[i]['thetas'] = thetas #The angles, in rad, between points
    routes[i]['pts_per_sumtheta'] = len(ll)/np.sum(thetas)

    pts_per_sumtheta.append(len(ll)/np.sum(thetas))

plt.figure(3)
plt.hist(pts_per_sumtheta,bins=50)
plt.show(block=False)



#number (''), port name, index number, coordinates
#ports = genfromtxt('ports.csv', delimiter=',')
port_num = list()
port_names = list()
port_latlon = list()
with open('ports.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if not row[''] == '': #i.e. skip the first row
            port_num.append(int(row['']))
            port_names.append(row['PORT_NAME'])
            tmp_coords = row['coords'].replace('(','').replace(')','').split(',')
            port_latlon.append([float(tmp_coords[1]),float(tmp_coords[0])])
port_latlon = np.asarray(port_latlon)

ports = dict()
for i in np.arange(len(port_num)):
    ports[port_num[i]] = {'PORT_NAME':port_names[i],'latlon':port_latlon[i]}
del port_num, port_names, port_latlon, tmp_coords



#number (''), trip_count, prev_port, next_port, distance, frequency
distances = genfromtxt('distances.csv', delimiter=',')
distances_num = np.asarray(distances[1:,0]).astype(int)
distances_trip_count = np.asarray(distances[1:,1]).astype(int)
distances_prev_port = np.asarray(distances[1:,2]).astype(int)
distances_next_port = np.asarray(distances[1:,3]).astype(int)
distances_distance = np.asarray(distances[1:,4])
distances_frequency = np.asarray(distances[1:,5])

for i in np.arange(len(distances_trip_count)):
    if distances_trip_count[i] in routes.keys():
        routes[distances_trip_count[i]]['distance'] = distances_distance[i]
        #if not(routes[distances_trip_count[i]]['prev_port'] == distances_prev_port[i]):
        #    print('not right prev_port!!!!' +  str(routes[distances_trip_count[i]]['prev_port']) + ' ' + str(distances_prev_port[i]))



plt.close(1)
plt.figure(1)
#for i in tqdm(np.arange(routes_lon.shape[0]-1)):
# for i in tqdm(range(100000)):
#     if np.abs(routes_lon[i] - routes_lon[i+1]) < 20:
#         plt.plot([routes_lon[i],routes_lon[i+1]],[routes_lat[i],routes_lat[i+1]],color='blue')
# plt.show(block=False)


#if np.abs(routes_lon[i] - routes_lon[i+1]) < 20:
num_routes = len(routes.keys())
#for i in tqdm(range(num_routes)):
for i in tqdm(range(3000)):
    route_num = list(routes.keys())[i]
    num_points = len(routes[route_num]['lat_lon'])
    for j in np.arange(num_points-1):
        pt0 = routes[route_num]['lat_lon'][j]
        pt1 = routes[route_num]['lat_lon'][j+1]
        if np.abs(pt1[1]-pt0[1]) < 20: #removes wrapping lines
            plt.plot([pt0[1],pt1[1]],[pt0[0],pt1[0]],color='blue')
#        plt.scatter(routes[route_num]['lat_lon'][i][1],routes[route_num]['lat_lon'][i][1],color='red')
    port_pt = ports[routes[route_num]['prev_port']]['latlon']
    plt.scatter(port_pt[1],port_pt[0],color='red',s=1)
plt.xlim([-180,180])
plt.ylim([-90,90])
plt.show(block=False)



