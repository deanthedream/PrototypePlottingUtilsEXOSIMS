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
#number (''), trip_count, prev_port, next_port, lat, lon, frequency
routes = genfromtxt('routes.csv', delimiter=',')
routes_num = routes[0,1:]
routes_trip_count = routes[1,1:]
routes_prev_port = routes[2,1:]
routes_next_port = routes[3,1:]
routes_lat = routes[4,1:]
routes_lon = routes[5,1:]
routes_freq = routes[6,1:]
#del routes

#number (''), port name, index number, coordinates
#ports = genfromtxt('ports.csv', delimiter=',')
port_num = list()
port_names = list()
port_latlon = list()
import csv
with open('ports.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if not row[''] == '': #i.e. skip the first row
            port_num.append(row[''])
            port_names.append(row['PORT_NAME'])
            tmp_coords = row['coords'].replace('(','').replace(')','').split(',')
            port_latlon.append([float(tmp_coords[0]),float(tmp_coords[1])])


#number (''), trip_count, prev_port, next_port, distance, frequency
distances = genfromtxt('distances.csv', delimiter=',')
distances_num = np.asarray(distances[0,1:])

distances_prev_port = np.asarray(distances[2,1:])
distances_next_port = np.asarray(distances[3,1:])
distances_distance = np.asarray(distances[4,1:])
distances_frequency = np.asarray(distances[5,1:])
