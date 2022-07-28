
from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv
from sgp4.api import jday
from sgp4.api import Satrec
import os
import numpy as np

tle_filepath = '/home/dean/Documents/exosims/PrototypePlottingUtilsEXOSIMS/ConstellationScheduler'
tle_filename = 'globalstar.txt'


satOBJs = list()

#### Open space-track TLE file ###################################
with open(os.path.join(tle_filepath,tle_filename), 'r') as f:
    # read all TLE from file and remove final line caused by split
    lines = f.read().split('\n')[:-1] 

#### Parsing space-track TLE data ################################
assert np.mod(len(lines),3) == 0, 'the number of lines for each TLE is not 3'
numTLE = len(lines)/3 # there should be 1 TLE for every 3 lines

satData = {} # stores all parsed satellite data
satNums = list() # stores all satellite identification numbers
for i in np.arange(numTLE, dtype='int'):
    line = lines[i*3]
    #assert line[0] == '0', 'Error: Not a Name Line'
    #The first character on every name line must be 0    
    satName = line.replace(" ","")
    TLE1 = lines[i*3+1] # first line of TLE
    TLE2 = lines[i*3+2] # second line of TLE

    satOBJ = Satrec.twoline2rv(TLE1, TLE2)#, wgs72)
    satNums.append(satOBJ.satnum)
    satData[satOBJ.satnum] = satOBJs.append({"satName":satName,"satelliteOBJ":satOBJ,"TLE0":line,"TLE1":TLE1,"TLE2":TLE2})
#return satData, lines, numTLE

for j in np.arange(len(satOBJs)):
    jds = list()
    frs = list()
    for i in np.arange(7):
        jd, fr = jday(2024, 12, 8, 9, 4+i, 0)
        jds.append(jd)
        frs.append(fr)
    frs = np.asarray(frs)
    jds = np.asarray(jds)
    e, r, v = satOBJs[j]['satelliteOBJ'].sgp4_array(jds, frs) # e = error, r = position vector, v = speed vector

    satOBJs[j]['e'] = e
    satOBJs[j]['r'] = r
    satOBJs[j]['v'] = v


r_earth = 6371.0

def genRandomLatLon(num=1,lat_low=-np.pi/2.,lat_high=np.pi/2.,lon_low=0,lon_high=2.*np.pi):
    """ Generates a random latitude and longitude
    """
    lons = np.random.uniform(low=lon_low,high=lon_high,size=num)
    lats = np.arccos(2.*np.random.uniform(low=np.sin(lat_low),high=np.sin(lat_high),size=num)-1.)
    return lons, lats

num_locs = 600    
lons, lats = genRandomLatLon(num_locs,lat_low=np.pi/4.,lat_high=np.pi/2.,lon_low=180.*np.pi/180.,lon_high=320.*np.pi/180.)

def lonLats_to_xyz(r,lons,lats):
    """ Computes x,y,z position from lons and lats
    """
    return r*np.asarray([np.cos(lats)*np.sin(lons),np.cos(lats)*np.sin(lons),np.sin(lats)])

#Randomly generate locations
r_locs = lonLats_to_xyz(r_earth+200,lons,lats)




