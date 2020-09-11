# Testing data volume and speed of cubic spline and RBS interpolators

import numpy as np
from scipy.interpolate import CubicSpline as CubicSpline
from scipy.interpolate import RectBivariateSpline as RBS
from EXOSIMS.util.phaseFunctions import quasiLambertPhaseFunction
import sys
import psutil
import itertools
import time
from scipy.signal import argrelextrema

w = np.asarray(np.linspace(start=0.,stop=2.*np.pi,num=100))
inc = np.linspace(start=0.,stop=np.pi/2.,num=100)
v = np.linspace(start=0.,stop=2.*np.pi,num=100)
e = np.linspace(start=0.,stop=1.,num=100)


#beta = np.arccos(np.sin(inc)*np.sin(v+w))

#The Left hand side of the expression
#lhs = 10.**(-0.4*dMag)*a**2.*(1.+e**2.)**2./p/Rp**2.


def rhs(inc,v,w,e):
    """
    The right hand side containing the variables of the planet dmag expression 
    """
    beta = np.arccos(np.sin(inc)*np.sin(v+w))
    rhs = quasiLambertPhaseFunction(beta)*(e*np.cos(v)+1.)**2.
    return rhs

#### Calculate CubicSpline
#Get initial memory usage
initMemory = dict(psutil.virtual_memory()._asdict())['used']/(1024.0 ** 3.)
t0 = time.time()
#cbs = CubicSpline(v,rhs(inc[0],v,w[0],e[0]))

cbss = dict()
for i,j,k in itertools.product(np.arange(len(inc)),np.arange(len(w)),np.arange(len(e))):
    cbss[(i,j,k)] = CubicSpline(v,rhs(inc[i],v,w[j],e[k]))


finalMemory = dict(psutil.virtual_memory()._asdict())['used']/(1024.0 ** 3.)
t1 = time.time()
usedMemory = finalMemory - initMemory
usedTime = t1-t0

print('For CubicSpline')
print('Memory Used (GB): ' + str(usedMemory) + ' for ' + str(len(w)*len(inc)*len(e)) + ' points')
print('Memory Used Per cbs (MB/cbs): ' + str(usedMemory/(len(w)*len(inc)*len(e))*10.**3.))
print('Time Used (s): ' + str(usedTime))
print('Time Used Per cbs (): ' + str(usedTime/(len(w)*len(inc)*len(e))))
####

# #### Calculate InterpolationGrid
# #Get initial memory usage
# initMemory = dict(psutil.virtual_memory()._asdict())['used']/(1024.0 ** 3.)
# t0 = time.time()
# #cbs = CubicSpline(v,rhs(inc[0],v,w[0],e[0]))

# interpArray = np.zeros((len(w),len(inc),len(e),len(v)))
# for i,j,k,l in itertools.product(np.arange(len(inc)),np.arange(len(w)),np.arange(len(e)),np.arange(len(v))):
#     interpArray[i,j,k,l] = rhs(inc[i],v[l],w[j],e[k])


# finalMemory = dict(psutil.virtual_memory()._asdict())['used']/(1024.0 ** 3.)
# t1 = time.time()
# usedMemory = finalMemory - initMemory
# usedTime = t1-t0

# print('For 4D InterpolationGrid Data')
# print('Memory Used (GB): ' + str(usedMemory) + ' for ' + str(len(w)*len(inc)*len(e)) + ' points')
# print('Memory Used Per cbs equivalent (GB/cbs): ' + str(usedMemory/(len(w)*len(inc)*len(e))))
# print('Time Used (s): ' + str(usedTime))
# print('Time Used Per cbs equivalent (): ' + str(usedTime/(len(w)*len(inc)*len(e))))
# ####

#### Calculate RBS
#Get initial memory usage
initMemory = dict(psutil.virtual_memory()._asdict())['used']/(1024.0 ** 3.)
t0 = time.time()
#cbs = CubicSpline(v,rhs(inc[0],v,w[0],e[0]))

rbsDict = dict()
extremaDict = dict()
extremaDict['min'] = dict() 
extremaDict['lmin'] = dict() 
extremaDict['lmax'] = dict() 
extremaDict['max'] = dict() 
for i,j in itertools.product(np.arange(len(inc)),np.arange(len(w))):
    #rhsArray = np.zeros((len(e),len(yarray)))
    #for k,l in itertools.product(np.arange(len(v)),np.arange(len(e))):
    #    rhsArray[k,l] = rhs(inc[i],v,w[j],e[l])
    rhsArray = np.zeros((len(e),len(v)))
    #for k,l in itertools.product(np.arange(len(e)),np.arange(len(v))):
    for k in np.arange(len(e)):
        rhsArray[k] = rhs(inc[i],v,w[j],e[k])
        lmaxInd = None
        lminInd = None
        arglmax = argrelextrema(rhsArray[k], np.greater)[0] #gives local max
        if len(arglmax) == 1:
            maxInd = np.argmax(rhsArray[k])
            lmax = np.nan
        elif len(arglmax) > 1: #there is more than one local maximum
            maxInd = np.argmax(rhsArray[k]) #ind where maximum occurs
            lmaxInd = list(arglmax).remove(maxInd)
            lmax = rhsArray[lmaxInd]
        elif len(arglmax) == 0: #all values are the same
            lmax = np.nan
            maxInd = 0
        else:
            print(error)
        arglmin = argrelextrema(rhsArray[k], np.less)[0] #gives local min
        if len(arglmin) == 1:
            minInd = np.argmin(rhsArray[k])
            lmin = np.nan
        elif len(arglmin) > 1: #there is more than one local minimum
            minInd = np.argmin(rhsArray[k]) #ind where minimum occurs
            lminInd = list(arglmin).remove(minInd)
            lmin = rhsArray[lminInd]
        elif len(arglmin) == 0: #all values are the same
            lmin = np.nan
            minInd = 0
        else:
            print(error)
        extremaDict['min'][(i,j,k)] = np.min(rhsArray[maxInd])
        extremaDict['lmin'][(i,j,k)] = lmin
        extremaDict['lmax'][(i,j,k)] = lmax
        extremaDict['max'][(i,j,k)] = np.max(rhsArray[minInd])
    rbsDict[(i,j)] = RBS(e,v,rhsArray)

    #Finds min, max, lmin, lmax of RBS vs nu at specific e
    
    


finalMemory = dict(psutil.virtual_memory()._asdict())['used']/(1024.0 ** 3.)
t1 = time.time()
usedMemory = finalMemory - initMemory
usedTime = t1-t0

print('For RBS')
print('Memory Used (GB): ' + str(usedMemory) + ' for ' + str(len(w)*len(inc)*len(e)) + ' points')
print('Memory Used Per vbs equivalent (MB/cbs): ' + str(usedMemory/(len(w)*len(inc)*len(e)*10.**3.)))
print('Time Used (s): ' + str(usedTime))
print('Time Used Per cbs equivalent (): ' + str(usedTime/(len(w)*len(inc)*len(e))))
####
