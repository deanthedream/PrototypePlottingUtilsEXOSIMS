#dynamicCompleteness
import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as random
import time
from astropy import constants as const
import astropy.units as u
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.planet_star_separation import planet_star_separation
import itertools
import datetime
import re

#### PLOT BOOL
plotBool = False
if plotBool == True:
    from plotProjectedEllipse import *
folder = './'
PPoutpath = './'

#### Randomly Generate Orbits
folder_load = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))
filename = 'HabEx_CKL2_PPKL2.json'
filename = 'WFIRSTcycle6core.json'
filename = 'HabEx_CSAG13_PPSAG13_compSubtype.json'
#filename = 'HabEx_CSAG13_PPSAG13_compSubtypeHighEccen.json'
scriptfile = os.path.join(folder_load,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
PPop = sim.PlanetPopulation
comp = sim.Completeness
TL = sim.TargetList
n = 10**5 #Dean's nice computer can go up to 10**8 what can atuin go up to?
inc, W, w = PPop.gen_angles(n,None)
W = W.to('rad').value
w = w.to('rad').value
#w correction caused in smin smax calcs
wReplacementInds = np.where(np.abs(w-1.5*np.pi)<1e-4)[0]
w[wReplacementInds] = w[wReplacementInds] - 0.001
wReplacementInds = np.where(np.abs(w-0.5*np.pi)<1e-4)[0]
w[wReplacementInds] = w[wReplacementInds] - 0.001
del wReplacementInds
inc = inc.to('rad').value
inc[np.where(inc>np.pi/2.)[0]] = np.pi - inc[np.where(inc>np.pi/2.)[0]]
sma, e, p, Rp = PPop.gen_plan_params(n)


#### Classify Planets
bini, binj, earthLike = comp.classifyPlanets(Rp, TL, np.arange(len(sma)), sma, e)
sma = sma.to('AU').value
####

#Separations
s_circle = np.ones(len(sma))
dmag = 25. #29.0
dmag_upper = 25. #29.0
IWA_HabEx = 0.045*u.arcsec #taken from a Habex Script in units of mas
IWA2=0.150*u.arcsec #Suggested by dmitry as analahous to WFIRST
OWA_HabEx = 6.*u.arcsec #from the HabEx Standards Team Final Report
s_inner = 10.*u.pc.to('AU')*IWA_HabEx.to('rad').value
s_outer = 10.*u.pc.to('AU')*OWA_HabEx.to('rad').value

#starMass
starMass = const.M_sun

periods = (2.*np.pi*np.sqrt((sma*u.AU)**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value

#Random time past periastron of first observation
tobs1 = np.random.rand(len(periods))*periods*u.year.to('day')

nus, planetIsVisibleBool = planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,plotBool, s_inner, s_outer, dmag_upper, dmag_lower=None) #Calculate planet-star nu edges and visible regions
ts = timeFromTrueAnomaly(nus,np.tile(periods,(18,1)).T*u.year.to('day'),np.tile(e,(18,1)).T) #Calculate the planet-star intersection edges
dt = ts[:,1:] - ts[:,:-1] #Calculate time region widths
maxIntTime = 0.
gtIntLimit = dt > maxIntTime #Create boolean array for inds
totalVisibleTimePerTarget = np.nansum(np.multiply(np.multiply(dt-maxIntTime,planetIsVisibleBool.astype('int')),gtIntLimit),axis=1) #We subtract the int time from the fraction of observable time
totalCompleteness = np.divide(totalVisibleTimePerTarget,periods*u.year.to('day')) # Fraction of time each planet is visible of its period

ts2 = ts[:,0:8] #cutting out all the nans
planetIsVisibleBool2 = planetIsVisibleBool[:,0:7] #cutting out all the nans

def dynamicCompleteness(ts2,planetIsVisibleBool2,tobs1,tpast_startTimes,ptypeBool=None):
    """
    TODO ADD SUBTYPE COMPLETENESS CAPABILITY
    Args:
    ptypeBool ():
        booleans indicating planet types. None means use all planets
    """
    if ptypeBool == None:
        ptypeBool = np.ones(len(tobs1))
    planetTypeBool = np.tile(ptypeBool,(7,1)).T
    planetIsVisibleBool2 = np.multiply(planetIsVisibleBool2,planetTypeBool) #Here we remove the planets that are not the desired type

    #Find all planets detectable at startTimes
    # startTime = 100. #startTime approach to planet visibility
    # startTimes = np.tile(np.mod(np.tile(startTime,(len(periods),1)),periods),(8,1)).T #startTime into properly sized array
    startTimes = np.tile(tobs1,(7,1)).T #startTime into properly sized array
    planetDetectedBools_times = np.multiply(ts2[:,:-1] < startTimes,np.multiply(ts2[:,1:] > startTimes,planetIsVisibleBool2)) #multiply time window bools by planetIsVisibleBool2. For revisit Completeness
    planetNotDetectedBools_times = np.logical_not(planetDetectedBools_times) #for dynamic completeness

    #tpast_startTimes = 50. #in days
    tobs2 = np.tile(np.mod(tobs1+np.tile(tpast_startTimes,(len(tobs1),1)).T,periods*u.year.to('day')),(7,1)).T
    planetDetectedBools2_times = np.multiply(ts2[:,:-1] < tobs2,np.multiply(ts2[:,1:] > tobs2,planetIsVisibleBool2)) #is the planet visible at this time segment in time 2?
    planetNotDetectedBools2_times = np.logical_not(planetDetectedBools2_times) #for dynamic completeness, the planet is not visible in this time segment at time 2

    planetDetectedBools = np.nansum(np.multiply(planetDetectedBools_times,planetDetectedBools2_times),axis=1) #each planet detected at time 1 and time 2 #planets detected and still in visible region
    planetNotDetectedBools = np.nansum(np.multiply(planetNotDetectedBools_times,planetDetectedBools2_times),axis=1) #each planet detected at time 1 and time 2 #planet not detected and now in visible region
    dynComp = np.sum(planetNotDetectedBools)/len(planetNotDetectedBools)
    revisitComp = np.sum(planetDetectedBools)/len(planetDetectedBools)

    return dynComp, revisitComp

timingStart = time.time()
trange = np.linspace(start=0.,stop=365.*13.,num=500)
dynComps = list()
revisitComps = list()
for k in np.arange(len(trange)):
    dynComp, revisitComp = dynamicCompleteness(ts2,planetIsVisibleBool2,tobs1,trange[k],None)
    dynComps.append(dynComp)
    revisitComps.append(revisitComp)
timingStop = time.time()
print('time: ' + str(timingStop-timingStart))

num=8883354345321844
plt.figure(num=num)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
plt.plot(trange,dynComps,color='blue')
plt.plot(trange,revisitComps,color='red')

plt.show(block=False)



dynCompDict = {}
for i,j in itertools.product(np.arange(np.min(bini),np.max(bini)+1), np.arange(np.min(binj),np.max(binj)+1)):
    print('(' + str(i) + ',' + str(j) + ')')
    dynCompDict[(i,j)] = dict()
    dynComps = list()
    revisitComps = list()
    barrayi = np.where(bini == i,1,0)
    barrayj = np.where(binj == j,1,0)
    for k in np.arange(len(trange)):
        dynComp, revisitComp = dynamicCompleteness(ts2,planetIsVisibleBool2,tobs1,trange[k],barrayi*barrayj)
        dynComps.append(dynComp)
        revisitComps.append(revisitComp)
    dynCompDict[(i,j)] = {'dynComp':dynComp,'revisitComps':revisitComps}


dynCompDict['all'] = dict()
for k in np.arange(len(trange)):
    dynComp, revisitComp = dynamicCompleteness(ts2,planetIsVisibleBool2,tobs1,trange[k],None)
    dynComps.append(dynComp)
    revisitComps.append(revisitComp)
dynCompDict['all'] = {'dynComp':dynComp,'revisitComps':revisitComps}
dynCompDict['earthLike'] = dict()
for k in np.arange(len(trange)):
    dynComp, revisitComp = dynamicCompleteness(ts2,planetIsVisibleBool2,tobs1,trange[k],earthLike.astype('int'))
    dynComps.append(dynComp)
    revisitComps.append(revisitComp)
dynCompDict['earthLike'] = {'dynComp':dynComp,'revisitComps':revisitComps}


num=8883354345321833
plt.figure(num=num)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
plt.plot(trange,dynComps,color='blue')
plt.plot(trange,revisitComps,color='red')

plt.show(block=False)

