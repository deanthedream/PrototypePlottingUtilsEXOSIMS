#Two Det Priors
"""
Purpose:
Given any two (s1,dmag1), (s2,dmag2), and dtheta pairs (and time between images???)
Find the probability the planet detected is of the Given Type
"""
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

# DELETE #Random time past periastron of first observation
# DELETE tobs1 = np.random.rand(len(periods))*periods*u.year.to('day')

#### Finding Test Planet
plotBool = False
ind=69
sma[ind] = 1.7354234901517238 
e[ind] = 0.3034481574237903 
inc[ind] = 0.7234687443868556 
w[ind] = 1.0943331760583406 
W[ind] = 0.19739778259085852 
p[ind] = 0.6714129374646385 
Rp[ind] = 0.9399498757082513*u.earthRad
nurange = np.linspace(start=0.,stop=2.*np.pi,num=300)

r=(sma[ind]*(1.-e[ind]**2.))/(1.+e[ind]*np.cos(nurange))
X = r*(np.cos(W[ind])* np.cos(w[ind] + nurange) - np.sin(W[ind])*np.sin(w[ind] + nurange)*np.cos(inc[ind]))
Y = r*(np.sin(W[ind])* np.cos(w[ind] + nurange) + np.cos(W[ind])*np.sin(w[ind] + nurange)*np.cos(inc[ind]))
Z = r*np.sin(inc[ind])* np.sin(w[ind] + nurange)
#Calculate dmag and s for all midpoints
Phi = (1.+np.sin(inc[ind])*np.sin(nurange+w[ind]))**2./4.
d = sma[ind]*u.AU*(1.-e[ind]**2.)/(e[ind]*np.cos(nurange)+1.)
dmags = deltaMag(p[ind],Rp[ind].to('AU'),d,Phi) #calculate dmag of the specified x-value
ss = planet_star_separation(sma[ind],e[ind],nurange,w[ind],inc[ind])
print('sma: ' + str(sma[ind]) + ' e: ' + str(e[ind]) + ' i: ' + str(inc[ind]) + ' w: ' + str(w[ind]) + ' W: ' + str(W[ind]) + ' p: ' + str(p[ind]) + ' Rp: ' + str(Rp[ind]))

num=1
plt.figure(num=num)
plt.plot(X,Y,color='black')
plt.xlabel('X')
plt.ylabel('Y')
plt.show(block=False)
num=2
plt.figure(num=num)
plt.plot(nurange,dmags,color='blue')
plt.ylabel('dmag')
plt.show(block=False)
num=3
plt.figure(num=num)
plt.plot(nurange,ss,color='red')
plt.ylabel('s')
plt.show(block=False)





# Instrument Uncertainty
uncertainty_dmag = 0.01 #HabEx requirement is 1%
uncertainty_s = 5.*u.mas.to('rad')*10.*u.pc.to('AU')

#Detection one #TODO make these of a specific planet
#nurange[50] used for determing location of first detection
tpastPeriastron1 = timeFromTrueAnomaly(nurange[50],periods[ind]*u.year.to('day'),e[ind]) #Calculate the planet-star intersection edges
sep1 = ss[50] #0.7 #AU
dmag1 = dmags[50] #23. #Luminosity Scaled Planet-star Difference in Magnitude, ' + r'$\Delta\mathrm{mag}-2.5\log_{10}(L)$'
nus1, planetIsVisibleBool1 = planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,plotBool, sep1-uncertainty_s, sep1+uncertainty_s, dmag1*(1.+uncertainty_dmag), dmag1*(1.-uncertainty_dmag)) #Calculate planet-star nu edges and visible regions
ts1 = timeFromTrueAnomaly(nus1,np.tile(periods,(18,1)).T*u.year.to('day'),np.tile(e,(18,1)).T) #Calculate the planet-star intersection edges
# dt = ts[:,1:] - ts[:,:-1] #Calculate time region widths
# maxIntTime = 0.
# gtIntLimit = dt > maxIntTime #Create boolean array for inds
# totalVisibleTimePerTarget = np.nansum(np.multiply(np.multiply(dt-maxIntTime,planetIsVisibleBool.astype('int')),gtIntLimit),axis=1) #We subtract the int time from the fraction of observable time
# totalCompleteness = np.divide(totalVisibleTimePerTarget,periods*u.year.to('day')) # Fraction of time each planet is visible of its period

# ts2 = ts[:,0:8] #cutting out all the nans
# planetIsVisibleBool2 = planetIsVisibleBool[:,0:7] #cutting out all the nans

numPlanetsInRegion1 = np.sum(np.any(planetIsVisibleBool1,axis=1))

#Dection 2
#nurange[75] #used for determining location of second detection
tpastPeriastron2 = timeFromTrueAnomaly(nurange[75],periods[ind]*u.year.to('day'),e[ind]) #Calculate the planet-star intersection edges
sep2 = ss[75] #0.7 #AU
dmag2 = dmags[75] #23. #Luminosity Scaled Planet-star Difference in Magnitude, ' + r'$\Delta\mathrm{mag}-2.5\log_{10}(L)$'
nus2, planetIsVisibleBool2 = planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,plotBool, sep2-uncertainty_s, sep2+uncertainty_s, dmag2*(1.+uncertainty_dmag), dmag2*(1.-uncertainty_dmag)) #Calculate planet-star nu edges and visible regions
ts2 = timeFromTrueAnomaly(nus2,np.tile(periods,(18,1)).T*u.year.to('day'),np.tile(e,(18,1)).T) #Calculate the planet-star intersection edges

numPlanetsInRegion2 = np.sum(np.any(planetIsVisibleBool2,axis=1))

#### Find Planet Inds With Both
detectableByBothBoolArray = np.any(planetIsVisibleBool2,axis=1)*np.any(planetIsVisibleBool1,axis=1)
numDetectableByBothArray = np.sum(detectableByBothBoolArray)
detectableByBothInds = np.where(detectableByBothBoolArray)[0] #inds of planets that are detectable at time 1 and time 2
#seems to successfully reduce numplanets by 1/100
actualPlanetTimeDifference = tpastPeriastron2-tpastPeriastron1 #the time that passed between image1 and image2

#Find Number of Times Planet Is Visible #TODO add error checking for if 3 visible regions doesn't exist
numberOfVisibleRegionsPerPlanets1 = np.sum(planetIsVisibleBool1[detectableByBothInds],axis=1)
indsWith1_1 = detectableByBothInds[np.where(numberOfVisibleRegionsPerPlanets1==1)[0]] #uses detectableByBothInds[np.where] format to ensure indsWith1_1 are inds of planetIsVisibleBool1
indsWith1_2 = detectableByBothInds[np.where(numberOfVisibleRegionsPerPlanets1==2)[0]]
indsWith1_3 = detectableByBothInds[np.where(numberOfVisibleRegionsPerPlanets1==3)[0]]
indsWith1_4 = detectableByBothInds[np.where(numberOfVisibleRegionsPerPlanets1==4)[0]]
numberOfVisibleRegionsPerPlanets2 = np.sum(planetIsVisibleBool2[detectableByBothInds],axis=1)
indsWith2_1 = detectableByBothInds[np.where(numberOfVisibleRegionsPerPlanets2==1)[0]]
indsWith2_2 = detectableByBothInds[np.where(numberOfVisibleRegionsPerPlanets2==2)[0]]
indsWith2_3 = detectableByBothInds[np.where(numberOfVisibleRegionsPerPlanets2==3)[0]]
indsWith2_4 = detectableByBothInds[np.where(numberOfVisibleRegionsPerPlanets2==4)[0]]

np.histogram(numberOfVisibleRegionsPerPlanets1)
np.histogram(numberOfVisibleRegionsPerPlanets2)

#Subdivide sets of inds where Image 1 has 1 visible region 
setNumVisTimes = dict()
setNumVisTimes[(1,1)] = dict() #image 1 has 1 visible region, image 2 has 1 visible region
setNumVisTimes[(1,2)] = dict() #image 1 has 1 visible region, image 2 has 2 visible region
setNumVisTimes[(1,3)] = dict() #image 1 has 1 visible region, image 2 has 3 visible region
setNumVisTimes[(1,4)] = dict() #image 1 has 1 visible region, image 2 has 4 visible region
setNumVisTimes[(2,1)] = dict() #image 1 has 2 visible region, image 2 has 1 visible region
setNumVisTimes[(2,2)] = dict() #image 1 has 2 visible region, image 2 has 2 visible region
setNumVisTimes[(2,3)] = dict() #image 1 has 2 visible region, image 2 has 3 visible region
setNumVisTimes[(2,4)] = dict() #image 1 has 2 visible region, image 2 has 4 visible region
setNumVisTimes[(3,1)] = dict() #image 1 has 3 visible region, image 2 has 1 visible region
setNumVisTimes[(3,2)] = dict() #image 1 has 3 visible region, image 2 has 2 visible region
setNumVisTimes[(3,3)] = dict() #image 1 has 3 visible region, image 2 has 3 visible region
setNumVisTimes[(3,4)] = dict() #image 1 has 3 visible region, image 2 has 4 visible region
setNumVisTimes[(4,1)] = dict() #image 1 has 4 visible region, image 2 has 1 visible region
setNumVisTimes[(4,2)] = dict() #image 1 has 4 visible region, image 2 has 2 visible region
setNumVisTimes[(4,3)] = dict() #image 1 has 4 visible region, image 2 has 3 visible region
setNumVisTimes[(4,4)] = dict() #image 1 has 4 visible region, image 2 has 4 visible region

setNumVisTimes[(1,1)]['inds'] = list(set(indsWith2_1).intersection(set(indsWith1_1))) #image 1 has 1 visible region, image 2 has 1 visible region
setNumVisTimes[(1,2)]['inds'] = list(set(indsWith2_2).intersection(set(indsWith1_1))) #image 1 has 1 visible region, image 2 has 2 visible region
setNumVisTimes[(1,3)]['inds'] = list(set(indsWith2_3).intersection(set(indsWith1_1))) #image 1 has 1 visible region, image 2 has 3 visible region
setNumVisTimes[(1,4)]['inds'] = list(set(indsWith2_4).intersection(set(indsWith1_1))) #image 1 has 1 visible region, image 2 has 4 visible region

setNumVisTimes[(2,1)]['inds'] = list(set(indsWith2_1).intersection(set(indsWith1_2))) #image 1 has 2 visible region, image 2 has 1 visible region
setNumVisTimes[(2,2)]['inds'] = list(set(indsWith2_2).intersection(set(indsWith1_2))) #image 1 has 2 visible region, image 2 has 2 visible region
setNumVisTimes[(2,3)]['inds'] = list(set(indsWith2_3).intersection(set(indsWith1_2))) #image 1 has 2 visible region, image 2 has 3 visible region
setNumVisTimes[(2,4)]['inds'] = list(set(indsWith2_4).intersection(set(indsWith1_2))) #image 1 has 2 visible region, image 2 has 4 visible region

setNumVisTimes[(3,1)]['inds'] = list(set(indsWith2_1).intersection(set(indsWith1_3))) #image 1 has 3 visible region, image 2 has 1 visible region
setNumVisTimes[(3,2)]['inds'] = list(set(indsWith2_2).intersection(set(indsWith1_3))) #image 1 has 3 visible region, image 2 has 2 visible region
setNumVisTimes[(3,3)]['inds'] = list(set(indsWith2_3).intersection(set(indsWith1_3))) #image 1 has 3 visible region, image 2 has 3 visible region
setNumVisTimes[(3,4)]['inds'] = list(set(indsWith2_4).intersection(set(indsWith1_3))) #image 1 has 3 visible region, image 2 has 4 visible region

setNumVisTimes[(4,1)]['inds'] = list(set(indsWith2_1).intersection(set(indsWith1_4))) #image 1 has 4 visible region, image 2 has 1 visible region
setNumVisTimes[(4,2)]['inds'] = list(set(indsWith2_2).intersection(set(indsWith1_4))) #image 1 has 4 visible region, image 2 has 2 visible region
setNumVisTimes[(4,3)]['inds'] = list(set(indsWith2_3).intersection(set(indsWith1_4))) #image 1 has 4 visible region, image 2 has 3 visible region
setNumVisTimes[(4,4)]['inds'] = list(set(indsWith2_4).intersection(set(indsWith1_4))) #image 1 has 4 visible region, image 2 has 4 visible region

print(len(setNumVisTimes[(1,1)]['inds'])+len(setNumVisTimes[(1,2)]['inds'])+len(setNumVisTimes[(1,3)]['inds'])+len(setNumVisTimes[(1,4)]['inds'])+\
    len(setNumVisTimes[(2,1)]['inds'])+len(setNumVisTimes[(2,2)]['inds'])+len(setNumVisTimes[(2,3)]['inds'])+len(setNumVisTimes[(2,4)]['inds'])+\
    len(setNumVisTimes[(3,1)]['inds'])+len(setNumVisTimes[(3,2)]['inds'])+len(setNumVisTimes[(3,3)]['inds'])+len(setNumVisTimes[(3,4)]['inds'])+\
    len(setNumVisTimes[(4,1)]['inds'])+len(setNumVisTimes[(4,2)]['inds'])+len(setNumVisTimes[(4,3)]['inds'])+len(setNumVisTimes[(4,4)]['inds']))
assert len(setNumVisTimes[(1,1)]['inds'])+len(setNumVisTimes[(1,2)]['inds'])+len(setNumVisTimes[(1,3)]['inds'])+len(setNumVisTimes[(1,4)]['inds'])+\
    len(setNumVisTimes[(2,1)]['inds'])+len(setNumVisTimes[(2,2)]['inds'])+len(setNumVisTimes[(2,3)]['inds'])+len(setNumVisTimes[(2,4)]['inds'])+\
    len(setNumVisTimes[(3,1)]['inds'])+len(setNumVisTimes[(3,2)]['inds'])+len(setNumVisTimes[(3,3)]['inds'])+len(setNumVisTimes[(3,4)]['inds'])+\
    len(setNumVisTimes[(4,1)]['inds'])+len(setNumVisTimes[(4,2)]['inds'])+len(setNumVisTimes[(4,3)]['inds'])+len(setNumVisTimes[(4,4)]['inds']) == len(detectableByBothInds),\
    'Whoops, missing a case where num visible regions > 3' #error checking number of inds

print('Determing if Times between s1,dmag1 and s2,dmag2 are appropriate')
#Calculate All Intersection Times For NumVisibleTimes (i,j)
timeTolerance = 3. #random tolerance on the time between two observations in days #TODO find a better number for this. Should be integration time x 2 since two detections must occur
planetsInVisibleRegionsInTimeWindow = list()
for (i,j) in [(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3),(2,4),(3,1),(3,2),(3,3),(3,4),(4,1)]: #iterate over sets of inds of intersections
    # i indicates the number of visible regions in image 1
    # j indicates the number of visible regions in image 2
    for planetj in np.arange(len(setNumVisTimes[(i,j)]['inds'])): #iterate over planet
        indsOfVisibleRegionsk = np.where(planetIsVisibleBool1[setNumVisTimes[(i,j)]['inds'][planetj]])[0] #should give us a new matrix with shape len(setNumVisTimes[(1,1)]['inds']), 17
        indsOfVisibleRegionsl = np.where(planetIsVisibleBool2[setNumVisTimes[(i,j)]['inds'][planetj]])[0]

        #Iterate over visible regions of image 1 and image 2, pick out smallest dt, pick out largest dt
        smallestdt = 10000000. #rediculously large number
        largestdt = 0.
        for k in indsOfVisibleRegionsk: #iterate over image 1 visible regions
            regionStart1 = ts1[setNumVisTimes[(i,j)]['inds'][planetj],k]
            regionEnd1 = ts1[setNumVisTimes[(i,j)]['inds'][planetj],k+1]
            #taverage1 = (regionStart1+regionEnd1)/2. #average time of time window
            for l in indsOfVisibleRegionsl: #iterate over image 2 visible regions
                regionStart2 = ts2[setNumVisTimes[(i,j)]['inds'][planetj],k]
                regionEnd2 = ts2[setNumVisTimes[(i,j)]['inds'][planetj],k+1]
                #taverage2 = (regionStart2+regionEnd2)/2. #average time of time window

                #Calculate largest and smallest visibility window
                dt1 = np.abs(regionEnd2-regionStart1) #time difference between starting region 1 and ending region 2
                dt2 = np.abs(regionEnd1-regionStart2) #time difference between starting region 2 and ending region 1
                smaller = np.min([dt1,dt2])
                larger = np.max([dt1,dt2])
                #dtaverage = taverage2-taverage1
                # if dtaverage < timeTolerance: #if the planet is in both windows within the allowed time tolerance
                #     planetsInVisibleRegionsInTimeWindow.append((i,j,planetj,k,l))
                if actualPlanetTimeDifference-timeTolerance < larger and actualPlanetTimeDifference+timeTolerance > smaller:
                    planetsInVisibleRegionsInTimeWindow.append((i,j,setNumVisTimes[(i,j)]['inds'][planetj],k,l))
                # if np.abs(np.abs(regionEnd2-regionStart1) - actualPlanetTimeDifference) < timeTolerance or\
                #     np.abs(np.abs(regionEnd1-regionStart2-) - actualPlanetTimeDifference) < timeTolerance:

#dts = ts2-ts1
print(len(planetsInVisibleRegionsInTimeWindow))

#### Check if (s1,dmag1) and (s2,dmag2) uncertainty boxes are overlapping. If so, add planets that are always in both regions
s1_upper = sep1+uncertainty_s
s1_lower = sep1-uncertainty_s
s2_upper = sep2+uncertainty_s
s2_lower = sep2-uncertainty_s
dmag1_upper = dmag1*(1.+uncertainty_dmag)
dmag1_lower = dmag1*(1.-uncertainty_dmag)
dmag2_upper = dmag2*(1.+uncertainty_dmag)
dmag2_lower = dmag2*(1.-uncertainty_dmag)

# Calculate planet extrema
mindmag, maxdmag, dmaglminAll, dmaglmaxAll, indsWith2, indsWith4, nuMinDmag, nuMaxDmag, nulminAll, nulmaxAll = calc_planet_dmagmin_dmagmax(e,inc,w,a,p,Rp)

#case 1
if (s2_lower < s1_upper) and (s2_lower > s1_lower) and (dmag2_upper < dmag1_upper) and (dmag2_upper > dmag1_lower):
    #find planets where smin > s2_lower, smax < s1_upper, dmag_max < dmag2_upper, dmag_min > dmag1_lower
#case 2
if (s2_lower < s1_upper) and (s2_lower > s1_lower) and (dmag2_lower < dmag1_upper) and (dmag2_lower > dmag1_lower):
    #find planets where smin > s2_lower, smax < s1_upper, dmag_max < s1_upper, dmag_min > dmag2_lower
#case 3
if (s2_upper < s1_upper) and (s2_upper > s1_lower) and (dmag2_lower < dmag1_upper) and (dmag2_lower > dmag1_lower):
    #find planets where smin > s1_lower, smax < s2_upper, dmag_max < s1_upper, dmag_min > dmag2_lower
#case 4
if (s2_upper < s1_upper) and (s2_upper > s1_lower) and (dmag2_upper < dmag1_upper) and (dmag2_upper > dmag1_lower):
    #find planets where smin > s1_lower, smax < s2_upper, dmag_max < dmag2_upper, dmag_min > dmag1_lower



