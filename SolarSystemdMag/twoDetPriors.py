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
import matplotlib.gridspec as gridspec

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
n = 4.*10**5 #Dean's nice computer can go up to 10**8 what can atuin go up to?
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
#inc[np.where(inc>np.pi/2.)[0]] = np.pi - inc[np.where(inc>np.pi/2.)[0]]
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
p[ind] = 0.34 #0.6714129374646385 
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
thetas = np.arctan2(Y,X) #angle of planet position from X-axis
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
theta1 = thetas[50]
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
theta2 = thetas[75]
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

#DELETE
#np.histogram(numberOfVisibleRegionsPerPlanets1)
#np.histogram(numberOfVisibleRegionsPerPlanets2)

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

print("Number of Planets Visible In Time Window: " + str(len(planetsInVisibleRegionsInTimeWindow)))
#TODO: Time sign matters between the first and second observation.
#We may be able to keep the planets within the time window if we change the planet parameter by pi or something.

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
mindmag, maxdmag, dmaglminAll, dmaglmaxAll, indsWith2, indsWith4, nuMinDmag, nuMaxDmag, nulminAll, nulmaxAll = calc_planet_dmagmin_dmagmax(e,inc,w,sma*u.AU,p,Rp)
nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds = calc_planet_sep_extrema(sma,e,W,w,inc)

indsCase1 = []
indsCase2 = []
indsCase3 = []
indsCase4 = []
#case 1 #det 1 up and left of det 2
if (s2_lower < s1_upper) and (s2_lower > s1_lower) and (dmag2_upper < dmag1_upper) and (dmag2_upper > dmag1_lower):
    #find planets where minSep > s2_lower, maxSep < s1_upper, maxdmag < dmag2_upper, mindmag > dmag1_lower
    indsCase1 = np.where((minSep > s2_lower)*(maxSep < s1_upper)*(maxdmag < dmag2_upper)*(mindmag > dmag1_lower))[0]
#case 2 #det 1 down and left of det 2
if (s2_lower < s1_upper) and (s2_lower > s1_lower) and (dmag2_lower < dmag1_upper) and (dmag2_lower > dmag1_lower):
    #find planets where minSep > s2_lower, maxSep < s1_upper, maxdmag < s1_upper, mindmag > dmag2_lower
    indsCase2 = np.where((minSep > s2_lower)*(maxSep < s1_upper)*(maxdmag < s1_upper)*(mindmag > dmag2_lower))[0]
#case 3 #det 1 down and right of det 2
if (s2_upper < s1_upper) and (s2_upper > s1_lower) and (dmag2_lower < dmag1_upper) and (dmag2_lower > dmag1_lower):
    #find planets where minSep > s1_lower, maxSep < s2_upper, maxdmag < s1_upper, mindmag > dmag2_lower
    indsCase3 = np.where((minSep > s1_lower)*(maxSep < s2_upper)*(maxdmag < s1_upper)*(mindmag > dmag2_lower))[0]
#case 4 #det 1 up and right of det 2
if (s2_upper < s1_upper) and (s2_upper > s1_lower) and (dmag2_upper < dmag1_upper) and (dmag2_upper > dmag1_lower):
    #find planets where minSep > s1_lower, maxSep < s2_upper, maxdmag < dmag2_upper, mindmag > dmag1_lower
    indsCase4 = np.where((minSep > s1_lower)*(maxSep < s2_upper)*(maxdmag < dmag2_upper)*(mindmag > dmag1_lower))[0]
print(str(len(indsCase1)) + " cases where det 1 up and left of det 2\n" +\
    str(len(indsCase2)) + " cases where det 1 down and left of det 2\n" +\
    str(len(indsCase3)) + " cases where det 1 down and right of det 2\n" +\
    str(len(indsCase4)) + " cases where det 1 up and right of det 2")
#TODO add these cases to the stack of inds in planetsInVisibleRegionsInTimeWindow

def calc_planetAngularXYPosition_FromXaxis(sma,e,w,W,inc,nu):
    """ Calculate the angular position of the planet from the X-axis at nu
    """
    r=(sma*(1.-e**2.))/(1.+e*np.cos(nu))
    X = r*(np.cos(W)* np.cos(w + nu) - np.sin(W)*np.sin(w + nu)*np.cos(inc))
    Y = r*(np.sin(W)* np.cos(w + nu) + np.cos(W)*np.sin(w + nu)*np.cos(inc))
    Z = r*np.sin(inc)* np.sin(w + nu)
    thetas = np.arctan2(Y,X) #angle of planet position from X-axis
    return thetas

#TODO left off here. Check for t
thetas1 = calc_planetAngularXYPosition_FromXaxis(sma[detectableByBothInds],e[detectableByBothInds],w[detectableByBothInds],W[detectableByBothInds],inc[detectableByBothInds],nus1[detectableByBothInds])
thetas2 = calc_planetAngularXYPosition_FromXaxis(sma[detectableByBothInds],e[detectableByBothInds],w[detectableByBothInds],W[detectableByBothInds],inc[detectableByBothInds],nus2[detectableByBothInds])
#Calc all dthetas of planets in pop.
#filter ones matching dTheta below

#Delta Theta reduction
actualDeltaTheta = theta2-theta1 #the change in theta observed
dTheta_1 = (theta2-np.abs(np.arctan2(uncertainty_s,sep2))) - (theta1+np.abs(np.arctan2(uncertainty_s,sep1))) #could be largest or smallest
dTheta_2 = (theta2+np.abs(np.arctan2(uncertainty_s,sep2))) - (theta1-np.abs(np.arctan2(uncertainty_s,sep1))) #could be largest of smallest
deltaTheta_min = np.min([dTheta_1,dTheta_2]) #minimum of range
delteTheta_max = np.max([dTheta_1,dTheta_2]) #maximum of range


#TODO: create function to calculate theta of each planet from X-axis given nu. Calculate these and use to find thetas of planets

#planetsInVisibleRegionsInTimeWindow
planetsInVisibleRegionsInTimeWindowInds = [planetsInVisibleRegionsInTimeWindow[i][2] for i in np.arange(len(planetsInVisibleRegionsInTimeWindow))]


# num=121321222112111
# plt.figure(num=num)
# plt.rc('axes',linewidth=2)
# plt.rc('lines',linewidth=2)
# plt.rcParams['axes.linewidth']=2
# plt.rc('font',weight='bold')
# plt.hist(sma,color='black',alpha=0.3,density=True,label='Pop.')
# plt.hist(sma[planetsInVisibleRegionsInTimeWindowInds],color='purple',alpha=0.3,density=True,label='Sub-pop.')
# plt.plot([sma[ind],sma[ind]],[0,1],color='black')
# plt.xlabel('Semi-major axis, in AU',weight='bold')
# plt.ylabel('Frequency',weight='bold')
# plt.legend()
# plt.yscale('log')
# plt.show(block=False)

# num=121321222112222
# plt.figure(num=num)
# plt.rc('axes',linewidth=2)
# plt.rc('lines',linewidth=2)
# plt.rcParams['axes.linewidth']=2
# plt.rc('font',weight='bold')
# plt.hist(e,color='black',alpha=0.3,density=True,label='Pop.')
# plt.hist(e[planetsInVisibleRegionsInTimeWindowInds],color='purple',alpha=0.3,density=True,label='Sub-pop.')
# plt.plot([e[ind],e[ind]],[0,1],color='black')
# plt.xlabel('Eccentricity',weight='bold')
# plt.ylabel('Frequency',weight='bold')
# plt.legend()
# plt.yscale('log')
# plt.show(block=False)

# num=121321222112333
# plt.figure(num=num)
# plt.rc('axes',linewidth=2)
# plt.rc('lines',linewidth=2)
# plt.rcParams['axes.linewidth']=2
# plt.rc('font',weight='bold')
# plt.hist(inc,color='black',alpha=0.3,density=True,label='Pop.')
# plt.hist(inc[planetsInVisibleRegionsInTimeWindowInds],color='purple',alpha=0.3,density=True,label='Sub-pop.')
# plt.plot([inc[ind],inc[ind]],[0,1],color='black')
# plt.xlabel('Inclination, in rad',weight='bold')
# plt.ylabel('Frequency',weight='bold')
# plt.legend()
# plt.yscale('log')
# plt.show(block=False)


# num=121321222112444
# plt.figure(num=num)
# plt.rc('axes',linewidth=2)
# plt.rc('lines',linewidth=2)
# plt.rcParams['axes.linewidth']=2
# plt.rc('font',weight='bold')
# plt.hist(w,color='black',alpha=0.3,density=True,label='Pop.')
# plt.hist(w[planetsInVisibleRegionsInTimeWindowInds],color='purple',alpha=0.3,density=True,label='Sub-pop.')
# plt.plot([w[ind],w[ind]],[0,1],color='black')
# plt.xlabel('Argument of periapsis, in rad',weight='bold')
# plt.ylabel('Frequency',weight='bold')
# plt.legend()
# plt.yscale('log')
# plt.show(block=False)

# num=121321222112555
# plt.figure(num=num)
# plt.rc('axes',linewidth=2)
# plt.rc('lines',linewidth=2)
# plt.rcParams['axes.linewidth']=2
# plt.rc('font',weight='bold')
# plt.hist(W,color='black',alpha=0.3,density=True,label='Pop.')
# plt.hist(W[planetsInVisibleRegionsInTimeWindowInds],color='purple',alpha=0.3,density=True,label='Sub-pop.')
# plt.plot([W[ind],W[ind]],[0,1],color='black')
# plt.xlabel('Longitude of the Ascending Node, in rad',weight='bold')
# plt.ylabel('Frequency',weight='bold')
# plt.legend()
# plt.yscale('log')
# plt.show(block=False)

# num=121321222112666
# plt.figure(num=num)
# plt.rc('axes',linewidth=2)
# plt.rc('lines',linewidth=2)
# plt.rcParams['axes.linewidth']=2
# plt.rc('font',weight='bold')
# plt.hist(p,color='black',alpha=0.3,density=True,label='Pop.')
# plt.hist(p[planetsInVisibleRegionsInTimeWindowInds],color='purple',alpha=0.3,density=True,label='Sub-pop.')
# plt.plot([p[ind],p[ind]],[0,1],color='black')
# plt.xlabel('Planet Albedo',weight='bold')
# plt.ylabel('Frequency',weight='bold')
# plt.legend()
# plt.yscale('log')
# plt.show(block=False)

# num=121321222112777
# plt.figure(num=num)
# plt.rc('axes',linewidth=2)
# plt.rc('lines',linewidth=2)
# plt.rcParams['axes.linewidth']=2
# plt.rc('font',weight='bold')
# plt.hist(Rp.value,color='black',alpha=0.3,density=True,label='Pop.')
# plt.hist(Rp[planetsInVisibleRegionsInTimeWindowInds].value,color='purple',alpha=0.3,density=True,label='Sub-pop.')
# plt.plot([Rp[ind].value,Rp[ind].value],[0,1],color='black')
# plt.xlabel('Planet Radius, in Earth Raddius',weight='bold')
# plt.ylabel('Frequency',weight='bold')
# plt.legend()
# plt.yscale('log')
# plt.show(block=False)

# num=121321222112888
# plt.figure(num=num)
# plt.rc('axes',linewidth=2)
# plt.rc('lines',linewidth=2)
# plt.rcParams['axes.linewidth']=2
# plt.rc('font',weight='bold')
# plt.hist(np.multiply(p,Rp.value),color='black',alpha=0.3,density=True,label='Pop.')
# plt.hist(np.multiply(p[planetsInVisibleRegionsInTimeWindowInds],Rp[planetsInVisibleRegionsInTimeWindowInds].value),color='purple',alpha=0.3,density=True,label='Sub-pop.')
# plt.plot([p[ind]*Rp[ind].value,p[ind]*Rp[ind].value],[0,1],color='black')
# plt.xlabel('p*Rp',weight='bold')
# plt.ylabel('Frequency',weight='bold')
# plt.legend()
# plt.yscale('log')
# plt.show(block=False)


#### Gridspec of above plots
num=8675309
plt.close(num)
fig = plt.figure(num=num,constrained_layout=True,figsize=(8,12))
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
gs = gridspec.GridSpec(ncols=2, nrows=4, figure=fig)
ax11 = fig.add_subplot(gs[0, 0])
ax12 = fig.add_subplot(gs[0, 1])
ax21 = fig.add_subplot(gs[1, 0])
ax22 = fig.add_subplot(gs[1, 1])
ax31 = fig.add_subplot(gs[2, 0])
ax32 = fig.add_subplot(gs[2, 1])
ax41 = fig.add_subplot(gs[3, 0])
ax42 = fig.add_subplot(gs[3, 1])

ax11.hist(sma,color='black',alpha=0.3,density=True,label='Pop.',bins=np.linspace(start=0.,stop=np.max(sma),endpoint=True,num=30))
ax11.hist(sma[planetsInVisibleRegionsInTimeWindowInds],color='purple',alpha=0.3,density=True,label='Sub-pop.',bins=np.linspace(start=0.,stop=np.max(sma),endpoint=True,num=30))
ax11.plot([sma[ind],sma[ind]],[0,1],color='black')
ax11.set_xlabel('Semi-major axis, in AU',weight='bold')
ax11.set_ylabel('Frequency',weight='bold')
ax11.legend()
#ax11.set_yscale('log')

ax12.hist(e,color='black',alpha=0.3,density=True,label='Pop.',bins=np.linspace(start=0.,stop=np.max(e),endpoint=True,num=30))
ax12.hist(e[planetsInVisibleRegionsInTimeWindowInds],color='purple',alpha=0.3,density=True,label='Sub-pop.',bins=np.linspace(start=0.,stop=np.max(e),endpoint=True,num=30))
ax12.plot([e[ind],e[ind]],[0,1],color='black')
ax12.set_xlabel('Eccentricity',weight='bold')
ax12.set_ylabel('Frequency',weight='bold')
ax12.legend()
#ax12.set_yscale('log')

ax21.hist(inc,color='black',alpha=0.3,density=True,label='Pop.',bins=np.linspace(start=0.,stop=np.pi,endpoint=True,num=30))
ax21.hist(inc[planetsInVisibleRegionsInTimeWindowInds],color='purple',alpha=0.3,density=True,label='Sub-pop.',bins=np.linspace(start=0.,stop=np.pi,endpoint=True,num=30))
tinc = inc[planetsInVisibleRegionsInTimeWindowInds]
tinc[np.where(tinc > np.pi/2.)[0]] = np.pi - tinc[np.where(tinc > np.pi/2.)[0]]
ax21.hist(tinc,color='red',alpha=0.3,density=True,label='Adjusted Inc.\nSub-pop.')
ax21.plot([inc[ind],inc[ind]],[0,1],color='black')
ax21.set_xlabel('Inclination, in rad',weight='bold')
ax21.set_ylabel('Frequency',weight='bold')
ax21.legend()
#ax21.set_yscale('log')

ax22.hist(w,color='black',alpha=0.3,density=True,label='Pop.',bins=np.linspace(start=0.,stop=2.*np.pi,endpoint=True,num=30))
ax22.hist(w[planetsInVisibleRegionsInTimeWindowInds],color='purple',alpha=0.3,density=True,label='Sub-pop.',bins=np.linspace(start=0.,stop=2.*np.pi,endpoint=True,num=30))
ax22.plot([w[ind],w[ind]],[0,1],color='black')
ax22.set_xlabel('Argument of periapsis, in rad',weight='bold')
ax22.set_ylabel('Frequency',weight='bold')
ax22.legend()
#ax22.set_yscale('log')

ax31.hist(W,color='black',alpha=0.3,density=True,label='Pop.',bins=np.linspace(start=0.,stop=2.*np.pi,endpoint=True,num=30))
ax31.hist(W[planetsInVisibleRegionsInTimeWindowInds],color='purple',alpha=0.3,density=True,label='Sub-pop.',bins=np.linspace(start=0.,stop=2.*np.pi,endpoint=True,num=30))
ax31.plot([W[ind],W[ind]],[0,1],color='black')
ax31.set_xlabel('Longitude of the Ascending Node, in rad',weight='bold')
ax31.set_ylabel('Frequency',weight='bold')
ax31.legend()
#ax31.set_yscale('log')

ax32.hist(p,color='black',alpha=0.3,density=True,label='Pop.',bins=np.linspace(start=0.,stop=np.max(p),endpoint=True,num=30))
ax32.hist(p[planetsInVisibleRegionsInTimeWindowInds],color='purple',alpha=0.3,density=True,label='Sub-pop.',bins=np.linspace(start=0.,stop=np.max(p),endpoint=True,num=30))
ax32.plot([p[ind],p[ind]],[0,1],color='black')
ax32.set_xlabel('Planet Albedo',weight='bold')
ax32.set_ylabel('Frequency',weight='bold')
ax32.legend()
#ax32.set_yscale('log')

ax41.hist(Rp.value,color='black',alpha=0.3,density=True,label='Pop.',bins=np.linspace(start=0.,stop=np.max(Rp),endpoint=True,num=30))
ax41.hist(Rp[planetsInVisibleRegionsInTimeWindowInds].value,color='purple',alpha=0.3,density=True,label='Sub-pop.',bins=np.linspace(start=0.,stop=np.max(Rp),endpoint=True,num=30))
ax41.plot([Rp[ind].value,Rp[ind].value],[0,1],color='black')
ax41.set_xlabel('Planet Radius, in Earth Raddius',weight='bold')
ax41.set_ylabel('Frequency',weight='bold')
ax41.legend()
#ax41.set_yscale('log')

ax42.hist(np.multiply(p,Rp.value),color='black',alpha=0.3,density=True,label='Pop.',bins=np.linspace(start=0.,stop=np.max(p*Rp.value),endpoint=True,num=30))
ax42.hist(np.multiply(p[planetsInVisibleRegionsInTimeWindowInds],Rp[planetsInVisibleRegionsInTimeWindowInds].value),color='purple',alpha=0.3,density=True,label='Sub-pop.',bins=np.linspace(start=0.,stop=np.max(p*Rp.value),endpoint=True,num=30))
ax42.plot([p[ind]*Rp[ind].value,p[ind]*Rp[ind].value],[0,1],color='black')
ax42.set_xlabel('p*Rp',weight='bold')
ax42.set_ylabel('Frequency',weight='bold')
ax42.legend()
#ax42.set_yscale('log')

plt.show(block=False)


#TODO: Plot COVARIANCE MATRICES FOR PLANETS find an old scatter plot or something




