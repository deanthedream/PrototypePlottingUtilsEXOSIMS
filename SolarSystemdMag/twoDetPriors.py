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
OS = sim.OpticalSystem
ZL = sim.ZodiacalLight
TL = sim.TargetList
TL.BV[0] = 0.65 #http://spiff.rit.edu/classes/phys440/lectures/color/color.html
TL.Vmag[0] = 1. #reference star
n = 4*10**3 #Dean's nice computer can go up to 10**8 what can atuin go up to?
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
uncertainty_theta_min = np.arctan2(uncertainty_s,s_inner)

#Detection one #TODO make these of a specific planet
#nurange[50] used for determing location of first detection
tpastPeriastron1 = timeFromTrueAnomaly(nurange[50],periods[ind]*u.year.to('day'),e[ind]) #Calculate the planet-star intersection edges
sep1 = ss[50] #0.7 #AU
dmag1 = dmags[50] #23. #Luminosity Scaled Planet-star Difference in Magnitude, ' + r'$\Delta\mathrm{mag}-2.5\log_{10}(L)$'
theta1 = thetas[50]
nus1, planetIsVisibleBool1 = planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,plotBool, sep1-uncertainty_s, sep1+uncertainty_s, dmag1*(1.+uncertainty_dmag), dmag1*(1.-uncertainty_dmag)) #Calculate planet-star nu edges and visible regions
ts1 = timeFromTrueAnomaly(nus1,np.tile(periods,(18,1)).T*u.year.to('day'),np.tile(e,(18,1)).T) #Calculate the planet-star intersection edges
uncertainty_theta1 = np.arctan2(uncertainty_s,sep1)
mode = [mode for mode in OS.observingModes if mode['detectionMode'] == True][0]
intTime1 = sim.OpticalSystem.calc_intTime(TL, [0], ZL.fZ0, ZL.fEZ0, dmag1, (sep1/(10.*u.pc.to('AU')))*u.rad, mode)
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
uncertainty_theta2 = np.arctan2(uncertainty_s,sep2)
intTime2 = sim.OpticalSystem.calc_intTime(TL, [0], ZL.fZ0, ZL.fEZ0, dmag2, (sep2/(10.*u.pc.to('AU')))*u.rad, mode)
numPlanetsInRegion2 = np.sum(np.any(planetIsVisibleBool2,axis=1))


#### Find Planet Inds With Both
detectableByBothBoolArray = np.any(planetIsVisibleBool2,axis=1)*np.any(planetIsVisibleBool1,axis=1)
numDetectableByBothArray = np.sum(detectableByBothBoolArray)
detectableByBothInds = np.where(detectableByBothBoolArray)[0] #inds of planets that are detectable at time 1 and time 2
#seems to successfully reduce numplanets by 1/100

#### Actual Planet Time Difference
actualPlanetTimeDifference = tpastPeriastron2-tpastPeriastron1 #the time that passed between image1 and image2

#### Actual Delta Theta
actualDeltaTheta = theta2-theta1 #the change in theta observed
dTheta_1 = (theta2-np.abs(np.arctan2(uncertainty_s,sep2))) - (theta1+np.abs(np.arctan2(uncertainty_s,sep1))) #could be largest or smallest
dTheta_2 = (theta2+np.abs(np.arctan2(uncertainty_s,sep2))) - (theta1-np.abs(np.arctan2(uncertainty_s,sep1))) #could be largest of smallest
deltaTheta_min = np.min([dTheta_1,dTheta_2]) #minimum of range
delteTheta_max = np.max([dTheta_1,dTheta_2]) #maximum of range



#### Calculates XY plane Angles
def calc_planetAngularXYPosition_FromXaxis(sma,e,w,W,inc,nu):
    """ Calculate the angular position of the planet from the X-axis at nu
    """
    r=(np.tile(sma,(18,1)).T*(1.-np.tile(e,(18,1)).T**2.))/(1.+np.tile(e,(18,1)).T*np.cos(nu))
    X = r*(np.cos(np.tile(W,(18,1)).T)* np.cos(np.tile(w,(18,1)).T + nu) - np.sin(np.tile(W,(18,1)).T)*np.sin(np.tile(w,(18,1)).T + nu)*np.cos(np.tile(inc,(18,1)).T))
    Y = r*(np.sin(np.tile(W,(18,1)).T)* np.cos(np.tile(w,(18,1)).T + nu) + np.cos(np.tile(W,(18,1)).T)*np.sin(np.tile(w,(18,1)).T + nu)*np.cos(np.tile(inc,(18,1)).T))
    #Z = r*np.sin(np.tile(inc,(18,1)).T)* np.sin(np.tile(w,(18,1)).T + nu)
    thetas = np.mod(np.arctan2(Y,X),2.*np.pi) #angle of planet position from X-axis, ranges from 0 to 2pi
    return thetas


def nuFromTheta(theta,sma,e,w,Omega,inc): #,nu):
    """ Calculates true anomaly from theta
    Args:
        theta (numpy array):
        sma (numpy array):
        e (numpy array):
        w (numpy array):
        Omega (numpy array):
        inc (numpy array):
        nu #DELETE FOR DEBUGGIN
    Returns:
        nu (float):
            true anomaly where theta occurs
    """
    tanTheta = np.tan(theta) #ranges from -inf to +inf

    if len(theta.shape) == 2:
        sma2 = np.tile(sma,(18,1)).T
        e2 = np.tile(e,(18,1)).T
        w2 = np.tile(w,(18,1)).T
        Omega2 = np.tile(Omega,(18,1)).T
        inc2 = np.tile(inc,(18,1)).T


    # #Coefficients of the quadratic equation in theatFromKOE.ipynb
    # a0 = tanTheta**2.*np.sin(Omega2)**2.*np.sin(w2)**2.*np.cos(inc2)**2. + tanTheta**2.*np.sin(Omega2)**2.*np.cos(inc2)**2.*np.cos(w2)**2. + tanTheta**2.*np.sin(w2)**2.*np.cos(Omega2)**2.\
    #     + tanTheta**2.*np.cos(Omega2)**2.*np.cos(w2)**2. + 2.*tanTheta*np.sin(Omega2)*np.sin(w2)**2.*np.cos(Omega2)*np.cos(inc2)**2. - 2.*tanTheta*np.sin(Omega2)*np.sin(w2)**2.*np.cos(Omega2)\
    #     + 2.*tanTheta*np.sin(Omega2)*np.cos(Omega2)*np.cos(inc2)**2.*np.cos(w2)**2. - 2.*tanTheta*np.sin(Omega2)*np.cos(Omega2)*np.cos(w2)**2. + np.sin(Omega2)**2.*np.sin(w2)**2.\
    #     + np.sin(Omega2)**2.*np.cos(w2)**2. + np.sin(w2)**2.*np.cos(Omega2)**2.*np.cos(inc2)**2. + np.cos(Omega2)**2.*np.cos(inc2)**2.*np.cos(w2)**2.
    # #a1 = 0. #this term is 0 so i just simplified the quadratic solution
    # a2 = -tanTheta**2.*np.sin(Omega2)**2.*np.cos(inc2)**2.*np.cos(w2)**2. - 2.*tanTheta**2.*np.sin(Omega2)*np.sin(w2)*np.cos(Omega2)*np.cos(inc2)*np.cos(w2) - tanTheta**2.*np.sin(w2)**2.*np.cos(Omega2)**2.\
    #     + 2.*tanTheta*np.sin(Omega2)**2.*np.sin(w2)*np.cos(inc2)*np.cos(w2) + 2.*tanTheta*np.sin(Omega2)*np.sin(w2)**2.*np.cos(Omega2) - 2.*tanTheta*np.sin(Omega2)*np.cos(Omega2)*np.cos(inc2)**2.*np.cos(w2)**2.\
    #     - 2.*tanTheta*np.sin(w2)*np.cos(Omega2)**2.*np.cos(inc2)*np.cos(w2) - np.sin(Omega2)**2.*np.sin(w2)**2. + 2.*np.sin(Omega2)*np.sin(w2)*np.cos(Omega2)*np.cos(inc2)*np.cos(w2) - np.cos(Omega2)**2.*np.cos(inc2)**2.*np.cos(w2)**2.
    # #Solve Quadratic
    # x0 = np.sqrt(-a0*a2)/a0
    # x1 = -np.sqrt(-a0*a2)/a0

    A = -tanTheta*np.sin(Omega2)*np.cos(inc2)*np.cos(w2) - tanTheta*np.sin(w2)*np.cos(Omega2) + np.sin(Omega2)*np.sin(w2) - np.cos(Omega2)*np.cos(inc2)*np.cos(w2)
    B = -tanTheta*np.sin(Omega2)*np.sin(w2)*np.cos(inc2) + tanTheta*np.cos(Omega2)*np.cos(w2) - np.sin(Omega2)*np.cos(w2) - np.sin(w2)*np.cos(Omega2)*np.cos(inc2)
    x0 = np.sqrt(1./(1.+(B/A)**2.)) #strictly ranges from 0 to 1
    x1 = -np.sqrt(1./(1.+(B/A)**2.)) #strictly ranges from -1 to 0

    #Calculate nus
    nu00 = np.arccos(np.clip(x0,-1,1)) #clip fixes float errors, ranges from 0 to pi/2
    nu01 = 2.*np.pi - nu00 #opposite solution, ranges from 3pi/2 to 2pi
    nu10 = np.arccos(np.clip(x1,-1,1)) #ramges from pi/2 to pi
    nu11 = 2.*np.pi - nu10 #ranges from pi to 3pi/2
    #shove nus into a (#plan, 18, 4) array
    nuArray = list()
    nuArray.append(nu00)
    nuArray.append(nu01)
    nuArray.append(nu10)
    nuArray.append(nu11)
    nuArray = np.asarray(nuArray)
    nuArray = np.swapaxes(nuArray,0,1)
    nuArray = np.swapaxes(nuArray,1,2)
    #nuArray = np.reshape(nuArray,(nu00.shape[0],nu00.shape[1],4)) #create an array with all errors in it so we can search along dimension for min

    #Calculate Thetas From nus
    thetas00 = calc_planetAngularXYPosition_FromXaxis(sma,e,w,W,inc,nu00) #nu00 has same shape as nu
    thetas01 = calc_planetAngularXYPosition_FromXaxis(sma,e,w,W,inc,nu01)
    thetas10 = calc_planetAngularXYPosition_FromXaxis(sma,e,w,W,inc,nu10)
    thetas11 = calc_planetAngularXYPosition_FromXaxis(sma,e,w,W,inc,nu11)
    thetasArray = list()
    thetasArray.append(thetas00)
    thetasArray.append(thetas01)
    thetasArray.append(thetas10)
    thetasArray.append(thetas11)
    thetasArray = np.asarray(thetasArray)
    thetasArray = np.swapaxes(thetasArray,0,1)
    thetasArray = np.swapaxes(thetasArray,1,2)
    #thetasArray = np.reshape(thetasArray,(thetas00.shape[0],thetas00.shape[1],4)) #create an array with all errors in it so we can search along dimension for min

    #Calculate errors between calculated and input
    error00 = np.abs(thetas00-theta)
    error01 = np.abs(thetas01-theta)
    error10 = np.abs(thetas10-theta)
    error11 = np.abs(thetas11-theta)

    #Calculate errors
    errorArray = list()
    errorArray.append(error00)
    errorArray.append(error01)
    errorArray.append(error10)
    errorArray.append(error11)
    errorArray = np.asarray(errorArray)
    errorArray = np.swapaxes(errorArray,0,1)
    errorArray = np.swapaxes(errorArray,1,2)
    #errorArray = np.reshape(errorArray,(error00.shape[0],error00.shape[1],4)) #create an array with all errors in it so we can search along dimension for min

    #Find the minimum error along star, 18 solutions
    #minErrorArray0 = np.zeros((error00.shape[0],error00.shape[1]))
    #outThetasArray = np.zeros((error00.shape[0],error00.shape[1]))
    outNuArray = np.zeros((error00.shape[0],error00.shape[1]))
    minErrorIndsArray0 = np.zeros((error00.shape[0],error00.shape[1]),dtype='int')
    for i in np.arange(error00.shape[0]): #iterate over each star
        for j in np.arange(error00.shape[1]):#iterate over all 18 solutions
            try:
                minErrorIndsArray0[i,j] = np.nanargmin(errorArray[i,j])
                #minErrorArray0[i,j] = errorArray[i,j,minErrorIndsArray0[i,j]]
                #outThetasArray[i,j] = thetasArray[i,j,minErrorIndsArray0[i,j]]
                outNuArray[i,j] = nuArray[i,j,minErrorIndsArray0[i,j]]
            except:
                minErrorIndsArray0[i,j] = 0
                #minErrorArray0[i,j] = np.nan
                #outThetasArray[i,j] = np.nan
                outNuArray[i,j] = np.nan

    return outNuArray

#TODO left off here. Check for t
# thetas1 = calc_planetAngularXYPosition_FromXaxis(sma[detectableByBothInds],e[detectableByBothInds],w[detectableByBothInds],W[detectableByBothInds],inc[detectableByBothInds],nus1[detectableByBothInds])
# thetas2 = calc_planetAngularXYPosition_FromXaxis(sma[detectableByBothInds],e[detectableByBothInds],w[detectableByBothInds],W[detectableByBothInds],inc[detectableByBothInds],nus2[detectableByBothInds])
thetas1 = calc_planetAngularXYPosition_FromXaxis(sma,e,w,W,inc,nus1)
thetas2 = calc_planetAngularXYPosition_FromXaxis(sma,e,w,W,inc,nus2)
nuFromTheta1m = nuFromTheta(thetas1-sigma_theta,sma,e,w,W,inc)
nuFromTheta1p = nuFromTheta(thetas1+sigma_theta,sma,e,w,W,inc)
nuFromTheta2m = nuFromTheta(thetas2-sigma_theta,sma,e,w,W,inc)
nuFromTheta2p = nuFromTheta(thetas2+sigma_theta,sma,e,w,W,inc)

#### Verify nuFromTheta function works
nu = nuFromTheta(thetas1,sma,e,w,W,inc) #doing this to validate this function

#???Calc all dthetas of planets in pop.
#???filter ones matching dTheta below


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
timeTolerance = np.max([intTime1.to('d').value, intTime2.to('d').value]) #3. #random tolerance on the time between two observations in days #TODO find a better number for this. Should be integration time x 2 since two detections must occur
planetsInVisibleRegionsInTimeWindow = list()
planetsInVisibleRegionsInTimeWindowInAngle = list()
planetsInVisibleRegionsInTimeWindow2 = list()
for (i,j) in [(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3),(2,4),(3,1),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4)]: #iterate over sets of inds of intersections
    # i indicates the number of visible regions in image 1
    # j indicates the number of visible regions in image 2
    for planetj in np.arange(len(setNumVisTimes[(i,j)]['inds'])): #iterate over all planets
        indsOfVisibleRegionsk = np.where(planetIsVisibleBool1[setNumVisTimes[(i,j)]['inds'][planetj]])[0] #should give us a new matrix with shape len(setNumVisTimes[(1,1)]['inds']), 17
        indsOfVisibleRegionsl = np.where(planetIsVisibleBool2[setNumVisTimes[(i,j)]['inds'][planetj]])[0]

        #Iterate over visible regions of image 1 and image 2, pick out smallest dt, pick out largest dt
        for k in indsOfVisibleRegionsk: #iterate over image 1 visible regions
            regionStart1 = ts1[setNumVisTimes[(i,j)]['inds'][planetj],k]
            regionEnd1 = ts1[setNumVisTimes[(i,j)]['inds'][planetj],k+1]
            #taverage1 = (regionStart1+regionEnd1)/2. #average time of time window
            angleStart1 = thetas1[setNumVisTimes[(i,j)]['inds'][planetj],k]
            angleStop1 = thetas1[setNumVisTimes[(i,j)]['inds'][planetj],k+1]

            #NOTE: This may be improved by (1) looking at only forward time cases (2) making backward in time cases have inc+pi 
            for l in indsOfVisibleRegionsl: #iterate over image 2 visible regions
                regionStart2 = ts2[setNumVisTimes[(i,j)]['inds'][planetj],l]
                regionEnd2 = ts2[setNumVisTimes[(i,j)]['inds'][planetj],l+1]
                #taverage2 = (regionStart2+regionEnd2)/2. #average time of time window
                angleStart2 = thetas2[setNumVisTimes[(i,j)]['inds'][planetj],l]
                angleStop2 = thetas2[setNumVisTimes[(i,j)]['inds'][planetj],l+1]


                #ADD DELTA THETA CALCULATION BOUNDS HERE???? I THINK ???? MIGHT NEED TO BE AN INSTRUMENT CALCULATION


                #Note whether region 1 starts first or region 2 starts first
                if regionStart1 < regionStart2: #the start time of region1 is before the start time of region 2
                    region1StartsFirstBool = True
                    maxdt = regionEnd2 - regionStart1
                    mindt = regionStart2 - regionEnd1
                    mindt = np.max([0.,mindt]) #ensures mindt is positive or 0

                    #Finding Min and max acceptable angles
                    if np.sign(angleStart2-angleStart1) == np.sign(actualDeltaTheta): #angle of region 2 relative to region 1 is in the same angular direction as the actual planet
                        maxDeltaAngle = (angleStop2 + uncertainty_theta2) - (angleStart1 - uncertainty_theta1) #largest angular change
                        minDeltaAngle = np.max([0.,(angleStart2 - uncertainty_theta2) - (angleStop1 + uncertainty_theta1)]) #smallest angular change
                    else: #otherwise angle of region 2 relative to region 1 is opposite what was actually observed
                        maxDeltaAngle = (angleStop1 + uncertainty_theta1) - (angleStart2 - uncertainty_theta2) #largest angular change
                        minDeltaAngle = np.max([0.,(angleStart1 - uncertainty_theta1) - (angleStop2 + uncertainty_theta2)]) #smallest angular change
                else:
                    region1StartsFirstBool = False
                    maxdt = regionEnd1 - regionStart2
                    mindt = regionStart1 - regionEnd2
                    mindt = np.max([0.,mindt]) #ensures mindt is positive or 0

                    #Finding Min and Max Acceptable Angles 
                    if -np.sign(angleStart2-angleStart1) == np.sign(actualDeltaTheta): #angle of region 2 relative to region 1 is in the same angular direction as the actual planet
                        maxDeltaAngle = (angleStop1 + uncertainty_theta1) - (angleStart2 - uncertainty_theta2)
                        minDeltaAngle = np.min([0.,(angleStart1 - uncertainty_theta1) - (angleStop2 + uncertainty_theta2)])
                    else: #otherwise angle of region 2 relative to region 1 is opposite what was actually observed
                        maxDeltaAngle = (angleStop2 + uncertainty_theta2) - (angleStart1 - uncertainty_theta1)
                        minDeltaAngle = np.min([0.,(angleStart2 - uncertainty_theta2) - (angleStop1 + uncertainty_theta1)])

                # #Note whether region 1 starts angularly smaller than region2
                # if angleStart1 < angleStart2:
                #     region1AngleFirstBool = True
                # else:
                #     region1AngleFirstBool = False

                #Calculate largest and smallest visibility window
                dt1 = np.abs(regionEnd2-regionStart1) #time difference between starting region 1 and ending region 2
                dt2 = np.abs(regionEnd1-regionStart2) #time difference between starting region 2 and ending region 1
                smallerInd = np.argmin([dt1,dt2]) #smaller of the two dts
                largerInd = np.argmax([dt1,dt2]) #larger of the two dts
                smaller = np.min([dt1,dt2]) #smaller of the two dts
                larger = np.max([dt1,dt2]) #larger of the two dts

                # dtheta1 = angleStop2-angleStart1
                # dtheta2 = angleStop1-angleStart2
                # mindTheta = np.min([dtheta1,dtheta2])
                # maxdTheta = np.max([dtheta1,dtheta2])

                #dtaverage = taverage2-taverage1
                # if dtaverage < timeTolerance: #if the planet is in both windows within the allowed time tolerance
                #     planetsInVisibleRegionsInTimeWindow.append((i,j,planetj,k,l))

                if actualPlanetTimeDifference > mindt - timeTolerance and actualPlanetTimeDifference < maxdt + timeTolerance:
                    planetsInVisibleRegionsInTimeWindow.append((i,j,setNumVisTimes[(i,j)]['inds'][planetj],k,l,region1StartsFirstBool))

                    if actualDeltaTheta < maxDeltaAngle and actualDeltaTheta > minDeltaAngle:
                        planetsInVisibleRegionsInTimeWindowInAngle.append((i,j,setNumVisTimes[(i,j)]['inds'][planetj],k,l,region1StartsFirstBool))



                # if actualPlanetTimeDifference-timeTolerance < larger and actualPlanetTimeDifference+timeTolerance > smaller:
                #     planetsInVisibleRegionsInTimeWindow.append((i,j,setNumVisTimes[(i,j)]['inds'][planetj],k,l,region1StartsFirstBool))

                #     if actualDeltaTheta < maxDeltaAngle and actualDeltaTheta > minDeltaAngle:
                #         planetsInVisibleRegionsInTimeWindowInAngle.append((i,j,setNumVisTimes[(i,j)]['inds'][planetj],k,l,region1StartsFirstBool))

                # if np.abs(np.abs(regionEnd2-regionStart1) - actualPlanetTimeDifference) < timeTolerance or\
                #     np.abs(np.abs(regionEnd1-regionStart2-) - actualPlanetTimeDifference) < timeTolerance:

print("Number of Planets Visible In Time Window: " + str(len(planetsInVisibleRegionsInTimeWindow)))
print("Number of Planets Visible In Time Window and Angle: " + str(len(planetsInVisibleRegionsInTimeWindowInAngle)))
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

ax41.hist(Rp.value,color='black',alpha=0.3,density=True,label='Pop.',bins=np.linspace(start=0.,stop=np.max(Rp.value),endpoint=True,num=30))
ax41.hist(Rp[planetsInVisibleRegionsInTimeWindowInds].value,color='purple',alpha=0.3,density=True,label='Sub-pop.',bins=np.linspace(start=0.,stop=np.max(Rp.value),endpoint=True,num=30))
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




