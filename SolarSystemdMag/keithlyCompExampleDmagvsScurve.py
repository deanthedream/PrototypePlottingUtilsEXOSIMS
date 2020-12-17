#### Keithly Completeness Example Dmags vs S curve plot
import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as random
import time
from astropy import constants as const
import astropy.units as u
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.planet_star_separation import planet_star_separation

from EXOSIMS.util.phaseFunctions import quasiLambertPhaseFunction
from EXOSIMS.util.phaseFunctions import betaFunc

import sys, os.path, EXOSIMS
import numpy as np
import copy
from copy import deepcopy
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))#'$HOME/Documents/exosims/Scripts'))#HabExTimeSweep_HabEx_CSAG13_PPSAG13'))#WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_40519'))#EXOSIMS/EXOSIMS/Scripts'))#EXOSIMS/EXOSIMS/Scripts'))
filename = 'HabEx_CSAG13_PPSAG13_compSubtype.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
#Note no completness specs in SAG13 SAG13
comp = sim.SurveySimulation.Completeness
TL= sim.SurveySimulation.TargetList
ZL = sim.ZodiacalLight
OS = sim.OpticalSystem


nu_smax = 0. 
pvenus = np.asarray([0.689])
Rpvenus = np.asarray([6051.8*1000.])*u.m
smavenus = np.asarray([108.21*10.**9.])*u.m #in AU
e=np.asarray([0.])
inc=np.asarray([0.])
W=np.asarray([0.])
w=np.asarray([0.])
nus = np.linspace(start=-np.pi/2,stop=np.pi/2.,num=300)
pneptune = np.asarray([0.442])
Rpneptune = np.asarray([24622.*1000.])*u.m
smaneptune = np.asarray([4495.*10.**9.])*u.m
#Setting these values. Need to get the ones for Our Sun at 10 pc
TL.BV[0] = 1.
TL.Vmag[0] = 1.


#starMass
starMass = const.M_sun

periods_venus = (2.*np.pi*np.sqrt((smavenus.to('AU'))**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value
periods_neptune = (2.*np.pi*np.sqrt((smaneptune.to('AU'))**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value

#Separations
#s_circle = np.ones(len(sma))
dmag = 25. #29.0
dmag_upper = 25. #29.0
IWA_HabEx = 0.045*u.arcsec #taken from a Habex Script in units of mas
IWA2=0.150*u.arcsec #Suggested by dmitry as analahous to WFIRST
OWA_HabEx = 6.*u.arcsec #from the HabEx Standards Team Final Report
s_inner = 15.*u.pc*IWA_HabEx.to('rad').value
s_outer = 15.*u.pc*OWA_HabEx.to('rad').value
s_circle = np.asarray([s_inner.to('AU').value])



#NEED TO MAKE GOOD HANDLING FOR E=0 ORBITS. SPECIFICALLY FOR MIN AND MAX SOLVING
# dmajorp,dminorp,_,_,Op,x,y,Phi,xreal,only2RealInds,yrealAllRealInds,fourIntInds,twoIntOppositeXInds,twoIntSameYInds,nu_minSepPoints,nu_maxSepPoints,\
#     nu_lminSepPoints,nu_lmaxSepPoints,nu_fourInt,nu_twoIntSameY,nu_twoIntOppositeX,nu_IntersectionsOnly2, yrealImagInds,\
#     t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,t_twoIntSameY0,\
#     t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
#     _,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_, periods = calcMasterIntersections(smavenus,e,W,w,inc,s_circle,starMass,False)


#need beta from true anomaly function
betas = betaFunc(inc,nus,w)
beta_smax = betaFunc(inc,0.,w)
Phis = quasiLambertPhaseFunction(betas)
Phi_smax = quasiLambertPhaseFunction(beta_smax)
rsvenus = smavenus*(1.-e**2.)/(1.+e*np.cos(nus))
dmags_venus = deltaMag(pvenus,Rpvenus,rsvenus[0],Phis)
dmag_venus_smax = deltaMag(pvenus,Rpvenus,rsvenus[0],Phi_smax)
seps_venus = planet_star_separation(smavenus,e,nus,w,inc)
WA_venus_smax = (smavenus.to('AU').value/(15.*u.pc.to('AU')))*u.rad.to('arcsec')*u.arcsec

#Calculate integration time at WA_venus_smax

#allModes = OS.observingModes
#det_mode = list(filter(lambda mode: mode['detectionMode'] == True, allModes))[0]
mode = OS.observingModes[0]

venus_intTime = OS.calc_intTime(TL,[0],ZL.fZ0,ZL.fEZ0,dmag_venus_smax,WA_venus_smax,mode)
mean_anomalyvenus = venus_intTime.to('year').value*2.*np.pi/periods_venus #total angle moved by planet
eccentric_anomalyvenus = mean_anomalyvenus#solve eccentric anomaly from mean anomaly
def trueAnomalyFromEccentricAnomaly(e,E):
    """ From https://en.wikipedia.org/wiki/True_anomaly #definitely exists in some other python scripts somewhere
    Args:
        e:
        E:
    Returns:
        nu:
    """
    nu = 2.*np.arctan2(np.sqrt(1.+e)*np.tan(E/2.),np.sqrt(1.-e))
    return nu
nus_venus = trueAnomalyFromEccentricAnomaly(e,eccentric_anomalyvenus)
#From optical ssytem nemati calc_intTime(TL, sInds, fZ, fEZ, dMag, WA, mode)




#1 Calculate smax of venus
#2 Calculate HabEx Int time for venus at that dmag and separation
#3 Iteratively Calculate s_inner that means venus  can't be imaged


#1 Pick and s_outer
#2 calculate true anomaly where neptune has s_outer
#3 ensure neptune's s_max is greater than the planet's limits
