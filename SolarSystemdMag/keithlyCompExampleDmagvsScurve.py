#### Keithly Completeness Example Dmags vs S curve plot
import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
from matplotlib import colors
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
import datetime
import re
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))#'$HOME/Documents/exosims/Scripts'))#HabExTimeSweep_HabEx_CSAG13_PPSAG13'))#WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_40519'))#EXOSIMS/EXOSIMS/Scripts'))#EXOSIMS/EXOSIMS/Scripts'))
filename = 'HabEx_CSAG13_PPSAG13_compSubtype.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
#Note no completness specs in SAG13 SAG13
comp = sim.SurveySimulation.Completeness
TL= sim.SurveySimulation.TargetList
ZL = sim.ZodiacalLight
OS = sim.OpticalSystem

folder = './'
PPoutpath = './'


#Getting observing mode
#allModes = OS.observingModes
#det_mode = list(filter(lambda mode: mode['detectionMode'] == True, allModes))[0]
mode = OS.observingModes[0]


nu_smax = 0. 
pvenus = np.asarray([0.689])
Rpvenus = (np.asarray([6051.8*1000.])*u.m).to('AU')
smavenus = (np.asarray([108.21*10.**9.])*u.m).to('AU') #in AU
e=np.asarray([1e-5])
inc=np.asarray([np.pi/2.-1e-5])
W=np.asarray([0.])
w=np.asarray([0.])
nus = np.linspace(start=-np.pi/2,stop=np.pi/2.,num=20000)
pneptune = np.asarray([0.442])
Rpneptune = (np.asarray([24622.*1000.])*u.m).to('AU')
smaneptune = (np.asarray([4495.*10.**9.])*u.m).to('AU')
#planProp['mars'] = {'R':3389.92*1000.,'a':227.92*10.**9.,'p':0.150}
pmars = np.asarray([0.150])
Rpmars = (np.asarray([3389.92*1000.])*u.m).to('AU')
smamars = (np.asarray([227.92*10.**9.])*u.m).to('AU')
#planProp['jupiter'] = {'R':69911.*1000.,'a':778.57*10.**9.,'p':0.538}
pjupiter = np.asarray([0.538])
Rpjupiter = (np.asarray([69911.*1000.])*u.m).to('AU')
smajupiter = (np.asarray([778.57*10.**9.])*u.m).to('AU')
#planProp['uranus'] = {'R':25362.*1000.,'a':2872.46*10.**9.,'p':0.488}
puranus = np.asarray([0.488])
Rpuranus = (np.asarray([25362.*1000.])*u.m).to('AU')
smauranus = (np.asarray([2872.46*10.**9.])*u.m).to('AU')
#Setting these values. Need to get the ones for Our Sun at 10 pc
TL.BV[0] = 1.
TL.Vmag[0] = 1.


#starMass
starMass = const.M_sun

periods_mars = (2.*np.pi*np.sqrt((smamars.to('AU'))**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value
periods_venus = (2.*np.pi*np.sqrt((smavenus.to('AU'))**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value
periods_neptune = (2.*np.pi*np.sqrt((smaneptune.to('AU'))**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value
periods_uranus = (2.*np.pi*np.sqrt((smauranus.to('AU'))**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value

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
#     _,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_, periods = calcMasterIntersections(smavenus,e,W,w,inc,s_circle,starMass,False)\

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

venus_intTime = OS.calc_intTime(TL,[0],ZL.fZ0*100000.,ZL.fEZ0*100000.,dmag_venus_smax,WA_venus_smax,mode)
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
nus_venus = trueAnomalyFromEccentricAnomaly(e,eccentric_anomalyvenus) #This is nominally the total true anomaly venus subtends as it is observed
sep_venus_edge0 = planet_star_separation(smavenus,e,0.,w,inc).to('AU').value #the separation where the true anomaly would not be observable
sep_venus_edge1 = planet_star_separation(smavenus,e,nus_venus/2.,w,inc).to('AU').value #the separation where the true anomaly would not be observable
sep_venus_edge2 = planet_star_separation(smavenus,e,-nus_venus/2.,w,inc).to('AU').value #the separation where the true anomaly would not be observable

#Calculate the points where 
beta_tmp1 = betaFunc(inc,nus_venus/2.,w)
Phi_tmp1 = quasiLambertPhaseFunction(beta_tmp1)
rsvenus = smavenus*(1.-e**2.)/(1.+e*np.cos(nus))
dmag_venus_tmp1 = deltaMag(pvenus,Rpvenus,rsvenus[0],Phi_tmp1)
beta_tmp2 = betaFunc(inc,-nus_venus/2.,w)
Phi_tmp2 = quasiLambertPhaseFunction(beta_tmp2)
rsvenus = smavenus*(1.-e**2.)/(1.+e*np.cos(nus))
dmag_venus_tmp2 = deltaMag(pvenus,Rpvenus,rsvenus[0],Phi_tmp2)

#From optical ssytem nemati calc_intTime(TL, sInds, fZ, fEZ, dMag, WA, mode)


#### Mars
betas = betaFunc(inc,nus,w)
beta_smax = betaFunc(inc,0.,w)
Phis = quasiLambertPhaseFunction(betas)
Phi_smax = quasiLambertPhaseFunction(beta_smax)
rsmars = smamars*(1.-e**2.)/(1.+e*np.cos(nus))
dmags_mars = deltaMag(pmars,Rpmars,rsmars[0],Phis)
dmag_mars_smax = deltaMag(pmars,Rpmars,rsmars[0],Phi_smax)
seps_mars = planet_star_separation(smamars,e,nus,w,inc)
WA_mars_smax = (smamars.to('AU').value/(22.87*u.pc.to('AU')))*u.rad.to('arcsec')*u.arcsec
mars_intTime = OS.calc_intTime(TL,[0],ZL.fZ0,ZL.fEZ0*1.,dmag_mars_smax,WA_mars_smax,mode)
mean_anomalymars = mars_intTime.to('year').value*2.*np.pi/periods_mars #total angle moved by planet
eccentric_anomalymars = mean_anomalymars#solve eccentric anomaly from mean anomaly
nus_mars = trueAnomalyFromEccentricAnomaly(e,eccentric_anomalymars) #This is nominally the total true anomaly venus subtends as it is observed
sep_mars_edge0 = planet_star_separation(smamars,e,0.,w,inc).to('AU') #the separation where the true anomaly would not be observable
sep_mars_edge1 = planet_star_separation(smamars,e,nus_mars/2.,w,inc).to('AU') #the separation where the true anomaly would not be observable
sep_mars_edge2 = planet_star_separation(smamars,e,-nus_mars/2.,w,inc).to('AU') #the separation where the true anomaly would not be observable
beta_tmp1 = betaFunc(inc,nus_mars/2.,w)
Phi_tmp1 = quasiLambertPhaseFunction(beta_tmp1)
rsmars = smamars*(1.-e**2.)/(1.+e*np.cos(nus))
dmag_mars_tmp1 = deltaMag(pmars,Rpmars,rsmars[0],Phi_tmp1)
beta_tmp2 = betaFunc(inc,-nus_mars/2.,w)
Phi_tmp2 = quasiLambertPhaseFunction(beta_tmp2)
rsmars = smamars*(1.-e**2.)/(1.+e*np.cos(nus))
dmag_mars_tmp2 = deltaMag(pmars,Rpmars,rsmars[0],Phi_tmp2)

#Telescope IWA From Mars
IWA_mars_pretty = (sep_mars_edge2.value/(22.87*u.pc.to('AU')))*u.rad.to('arcsec') #The IWA that causes the separation for mars in arcsec

#### Neptune
#true anomaly of intersection
plotBool = False
# dmajorp,dminorp,theta_OpQ_X,theta_OpQp_X,Op,x,y,Phi,xreal,only2RealInds,yrealAllRealInds,\
#         fourIntInds,twoIntOppositeXInds,twoIntSameYInds,nu_minSepPoints,nu_maxSepPoints,nu_lminSepPoints,nu_lmaxSepPoints,nu_fourInt,\
#         nu_twoIntSameY,nu_twoIntOppositeX,nu_IntersectionsOnly2, yrealImagInds,\
#         t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,t_twoIntSameY0,\
#         t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
#         minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep,\
#         errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,errors_twoIntSameY0,\
#         errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,type0_0Inds,\
#         type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,type2_0Inds,type2_1Inds,type2_2Inds,\
#         type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,fourInt_x,fourInt_y,twoIntSameY_x,twoIntSameY_y,twoIntOppositeX_x,\
#         twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3, periods = calcMasterIntersections(np.asarray(smamars.to('AU').value),e,W,w,inc,np.asarray(sep_mars_edge2.value),starMass,plotBool)
# print(nu_fourInt)
dmajorp,dminorp,theta_OpQ_X,theta_OpQp_X,Op,x,y,Phi,xreal,only2RealInds,yrealAllRealInds,\
        fourIntInds,twoIntOppositeXInds,twoIntSameYInds,nu_minSepPoints,nu_maxSepPoints,nu_lminSepPoints,nu_lmaxSepPoints,nu_fourInt,\
        nu_twoIntSameY,nu_twoIntOppositeX,nu_IntersectionsOnly2, yrealImagInds,\
        t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,t_twoIntSameY0,\
        t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
        minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep,\
        errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,errors_twoIntSameY0,\
        errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,type0_0Inds,\
        type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,type2_0Inds,type2_1Inds,type2_2Inds,\
        type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,fourInt_x,fourInt_y,twoIntSameY_x,twoIntSameY_y,twoIntOppositeX_x,\
        twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3, periods = calcMasterIntersections(smaneptune.to('AU').value,e,W,w,inc,sep_mars_edge2.value,starMass,plotBool)
print(nu_fourInt)
# nus[only2RealInds,4:6] = nu_IntersectionsOnly2
# nus[yrealAllRealInds[fourIntInds],4:8] = nu_fourInt
# nus[yrealAllRealInds[twoIntOppositeXInds],4:6] = nu_twoIntOppositeX
# nus[yrealAllRealInds[twoIntSameYInds],4:6] = nu_twoIntSameY
#nu_fourInt
betas_fourInt = betaFunc(inc,nu_fourInt,w)
Phis_fourInt = quasiLambertPhaseFunction(betas_fourInt)
rsneptune_fourInt = smaneptune.to('AU')*(1.-e**2.)/(1.+e*np.cos(nu_fourInt))
seps_neptune_fourInt = planet_star_separation(smaneptune.to('AU'),e,nu_fourInt,w,inc).to('AU')
dmags_neptune_fourInt = deltaMag(pneptune,Rpneptune,rsneptune_fourInt,Phis_fourInt)
#WA_neptune_IWA = (smaneptune.to('AU').value/(22.87*u.pc.to('AU')))*u.rad.to('arcsec')*u.arcsec
neptune_intTimes_pretty = OS.calc_intTime(TL,[0],ZL.fZ0,ZL.fEZ0,dmags_neptune_fourInt,IWA_mars_pretty*u.arcsec,mode)
mean_anomalyneptune_pretty = neptune_intTimes_pretty.to('year').value*2.*np.pi/periods_neptune #total angle moved by planet
trueanomaly_neptune_pretty = nu_fourInt + mean_anomalyneptune_pretty # for e=0, cos(nu)=cos(E)=cos(M)
seps_neptune_pretty = planet_star_separation(smaneptune,e,trueanomaly_neptune_pretty,w,inc).to('AU')
betas_fourInt_pretty = betaFunc(inc,trueanomaly_neptune_pretty,w)
Phis_fourInt_pretty = quasiLambertPhaseFunction(betas_fourInt_pretty)
rsneptune_fourInt_pretty = smaneptune*(1.-e**2.)/(1.+e*np.cos(trueanomaly_neptune_pretty))
dmags_neptune_fourInt_pretty = deltaMag(pneptune,Rpneptune,rsneptune_fourInt_pretty,Phis_fourInt_pretty)

# #Can Delete
# a = smaneptune.to('AU')
# v = nu_fourInt[0,1]
# v=1.544
# r = a*(1-e**2.)/(1.+e*np.cos(v))
# X = r*(np.cos(W)*np.cos(w+v) - np.sin(W)*np.sin(w+v)*np.cos(inc))
# Y = r*(np.sin(W)*np.cos(w+v) + np.cos(W)*np.sin(w+v)*np.cos(inc))
# Z = r*(np.sin(inc)*np.sin(w+v))
# print(r)
# print(X)
# print(Y)
# print(Z)

# Calculate Seps vs True Anomaly
betas = betaFunc(inc,nus,w)
beta_smax = betaFunc(inc,0.,w)
Phis = quasiLambertPhaseFunction(betas)
Phi_smax = quasiLambertPhaseFunction(beta_smax)
rsneptune = smaneptune*(1.-e**2.)/(1.+e*np.cos(nus))
dmags_neptune = deltaMag(pneptune,Rpneptune,rsneptune[0],Phis)
dmag_neptune_smax = deltaMag(pneptune,Rpneptune,rsneptune[0],Phi_smax)
seps_neptune = planet_star_separation(smaneptune,e,nus,w,inc)
WA_neptune_smax = (smaneptune.to('AU').value/(22.87*u.pc.to('AU')))*u.rad.to('arcsec')*u.arcsec
neptune_intTime = OS.calc_intTime(TL,[0],ZL.fZ0,ZL.fEZ0,dmag_neptune_smax,WA_neptune_smax,mode)
mean_anomalyneptune = neptune_intTime.to('year').value*2.*np.pi/periods_neptune #total angle moved by planet


#### Uranus intersections
uranusOWASep = np.asarray([11.83])*u.AU
dmajorp,dminorp,theta_OpQ_X,theta_OpQp_X,Op,x,y,Phi,xreal,only2RealInds,yrealAllRealInds,\
        fourIntInds,twoIntOppositeXInds,twoIntSameYInds,nu_minSepPoints,nu_maxSepPoints,nu_lminSepPoints,nu_lmaxSepPoints,nu_fourInt2,\
        nu_twoIntSameY,nu_twoIntOppositeX,nu_IntersectionsOnly2, yrealImagInds,\
        t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,t_twoIntSameY0,\
        t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
        minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep,\
        errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,errors_twoIntSameY0,\
        errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,type0_0Inds,\
        type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,type2_0Inds,type2_1Inds,type2_2Inds,\
        type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,fourInt_x,fourInt_y,twoIntSameY_x,twoIntSameY_y,twoIntOppositeX_x,\
        twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3, periods = calcMasterIntersections(smauranus.to('AU').value,e,W,w,inc,uranusOWASep.value,starMass,plotBool)
print(nu_fourInt)
betas_fourInt2 = betaFunc(inc,nu_fourInt2,w)
Phis_fourInt2 = quasiLambertPhaseFunction(betas_fourInt2)
rsuranus_fourInt2 = smauranus.to('AU')*(1.-e**2.)/(1.+e*np.cos(nu_fourInt2))
seps_uranus_fourInt2 = planet_star_separation(smauranus.to('AU'),e,nu_fourInt2,w,inc).to('AU')
dmags_uranus_fourInt2 = deltaMag(puranus,Rpuranus,rsuranus_fourInt2,Phis_fourInt2)
#WA_neptune_IWA = (smaneptune.to('AU').value/(22.87*u.pc.to('AU')))*u.rad.to('arcsec')*u.arcsec
WA_uranus = (uranusOWASep.value/(22.87*u.pc.to('AU')))*u.rad.to('arcsec') #The IWA that causes the separation for mars in arcsec
uranus_intTimes_pretty = OS.calc_intTime(TL,[0],ZL.fZ0,ZL.fEZ0,dmags_uranus_fourInt2,WA_uranus*u.arcsec,mode)
mean_anomalyuranus_pretty = uranus_intTimes_pretty.to('year').value*2.*np.pi/periods_uranus #total angle moved by planet
trueanomaly_uranus_pretty = nu_fourInt2 + mean_anomalyuranus_pretty # for e=0, cos(nu)=cos(E)=cos(M)
seps_uranus_pretty = planet_star_separation(smauranus,e,trueanomaly_uranus_pretty,w,inc).to('AU')
betas_fourInt2_pretty = betaFunc(inc,trueanomaly_uranus_pretty,w)
Phis_fourInt2_pretty = quasiLambertPhaseFunction(betas_fourInt2_pretty)
rsuranus_fourInt2_pretty = smauranus*(1.-e**2.)/(1.+e*np.cos(trueanomaly_uranus_pretty))
dmags_uranus_fourInt2_pretty = deltaMag(puranus,Rpuranus,rsuranus_fourInt2_pretty,Phis_fourInt2_pretty)

#all the separations
betas = betaFunc(inc,nus,w)
Phis = quasiLambertPhaseFunction(betas)
rsuranus = smauranus*(1.-e**2.)/(1.+e*np.cos(nus))
dmags_uranus = deltaMag(puranus,Rpuranus,rsuranus[0],Phis)
seps_uranus = planet_star_separation(smauranus,e,nus,w,inc)


maxdmaguranus = 31.
dmag_upper = maxdmaguranus
#### Solving for dmag_min and dmag_max for each planet ################
mindmag, maxdmag, dmaglminAll, dmaglmaxAll, indsWith2, indsWith4, nuMinDmag, nuMaxDmag, nulminAll, nulmaxAll = calc_planet_dmagmin_dmagmax(e,inc,w,smauranus,puranus,Rpuranus)
#### nu From dmag_upper
print('Num Planets with At Least 2 Int given dmag: ' + str(np.sum((mindmag < dmag_upper)*(maxdmag > dmag_upper))))
print('Num Planets with dmag local extrema: ' + str(len(indsWith4)))
print('Num Planets with given 4 Int given dmag: ' + str(np.sum((dmaglminAll < dmag_upper)*(dmaglmaxAll > dmag_upper))))
indsWith4Int = indsWith4[np.where((dmaglminAll < dmag_upper)*(dmaglmaxAll > dmag_upper))[0]]
indsWith2Int = list(set(np.where((mindmag < dmag_upper)*(maxdmag > dmag_upper))[0]) - set(indsWith4Int))
nus2Int, nus4Int, dmag2Int, dmag4Int = calc_planetnu_from_dmag(dmag_upper,e,inc,w,smauranus,puranus,Rpuranus,mindmag, maxdmag, indsWith2Int, indsWith4Int)
#nus[indsWith2Int,8:10] = nus2Int
#nus[indsWith4Int,8:12] = nus4Int
#nus2Int
betas_fourInt3 = betaFunc(inc,nus2Int,w)
Phis_fourInt3 = quasiLambertPhaseFunction(betas_fourInt3)
rsuranus_fourInt3 = smauranus.to('AU')*(1.-e**2.)/(1.+e*np.cos(nus2Int))
seps_uranus_fourInt3 = planet_star_separation(smauranus.to('AU'),e,nus2Int,w,inc).to('AU')
dmags_uranus_fourInt3 = deltaMag(puranus,Rpuranus,rsuranus_fourInt3,Phis_fourInt3)

#Ensure intTime of uranus is shorter than the visibility window
uranusVisibilityWindowCorner = (nu_fourInt2[0,1] - nus2Int[0,0])/(2.*np.pi)*periods_uranus


#Uranus
dmajorp,dminorp,theta_OpQ_X,theta_OpQp_X,Op,x,y,Phi,xreal,only2RealInds,yrealAllRealInds,\
        fourIntInds,twoIntOppositeXInds,twoIntSameYInds,nu_minSepPoints,nu_maxSepPoints,nu_lminSepPoints,nu_lmaxSepPoints,nu_fourInt4,\
        nu_twoIntSameY,nu_twoIntOppositeX,nu_IntersectionsOnly2, yrealImagInds,\
        t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,t_twoIntSameY0,\
        t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
        minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep,\
        errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,errors_twoIntSameY0,\
        errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,type0_0Inds,\
        type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,type2_0Inds,type2_1Inds,type2_2Inds,\
        type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,fourInt_x,fourInt_y,twoIntSameY_x,twoIntSameY_y,twoIntOppositeX_x,\
        twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3, periods = calcMasterIntersections(smauranus.to('AU').value,e,W,w,inc,sep_mars_edge2.value,starMass,plotBool)
print(nu_fourInt4)
# nus[only2RealInds,4:6] = nu_IntersectionsOnly2
# nus[yrealAllRealInds[fourIntInds],4:8] = nu_fourInt
# nus[yrealAllRealInds[twoIntOppositeXInds],4:6] = nu_twoIntOppositeX
# nus[yrealAllRealInds[twoIntSameYInds],4:6] = nu_twoIntSameY
#nu_fourInt
betas_fourInt4 = betaFunc(inc,nu_fourInt4,w)
Phis_fourInt4 = quasiLambertPhaseFunction(betas_fourInt4)
rsuranus_fourInt4 = smauranus.to('AU')*(1.-e**2.)/(1.+e*np.cos(nu_fourInt4))
seps_uranus_fourInt4 = planet_star_separation(smauranus.to('AU'),e,nu_fourInt4,w,inc).to('AU')
dmags_uranus_fourInt4 = deltaMag(puranus,Rpuranus,rsuranus_fourInt4,Phis_fourInt4)

#Uranus not visible range upper
upper = 0.9070013
nus_uranus_notVisible1 = np.linspace(start=3.*np.pi/2.,stop=2.*np.pi+upper,num=1000)
betas = betaFunc(inc,nus_uranus_notVisible1,w)
Phis = quasiLambertPhaseFunction(betas)
rsuranus = smauranus*(1.-e**2.)/(1.+e*np.cos(nus_uranus_notVisible1))
dmags_uranusNotVisible = deltaMag(puranus,Rpuranus,rsuranus[0],Phis)
seps_uranusNotVisible = planet_star_separation(smauranus,e,nus_uranus_notVisible1,w,inc)
#Uranus not visible range lower
lower = 0.08
nus_uranus_notVisible2 = np.linspace(start=np.pi/2.,stop=np.pi/2.+lower,num=1000)
betas = betaFunc(inc,nus_uranus_notVisible2,w)
Phis = quasiLambertPhaseFunction(betas)
rsuranus2 = smauranus*(1.-e**2.)/(1.+e*np.cos(nus_uranus_notVisible2))
dmags_uranusNotVisible2 = deltaMag(puranus,Rpuranus,rsuranus[0],Phis)
seps_uranusNotVisible2 = planet_star_separation(smauranus,e,nus_uranus_notVisible2,w,inc)


OWA_uranus_edge2 = (uranusOWASep.value/(22.87*u.pc.to('AU')))*(u.rad.to('arcsec')) #The IWA that causes the separation for mars in arcsec


miny = 25.
maxy = 34.
# fig = plt.figure(num=1)
# plt.rc('axes',linewidth=2)
# plt.rc('lines',linewidth=2)
# plt.rcParams['axes.linewidth']=2
# plt.rc('font',weight='bold')
# plt.plot([22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),uranusOWASep.value],[maxdmaguranus,maxdmaguranus],color='black')
# plt.plot(np.asarray([1,1])*22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),[miny,maxdmaguranus],color='black')
# plt.plot(np.asarray([1,1])*uranusOWASep.value,[miny,maxdmaguranus],color='black')
# plt.plot(seps_mars.to('AU'),dmags_mars,color='red')
# #plt.plot(seps_mars.to('AU'),dmags_mars,color='black',linestyle='--')
# plt.scatter(sep_mars_edge1,dmag_mars_tmp1,color='red')
# plt.scatter(sep_mars_edge2,dmag_mars_tmp2,color='red')
# plt.plot(seps_venus.to('AU'),dmags_venus,color='gold')
# plt.scatter(sep_venus_edge1,dmag_venus_tmp1,color='gold')
# plt.scatter(sep_venus_edge2,dmag_venus_tmp2,color='gold')
# #Neptune
# #plt.plot(seps_neptune.to('AU'),dmags_neptune,color=colors.to_rgba('deepskyblue'))
# #plt.scatter(seps_neptune_fourInt[0,2].to('AU'),dmags_neptune_fourInt[0,2],color=colors.to_rgba('deepskyblue'))
# #plt.scatter(seps_neptune_pretty[0,2].to('AU'),dmags_neptune_fourInt_pretty[0,2],color=colors.to_rgba('deepskyblue'))
# #Uranus
# plt.plot(seps_uranus.to('AU'),dmags_uranus,color=colors.to_rgba('darkblue'))
# #uranus upper points
# plt.scatter(seps_uranus_fourInt2[0,1].to('AU'),dmags_uranus_fourInt2[0,1],color=colors.to_rgba('darkblue'))
# plt.scatter(seps_uranus_pretty[0,0].to('AU'),dmags_uranus_fourInt2_pretty[0,0],color=colors.to_rgba('darkblue'))
# plt.scatter(seps_uranus_fourInt3[0,0].to('AU'),dmags_uranus_fourInt3[0,0],color=colors.to_rgba('darkblue'))
# #uranus lower points
# plt.scatter(seps_uranus_fourInt2[0,3].to('AU'),dmags_uranus_fourInt2[0,3],color=colors.to_rgba('darkblue'))
# plt.scatter(seps_uranus_pretty[0,3].to('AU'),dmags_uranus_fourInt2_pretty[0,3],color=colors.to_rgba('darkblue'))

# plt.ylabel(r'$\Delta \mathrm{mag}$',weight='bold')
# plt.ylim([miny,maxy])
# plt.xscale('log')
# plt.xlim([10.**0.,40])#10.**2.])
# plt.xlabel('Planet-Star Separation, s, in AU',weight='bold')

# plt.show(block=False)

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# inset_axes0 = inset_axes(plt.gca(), width="30%", height=1., loc=0) # width = 30% of parent_bbox # height : 1 inch
# inset_axes0.plot(np.asarray([1,1])*22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),[miny,maxdmaguranus],color='black')
# inset_axes0.scatter(sep_mars_edge1,dmag_mars_tmp1,color='red')
# inset_axes0.scatter(sep_mars_edge2,dmag_mars_tmp2,color='red')
# inset_axes0.set_xlim([1.50,1.525])
# inset_axes0.set_ylim([27.3,28.1])
# inset_axes1 = inset_axes(plt.gca(), width="30%", height=1., loc=1) # width = 30% of parent_bbox # height : 1 inch
# inset_axes1.plot(np.asarray([1,1])*uranusOWASep.value,[miny,maxdmaguranus],color='black')
# inset_axes1.plot([22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),uranusOWASep.value],[maxdmaguranus,maxdmaguranus],color='black')
# inset_axes1.plot(seps_uranus.to('AU'),dmags_uranus,color=colors.to_rgba('darkblue'))
# inset_axes1.scatter(seps_uranus_fourInt2[0,1].to('AU'),dmags_uranus_fourInt2[0,1],color=colors.to_rgba('darkblue'))
# inset_axes1.scatter(seps_uranus_fourInt3[0,0].to('AU'),dmags_uranus_fourInt3[0,0],color=colors.to_rgba('darkblue'))
# inset_axes1.set_xlim([11.55,11.9])
# inset_axes1.set_ylim([30.80,31.20])
# inset_axes2 = inset_axes(plt.gca(), width="30%", height=1., loc=2) # width = 30% of parent_bbox # height : 1 inch
# inset_axes2.plot(np.asarray([1,1])*uranusOWASep.value,[miny,maxdmaguranus],color='black')
# inset_axes2.plot(seps_uranus.to('AU'),dmags_uranus,color=colors.to_rgba('darkblue'))
# inset_axes2.scatter(seps_uranus_fourInt2[0,3].to('AU'),dmags_uranus_fourInt2[0,3],color=colors.to_rgba('darkblue'))
# inset_axes2.scatter(seps_uranus_pretty[0,3].to('AU'),dmags_uranus_fourInt2_pretty[0,3],color=colors.to_rgba('darkblue'))
# inset_axes2.set_xlim([11.8299,11.8301])
# inset_axes2.set_ylim([25.5,27.0])

# plt.show(block=False)

plt.close(2)
from matplotlib import gridspec
import matplotlib
from matplotlib import lines
rect1 = matplotlib.patches.Rectangle((uranusOWASep.value,20.), 50, 50, color='grey',alpha=0.2,linewidth=0) #outside OWA
rect2 = matplotlib.patches.Rectangle((22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),20.), -50, 50, color='grey',alpha=0.2, linewidth=0) #Inside IWA
rect3 = matplotlib.patches.Rectangle((22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),maxdmaguranus), (uranusOWASep.value-22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad')), 50, color='grey',alpha=0.2, linewidth=0)
fig3 = plt.figure(constrained_layout=True,num=2,figsize=(7.5,5))
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
gs = fig3.add_gridspec(ncols=2, nrows=3,width_ratios=[3,1],height_ratios=[1,1,1])
f3_ax0 = fig3.add_subplot(gs[:, 0])
f3_ax0.plot([11.,11.],[25.5,27.0],color='grey')
f3_ax0.plot([11.,12.5],[27.0,27.0],color='grey')
f3_ax0.plot([12.5,12.5],[27.0,25.5],color='grey')
f3_ax0.plot([12.5,11.],[25.5,25.5],color='grey')
f3_ax0.add_patch(rect1)
f3_ax0.plot([11.,11.],[30.5,31.5],color='grey')
f3_ax0.plot([11.,12.5],[31.5,31.5],color='grey')
f3_ax0.plot([12.5,12.5],[31.5,30.5],color='grey')
f3_ax0.plot([12.5,11.],[30.5,30.5],color='grey')
f3_ax0.add_patch(rect2)
f3_ax0.plot([1.4,1.4],[27.25,28.25],color='grey')
f3_ax0.plot([1.4,1.6],[28.25,28.25],color='grey')
f3_ax0.plot([1.6,1.6],[28.25,27.25],color='grey')
f3_ax0.plot([1.6,1.4],[27.25,27.25],color='grey')
f3_ax0.add_patch(rect3)

f3_ax0.plot([22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),uranusOWASep.value],[maxdmaguranus,maxdmaguranus],color='black')
f3_ax0.plot(np.asarray([1,1])*22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),[miny,maxdmaguranus],color='black')
f3_ax0.plot(np.asarray([1,1])*uranusOWASep.value,[miny,maxdmaguranus],color='black')
f3_ax0.plot(seps_mars.to('AU'),dmags_mars,color='red')
f3_ax0.plot(seps_mars.to('AU'),dmags_mars,color='black',linestyle='--')
f3_ax0.scatter(sep_mars_edge1,dmag_mars_tmp1,color='red')
f3_ax0.scatter(sep_mars_edge2,dmag_mars_tmp2,color='red')
# f3_ax0.plot(seps_venus.to('AU'),dmags_venus,color='gold')
# f3_ax0.scatter(sep_venus_edge1,dmag_venus_tmp1,color='gold')
# f3_ax0.scatter(sep_venus_edge2,dmag_venus_tmp2,color='gold')
#Neptune
#plt.plot(seps_neptune.to('AU'),dmags_neptune,color=colors.to_rgba('deepskyblue'))
#plt.scatter(seps_neptune_fourInt[0,2].to('AU'),dmags_neptune_fourInt[0,2],color=colors.to_rgba('deepskyblue'))
#plt.scatter(seps_neptune_pretty[0,2].to('AU'),dmags_neptune_fourInt_pretty[0,2],color=colors.to_rgba('deepskyblue'))
#Uranus
f3_ax0.plot(seps_uranus.to('AU'),dmags_uranus,color=colors.to_rgba('skyblue'))
f3_ax0.plot(seps_uranusNotVisible,dmags_uranusNotVisible,color='black',linestyle='--')
f3_ax0.plot(seps_uranusNotVisible2,dmags_uranusNotVisible2,color='black',linestyle='--')
#uranus upper points
f3_ax0.scatter(seps_uranus_fourInt2[0,1].to('AU'),dmags_uranus_fourInt2[0,1],color=colors.to_rgba('skyblue'))
f3_ax0.scatter(seps_uranus_pretty[0,0].to('AU'),dmags_uranus_fourInt2_pretty[0,0],color=colors.to_rgba('skyblue'))
f3_ax0.scatter(seps_uranus_fourInt3[0,0].to('AU'),dmags_uranus_fourInt3[0,0],color=colors.to_rgba('skyblue'))
f3_ax0.scatter(seps_uranus_fourInt4[0,3].to('AU'),dmags_uranus_fourInt4[0,3],color=colors.to_rgba('skyblue'))
#uranus lower points
f3_ax0.scatter(seps_uranus_fourInt2[0,3].to('AU'),dmags_uranus_fourInt2[0,3],color=colors.to_rgba('skyblue'))
f3_ax0.scatter(seps_uranus_pretty[0,3].to('AU'),dmags_uranus_fourInt2_pretty[0,3],color=colors.to_rgba('skyblue'))
f3_ax0.set_ylabel(r'$\Delta \mathrm{mag}$',weight='bold')
f3_ax0.set_ylim([miny,maxy])
f3_ax0.set_xscale('log')
f3_ax0.set_xlim([10.**0.,40])#10.**2.])
f3_ax0.set_xlabel('Planet-Star Separation, s, in AU',weight='bold')

dotted_line1 = lines.Line2D([], [], linestyle="--", color='black')
dotted_line2 = lines.Line2D([], [], linestyle="-", color=colors.to_rgba('skyblue'))
dotted_line3 = lines.Line2D([], [], linestyle="--", color='black')
dotted_line4 = lines.Line2D([], [], linestyle="-", color='red')
f3_ax0.legend([(dotted_line4,dotted_line3),(dotted_line2, dotted_line1),(dotted_line2)], ["Mars Not Visible","Neptune Not Visible","Neptune Visible"])
#f3_ax0.legend([])

f3_ax1 = fig3.add_subplot(gs[0, 1])
f3_ax1.plot(np.asarray([1,1])*22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),[miny,maxdmaguranus],color='black')
f3_ax1.plot(seps_mars.to('AU'),dmags_mars,color='red')
f3_ax1.plot(seps_mars.to('AU'),dmags_mars,color='black',linestyle='--')
f3_ax1.scatter(sep_mars_edge1,dmag_mars_tmp1,color='red')
f3_ax1.scatter(sep_mars_edge2,dmag_mars_tmp2,color='red')
f3_ax1.set_xlim([1.50,1.525])
f3_ax1.set_ylim([27.3,28.1])
f3_ax1.set_ylabel(r'$\Delta \mathrm{mag}$', weight='bold')
f3_ax1.set_xlabel('s, in AU', weight='bold')
rect1_ax1 = matplotlib.patches.Rectangle((uranusOWASep.value,20.), 50, 50, color='grey',alpha=0.2,linewidth=0) #outside OWA
rect2_ax1 = matplotlib.patches.Rectangle((22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),20.), -50, 50, color='grey',alpha=0.2, linewidth=0) #Inside IWA
rect3_ax1 = matplotlib.patches.Rectangle((22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),maxdmaguranus), (uranusOWASep.value-22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad')), 50, color='grey',alpha=0.2, linewidth=0)
f3_ax1.plot([11.,11.],[25.5,27.0],color='grey')
f3_ax1.plot([11.,12.5],[27.0,27.0],color='grey')
f3_ax1.plot([12.5,12.5],[27.0,25.5],color='grey')
f3_ax1.plot([12.5,11.],[25.5,25.5],color='grey')
f3_ax1.add_patch(rect1_ax1)
f3_ax1.plot([11.,11.],[30.5,31.5],color='grey')
f3_ax1.plot([11.,12.5],[31.5,31.5],color='grey')
f3_ax1.plot([12.5,12.5],[31.5,30.5],color='grey')
f3_ax1.plot([12.5,11.],[30.5,30.5],color='grey')
f3_ax1.add_patch(rect2_ax1)
f3_ax1.plot([1.4,1.4],[27.25,28.25],color='grey')
f3_ax1.plot([1.4,1.6],[28.25,28.25],color='grey')
f3_ax1.plot([1.6,1.6],[28.25,27.25],color='grey')
f3_ax1.plot([1.6,1.4],[27.25,27.25],color='grey')
f3_ax1.add_patch(rect3_ax1)

f3_ax2 = fig3.add_subplot(gs[1, 1])
f3_ax2.plot(np.asarray([1,1])*uranusOWASep.value,[miny,maxdmaguranus],color='black')
f3_ax2.plot([22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),uranusOWASep.value],[maxdmaguranus,maxdmaguranus],color='black')
f3_ax2.plot(seps_uranus.to('AU'),dmags_uranus,color=colors.to_rgba('skyblue'))
f3_ax2.scatter(seps_uranus_fourInt2[0,1].to('AU'),dmags_uranus_fourInt2[0,1],color=colors.to_rgba('skyblue'))
f3_ax2.scatter(seps_uranus_fourInt3[0,0].to('AU'),dmags_uranus_fourInt3[0,0],color=colors.to_rgba('skyblue'))
f3_ax2.plot(seps_uranusNotVisible,dmags_uranusNotVisible,color='black',linestyle='--')
f3_ax2.set_xlim([11.55,11.9])
f3_ax2.set_ylim([30.80,31.20])
f3_ax2.set_ylabel(r'$\Delta \mathrm{mag}$', weight='bold')
f3_ax2.set_xlabel('s, in AU', weight='bold')
rect1_ax2 = matplotlib.patches.Rectangle((uranusOWASep.value,20.), 50, 50, color='grey',alpha=0.2,linewidth=0) #outside OWA
rect2_ax2 = matplotlib.patches.Rectangle((22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),20.), -50, 50, color='grey',alpha=0.2, linewidth=0) #Inside IWA
rect3_ax2 = matplotlib.patches.Rectangle((22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),maxdmaguranus), (uranusOWASep.value-22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad')), 50, color='grey',alpha=0.2, linewidth=0)
f3_ax2.plot([11.,11.],[25.5,27.0],color='grey')
f3_ax2.plot([11.,12.5],[27.0,27.0],color='grey')
f3_ax2.plot([12.5,12.5],[27.0,25.5],color='grey')
f3_ax2.plot([12.5,11.],[25.5,25.5],color='grey')
f3_ax2.add_patch(rect1_ax2)
f3_ax2.plot([11.,11.],[30.5,31.5],color='grey')
f3_ax2.plot([11.,12.5],[31.5,31.5],color='grey')
f3_ax2.plot([12.5,12.5],[31.5,30.5],color='grey')
f3_ax2.plot([12.5,11.],[30.5,30.5],color='grey')
f3_ax2.add_patch(rect2_ax2)
f3_ax2.plot([1.4,1.4],[27.25,28.25],color='grey')
f3_ax2.plot([1.4,1.6],[28.25,28.25],color='grey')
f3_ax2.plot([1.6,1.6],[28.25,27.25],color='grey')
f3_ax2.plot([1.6,1.4],[27.25,27.25],color='grey')
f3_ax2.add_patch(rect3_ax2)

f3_ax3 = fig3.add_subplot(gs[2, 1])
f3_ax3.plot(np.asarray([1,1])*uranusOWASep.value,[miny,maxdmaguranus],color='black')
f3_ax3.plot(seps_uranus.to('AU'),dmags_uranus,color=colors.to_rgba('skyblue'))
f3_ax3.plot(seps_uranusNotVisible,dmags_uranusNotVisible,color='black',linestyle='--')
f3_ax3.scatter(seps_uranus_fourInt2[0,3].to('AU'),dmags_uranus_fourInt2[0,3],color=colors.to_rgba('skyblue'))
f3_ax3.scatter(seps_uranus_pretty[0,3].to('AU'),dmags_uranus_fourInt2_pretty[0,3],color=colors.to_rgba('skyblue'),marker='x')
f3_ax3.set_xlim([11.829985,11.830015])
f3_ax3.set_ylim([25.5,27.0])
#f3_ax3.ticklabel_format(axis='x',style='sci',useOffset=11.83,useMathText=True)
f3_ax3.set_ylabel(r'$\Delta \mathrm{mag}$', weight='bold')
f3_ax3.set_xlabel('s ' + r'$\times 10^{-5}+11.83$' + ', in AU', weight='bold')#,labelpad=11)
rect1_ax3 = matplotlib.patches.Rectangle((uranusOWASep.value,20.), 50, 50, color='grey',alpha=0.2,linewidth=0) #outside OWA
rect2_ax3 = matplotlib.patches.Rectangle((22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),20.), -50, 50, color='grey',alpha=0.2, linewidth=0) #Inside IWA
rect3_ax3 = matplotlib.patches.Rectangle((22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad'),maxdmaguranus), (uranusOWASep.value-22.87*u.pc.to('AU')*IWA_mars_pretty*u.arcsec.to('rad')), 50, color='grey',alpha=0.2, linewidth=0)
f3_ax3.plot([11.,11.],[25.5,27.0],color='grey')
f3_ax3.plot([11.,12.5],[27.0,27.0],color='grey')
f3_ax3.plot([12.5,12.5],[27.0,25.5],color='grey')
f3_ax3.plot([12.5,11.],[25.5,25.5],color='grey')
f3_ax3.add_patch(rect1_ax3)
f3_ax3.plot([11.,11.],[30.5,31.5],color='grey')
f3_ax3.plot([11.,12.5],[31.5,31.5],color='grey')
f3_ax3.plot([12.5,12.5],[31.5,30.5],color='grey')
f3_ax3.plot([12.5,11.],[30.5,30.5],color='grey')
f3_ax3.add_patch(rect2_ax3)
f3_ax3.plot([1.4,1.4],[27.25,28.25],color='grey')
f3_ax3.plot([1.4,1.6],[28.25,28.25],color='grey')
f3_ax3.plot([1.6,1.6],[28.25,27.25],color='grey')
f3_ax3.plot([1.6,1.4],[27.25,27.25],color='grey')
f3_ax3.add_patch(rect3_ax3)
f3_ax3.xaxis.offsetText.set_visible(False)
#f3_ax3.xaxis.offsetText.set_text(r'$\times 10^{-5}+11.83$')
#tx = f3_ax3.xaxis.get_offset_text()
#tx.set_text(r'$\times10^{-5}+11.83$')

#tx.draw()
#tx.set_fontsize(7)
#f3_ax3.text(x=11.83,y=25.,s=r"$\times 10^{-5}+11.83$",backgroundcolor="white",bbox={'xy':(11.83,25.),'width':0.01,'height':1.})

plt.gcf().text(0.005, 0.96, 'a)', fontsize=14)

plt.gcf().text(0.09, 0.35, 'b)', fontsize=14)
plt.gcf().text(0.49, 0.705, 'c)', fontsize=14)
plt.gcf().text(0.43, 0.25, 'd)', fontsize=14)
plt.gcf().text(0.68, 0.95, 'b)', fontsize=14)
plt.gcf().text(0.68, 0.65, 'c)', fontsize=14)
plt.gcf().text(0.68, 0.33, 'd)', fontsize=14)

plt.show(block=False)
plt.gcf().canvas.draw()
# Save to a File
date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'dmagVsSIntersectionPlot' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)
print('Done plotting dmagVsSIntersectionPlot')

