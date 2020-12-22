####
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
import itertools
from EXOSIMS.util.phaseFunctions import quasiLambertPhaseFunction
from EXOSIMS.util.phaseFunctions import betaFunc
from keplertools.fun import eccanom
import datetime
import re
import time
try:
    import cPickle as pickle
except:
    import pickle

# #### PLOT BOOL
plotBool = False
# if plotBool == True:
#     from plotProjectedEllipse import *
PPoutpath = './'

#### Randomly Generate Orbits
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))
filename = 'HabEx_CKL2_PPKL2.json'
filename = 'WFIRSTcycle6core.json'
filename = 'HabEx_CSAG13_PPSAG13_compSubtype.json'
#filename = 'HabEx_CSAG13_PPSAG13_compSubtypeHighEccen.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
PPop = sim.PlanetPopulation
comp = sim.Completeness
TL = sim.TargetList
n = 5*10**5 #Dean's nice computer can go up to 10**8 what can atuin go up to?
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
inc[np.where(inc>np.pi/2)[0]] = np.pi - inc[np.where(inc>np.pi/2)[0]]
sma, e, p, Rp = PPop.gen_plan_params(n)

#### Adjustment for Planets Causing Errors
#Planet to be removed
ar = 0.6840751713914676*u.AU #sma
er = 0.12443160036480415 #e
Wr = 6.1198652952593 #W
wr = 2.661645323283813 #w
incr = 0.8803680245150818 #inc
sma,e,W,w,inc = nukeKOE(sma,e,W,w,inc,ar,er,Wr,wr,incr)
ar = 1.1300859542315127*u.AU
er = 0.23306811746716588
Wr = 5.480292250277455
wr = 2.4440871464730183
incr = 1.197618937201339
sma,e,W,w,inc = nukeKOE(sma,e,W,w,inc,ar,er,Wr,wr,incr)

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

#### SAVED PLANET FOR Plot 3D Ellipse to 2D Ellipse Projection Diagram
ellipseProjection3Dto2DInd = 23 #22
sma[ellipseProjection3Dto2DInd] = 1.2164387563540457
e[ellipseProjection3Dto2DInd] = 0.531071885292766
w[ellipseProjection3Dto2DInd] = 3.477496280463054
W[ellipseProjection3Dto2DInd] = 5.333215834002414
inc[ellipseProjection3Dto2DInd] = 1.025093642138022
####
#### SAVED PLANET FOR Plot Projected, derotated, centered ellipse 
derotatedInd = 33
sma[derotatedInd] = 5.738800898338014
e[derotatedInd] = 0.29306873405223816
w[derotatedInd] = 4.436383063578559
W[derotatedInd] = 4.240810639711751
inc[derotatedInd] = 1.072680736014668
####
#### SAVED PLANET FOR Plot Sep vs nu
sepvsnuInd = 24
sma[sepvsnuInd] = 1.817006521549392
e[sepvsnuInd] = 0.08651509983996385
W[sepvsnuInd] = 3.3708439025758006
w[sepvsnuInd] = 4.862116908343989
inc[sepvsnuInd] = 1.2491324942585256
####
#### SAVED PLANET FOR Plot Sep vs t
sepvstInd = 25
sma[sepvstInd] = 2.204556035394906
e[sepvstInd] = 0.2898368164549611
W[sepvstInd] = 4.787284415551434
w[sepvstInd] = 2.71176523941224
inc[sepvstInd] = 1.447634036719772
####
#### A NICE 4 INTERSECTION EXAMPLE
fourIntersectionInd = 2173 #33
sma[fourIntersectionInd] = 5.363760022304063
e[fourIntersectionInd] = 0.557679118292977
w[fourIntersectionInd] = 5.058312201296985
W[fourIntersectionInd] = 0.6867396268911974
inc[fourIntersectionInd] = 0.8122666711110185
####


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
    twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3, periods = calcMasterIntersections(sma,e,W,w,inc,s_circle,starMass,plotBool)


#### nu From dMag #####################################################
#### Solving for dmag_min and dmag_max for each planet ################
mindmag, maxdmag, dmaglminAll, dmaglmaxAll, indsWith2, indsWith4, nuMinDmag, nuMaxDmag, nulminAll, nulmaxAll = calc_planet_dmagmin_dmagmax(e,inc,w,sma*u.AU,p,Rp)
print('Num Planets with At Least 2 Int given dmag: ' + str(np.sum((mindmag < dmag)*(maxdmag > dmag))))
print('Num Planets with dmag local extrema: ' + str(len(indsWith4)))
print('Num Planets with given 4 Int given dmag: ' + str(np.sum((dmaglminAll < dmag)*(dmaglmaxAll > dmag))))
indsWith4Int = indsWith4[np.where((dmaglminAll < dmag)*(dmaglmaxAll > dmag))[0]]
indsWith2Int = np.asarray(list(set(np.where((mindmag < dmag)*(maxdmag > dmag))[0]) - set(indsWith4Int)))
######################################################################

#### Dmag Extrema Times ##############################################
time_dmagmin = timeFromTrueAnomaly(nuMinDmag,periods,e)
time_dmagmax = timeFromTrueAnomaly(nuMaxDmag,periods,e)
time_dmaglmin = timeFromTrueAnomaly(nulminAll,periods[indsWith4],e[indsWith4])
time_dmaglmax = timeFromTrueAnomaly(nulmaxAll,periods[indsWith4],e[indsWith4])
######################################################################

######################################################################
#### Solving for nu, dmag intersections ##############################
nus2Int, nus4Int, dmag2Int, dmag4Int = calc_planetnu_from_dmag(dmag,e,inc,w,sma*u.AU,p,Rp,mindmag, maxdmag, indsWith2Int, indsWith4Int)
time_dmagInts = np.zeros((len(e),4))*np.nan
time_dmagInts[indsWith2Int,0] = timeFromTrueAnomaly(nus2Int[:,0],periods[indsWith2Int],e[indsWith2Int])
time_dmagInts[indsWith2Int,1] = timeFromTrueAnomaly(nus2Int[:,1],periods[indsWith2Int],e[indsWith2Int])
if not indsWith4Int is None and not nus4Int is None:
    time_dmagInts[indsWith4Int,0] = timeFromTrueAnomaly(nus4Int[:,0],periods[indsWith4Int],e[indsWith4Int])
    time_dmagInts[indsWith4Int,1] = timeFromTrueAnomaly(nus4Int[:,1],periods[indsWith4Int],e[indsWith4Int])
    time_dmagInts[indsWith4Int,2] = timeFromTrueAnomaly(nus4Int[:,2],periods[indsWith4Int],e[indsWith4Int])
    time_dmagInts[indsWith4Int,3] = timeFromTrueAnomaly(nus4Int[:,3],periods[indsWith4Int],e[indsWith4Int])
######################################################################

#### Bulking all Times Together ######################################
times_s = calc_t_sInnersOuter(sma,e,W,w,inc,s_inner*np.ones(len(sma)),s_outer*np.ones(len(sma)),starMass,plotBool)
times = np.concatenate((np.zeros((len(e),1)),times_s,time_dmagInts,np.reshape(periods,(len(periods),1))),axis=1)
timesSortInds = np.argsort(times,axis=1)
times2 = np.sort(times,axis=1) #sorted from smallest to largest
indsWithAnyInt = np.where(np.sum(~np.isnan(times2),axis=1))[0] #Finds the planets which have any intersections
#####################################################################


#Check visibility in all given bounds (For Completeness)
#NEED TO BE ABLE TO PUT BOUNDS INTO BOX WITH 4 SIDES
#AND BOX WITH 3 SIDES


nus, planetIsVisibleBool = planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,plotBool, s_inner, s_outer, dmag_upper, dmag_lower=None) #Calculate planet-star nu edges and visible regions
ts = timeFromTrueAnomaly(nus,np.tile(periods,(18,1)).T*u.year.to('day'),np.tile(e,(18,1)).T) #Calculate the planet-star intersection edges
dt = ts[:,1:] - ts[:,:-1] #Calculate time region widths
maxIntTime = 0.
gtIntLimit = dt > maxIntTime #Create boolean array for inds
totalVisibleTimePerTarget = np.nansum(np.multiply(np.multiply(dt-maxIntTime,planetIsVisibleBool.astype('int')),gtIntLimit),axis=1) #We subtract the int time from the fraction of observable time
totalCompleteness = np.divide(totalVisibleTimePerTarget,periods*u.year.to('day')) # Fraction of time each planet is visible of its period

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


#### Verifying Planet Visibility Windows #################################
nus_range = np.linspace(start=0,stop=2.*np.pi,num=1000)
#beta = list()
#Phis = list()
visbools = list()
compMethod2 = list()
total_time_visible_error = list()
numPoints = 10**5 #number of points to evaluate at
numPlans = 25*1000 #number of planets to iterate over
path = PPoutpath + 'planetVisibilityWindowsNpts' + str(numPoints) + 'Nplans' + str(numPlans) + '.pkl'
if os.path.exists(path):
    try:
        with open(path, "rb") as ff:
            rawdata = pickle.load(ff)
    except UnicodeDecodeError:
        with open(path, "rb") as ff:
            rawdata = pickle.load(ff,encoding='latin1')
    compMethod2 = rawdata['compMethod2']
    total_time_visible_error = rawdata['total_time_visible_error']
    visbools = rawdata['visbools']
else:
    start = time.time()
    for i in np.arange(numPlans):#len(inc)):
        print('Working on: ' + str(i) + '/' + str(numPlans))
        #period = periods[i]
        #np.linspace(start=0.,stop=periods[i])
        M = np.linspace(start=0.,stop=2.*np.pi,num=numPoints) #Even distribution across mean anomaly
        E = np.asarray([eccanom(M[j],e[i]) for j in np.arange(len(M))])
        nus_range = trueAnomalyFromEccentricAnomaly(e[i],E)
        betas = betaFunc(inc[i],nus_range,w[i])
        Phis = quasiLambertPhaseFunction(betas)
        rs = sma[i]*u.AU*(1.-e[i]**2.)/(1.+e[i]*np.cos(nus_range))
        dmags = deltaMag(p[i],Rp[i],rs,Phis)
        seps = planet_star_separation(sma[i],e[i],nus_range,w[i],inc[i])
        visibleBool = (seps > s_inner)*(seps < s_outer)*(dmags < dmag_upper)
        visbools.append(visibleBool)

        compMethod2.append(np.sum(visibleBool.astype('int'))/numPoints)

        total_time_visible_error.append(np.abs(compMethod2[i]-totalCompleteness[i]))
    stop = time.time()
    print('Execution Time (s): ' + str(stop-start))

    compMethod2 = np.asarray(compMethod2)
    print(compMethod2)
    print(totalCompleteness[0:len(compMethod2)])
    print(total_time_visible_error)
    print(np.max(total_time_visible_error))

    rawdata = {}
    rawdata['compMethod2'] = compMethod2
    rawdata['total_time_visible_error'] = total_time_visible_error
    rawdata['visbools'] = visbools


    # store 2D completeness pdf array as .comp file
    with open(path, 'wb') as ff:
        pickle.dump(rawdata, ff)


with open("./keithlyCompConvergence.csv", "a") as myfile:
    myfile.write(str(ns[i]) + "," + str(np.mean(totalCompleteness)) + "," + str(stop-start) + "\n")

num = 1000
plt.figure(num=num)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
logbins = np.logspace(np.log10(10**-10),np.log10(10**-4),24)
plt.hist(total_time_visible_error,color='purple',bins=logbins)
plt.xscale('log')
plt.yscale('log')
plt.xlim([10**-10,10**-4])
plt.xlabel('Completeness Error',weight='bold')
plt.ylabel('Frequency', weight='bold')
plt.show(block=False)
plt.gcf().canvas.draw()
# Save to a File
date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'CompletenessErrorHistogram' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)
print('Done plotting Completeness Error Histogram')
    #check visibility time difference between completeness method and 

#I THINK I NEED TO CONVERT THIS INTO TIME
##########################################################################





# #### Data Struct of Completeness
# compDict = dict()
# maxIntTimes = [0.,30.,60.,90.] #in days
# starDistances = [5.,10.,15.] #in pc
# for i in np.arange(len(starDistances)):
#     starDistance = starDistances[i]
#     s_inner = starDistance*u.pc.to('AU')*IWA_HabEx.to('rad').value
#     s_outer = starDistance*u.pc.to('AU')*OWA_HabEx.to('rad').value #RANDOMLY MULTIPLY BY 3 HERE
#     #will need to recalculate separations
#     for j in np.arange(len(maxIntTimes)):
#         maxIntTime = maxIntTimes[j]
#         compDict[(i,j)] = dict()
#         compDict[(i,j)]['maxIntTime'] = maxIntTime
#         compDict[(i,j)]['stardistance'] = starDistance
#         compDict[(i,j)]['s_inner'] = s_inner
#         compDict[(i,j)]['s_outer'] = s_outer
#         nus, planetIsVisibleBool = planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,plotBool, s_inner, s_outer, dmag_upper, dmag_lower=None) #Calculate planet-star nu edges and visible regions
#         ts = timeFromTrueAnomaly(nus,np.tile(periods,(18,1)).T*u.year.to('day'),np.tile(e,(18,1)).T) #Calculate the planet-star intersection edges
#         dt = ts[:,1:] - ts[:,:-1] #Calculate time region widths
#         # Completeness Calculated Based On Planets In the instrument's visibility limits
#         compDict[(i,j)]['totalVisibleTimePerTarget'] = np.nansum(np.multiply(dt,planetIsVisibleBool.astype('int')),axis=1) #The traditional calculation, accounting for how long the planet is in the visible region
#         compDict[(i,j)]['totalCompletenessPerTarget'] = np.divide(compDict[(i,j)]['totalVisibleTimePerTarget'],periods*u.year.to('day')) # Fraction of time each planet is visible of its period
#         compDict[(i,j)]['totalCompleteness'] = np.sum(compDict[(i,j)]['totalCompletenessPerTarget'])/len(compDict[(i,j)]['totalCompletenessPerTarget']) #Calculates the total completenss by summing all the fractions and normalize by number of targets
#         assert np.all(compDict[(i,j)]['totalCompletenessPerTarget'] >= 0), 'Not all positive comp'
#         assert compDict[(i,j)]['totalCompleteness'] >= 0, 'Not positive comp'
#         # Completeness 
#         gtIntLimit = dt > maxIntTime #Create boolean array for inds
#         compDict[(i,j)]['totalVisibleTimePerTarget_maxIntTimeCorrected'] = np.nansum(np.multiply(np.multiply(dt-maxIntTime,planetIsVisibleBool.astype('int')),gtIntLimit.astype('int')),axis=1) #We subtract the int time from the fraction of observable time
#         compDict[(i,j)]['totalCompletenessPerTarget_maxIntTimeCorrected'] = np.divide(compDict[(i,j)]['totalVisibleTimePerTarget_maxIntTimeCorrected'],periods*u.year.to('day')) # Fraction of time each planet is visible of its period
#         compDict[(i,j)]['totalCompleteness_maxIntTimeCorrected'] = np.sum(compDict[(i,j)]['totalCompletenessPerTarget_maxIntTimeCorrected'])/len(compDict[(i,j)]['totalCompletenessPerTarget_maxIntTimeCorrected']) #Calculates the total completenss by summing all the fractions and normalize by number of targets
#         compDict[(i,j)]['SubTypeCompPerTarget'] = dict()
#         compDict[(i,j)]['SubTypeCompPerTarget_maxIntTimeCorrected'] = dict()
#         compDict[(i,j)]['SubTypeComp'] = dict()
#         compDict[(i,j)]['SubTypeComp_maxIntTimeCorrected'] = dict()
#         for overi, overj in itertools.product(np.arange(len(comp.Rp_hi)),np.arange(len(comp.L_lo[0,:]))):
#             compDict[(i,j)]['SubTypeCompPerTarget'][(overi,overj)] = np.multiply(compDict[(i,j)]['totalCompletenessPerTarget'],((bini==overi)*(binj==overj)).astype('int'))/np.sum(np.multiply(periods*u.year.to('day'),((bini==overi)*(binj==overj)).astype('int'))) #Calculate completeness for this specific planet subtype
#             compDict[(i,j)]['SubTypeCompPerTarget_maxIntTimeCorrected'][(overi,overj)] = np.multiply(compDict[(i,j)]['totalCompletenessPerTarget_maxIntTimeCorrected'],((bini==overi)*(binj==overj)).astype('int'))/np.sum(np.multiply(periods*u.year.to('day'),((bini==overi)*(binj==overj)).astype('int'))) #Calculate completeness for this specific planet subtype
#             compDict[(i,j)]['SubTypeComp'][(overi,overj)] = np.sum(compDict[(i,j)]['SubTypeCompPerTarget'][(overi,overj)])
#         compDict[(i,j)]['SubTypeComp_maxIntTimeCorrected'][(overi,overj)] = np.sum(compDict[(i,j)]['SubTypeCompPerTarget_maxIntTimeCorrected'][(overi,overj)])
        
#         #Earth-Like Completeness, The probability of the detected planet being Earth-Like
#         compDict[(i,j)]['EarthlikeCompPerTarget'] = np.multiply(compDict[(i,j)]['totalCompletenessPerTarget'],(earthLike).astype('int')) #/np.sum(np.multiply(periods*u.year.to('day'),(earthLike).astype('int'))) #Calculates the completeness for Earth-Like Planets
#         compDict[(i,j)]['EarthlikeCompPerTarget_maxIntTimeCorrected'] = np.multiply(compDict[(i,j)]['totalCompletenessPerTarget_maxIntTimeCorrected'],(earthLike).astype('int')) #/np.sum(np.multiply(periods*u.year.to('day'),(earthLike).astype('int'))) #Calculates the completeness for Earth-Like Planets
#         compDict[(i,j)]['EarthlikeComp'] = np.sum(compDict[(i,j)]['EarthlikeCompPerTarget'])/len(earthLike) #np.sum(earthLike.astype('int'))
#         compDict[(i,j)]['EarthlikeComp_maxIntTimeCorrected'] = np.sum(compDict[(i,j)]['EarthlikeCompPerTarget_maxIntTimeCorrected'])/len(earthLike) #np.sum(earthLike.astype('int'))

#         #Earth-Like Completeness
#         compDict[(i,j)]['EarthlikeComp2'] = np.sum(compDict[(i,j)]['EarthlikeCompPerTarget'])/np.sum(earthLike.astype('int'))
#         compDict[(i,j)]['EarthlikeComp2_maxIntTimeCorrected'] = np.sum(compDict[(i,j)]['EarthlikeCompPerTarget_maxIntTimeCorrected'])/np.sum(earthLike.astype('int'))

# maxIntTime = 30. #days
# gtIntLimit = dt > maxIntTime #Create boolean array for inds
# totalCompletenessIntLimit = np.nansum(np.multiply(np.multiply(dt-maxIntTime,planetIsVisibleBool.astype('int')),gtIntLimit),axis=1) #We subtract the int time from the fraction of observable time
# totalCompletenessIntLimit = np.divide(totalVisibleTimePerTarget,periods*u.year.to('day')) # Fraction of time each planet is visible of its period






#### Calculate Planet Time Windows






#Determine Planet Visibility For A Ton Of Nus





