#Written


import numpy as np
import math
from pylab import *
try:
    import cPickle as pickle
except:
    import pickle
import os
import numpy as np
from pylab import *
from numpy import nan
import matplotlib.pyplot as plt
import argparse
import json
from EXOSIMS.util.vprint import vprint
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from physt import *
import physt
from mpl_toolkits.mplot3d import Axes3D #required for 3d plot
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline


def generateEquadistantPointsOnSphere(N=100,PPoutpath='/home/dean/Documents/exosims/cache/'):
    """Generate uniform points on a sphere
    #https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    Args
    Return:
        xyzpoint (numpy array) - of points with dimensions [~N,3], xyzpoint[~N] = [x,y,z]
        ra_dec (numpy array) - of ra and dec of points with dimensions [~N,2] 
            where [0,0] = firstind, declination
            where [100,1] = 100th ind, right ascension
            ra is from 0 to 2pi
            dec is from 0 to pi
    #designed to be run from EXOSIMS/EXOSIMS/util
    """
    #close('all')
    point = list()
    ra_dec = list()
    r=1.
    Ncount = 0
    a = 4*np.pi*r**2./N
    d = np.sqrt(a)
    Mtheta = np.round(np.pi/d)
    dtheta = np.pi/Mtheta
    dphi = a/dtheta
    for m in np.arange(0,Mtheta):
        theta = np.pi*(m+0.5)/Mtheta
        Mphi = np.round(2*np.pi*np.sin(theta)/dphi)
        for n in np.arange(0,Mphi):
            phi = 2*np.pi*n/Mphi
            x = np.sin(theta)*np.cos(phi)
            y = np.sin(theta)*np.sin(phi)
            z = np.cos(theta)
            point.append([x,y,z])
            ra_dec.append([theta,phi])
            Ncount += 1

    fig = figure(num=5000)
    ax = fig.add_subplot(111, projection='3d')
    xyzpoint = np.asarray(point)
    ra_dec = np.asarray(ra_dec)
    ax.scatter(xyzpoint[:,0], xyzpoint[:,1], xyzpoint[:,2], color='k', marker='o')
    title('Points Evenly Distributed on a Unit Sphere')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    show(block=False)

    fname = 'PointsEvenlyDistributedOnaUnitSphere'
    savefig(PPoutpath + fname + '.png')
    savefig(PPoutpath + fname + '.svg')
    savefig(PPoutpath + fname + '.eps')

    #output of form ra_dec[ind,ra/dec]
    return xyzpoint, ra_dec

def generateHistHEL(hEclipLon,PPoutpath='/home/dean/Documents/exosims/cache/'):
    """ Generates a Heliocentric Ecliptic Longitude Histogram
    Returns:
        numVsLonInterp2 - (interpolant) - this is the interpolant of the histogram of stars 
            located along the heliocentric ecliptic longitude
    """
    figure(num=2000)
    rc('axes',linewidth=2)
    rc('lines',linewidth=2)
    rcParams['axes.linewidth']=2
    rc('font',weight='bold')
    h, edges = np.histogram(hEclipLon)
    xdiff = np.diff(edges)
    xcents = xdiff/2.+edges[:-1]
    numVsLonInterp = interp1d(xcents, h, kind='cubic',fill_value='extrapolate')
    ystart = numVsLonInterp(np.pi)
    yend = numVsLonInterp(-np.pi)
    yse = (ystart+yend)/2.
    h = np.append(h,yse)
    xcents = np.append(xcents,np.pi)
    h = np.insert(h, 0, yse, axis=0)
    xcents = np.insert(xcents, 0, -np.pi, axis=0)
    numVsLonInterp2 = CubicSpline(xcents,h,bc_type='periodic')
    tp = np.linspace(-np.pi,np.pi,num=100)
    hist(hEclipLon)
    plot(tp,numVsLonInterp2(tp))
    xlim(-np.pi,np.pi)
    xlabel('Heliocentric Ecliptic Longitude of Targets (rad)')
    title('Histogram of Planned to Observe Targets')

    fname = 'HistogramPlannedTargetsToObserve'
    savefig(PPoutpath + fname + '.png')
    show(block=False)

    targUnderSpline = numVsLonInterp2.integrate(-np.pi,np.pi)/xdiff#Integral of spline, tells how many targets are under spline
    sumh = sum(h[1:-1])#*xdiff[0]
    return numVsLonInterp2, targUnderSpline, sumh, xdiff, edges

def generatePlannedObsTimeHistHEL(edges,t_dets,comp,hEclipLon,PPoutpath='/home/dean/Documents/exosims/cache/'):
    edges[0] = -np.pi#Force edges to -pi and pi
    edges[-1] = np.pi
    t_bins = list()
    t_dets_tmp = t_dets[comp > 0.]
    for i in np.arange(len(edges)-1):
        t_bins.append(sum(t_dets_tmp[np.where((edges[i] <= hEclipLon)*(hEclipLon <= edges[i+1]))[0]]).value)
    #consistency check
    sum(t_dets)
    sum(np.asarray(t_bins))

    #Plot t_bins in Histogram
    left_edges = edges[:-1]
    right_edges = edges[1:]
    centers = (left_edges+right_edges)/2.
    t_bins = np.asarray(t_bins)
    widths = np.diff(edges)
    figure(num=2002)
    bar(centers,t_bins,width=widths)
    xlabel('Heliocentric Ecliptic Longitude of Targets (rad)')
    ylabel('Sum Integration Time (days)')
    xlim([-np.pi,np.pi])
    title('Histogram of Planned Time to Observe Targets')

    fname = 'HistogramPlannedTargetTimeToObserve'
    savefig(PPoutpath + fname + '.png')
    show(block=False)



close('all')

#### Inputs ##########################################################
exoplanetObsTime = 365.25#Maximum amount of observing time
maxNumYears = 6.#maximum number of years we will allow
maxNumDays = maxNumYears*365.25

OBdur = 1.
#####################################################################

def maxNumRepInTime(OBdur,Time):
    #Time in days
    #OBdur in days
    maxNumRepInTime = np.asarray([math.ceil(Time/OBduri) for OBduri in OBdur])#.astype('int')
    maxRepsPerYear = np.asarray([math.ceil(365.25/OBduri) for OBduri in OBdur])#.astype('int')
    return maxNumRepInTime, maxRepsPerYear


missionPortion = exoplanetObsTime/(maxNumYears*365.25) # This is constrained


#Create List of OB durations
OBdur2 = list(set(np.logspace(np.log10(0.1),np.log10(365.),num=50,base=10.).astype(int)))
tmp = list(np.asarray(range(10))+1.5)
OBdur2.remove(0)
OBdur2 = sort(np.asarray(OBdur2 + tmp))


#Calculate Maximum number of repetitions within exoplanetObsTime
maxNumRepTot2, maxRepsPerYear2 = maxNumRepInTime(OBdur2,exoplanetObsTime)
fig = figure(1)
#loglog(OBdur2,maxNumRepTot2,marker='o')
semilogx(OBdur2,maxNumRepTot2,marker='o')
xlabel('Num Days')
ylabel('Max Num Reps')
show(block=False)


# ##### Distribute Observing Blocks throughout years ##################################
# #Even Distribution
# def evenDist(numOB,OBdur,maxNumDays):
#     """
#     maxNumDays - missionLife in days
#     OBdur - duration of an OB in days
#     numOB - number of Observing blocks to create
#     """
#     OBstartTimes = np.linspace(0,maxNumDays-OBdur,num=numOB, endpoint=True)
#     #Check maxNumDays - OBstartTimes[-1] < OBdur
#     return OBstartTimes

# #### GeomDist OB Spacing ############################################
# def geomDist(numOB,OBdur,maxNumDays):
#     """
#     maxNumDays - missionLife in days
#     OBdur - duration of an OB in days
#     numOB - number of Observing blocks to create
#     """
#     OBstartTimes = np.geomspace(1e-10,maxNumDays-OBdur,num=numOB, endpoint=True)
#     #Check maxNumDays - OBstartTimes[-1] < OBdur
#     return OBstartTimes

# GeomDistOBstartTimes = list()
# for i in np.arange(len(OBdur2)):
#     GeomDistOBstartTimes.append(geomDist(maxNumRepTot2[i], OBdur2[i], maxNumDays))
# ####################################################################


#### HarmonicDist ##################################################
def harmonicDist(numOB, OBdur, exoplanetObsTime):#, missionPortion):
    """
    """
    daysInYear = 365.25 #Days in a year
    numYears = exoplanetObsTime/daysInYear#Number of years

    numOBperYear = np.ceil(numOB/numYears)#Number of Observing blocks that can fit into 1 year

    OneYrOBstartTimes = np.asarray([])
    OneYrOBendTimes = np.asarray([])
    for i in range(int(np.ceil(numYears))):#Note it should be fine if we fully fill out the remaining year
        OneYrOBstartTimes = np.append(OneYrOBstartTimes,np.linspace(0.,365.25,num=numOBperYear, endpoint=False)+i*daysInYear)
        OneYrOBendTimes = np.append(OneYrOBendTimes,np.linspace(0.,365.25,num=numOBperYear, endpoint=False)+float(i*daysInYear+OBdur))
    return OneYrOBstartTimes, OneYrOBendTimes

HarmonicDistOB = list()
for i in np.arange(len(OBdur2)):
    tmpStart, tmpEnd = harmonicDist(maxNumRepTot2[i], OBdur2[i], maxNumDays)
    HarmonicDistOB.append([tmpStart, tmpEnd])
##########

#Write to output files
writeHarmonicToOutputFiles = False#Change this to true to create each of these start and end time Observing Blocks as .csv files
if writeHarmonicToOutputFiles == True:
    path = '/home/dean/Documents/exosims/Scripts/'
    tmp = ''
    for i in np.arange(len(OBdur2)):
        myList = list()
        for j in range(len(HarmonicDistOB[i][0])):
            myList.append(str(HarmonicDistOB[i][0][j]) + ',' + str(HarmonicDistOB[i][1][j]) + '\n')
        outString = ''.join(myList)
        #print outString
        fname = path + 'harmonicOB' + str(i) + '.csv'
        f = open(fname, "w")
        f.write(outString)
        print '"' + fname.split('/')[-1] + '",'
#####################################################################





#######################################################################################
#maxNumReps = maxNumYears*maxRepsPerYear2#number of Reps/ number of years #The minimum number of repetitions to go into 1 year in order to finish before 6 years

figure(2)
num = np.linspace(0,50,num=50)
tmp = np.geomspace(0.0001,maxNumDays,num=50)
frac = 0.6
tmp2 = frac*np.geomspace(0.0001,maxNumDays,num=50) + (1-frac)*np.linspace(0,maxNumDays,num=50)
tmp3 = frac*np.geomspace(0.0001,maxNumDays,num=50) + (1-frac)*np.linspace(0,maxNumDays,num=50)*np.geomspace(0.0001,maxNumDays,num=50)/maxNumDays
tmp4 = np.linspace(0,maxNumDays,num=50)
def func(x,valMax):
    m=200.
    frac = 0.8
    val1 = m*x
    val2 = x**3#12*x**2#np.exp(x)
    val = frac*val1 + (1-frac)*val2
    val = val*valMax/max(val)
    return val
def dfunc(x):
    dval = 0.8*200 + 3*(1-0.8)*x**2.
    return dval
tmp5 = func(num,maxNumDays)

plot(tmp,num,marker='o',color='blue')
plot(tmp2,num,marker='o',color='black')
plot(tmp3,num,marker='o',color='red')
plot(tmp4,num,marker='o',color='green')
plot(tmp5,num,marker='o',color='orange')
ylabel('Points Number')
xlabel('Start Times')
show(block=False)



import scipy.integrate as integrate
tmp5L = integrate.quad(dfunc,0,max(num))#This is the total length of the path from (0,0) to (num,maxNumDays)

minNumReps = 0

numRep = 10#number of repetitions in 1 year
assert(numRep <= 365.25/OBdur - 365.25%OBdur, 'numRep too large')
missionPortion = numRep*OBdur/365.25
missionLife = exoplanetObsTime/missionPortion

def isoMissionDuration(mL,mP,mdur):
    #missionLife,missionPortion,mission duration
    #mdur = mL*mP #total amount of time to elapse during the mission
    if mL is None:
        mL = mdur/mP
    elif mP is None:
        mP = mdur/mL
    return mL, mP

# def OBharmonics(num,mP):
#     #num of repetitions to occur within one year
#     #mP missionPortion



tmp = np.asarray(range(30))*12.
tmp1 = np.asarray(range(30))
tmp2 = np.asarray(range(12))+0.5
denom = np.asarray(range(30),)+1.
tmp3 = 365.25/denom

OBdurs = list()
[OBdurs.append(x) for x in tmp.tolist()]
[OBdurs.append(x) for x in tmp1.tolist()]
[OBdurs.append(x) for x in tmp2.tolist()]
[OBdurs.append(x) for x in tmp3.tolist()]




#### Calculate the Maximum Star Completeness of all 651 Targets under Consideration #####################
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import astropy.units as u
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))#EXOSIMS/EXOSIMS/Scripts'))#EXOSIMS/EXOSIMS/Scripts'))
filename = 'WFIRSTcycle6core.json'#'Dean3June18RS26CXXfZ01OB66PP01SU01.json'#'Dean1June18RS26CXXfZ01OB56PP01SU01.json'#'./TestScripts/04_KeplerLike_Occulter_linearJScheduler.json'#'Dean13May18RS09CXXfZ01OB01PP03SU01.json'#'sS_AYO7.json'#'ICDcontents.json'###'sS_protoTimeKeeping.json'#'sS_AYO3.json'#sS_SLSQPstatic_parallel_ensembleJTWIN.json'#'sS_JTwin.json'#'sS_AYO4.json'#'sS_AYO3.json'
#filename = 'sS_intTime6_KeplerLike2.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile,nopar=True)

TL = sim.TargetList#OK
ZL = sim.ZodiacalLight#OK
sInds = np.arange(TL.nStars)#OK
fZ = sim.ZodiacalLight.fEZ0#fZmin/u.arcsec**2#
fEZ = ZL.fEZ0
OS = sim.OpticalSystem
WA = OS.WA0
mode = sim.SurveySimulation.detmode
dMagLim = sim.Completeness.dMagLim#OK #This is set based off the 10**-9 contrast of WFIRST
dmag = np.linspace(1, dMagLim, num=1500,endpoint=True)
Cp = np.zeros([sInds.shape[0],dmag.shape[0]])
Cb = np.zeros(sInds.shape[0])/u.s
Csp = np.zeros(sInds.shape[0])/u.s
for i in xrange(dmag.shape[0]):
    Cp[:,i], Cb[:], Csp[:] = OS.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dmag[i], WA, mode)
#Cb = np.zeros(sInds.shape[0])/u.s#Technically, forcing this to be zero results in the dMag limit that can be achieved with inf time

#Look 
t_dets_inf = np.asarray([t.value + np.inf if t.value > 0. else t.value for t in sim.SurveySimulation.t0])*u.d#sim.SurveySimulation.t0+np.inf*u.d#OK
comp_inf = sim.Completeness.comp_per_intTime(t_dets_inf, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)
sum_comp_inf = sum(comp_inf[comp_inf>0.])
t_dets_BIG = np.asarray([t.value + 1000. if t.value > 0. else t.value for t in sim.SurveySimulation.t0])*u.d#sim.SurveySimulation.t0+1000.*u.d#OK
comp_BIG = sim.Completeness.comp_per_intTime(t_dets_BIG, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)
sum_comp_BIG = sum(comp_inf[comp_BIG>0.])
t_dets = sim.SurveySimulation.t0#OK
comp = sim.Completeness.comp_per_intTime(t_dets, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)
sum_comp = sum(comp[comp>0.])

Cb = np.zeros(sInds.shape[0])/u.s#Technically, forcing this to be zero results in the dMag limit that can be achieved with inf time
comp_Cb0 = sim.Completeness.comp_per_intTime(t_dets, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)
sum_comp_Cb0 = sum(comp_Cb0[comp_Cb0>0.])
#########################################################################################################


#To Get RA of All Targets
sim.TargetList.coords.ra
#To Get DEC of All Targets
sim.TargetList.coords.dec


def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))
    lens = map(len, arrs)
    dim = len(arrs)
    sz = 1
    for s in lens:
       sz *= s
    ans = []
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)
    return tuple(ans)

# def observedCompHammer(ra,dec,comp):
#     """
#     Args:
#         ra (numpy array) - numpy array of floats (unitless) with length number of stars in target list
#             containing the right ascension of each star in deg
#         dec (numpy array) - numpy array of floats (unitless) with length number of stars in target list
#             containing the declination of each star in deg
#     """

# #Simple Plot of Temporal Coverage of Sky
# tmpfig2 = figure(figsize=(14,4))#,num=3)
# #gs = GridSpec(2,5, width_ratios=[4,1,0.3,4,1.25], height_ratios=[1,4])
# #gs.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 
# rc('axes',linewidth=2)
# rc('lines',linewidth=2)
# rcParams['axes.linewidth']=2
# rc('font',weight='bold')
# sb1 = subplot(111,projection="hammer")
# grid(True)







close('all')
#### Calculate Ecliptic Latitude and Longitude of Stars
ra = sim.TargetList.coords.ra.value
dec = sim.TargetList.coords.dec.value
ra = ra*np.pi/180. -np.pi#The right ascension of the stars in the heliocentric ecliptic fixed frame
ra2 = ra[comp > 0.]
dec = dec*np.pi/180.#The declinations of the stars in the heliocentric ecliptic fixed frame
dec2 = dec[comp > 0.]
x = np.cos(dec2)*np.cos(ra2)#When dec2 =0, x/y=1
y = np.cos(dec2)*np.sin(ra2)
z = np.sin(dec2)
r_stars_equat = np.asarray([[x[i],y[i],z[i]] for i in np.arange(len(x))])
r_stars_equat = np.divide(r_stars_equat,np.asarray([np.linalg.norm(r_stars_equat,axis=1).tolist()]).T)*u.AU#Target stars in the equatorial coordinate frame
r_stars_eclip = sim.Observatory.equat2eclip(r_stars_equat,sim.TimeKeeping.currentTimeAbs,rotsign=1).value#target stars in the heliocentric ecliptic frame
hEclipLat = np.arcsin(r_stars_eclip[:,2])
hEclipLon = np.arctan2(r_stars_eclip[:,1],r_stars_eclip[:,0])
#######

#Generate evenly distributed points of reference sphere###############
xyzpoints, lat_lon = generateEquadistantPointsOnSphere(N=30)
lon_sphere = lat_lon[:,1] - np.pi #lon of points distributed over sphere
lat_sphere = lat_lon[:,0] - np.pi/2. #lat of points distributed over sphere
x = np.cos(lat_sphere)*np.cos(lon_sphere)
y = np.cos(lat_sphere)*np.sin(lon_sphere)
z = np.sin(lat_sphere)
r_sphere = np.asarray([[x[i],y[i],z[i]] for i in np.arange(len(x))])
r_sphere = np.divide(r_sphere,np.asarray([np.linalg.norm(r_sphere,axis=1).tolist()]).T)
lat_lon2 = np.asarray([lon_sphere,lat_sphere])
#################################################

#Calculate distances between stars and points##########
hStars = np.zeros(len(r_sphere[:,0]))
for ind in np.arange(len(r_stars_eclip[:,0])):
    r_diff = r_sphere - r_stars_eclip[ind]
    d_diff = np.linalg.norm(r_diff,axis=1)
    minInd = np.argmin(d_diff)
    hStars[minInd] += 1
########################################################
#TODO Create method of plotting these bins on the celestial sphere

#### Histograms of Star Count vs Heliocentric Ecliptic Longitude ##################################
histInterp, targUnderSpline, sumh, xdiff, edges = generateHistHEL(hEclipLon)
#histIntep is a periodic interpolant from -pi to pi of planned observation planet occurence frequency
#targUnderSpline and sumh are here to ensure consistency of the integral
###################################################################################

#### Generate Random OB distributions #####################
# 1 Use Observing Block Durations Previously Defined OBdur2
# 2 Determine 
generatePlannedObsTimeHistHEL(edges,t_dets,comp,hEclipLon)
###########################################################


#####Seaborn Plot Attempt
tmpfig0 = figure(figsize=(14,4),num=1000)#,num=3)
#sb2 = subplot(111,projection="rectilinear")
sb2 = subplot(111,projection="hammer")
cm3 = cm.get_cmap('winter')
#gca()
sns.kdeplot(hEclipLon,hEclipLat,cbar=cm3, shade=True, shade_lowest=False)
cm1 = cm.get_cmap('autumn')
scatter(hEclipLon,hEclipLat,c=comp[comp > 0.],cmap=cm1)
#sb2.colorbar()
show(block=False)
###########



#Simple Plot of Temporal Coverage of Sky
tmpfig2 = figure(figsize=(14,4),num=3000)#,num=3)
#gs = GridSpec(2,5, width_ratios=[4,1,0.3,4,1.25], height_ratios=[1,4])
#gs.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 
rc('axes',linewidth=2)
rc('lines',linewidth=2)
rcParams['axes.linewidth']=2
rc('font',weight='bold')
sb1 = subplot(111,projection="hammer")
grid(True)

num1 = 15#+1
num2 = num1/2.#+1
xmin =-np.pi
xmax=np.pi
ymin=-np.pi/2.
ymax=np.pi/2.
x = np.linspace(xmin,xmax,num=int(num1),endpoint=True)#np.arange(0.,360.)
y = np.linspace(ymin,ymax,num=int(num2),endpoint=True)#np.arange(0.,180.)-90.
#x = np.append(x,x[-1]+(x[-1]-x[-2]))
#y = np.append(y,y[-1]+(y[-1]-y[-2]))

bins = (x,y)#(x[::10],y[::10])
h, xedges, yedges = np.histogram2d(hEclipLon,hEclipLat,bins=bins,density=True)#,normed=True#,range=[[xmin,xmax],[ymin,ymax]])
cm2 = cm.get_cmap('winter')
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])


xbins = x
ybins = y
xcents = np.diff(xbins)/2.+xbins[:-1]
ycents = np.diff(ybins)/2.+ybins[:-1]
pcolormesh(xcents,ycents,h.T,cmap=cm2,edgecolors='face')
#contourf(xcents, ycents, h.T, num=15, cmap=cm2,extent=(xmin, xmax, ymin, ymax))#,origin='lower')# intepolation='nearest')
#contourf(X.T, Y.T, h, num=15, cmap=cm2)
colorbar()
# contour(X.T, Y.T, h, num=3, colors='k')
# colorbar()
#contourf(xedges[:-1],yedges[:-1],h,cmap=cm2)
#contourf(X, Y, h)#(xedges, yedges, h)
scatter(xcents[0],ycents[0],color='white')

smhist = gaussian_filter(h, (5,5), mode='wrap')

cm1 = cm.get_cmap('autumn')
scatter(hEclipLon,hEclipLat,c=comp[comp > 0.],cmap=cm1)
title('Observed Targets in the sky',weight='bold',fontsize=12)
colorbar()
tight_layout()
#add colorbar label

tmpfig2 = figure(figsize=(14,4),num=4000)#,num=3)
sb2 = subplot(111,projection="rectilinear")
cm3 = cm.get_cmap('winter')
contourf(xcents, ycents, h.T, num=15, cmap=cm3, extent=(xmin, xmax, ymin, ymax),origin='lower')
colorbar()
scatter(hEclipLon,hEclipLat,c=comp[comp > 0.],cmap=cm1)
colorbar()
scatter(xcents[0],ycents[0],color='white',cmap=cm1)
# rect = Rectangle((start_angles[i],-np.pi/2),angular_widths[i],np.pi,angle=0.0, alpha=0.2)
# sb2.add_patch(rect)
#colorbar()
show(block=False)



#### 2d spline fit of points #######################
ra = sim.TargetList.coords.ra.value
dec = sim.TargetList.coords.dec.value
ra = ra*np.pi/180. -np.pi#The right ascension of the stars in the heliocentric ecliptic fixed frame
ra2 = ra[comp > 0.]
dec = dec*np.pi/180.#The declinations of the stars in the heliocentric ecliptic fixed frame
dec2 = dec[comp > 0.]

####################################################














# right_ascensions = sim.TargetList.coords.ra.value*np.pi/180. -np.pi#The right ascension of the stars in the heliocentric ecliptic fixed frame
# right_ascensions = right_ascensions[comp > 0.]
# declinations = sim.TargetList.coords.dec.value*np.pi/180.#The declinations of the stars in the heliocentric ecliptic fixed frame
# declinations = declinations[comp > 0.]

ra = ra*np.pi/180. -np.pi#The right ascension of the stars in the heliocentric ecliptic fixed frame
ra2 = ra[comp > 0.]
dec = dec*np.pi/180.#The declinations of the stars in the heliocentric ecliptic fixed frame
dec2 = dec[comp > 0.]

ra_dec = np.vstack([ra2,dec2])#vstack of points to create gaussian_kde over
x = np.arange(0.,361.)
y = np.arange(0.,362.)/2.-90.
#xy = np.vstack([x,y])
#xy = np.meshgrid(x,y)

X, Y = np.meshgrid(x, y)
#positions = np.vstack([X.ravel(), Y.ravel()])

# g = meshgrid2(x, y)
# positions = np.vstack(map(np.ravel, g))
# z = gaussian_kde(ra_dec)(positions)

# z2 = np.zeros([len(x),len(y)])
# minX = int(min(positions[0,:]))
# minY = int(min(positions[1,:]))
# for ind in np.arange(len(positions[0,:])):
#     xi = int(positions[0,ind])#this is the ra
#     yi = int(positions[1,ind])#This is the dec

#     #count number of targets around coord in sky
#     cnt = 0
#     for ind in np.arange(len(comp)):
#         if comp[ind]>0.:
#             #do stuff
#             r_coord = np.asarray([np.cos(xi*np.pi/180.),np.sin(xi*np.pi/180.),np.sin(yi*np.pi/180.)])
#             r_star = np.asarray([np.cos(ra[ind]*np.pi/180.),np.sin(ra[ind]*np.pi/180.),np.sin(dec[ind]*np.pi/180.)])
#             if np.abs(np.arccos(np.dot(r_coord,r_star)/np.linalg.norm(r_coord)/np.linalg.norm(r_star))) < 20.*np.pi/180.:#checks if star is within 5 def of location
#                 cnt += 1
#     z2[xi-minX,yi-minY] = cnt


#     #Uses z from the gaussian_kde but doesn't give what I want
#     #z2[xi-minX,yi-minY] = z[ind]

# contourf(X,Y,np.asarray(z2).T)
# #contour(x,y,z2)
# #contour([x,y,] z)

x = np.arange(0.,360.)
y = np.arange(0.,180.)-90.
xmin =0
xmax=360
ymin=-90
ymax=90
#bins = .#doing evert 10x10deg grid
h, xedges, yedges = np.histogram2d(ra2*180./np.pi,dec2*180./np.pi,bins=(x[::10],y[::10]),normed=True)#bins,range=[[xmin,xmax],[ymin,ymax]])
cm2 = cm.get_cmap('winter')
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
contourf(X, Y, h, 100, cmap=cm2)
colorbar()
contour(X, Y, h, 10, colors='k')
#contourf(xedges[:-1],yedges[:-1],h,cmap=cm2)
colorbar()
#contourf(X, Y, h)#(xedges, yedges, h)

cm1 = cm.get_cmap('autumn')
sc = scatter(ra2,dec2,c=comp[comp > 0.],cmap=cm1)
title('Observed Targets in the sky',weight='bold',fontsize=12)
colorbar()
#add colorbar label
show(block=False)


# savefig('/'.join(pklfile.split('/')[:-1]) + '/' + pkldir + 'SkyCoverage' + '.png')
# savefig('/'.join(pklfile.split('/')[:-1]) + '/' + pkldir + 'SkyCoverage' + '.svg')
# savefig('/'.join(pklfile.split('/')[:-1]) + '/' + pkldir + 'SkyCoverage' + '.eps')

ra = sim.TargetList.coords.ra.value
dec = sim.TargetList.coords.dec.value
observedCompHammer(ra,dec,comp)

