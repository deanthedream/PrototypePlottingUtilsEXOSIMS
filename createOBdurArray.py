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


#### PeriodicDist ##################################################
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
assert numRep <= 365.25/OBdur - 365.25%OBdur, 'numRep too large'
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

Cb2 = np.zeros(sInds.shape[0])/u.s#Technically, forcing this to be zero results in the dMag limit that can be achieved with inf time
comp_Cb0 = sim.Completeness.comp_per_intTime(t_dets, TL, sInds, fZ, fEZ, WA, mode, Cb2, Csp)

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

#### From EXOSIMS/util/evenlyDistributePointsOnSphere.py
from evenlyDistributePointsOnSphere import splitOut, nlcon2, f, pt_pt_distances, secondSmallest, setupConstraints, initialXYZpoints
from scipy.optimize import minimize
x, y, z, v = initialXYZpoints(num_pts=30) # Generate Initial Set of XYZ Points
con = setupConstraints(v,nlcon2) # Define constraints on each point of the sphere
x0 = v.flatten() # takes v and converts it into [x0,y0,z0,x1,y1,z1,...,xn,yn,zn]
out1k = minimize(f,x0, method='SLSQP',constraints=(con), options={'ftol':1e-4, 'maxiter':1000}) # run optimization problem for 1000 iterations
out1kx, out1ky, out1kz = splitOut(out1k)
out1kv = np.asarray([[out1kx[i], out1ky[i], out1kz[i]] for i in np.arange(len(out1kx))])
dist1k = pt_pt_distances(out1kv)
####################################################################

close(50067)
fig = figure(num=50067)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(out1kv[:,0], out1kv[:,1], out1kv[:,2], color='black')
show(block=False)

d_diff_pts_array = list()
inds_of_closest = list()
diff_closest = list()
for i in np.arange(len(out1kv)):
    xyzpoint = out1kv[i] # extract a single xyz point on sphere
    diff_pts = out1kv - xyzpoint # calculate angular difference between point spacing
    d_diff_pts = np.linalg.norm(diff_pts,axis=1) # calculate linear distance between points
    d_diff_pts_array.append(d_diff_pts)
    inds_of_closest.append(d_diff_pts_array[i].argsort()[:7])
    diff_closest.append(d_diff_pts_array[i][inds_of_closest[i]])

    #### Plot closest segments
    plotted = list() #keeps track of index-to-index lines plotted
    for j in np.delete(inds_of_closest[i],0):
        if [i,j] in plotted or [j,i] in plotted:
            continue
        ax.plot([xyzpoint[0],out1kv[j,0]],[xyzpoint[1],out1kv[j,1]],[xyzpoint[2],out1kv[j,2]],color='red')
        plotted.append([i,j])
show(block=False)
d_diff_pts_array = np.asarray(d_diff_pts_array)



# #### Find Closest Distance between two arbitrary lines ##############
# fig_alines = figure(num=1001010)
# ax = fig_alines.add_subplot(111, projection='3d')
# v1 = np.asarray([1.,2.,3.])
# v1 = v1/np.linalg.norm(v1)
# v2 = np.asarray([-1.,-5.,1.])
# v2 = v2/np.linalg.norm(v2)
# pt1 = np.asarray([0.,0.,0.])
# pt2 = np.asarray([1.,1.,1.])
# # d1 = np.dot(v1,pt1)
# # d2 = np.dot(v2,pt2)
# # v3 = np.cross(v1,v2)/np.linalg.norm(np.cross(v1,v2))
# t1 = 2.
# ax.plot([pt1[0],pt1[0]+v1[0]*t1],[pt1[1],pt1[1]+v1[1]*t1],[pt1[2],pt1[2]+v1[2]*t1],color='red')
# t2 = 2.
# ax.plot([pt2[0],pt2[0]+v2[0]*t2],[pt2[1],pt2[1]+v2[1]*t2],[pt2[2],pt2[2]+v2[2]*t2],color='blue')
# # #The question, What point do I choose as my starting point for v3?
# # #FALSE: I think if d1=d2=0, then those two lines intersect... 
# # c_eqn1 = np.append(v1, d1)# coefficients of equation 1
# # c_eqn2 = np.append(v2, d2)
# # d3=0.#d3 is unknown. give it value of 0
# # c_eqn3 = np.append(v3, d3) 
# # #hmmm I shouldn't need to do these things... 
# # #Do row ops to make v1[0] = 1 and all others 0
# # c_eqn2 = c_eqn2 - (c_eqn2[0]/c_eqn1[0])*c_eqn1 #should make coeff of c_eqn2[0]=0
# # c_eqn3 = c_eqn3 - (c_eqn3[0]/c_eqn1[0])*c_eqn1 #should make coeff of c_eqn3[0]=0
# # c_eqn1 = c_eqn1/c_eqn1[0] #make coeff of c_eqn1[0]=1
# # #Do row ops equation 2
# # c_eqn1 = c_eqn1 - (c_eqn1[1]/c_eqn2[1])*c_eqn2 #make coeff of c_eqn1[1]=0
# # c_eqn3 = c_eqn3 - (c_eqn3[1]/c_eqn2[1])*c_eqn2 #make coeff of c_eqn3[1]=0
# # c_eqn2 = c_eqn2/c_eqn2[1] #make coeff of c_eqn2[1]=1
# # #Do row ops equation 3
# # c_eqn1 = c_eqn1 - (c_eqn1[2]/c_eqn3[2])*c_eqn3 #make coeff of c_eqn1[2]=0
# # c_eqn2 = c_eqn2 - (c_eqn2[2]/c_eqn3[2])*c_eqn3 #make coeff of c_eqn2[2]=0
# # c_eqn3 = c_eqn3/c_eqn3[2] #make coeff of c_eqn3[2]=1

# # #distances along vectors to get to P1 and P2
# # t2 = (-np.dot(pt2,v2) +np.dot(pt1,v1) +(np.dot(pt2,v1) -np.dot(pt1,v1))*np.dot(v1,v2))/(1.-np.dot(v2,v1)*np.dot(v1,v2))
# # t1 = np.dot(pt2,v1)+ t2*np.dot(v2,v1)-np.dot(pt1,v1)

# # #distances along vectors to get to P1 and P2
# # t2 = (-np.dot(pt2,v2) +np.dot(pt1,v1) +(np.dot(pt2,v1) -np.dot(pt1,v1))*np.dot(v1,v2))/(1.-np.dot(v2,v1)*np.dot(v1,v2))
# # t1 = np.dot(pt2,v1)+ t2*np.dot(v2,v1)-np.dot(pt1,v1)

# # PQ = pt2 + t2 * v2 - pt1 - t1 * v1

# # P1 = pt1 + t1*v1
# # P2 = pt2 + t2*v2
# # P1P2 = P2-P1
# # P1P2hat = P1P2/np.linalg.norm(P1P2)
# # dp1 = np.dot(P1P2,v1)
# # dp2 = np.dot(P1P2,v2)


# A = np.asarray([[-1.,np.dot(v2,v1)],[-1.*np.dot(v1,v2),1.]])
# Ainv = np.linalg.inv(A)
# b = np.asarray([[np.dot(pt1,v1)-np.dot(pt2,v1)],[np.dot(pt1,v2)-np.dot(pt2,v2)]])
# x = np.dot(Ainv,b)
# t1 = x[0][0]
# t2 = x[1][0]
# P1 = pt1 + t1*v1
# P2 = pt2 + t2*v2
# P1P2 = P2-P1
# P1P2hat = P1P2/np.linalg.norm(P1P2)
# dp1 = np.dot(P1P2,v1)
# dp2 = np.dot(P1P2,v2)



# pt1 = np.asarray([0.,0.,0.])
# pt2 = np.asarray([1.,1.,1.])
# v1 = np.asarray([1.,2.,3.])
# v2 = np.asarray([-1.,-5.,1.])
# A = np.asarray([[14.,8.],[8.,27.]])
# Ainv = np.linalg.inv(A)
# b = np.asarray([[6.],[-5.]])
# x = np.matmul(Ainv,b)
# t1 = x[0][0]
# t2 = x[1][0]
# P1 = pt1 + t1*v1
# P2 = pt2 + t2*v2
# P1P2 = P2-P1
# P1P2hat = P1P2/np.linalg.norm(P1P2)
# dp1 = np.dot(P1P2,v1)
# dp2 = np.dot(P1P2,v2)


# p1 = np.asarray([[0.,0.,0.]]).T
# p2 = np.asarray([[1.,1.,1.]]).T
# zeros = np.zeros([3,1])
# v1 = np.asarray([1.,2.,3.])
# v2 = np.asarray([-1.,-5.,1.])
# v3 = np.cross(v1,v2)
# v1 = np.asarray([v1]).T
# v2 = np.asarray([v2]).T
# v3 = np.asarray([v3]).T
# I = np.eye(3,3)
# zm31 = np.zeros([3,1])
# zm33 = np.zeros([3,3])

# A = [   [zm31,     v2,   zm33,     -I, zm31   ],
#         [  v1,   zm31,   zm33,     -I,   v3   ],
#         [  v1,   zm31,     -I,   zm33, zm31   ],
#         [zm31,     v2,      I,   zm33,  -v3   ],
#         [zm31,   zm31,      I,     -I,   v3   ]]
# A2 = np.zeros([3*5,15])
# # for row in np.arange(len(A)):
# #     colCount1 = 0
# #     for col in np.arange(len(A[row])):
# #         for row2 in np.arange(len(A[row][col])):
# #             colCount2 = colCount1
# #             for col2 in np.arange(len(A[row][col][row2])):
# #                 A2[3*row+row2][colCount1 + colCount2] = A[row][col][row2][col2]
# #                 colCount2 += 1
# #         colCount1 += 1
# # #A2inv = np.linalg.inv(A2)
# v1 = v1.T[0]
# v2 = v2.T[0]
# v3 = v3.T[0]
# A2[0:3,3:6] = np.diag(v2[0:3])
# A2[3:6,0:3] = np.diag(v1[0:3])
# A2[6:9,0:3] = np.diag(v1[0:3])
# A2[9:12,3:6] = np.diag(v2[0:3])
# A2[0:3,11:14] = -np.eye(3,3)
# A2[3:6,11:14] = -np.eye(3,3)
# A2[3:6,14] = v3[0:3]
# A2[6:9,8:11] = -np.eye(3,3)
# A2[9:12,8:11] = np.eye(3,3)
# A2[12:15,8:11] = np.eye(3,3)
# A2[12:15,11:14] = -np.eye(3,3)
# A2[9:12,14] = -v3[0:3]
# A2[12:15,14] = v3[0:3]



# b = [   -p2,
#         -p1,
#         -p1,
#         -p2,
#         np.asarray([[0,0,0]]).T]
# ### Reshape so b isn't arrays of arrays
# b2 = np.zeros([1,14])
# for row in np.arange(len(b)):
#     for row2 in np.arange(len(b[row])):
#         b2[0,3*row+row2] = b[row][row2]
# x = np.linalg.lstsq(A2.T,b2.T)
# #todo convert these into proper A and b matrices

# x = np.matmul(A2inv,b2)


# ax.scatter(P1[0],P1[1],P1[2],color='black')
# ax.scatter(P2[0],P2[1],P2[2],color='green')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# show(block=False)


# #for t=
# c11 = np.dot(pt2,v1)
# c12 = np.dot(v2,v1)
# c13 = -np.dot(pt1,v1)
# #for s=
# c21 = -np.dot(pt2,v2)
# c22 = np.dot(pt1,v2)
# c23 = np.dot(v1,v2)

# # s = c21 + c22 + c23*(c11+c12*s+c13)
# # s - c23*c12*s = c21 + c22 + c23*(c11+c13)
# s = (c21 + c22 + c23*(c11+c13))/(1.-c23*c12)
# # we don't know what point v3 goes through so we can't calculate d3
# # technically, we want to ???minimize??? abs(d3) (I think)
# #d3 = np.dot(v3,pt1) # we are able to use either pt1 or pt2 to find d3 since this line goes through both


# #I have found algorithm to calculate shortest distance between two lines in 3D and rewrite it in Python.But also I would like to enchanced it that it could return not only distance,but also the postion of closest points.
# def line2line(p0,v0,p1,v1):#(-0.073455669 4.9843092 0.26107353 0.0 0.0 -1.0 -3.85838175 12.1999998 -4.50372314 0.405142069 -0.76723671 0.497199893):
#     epsilon = 0.00000001
#     #L1P0 = np.array([xbeam,ybeam,zbeam]) #position of P0 on first line
#     #L2P0= np.array([xout,yout,zout]) #position of P0 on first line
#     #L1P1 = np.array([xbeam + ubeam ,ybeam + vbeam ,zbeam + wbeam]) #ubeam,vbeam and wbeam are direction cosines
#     #L2P1 = np.array([xout + cx,yout + cy,zout + cz]) #cx,cy,cz are direction cosines
#     # u = v0#L1P1 - L1P0
#     # v = v1#L2P1 - L2P0
#     # w = p1-p0#L1P0 - L2P0
#     # a = np.dot(u,u)
#     # b = np.dot(u,v)
#     # c = np.dot(v,v)
#     # d = np.dot(u,w)
#     # e = np.dot(v,w)
#     # D = a*c - b*b
#     # if D < epsilon:
#     #     sc = 0.0
#     #     tc = d/b if b>c else e/c
#     # else:
#     #     sc = (b*e - c*d) / D
#     #     tc = (a*e - b*d) / D
#     # dP = w + (sc * u) - (tc * v)

#     a = np.dot(v0,v0)
#     b = np.dot(v0,v1)
#     c = np.dot(v1,v1)
#     d = np.dot(v0,p1-p0)
#     e = np.dot(v1,p1-p0)
#     D = a*c - b*b
#     if D < epsilon:
#         sc = 0.0
#         tc = d/b if b>c else e/c
#     else:
#         sc = (b*e - c*d) / D
#         tc = (a*e - b*d) / D
#     dP = p1-p0 + (sc * v0) - (tc * v1)
#     return np.linalg.norm(dP)


def line2linev2(p0,v0,p1,v1):
    epsilon = 0.00000001
    a = np.dot(v0,v0)
    b = np.dot(v0,v1)
    c = np.dot(v1,v1)
    d = np.dot(v0,p1-p0)
    e = np.dot(v1,p1-p0)
    D = a*c - b*b#had 
    if D < epsilon:
        t0 = 0.0
        t1 = d/b if b>c else e/c
    else:
        t0 = (b*e - c*d) / D
        t1 = (a*e - b*d) / D
    dP = p1-p0 + t0*v0 - t1*v1#p1-p0 + (sc * v0) - (tc * v1)
    return np.linalg.norm(dP), dP, p0-t0*v0, p1-t1*v1, t0, t1
p0 = np.asarray([0., 0., 0.])
p1 = np.asarray([1., 1., 2.])
v0 = np.asarray([1., 0., 1.])
v1 = np.asarray([0., 1., 0.])
out = line2linev2(p0,v0,p1,v1)
t0 = out[4]
t1 = out[5]
q0 = out[2]
q1 = out[3]
dP = out[1]
close(2055121)
fig = figure(num=2055121)
ax = fig.add_subplot(111, projection='3d')
ax.plot([p0[0],p0[0]-t0*v0[0]],[p0[1],p0[1]-t0*v0[1]],[p0[2],p0[2]-t0*v0[2]],color='red')
ax.plot([p1[0],p1[0]-t1*v1[0]],[p1[1],p1[1]-t1*v1[1]],[p1[2],p1[2]-t1*v1[2]],color='blue')
ax.scatter(p0[0],p0[1],p0[2],color='black')#starting points
ax.scatter(p1[0],p1[1],p1[2],color='black')#starting points
ax.plot([q0[0],q1[0]],[q0[1],q1[1]],[q0[2],q1[2]],color='purple')
ax.scatter(q0[0],q0[1],q0[2],color='purple')#ending points
ax.scatter(q1[0],q1[1],q1[2],color='purple')#ending points
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(-1,-1,-1,color='white')
ax.scatter(3,3,3,color='white')
show(block=False)
#####################################################################



#Generate evenly distributed points of reference sphere###############
xyzpoints, lat_lon = generateEquadistantPointsOnSphere(N=40)
lon_sphere = lat_lon[:,1] - np.pi #lon of points distributed over sphere
lat_sphere = lat_lon[:,0] - np.pi/2. #lat of points distributed over sphere
x = np.cos(lat_sphere)*np.cos(lon_sphere)
y = np.cos(lat_sphere)*np.sin(lon_sphere)
z = np.sin(lat_sphere)
r_sphere = np.asarray([[x[i],y[i],z[i]] for i in np.arange(len(x))])
r_sphere = np.divide(r_sphere,np.asarray([np.linalg.norm(r_sphere,axis=1).tolist()]).T)
lat_lon2 = np.asarray([lon_sphere,lat_sphere])
#################################################
#ATTEMPT USING GABE's Code
from EXOSIMS.StarCatalog import FakeCatalog
FC = FakeCatalog.FakeCatalog()
gabe_coords = FC.partitionSphere(40,1)
gra = gabe_coords.ra.value*np.pi/180.
gdec = gabe_coords.dec.value*np.pi/180.
##########################

#Calculate distances between stars and points##########
hStars = np.zeros(len(r_sphere[:,0]))
for ind in np.arange(len(r_stars_eclip[:,0])):
    r_diff = r_sphere - r_stars_eclip[ind]#skycoords
    d_diff = np.linalg.norm(r_diff,axis=1)
    minInd = np.argmin(d_diff)
    hStars[minInd] += 1
########################################################
# Now imported from evenlyDistributePointsOnSphere.py
# def secondSmallest(d_diff_pts):
#     """For a list of points, return the value and ind of the second smallest
#     args:
#         d_diff_pts - numy array of floats of distances between points
#     returns:
#         secondSmallest_value - 
#         secondSmallest_ind - 
#     """
#     tmp_inds = np.arange(len(d_diff_pts))
#     tmp_inds_min0 = np.argmin(d_diff_pts)
#     tmp_inds = np.delete(tmp_inds, tmp_inds_min0)
#     tmp_d_diff_pts =np.delete(d_diff_pts, tmp_inds_min0)
#     secondSmallest_value = min(tmp_d_diff_pts)
#     secondSmallest_ind = np.argmin(np.abs(d_diff_pts - secondSmallest_value))
#     return secondSmallest_value, secondSmallest_ind

#TODO Create method of plotting these bins on the celestial sphere
#We will use the points as the vertices of the edges. Divide all into triangles
#Iterate over each xyzpoint in xyzpoints and find top N closest to current point (save as indexes)
closest_point_inds = list() # list of numpy arrays containing closest points to a given ind
for i in np.arange(len(xyzpoints)):
    xyzpoint = xyzpoints[i] # extract a single xyz point on sphere
    diff_pts = xyzpoints - xyzpoint # calculate angular difference between point spacing
    d_diff_pts = np.linalg.norm(diff_pts,axis=1) # calculate linear distance between points
    #base distance off closest point (that is not ~0, itself)
    # tmp_inds = np.arange(len(d_diff_pts))
    # tmp_inds_min0 = np.argmin(d_diff_pts)
    # tmp_inds = np.delete(tmp_inds, tmp_inds_min0)
    # tmp_d_diff_pts =np.delete(d_diff_pts, tmp_inds_min0)
    # secondSmallest_value = min(tmp_d_diff_pts)
    # secondSmallest_ind = np.argmin(np.abs(d_diff_pts - secondSmallest_value))
    ss_d, ss_ind = secondSmallest(d_diff_pts)
    ss_d = ss_d*1.5 # This is a factor that defines the distance that all other points will be away from the current point. Just a guess
    tmp_closest_point_inds = np.where((d_diff_pts < ss_d)*(d_diff_pts != min(d_diff_pts)))[0]
    
    r_closest = list()
    for j in np.arange(len(tmp_closest_point_inds)-1):
        r_closest.append(xyzpoints[tmp_closest_point_inds[j]] - xyzpoint) # gets vector from center point to closest points

    #r_closest[0]
    #np.cross()
    closest_point_inds.append(tmp_closest_point_inds)


    

fig = figure(num=5000)
ax = fig.axes[0]#add_subplot(111, projection='3d')
# ax = gca()
for i in closest_point_inds[0]:
    ax.scatter(xyzpoints[i,0], xyzpoints[i,1], xyzpoints[i,2], color='r', marker='+',s=200)
for i in np.arange(len(xyzpoints)):
    for j in np.arange(len(closest_point_inds[i])):
        ind = closest_point_inds[i][j]
        ax.plot([xyzpoints[i,0],xyzpoints[ind,0]],[xyzpoints[i,1],xyzpoints[ind,1]],[xyzpoints[i,2],xyzpoints[ind,2]])
show(block=False)






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














# # right_ascensions = sim.TargetList.coords.ra.value*np.pi/180. -np.pi#The right ascension of the stars in the heliocentric ecliptic fixed frame
# # right_ascensions = right_ascensions[comp > 0.]
# # declinations = sim.TargetList.coords.dec.value*np.pi/180.#The declinations of the stars in the heliocentric ecliptic fixed frame
# # declinations = declinations[comp > 0.]

# ra = ra*np.pi/180. -np.pi#The right ascension of the stars in the heliocentric ecliptic fixed frame
# ra2 = ra[comp > 0.]
# dec = dec*np.pi/180.#The declinations of the stars in the heliocentric ecliptic fixed frame
# dec2 = dec[comp > 0.]

# ra_dec = np.vstack([ra2,dec2])#vstack of points to create gaussian_kde over
# x = np.arange(0.,361.)
# y = np.arange(0.,362.)/2.-90.
# #xy = np.vstack([x,y])
# #xy = np.meshgrid(x,y)

# X, Y = np.meshgrid(x, y)
# #positions = np.vstack([X.ravel(), Y.ravel()])

# # g = meshgrid2(x, y)
# # positions = np.vstack(map(np.ravel, g))
# # z = gaussian_kde(ra_dec)(positions)

# # z2 = np.zeros([len(x),len(y)])
# # minX = int(min(positions[0,:]))
# # minY = int(min(positions[1,:]))
# # for ind in np.arange(len(positions[0,:])):
# #     xi = int(positions[0,ind])#this is the ra
# #     yi = int(positions[1,ind])#This is the dec

# #     #count number of targets around coord in sky
# #     cnt = 0
# #     for ind in np.arange(len(comp)):
# #         if comp[ind]>0.:
# #             #do stuff
# #             r_coord = np.asarray([np.cos(xi*np.pi/180.),np.sin(xi*np.pi/180.),np.sin(yi*np.pi/180.)])
# #             r_star = np.asarray([np.cos(ra[ind]*np.pi/180.),np.sin(ra[ind]*np.pi/180.),np.sin(dec[ind]*np.pi/180.)])
# #             if np.abs(np.arccos(np.dot(r_coord,r_star)/np.linalg.norm(r_coord)/np.linalg.norm(r_star))) < 20.*np.pi/180.:#checks if star is within 5 def of location
# #                 cnt += 1
# #     z2[xi-minX,yi-minY] = cnt


# #     #Uses z from the gaussian_kde but doesn't give what I want
# #     #z2[xi-minX,yi-minY] = z[ind]

# # contourf(X,Y,np.asarray(z2).T)
# # #contour(x,y,z2)
# # #contour([x,y,] z)

# x = np.arange(0.,360.)
# y = np.arange(0.,180.)-90.
# xmin =0
# xmax=360
# ymin=-90
# ymax=90
# #bins = .#doing evert 10x10deg grid
# h, xedges, yedges = np.histogram2d(ra2*180./np.pi,dec2*180./np.pi,bins=(x[::10],y[::10]),normed=True)#bins,range=[[xmin,xmax],[ymin,ymax]])
# cm2 = cm.get_cmap('winter')
# X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
# contourf(X, Y, h, 100, cmap=cm2)
# colorbar()
# contour(X, Y, h, 10, colors='k')
# #contourf(xedges[:-1],yedges[:-1],h,cmap=cm2)
# colorbar()
# #contourf(X, Y, h)#(xedges, yedges, h)

# cm1 = cm.get_cmap('autumn')
# sc = scatter(ra2,dec2,c=comp[comp > 0.],cmap=cm1)
# title('Observed Targets in the sky',weight='bold',fontsize=12)
# colorbar()
# #add colorbar label
# show(block=False)


# # savefig('/'.join(pklfile.split('/')[:-1]) + '/' + pkldir + 'SkyCoverage' + '.png')
# # savefig('/'.join(pklfile.split('/')[:-1]) + '/' + pkldir + 'SkyCoverage' + '.svg')
# # savefig('/'.join(pklfile.split('/')[:-1]) + '/' + pkldir + 'SkyCoverage' + '.eps')

# ra = sim.TargetList.coords.ra.value
# dec = sim.TargetList.coords.dec.value
# observedCompHammer(ra,dec,comp)

