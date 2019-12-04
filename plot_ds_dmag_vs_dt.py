""" ds and dmag vs dt
This function loads a series of .pkl files, takes all detected planets, plots (r,dMag, s) and propagates systems for varying dt

Written By: Dean Keithly
Written On: 8/22/2019
"""

import glob
import random as myRand
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
try:
    import cPickle as pickle
except:
    import pickle
import os
import numpy as np
#from pylab import *
from numpy import nan
if not 'DISPLAY' in os.environ.keys(): #Check environment for keys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
else:
    import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import json
from EXOSIMS.util.read_ipcluster_ensemble import gen_summary
from EXOSIMS.util.read_ipcluster_ensemble import read_all
from numpy import linspace
from matplotlib.ticker import NullFormatter, MaxNLocator
from matplotlib import ticker
import astropy.units as u
import matplotlib.patheffects as PathEffects
import datetime
import re
from EXOSIMS.util.vprint import vprint
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


#folder = '/home/dean/Documents/SIOSlab/EXOSIMSres/HabExCompSpecPriors_HabEx_4m_TSDD_pop100DD_revisit_20180424/HabEx_CSAG13_PPSAG13'
folder = '/home/dean/Documents/SIOSlab/EXOSIMSres/HabExCompSpecPriors_HabEx_4m_50119/HabEx_CKL2_PPKL2'#HABEX stuff
folder = '/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_405_19/WFIRSTcycle6core_CSAG13_PPSAG13'#WFIRST stuff
PPoutpath = './'

if not os.path.exists(folder):#Folder must exist
    raise ValueError('%s not found'%folder)
if not os.path.exists(PPoutpath):#PPoutpath must exist
    raise ValueError('%s not found'%PPoutpath) 
outspecfile = os.path.join(folder,'outspec.json')
if not os.path.exists(outspecfile):#outspec file not found
    raise ValueError('%s not found'%outspecfile) 
try:
    with open(outspecfile, 'rb') as g:
        outspec = json.load(g, encoding='latin1')
except:
    vprint('Failed to open outspecfile %s'%outspecfile)
    pass


#Extract Data from folder containing pkl files
out = gen_summary(folder)#out contains information on the detected planets
#allres = read_all(folder)# contains all drm from all missions in folder

outspec['cachedir'] = '/home/dean/.EXOSIMS/cache'

#Create Mission Object To Extract Some Plotting Limits
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=None, nopar=True, **outspec)
#DELETE? ymax = np.nanmax(sim.PlanetPhysicalModel.ggdat['radii']).to('earthRad').value

SU = sim.SimulatedUniverse
TL = sim.TargetList
#I care about 
#SU.r
#SU.dmag
#SU.S

def calc_az(r_init, SU):
    """
    Args:
        r_init (numpy array) - array of initial planet positions
        SU (object) - simulated universe object
    Returns:
        az (numpy array) - the delta in azimuth between the two vectors
    """
    az_init = np.arctan2(r_time[0][:,1].value.copy(), r_time[0][:,0].value.copy())
    az_time = np.arctan2(SU.r[:,1].value.copy(),SU.r[:,0].value.copy())
    az1 = abs(az_time-az_init) #turns into absolute angles
    az2 = [az if az < np.pi else 2.*np.pi-az for az in az1] #converts to +/-np.pi
    return az2

# Declare Storage Arrays for dMag[planet_ind, time_ind]
dMag_time_agg = np.asarray([])#list()
s_time_agg = np.asarray([])#list()
r_time_agg = np.asarray([])#list()
az_time_agg = np.asarray([])#list()

pklfiles = glob.glob(os.path.join(folder,'*.pkl'))
for counter,f in enumerate(pklfiles[0:20]):
    print("%d/%d"%(counter,len(pklfiles)))
    with open(f, 'rb') as g:
        res = pickle.load(g, encoding='latin1')

    #This must happen here bc each res will have planets around the same host star therefore causing multiple times
    #Take SU object
    #SU.init_systems

    #Figure out planet inds that have been detected. Then filter below
    pInds = list()
    obsInd = list() # list of observations where detections were made
    for i in np.arange(len(res['DRM'])):#iterate over each observation
        if len(res['DRM'][i]['det_status']) == 0:#skip if no detections
            continue
        obsInd.append(i)
        for j in np.arange(len(res['DRM'][i]['det_status'])): #iterate over all det_status
            if res['DRM'][i]['det_status'][j] == 1: #if the planet was detected
                pInds.append(res['DRM'][i]['plan_inds'][j])#add to list of pInds detected


    ######## HERE IS THE ISSUE! I NEED TO COME UP WITH OBS CONTAINING DETECTIONS

    #Extract a,e,I,O,w,M0,Mp, plan2star ,TL.MsTrue from pkl file
    SU.a = res['systems']['a'][pInds]
    SU.e = res['systems']['e'][pInds]
    SU.I = res['systems']['I'][pInds]
    SU.O = res['systems']['O'][pInds]
    SU.w = res['systems']['w'][pInds]
    SU.M0 = res['systems']['M0'][pInds]
    SU.Mp = res['systems']['Mp'][pInds]
    SU.Rp = res['systems']['Rp'][pInds]
    SU.p = res['systems']['p'][pInds]
    SU.plan2star = res['systems']['plan2star'][pInds]
    TL.MsTrue = res['systems']['MsTrue']

    #Keep for debugging
    # print(len(SU.a))
    # print(len(SU.e))
    # print(len(SU.I))
    # print(len(SU.O))
    # print(len(SU.w))
    # print(len(SU.M0))
    # print(len(SU.Mp))
    # print(len(SU.Rp))
    # print(len(SU.p))
    # print(len(SU.plan2star))
    # print(len(TL.MsTrue))

    SU.init_systems() #this should calculate self.r, self.d, ...
    dt_i = list()
    sInd_i = list()
    #TODO propagate each system to the time of detection
    for i in obsInd:#np.arange(len(res['DRM'])):
        dt_i.append(res['DRM'][i]['arrival_time'])
        sInd_i.append(res['DRM'][i]['star_ind']) #
        SU.propag_system(sInd_i[-1], dt_i[-1]) #propagates the system to the time of detection
        #Note all planets will be moved to position when they were

    #extract initial dMag and s of each planet here
    dMag_time = [SU.dMag.copy()]
    s_time = [SU.s.copy()]
    r_time = [SU.r.copy()] #will be sInds x 3 x numdt
    r_init = SU.r.copy()
    az_time = [np.zeros(len(SU.s.copy()))]

    #propagate all by a fixed time, dt, repeatedly
    dt = 1.*u.d
    for i in np.arange(300): #300 was a number I pulled out of my butt
        for j in np.arange(len(sInd_i)):
            SU.propag_system(sInd_i[j],dt)#propagate all systems
        dMag_time.append(SU.dMag.copy())
        s_time.append(SU.s.copy())
        r_time.append(SU.r.copy()) # will be sInds x 3

        #calculate az too!
        az_time.append(calc_az(r_init,SU))

    #convert into numpy arrays
    dMag_time = np.asarray(dMag_time).T
    s_time = np.asarray(s_time).T
    r_time = np.asarray(r_time).T
    az_time = np.asarray(az_time).T


    # dMag_time_agg.append(dMag_time)
    # s_time_agg.append(s_time)
    # r_time_agg.append(r_time)
    # az_time_agg.append(az_time)
    try:
        dMag_time_agg = np.vstack((dMag_time_agg,dMag_time))
        s_time_agg = np.vstack((s_time_agg,s_time)) #.append(s_time)
        r_time_agg = np.concatenate((r_time_agg, r_time), axis=1)
        az_time_agg = np.vstack((az_time_agg,az_time))
    except:
        print('except')
        dMag_time_agg = dMag_time
        s_time_agg = s_time
        r_time_agg = r_time
        az_time_agg = az_time

    #extract new dMag and s of each planet

#### Astrometric and Photometric Uncertainty Assumptions######
U_dmag = 0.1 # pulled out of butt, 1/10 of order of magnitude
U_az = 
U_s = 
##############################################################


dMag_time_agg = np.asarray(dMag_time_agg)
s_time_agg = np.asarray(s_time_agg)
r_time_agg = np.asarray(r_time_agg)
az_time_agg = np.asarray(az_time_agg)

########### Make Plots ##################################
#### dMag vs s
plt.close(10987)
fig = plt.figure(10987,figsize=(10,10))
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
ax10= fig.add_subplot(111)#, projection= '3d')
ax10.scatter(s_time_agg[:,0],dMag_time_agg[:,0], color='red',s=5)#, alpha=1.
#for j in np.arange(len(s_time_agg[:,0])):
for i in np.arange(300):
    ax10.scatter(s_time_agg[:,i],dMag_time_agg[:,i], color='blue', alpha=0.05, s=3)
    #plt.show(block=False)
    # plt.draw()
    # plt.pause(0.1)
    #input('press a key...')
#ax10.scatter(s_time_agg[:,10],dMag_time_agg[:,10], color='orange', alpha=0.2)
#ax10.scatter(s_time_agg[:,40],dMag_time_agg[:,40], color='red', alpha=0.2)

ax10.set_xlabel('s in AU',weight='bold')
ax10.set_ylabel(r'$\Delta \mathrm{mag}$' + ' in AU', weight='bold')
plt.show(block=False)



#### dMag vs s vs az
plt.close(11256)
fig1 = plt.figure(11256,figsize=(20,10))
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
ax11= fig1.add_subplot(121, projection= '3d')
ax12= fig1.add_subplot(122)#, projection= '3d')
ax11.scatter(s_time_agg[:,0],dMag_time_agg[:,0], az_time_agg[:,0], color='red',s=5)#, alpha=1.
ax12.scatter(s_time_agg[:,0],az_time_agg[:,0], color='red', s=5)
ax11.set_xlabel(r'$\Delta s$')
ax11.set_ylabel(r'$\Delta \mathrm{mag}$')
ax11.set_zlabel(r'$\Delta \theta$')
ax11.set_zlim([0.,2.*np.pi])
ax11.set_xlim([0.,np.max(s_time_agg)])
ax12.set_xlabel(r'$\Delta s$')
ax12.set_ylabel(r'$\Delta \theta$')
ax12.set_ylim([0.,2.*np.pi])
ax12.set_xlim([0.,np.max(s_time_agg)])
for i in np.arange(300):
    ax11.scatter(s_time_agg[:,i],dMag_time_agg[:,i], az_time_agg[:,i], color='blue', alpha=0.05, s=3)
    ax12.scatter(s_time_agg[:,i],az_time_agg[:,i], color='blue', alpha=0.05, s=3)
    # plt.show(block=False)
    # plt.draw()
    # plt.pause(0.001)

plt.show(block=False)


#### dMag vs s vs t
plt.close(35486)
fig2 = plt.figure(35486,figsize=(30,10))
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
ax21= fig2.add_subplot(131, projection= '3d')
ax22= fig2.add_subplot(132)#, projection= '3d')
ax23= fig2.add_subplot(133)
ax21.scatter(s_time_agg[:,0],dMag_time_agg[:,0], az_time_agg[:,0], color='red',s=5)#, alpha=1.
ax22.scatter(s_time_agg[:,0],az_time_agg[:,0], color='red', s=10)
ax23.scatter(np.zeros(len(dMag_time_agg[:,0])),dMag_time_agg[:,0], color='red', s=10)
ax21.set_xlabel(r'$s$')
ax21.set_ylabel(r'$\Delta \mathrm{mag}$')
ax21.set_zlabel('Time Since First Observation, ' + r'$\Delta t$' + ' in Days', weight='bold')
ax21.set_zlim([0.,300.])
ax21.set_xlim([0.,np.max(s_time_agg)])
ax22.set_xlabel('Time Since First Observation, ' + r'$\Delta t$' + ' in Days', weight='bold')
ax22.set_ylabel(r'$s$')
ax22.set_ylim([0.,np.max(s_time_agg)])
ax22.set_xlim([0.,300.])
ax23.set_xlabel('Time Since First Observation, ' + r'$\Delta t$' + ' in Days', weight='bold')
ax23.set_ylabel(r'$\Delta \mathrm{mag}$')
ax23.set_xlim([0.,300.])
ax23.set_ylim([0.,np.max(dMag_time_agg)])
for i in np.arange(300):
    ax21.scatter(s_time_agg[:,i],dMag_time_agg[:,i], i, color='blue', alpha=0.05, s=3)
    ax22.scatter(np.zeros(len(s_time_agg[:,0]))+i,s_time_agg[:,i], color='blue', alpha=0.05, s=3)
    ax23.scatter(np.zeros(len(dMag_time_agg[:,0]))+i,dMag_time_agg[:,i], color='blue', alpha=0.05, s=3)
    # plt.show(block=False)
    # plt.draw()
    # plt.pause(0.001)
plt.show(block=False)


#### Calculate ddMag, ds
ddMag_time_agg = np.zeros(list(dMag_time_agg.shape))
ds_time_agg = np.zeros(list(s_time_agg.shape))
for i in np.arange(s_time_agg.shape[0]):
    ddMag_time_agg[i,:] = dMag_time_agg[i,:] - dMag_time_agg[i,0]
    ds_time_agg[i,:] = s_time_agg[i,:] - s_time_agg[i,0]

#### Plotting ddMag, ds, dTheta, dt
plt.close(23831)
fig3 = plt.figure(23831,figsize=(40,10))
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
ax31= fig3.add_subplot(141, projection= '3d')
ax32= fig3.add_subplot(142)#, projection= '3d')
ax33= fig3.add_subplot(143)
ax34= fig3.add_subplot(144)
ax31.scatter(ds_time_agg[:,0],ddMag_time_agg[:,0], az_time_agg[:,0], color='red',s=5)#, alpha=1.
ax32.scatter(np.zeros(len(ddMag_time_agg[:,0])),ds_time_agg[:,0], color='red', s=10)
ax33.scatter(np.zeros(len(ddMag_time_agg[:,0])),ddMag_time_agg[:,0], color='red', s=10)
ax34.scatter(np.zeros(len(ddMag_time_agg[:,0])),az_time_agg[:,0], color='red', s=10)
#labels
ax31.set_xlabel(r'$\Delta s$')
ax31.set_ylabel(r'$\Delta \Delta \mathrm{mag}$')
ax31.set_zlabel(r'$\Delta \Theta$')
ax32.set_ylabel(r'$\Delta s$')
ax32.set_xlabel('Time Since First Observation, ' + r'$\Delta t$' + ' in Days', weight='bold')
ax33.set_ylabel(r'$\Delta \Delta \mathrm{mag}$')
ax33.set_xlabel('Time Since First Observation, ' + r'$\Delta t$' + ' in Days', weight='bold')
ax34.set_ylabel(r'$\Delta \Theta$')
ax34.set_xlabel('Time Since First Observation, ' + r'$\Delta t$' + ' in Days', weight='bold')
#limits
ax31.set_xlim([np.min(ds_time_agg),np.max(ds_time_agg)])
ax31.set_ylim([np.min(ddMag_time_agg),np.max(ddMag_time_agg)])
ax31.set_zlim([0.,np.max(az_time_agg)])
ax32.set_ylim([np.min(ds_time_agg),np.max(ds_time_agg)])
ax32.set_xlim([0.,300.])
ax33.set_ylim([np.min(ddMag_time_agg),np.max(ddMag_time_agg)])
ax33.set_xlim([0.,300.])
ax34.set_ylim([0.,np.max(az_time_agg)])
ax34.set_xlim([0.,300.])
#Plot trails
for i in np.arange(300):
    ax31.scatter(ds_time_agg[:,i],ddMag_time_agg[:,i], az_time_agg[:,i], color='blue', alpha=0.05, s=3)
    ax32.scatter(np.zeros(len(ds_time_agg[:,0]))+i,ds_time_agg[:,i], color='blue', alpha=0.05, s=3)
    ax33.scatter(np.zeros(len(ddMag_time_agg[:,0]))+i,ddMag_time_agg[:,i], color='blue', alpha=0.05, s=3)
    ax34.scatter(np.zeros(len(ddMag_time_agg[:,0]))+i,az_time_agg[:,i], color='blue', alpha=0.05, s=3)
    # plt.show(block=False)
    # plt.draw()
    # plt.pause(0.001)
plt.show(block=False)


print(salktyburrito)

#1 check res for the actual data I need to reconstruct planet position
#res['seed']#useless.... unless I can recreate the simulation object!
"""
##res['DRM'] #contains
'det_params':
    d - planet-star distance in AU

res['systems'] #contains
Rp - planet radius in km
I - planet inclination in deg
O - planet right ascension of the ascending node in deg
M0 - initial mean anomaly in def
star
plan2star - indices mapping planets to target stars
a - planet semi major axis
MsTrue
e - planet eccentricty
mu
p - planet albedo
w - planet argument of perigee in deg
MeEst
Mp
"""





#CREATE CUSTOM GEN_SUMMARY WHICH EXTRACTS DETECTED PLANETS FROM RES AND 
#res['DRM'] #contains phi's, fEZ's, dMag's, 
#will need to reconstruct r and v from the orbital parameters in res['systems']
#################################################

#out keys
#['fname', 'detected', 'fullspectra', 'partspectra', 'Rps', 'Mps', 
#'tottime', 'starinds', 'smas', 'ps', 'es', 'WAs', 'SNRs', 'fZs', 'fEZs', 
#'allsmas', 'allRps', 'allps', 'alles', 'allMps', 'dMags', 'rs'])
#DELETe out['WAs'].append(np.hstack([row['det_params']['WA'][row['det_status'] == 1].to('arcsec').value for row in res['DRM']]))
#DELETE out['dMags'].append(np.hstack([row['det_params']['dMag'][row['det_status'] == 1] for row in res['DRM']]))
# out['rs'].append(np.hstack([row['det_params']['d'][row['det_status'] == 1].to('AU').value for row in res['DRM']]))
#DELETE out['fEZs'].append(np.hstack([row['det_params']['fEZ'][row['det_status'] == 1].value for row in res['DRM']]))
#DELETE out['fZs'].append(np.hstack([[row['det_fZ'].value]*len(np.where(row['det_status'] == 1)[0]) for row in res['DRM']]))
#DELETE out['fullspectra'].append(np.hstack([row['plan_inds'][row['char_status'] == 1]  for row in res['DRM']]))
#DELETE out['partspectra'].append(np.hstack([row['plan_inds'][row['char_status'] == -1]  for row in res['DRM']]))
#DELETE out['tottime'].append(np.sum([row['det_time'].value+row['char_time'].value for row in res['DRM']]))
# out['SNRs'].append(np.hstack([row['det_SNR'][row['det_status'] == 1]  for row in res['DRM']]))
# out['Rps'].append((res['systems']['Rp'][dets]/u.R_earth).decompose().value)
# out['smas'].append(res['systems']['a'][dets].to(u.AU).value)
# out['ps'].append(res['systems']['p'][dets])
# out['es'].append(res['systems']['e'][dets])
# out['Mps'].append((res['systems']['Mp'][dets]/u.M_earth).decompose())
# out['starinds'].append(np.hstack([[row['star_ind']]*len(np.where(row['det_status'] == 1)[0]) for row in res['DRM']]))

#need to redo
#14336
#self.r self.v self.Mp self.d self.s self.phi self.fEZ self.dMag self.WA

#DELETA?
# SU.r = out[''] # planet position x,y,z
# SU.v = out[''] #planet velocity
# SU.Mp = out['Mps'] #planet masses
# SU.d = out['rs'] #planet star distance
# SU.s = out[''] #planet star separation
# SU.phi = out['']#
# SU.fEZ = out['fEZs']
# SU.dMag = out['dMags']
# SU.WA = out['WAs']



#### Plot dMag vs s of Detected Exoplanets



