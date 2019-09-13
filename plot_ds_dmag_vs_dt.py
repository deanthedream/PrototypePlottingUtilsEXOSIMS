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


pklfiles = glob.glob(os.path.join(folder,'*.pkl'))
for counter,f in enumerate(pklfiles[0:2]):
    print("%d/%d"%(counter,len(pklfiles)))
    with open(f, 'rb') as g:
        res = pickle.load(g, encoding='latin1')

    #This must happen here bc each res will have planets around the same host star therefore causing multiple times
    #Take SU object
    #SU.init_systems
    #requires a,e,I,O,w,M0,Mp, plan2star ,TL.MsTrue
    SU.a = res['systems'].a
    SU.e = res['systems'].e
    SU.I = res['systems'].I
    SU.O = res['systems'].O
    SU.w = res['systems'].w
    SU.M0 = res['systems'].M0
    SU.Mp = res['systems'].Mp
    SU.plan2star = res['systems'].plan2star
    TL.MsTrue = res['systems'].MsTrue


    SU.init_systems()
    #TODO propagate each system to the time of detection
    for i in np.arange(len(res['DRM'])):
        dt_i.append(res['DRM'][i]['arrival_time'].value)
        sInd_i.append(res['DRM'][i]['star_ind']) #
        SU.propag_system(sInd_i[-1], dt_i[-1]) #propagates the system to the time of detection
        #Note all planets will be moved to position when they were

    #extract initial dMag and s of each planet here

    #propagate all by a fixed time, dt, giving 

    #extract new dMag and s of each planet


#1 check res for the actual data I need to reconstruct planet position
res['seed']#useless.... unless I can recreate the simulation object!
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

SU.r = out[''] # planet position x,y,z
SU.v = out[''] #planet velocity
SU.Mp = out['Mps'] #planet masses
SU.d = out['rs'] #planet star distance
SU.s = out[''] #planet star separation
SU.phi = out['']#
SU.fEZ = out['fEZs']
SU.dMag = out['dMags']
SU.WA = out['WAs']



#### Plot dMag vs s of Detected Exoplanets



