#SolarSystem Planet Type Detection

import numpy as np
import os
#from exodetbox.projectedEllipse import *
#from exodetbox.stats import *
#from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
import matplotlib
#from mpl_toolkits.mplot3d import Axes3D
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
#from pandas.plotting import scatter_matrix
#import pandas as pd 
#import corner
from EXOSIMS.util.eccanom import *
#from scipy.stats import multivariate_normal
#from scipy.stats import norm
#from exodetbox.trueAnomalyFromEccentricAnomaly import trueAnomalyFromEccentricAnomaly
#from statsmodels.stats.weightstats import DescrStatsW
#import csv
import pickle
import glob
from matplotlib import colors

def gen_summary_kopparapu(folder):
    """
    """
    pklfiles = glob.glob(os.path.join(folder,'*.pkl'))

    out = {'fname':[],
           'detected':[],
           #'fullspectra':[],
           #'partspectra':[],
           'Rps':[],
           #'Mps':[],
           #'tottime':[],
           'starinds':[],
           'smas':[],
           #'ps':[],
           'es':[],
           #'WAs':[],
           #'SNRs':[],
           #'fZs':[],
           #'fEZs':[],
           #'allsmas':[],
           #'allRps':[],
           #'allps':[],
           #'alles':[],
           #'allMps':[],
           #'dMags':[],
           #'rs':[]}
           }

    for counter,f in enumerate(pklfiles):
        print("%d/%d"%(counter,len(pklfiles)))
        with open(f, 'rb') as g:
            res = pickle.load(g, encoding='latin1')

        out['fname'].append(f)
        dets = np.hstack([row['plan_inds'][row['det_status'] == 1]  for row in res['DRM']])
        out['detected'].append(dets) # planet inds

        #out['WAs'].append(np.hstack([row['det_params']['WA'][row['det_status'] == 1].to('arcsec').value for row in res['DRM']]))
        #out['dMags'].append(np.hstack([row['det_params']['dMag'][row['det_status'] == 1] for row in res['DRM']]))
        #out['rs'].append(np.hstack([row['det_params']['d'][row['det_status'] == 1].to('AU').value for row in res['DRM']]))
        #out['fEZs'].append(np.hstack([row['det_params']['fEZ'][row['det_status'] == 1].value for row in res['DRM']]))
        #out['fZs'].append(np.hstack([[row['det_fZ'].value]*len(np.where(row['det_status'] == 1)[0]) for row in res['DRM']]))
        #out['fullspectra'].append(np.hstack([row['plan_inds'][row['char_status'] == 1]  for row in res['DRM']]))
        #out['partspectra'].append(np.hstack([row['plan_inds'][row['char_status'] == -1]  for row in res['DRM']]))
        #out['tottime'].append(np.sum([row['det_time'].value+row['char_time'].value for row in res['DRM']]))
        #out['SNRs'].append(np.hstack([row['det_SNR'][row['det_status'] == 1]  for row in res['DRM']]))
        out['Rps'].append((res['systems']['Rp'][dets]/u.R_earth).decompose().value)
        out['smas'].append(res['systems']['a'][dets].to(u.AU).value)
        #out['ps'].append(res['systems']['p'][dets])
        out['es'].append(res['systems']['e'][dets])
        #out['Mps'].append((res['systems']['Mp'][dets]/u.M_earth).decompose())
        out['starinds'].append(np.hstack([[row['star_ind']]*len(np.where(row['det_status'] == 1)[0]) for row in res['DRM']]))
        #DELETE out['starinds'].append(np.hstack([row['star_ind'][row['det_status'] == 1] for row in res['DRM']]))

        #if includeUniversePlanetPop == True:
        #  out['allRps'].append((res['systems']['Rp']/u.R_earth).decompose().value)
        #  out['allMps'].append((res['systems']['Mp']/u.M_earth).decompose())
        #  out['allsmas'].append(res['systems']['a'].to(u.AU).value)
        #  out['allps'].append(res['systems']['p'])
        #  out['alles'].append(res['systems']['e'])
        del res
        
    return out


folder = "/home/dean/Documents/exosims/EXOSIMSres/HabEx_SAG13HabZone_6621/HabEx_SolarSystem/"
PPoutpath = '/home/dean/Documents/exosims/EXOSIMSres/HabEx_SAG13HabZone_6621/'


#Compile all DRMs
out = gen_summary_kopparapu(folder)


out['detected']
pInds = np.concatenate([i for i in out['detected']])
numOfEach = dict()
numOfEach['numMercury'] = len(np.where(np.mod(pInds,8)==0)[0])
numOfEach['numVenus'] = len(np.where(np.mod(pInds,8)==1)[0])
numOfEach['numEarth'] = len(np.where(np.mod(pInds,8)==2)[0])
numOfEach['numMars'] = len(np.where(np.mod(pInds,8)==3)[0])
numOfEach['numJupiter'] = len(np.where(np.mod(pInds,8)==4)[0])
numOfEach['numSaturn'] = len(np.where(np.mod(pInds,8)==5)[0])
numOfEach['numUranus'] = len(np.where(np.mod(pInds,8)==6)[0])
numOfEach['numNeptune'] = len(np.where(np.mod(pInds,8)==7)[0])


pColors = [colors.to_rgba('grey'),colors.to_rgba('gold'),colors.to_rgba('blue'),colors.to_rgba('red'),\
    colors.to_rgba('orange'),colors.to_rgba('goldenrod'),colors.to_rgba('darkblue'),colors.to_rgba('deepskyblue')]


num=35989878906789689768
plt.figure(num=num,figsize=(8,4))
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
plt.bar(['Mercury'],numOfEach['numMercury']/len(out['fname']),width=0.8,color=pColors[0])
plt.bar(['Venus'],numOfEach['numVenus']/len(out['fname']),width=0.8,color=pColors[1])
plt.bar(['Earth'],numOfEach['numEarth']/len(out['fname']),width=0.8,color=pColors[2])
plt.bar(['Mars'],numOfEach['numMars']/len(out['fname']),width=0.8,color=pColors[3])
plt.bar(['Jupiter'],numOfEach['numJupiter']/len(out['fname']),width=0.8,color=pColors[4])
plt.bar(['Saturn'],numOfEach['numSaturn']/len(out['fname']),width=0.8,color=pColors[5])
plt.bar(['Uranus'],numOfEach['numUranus']/len(out['fname']),width=0.8,color=pColors[6])
plt.bar(['Neptune'],numOfEach['numNeptune']/len(out['fname']),width=0.8,color=pColors[7])
plt.ylabel('Normalized Yield Frequency', weight='bold')
plt.show(block=False)
fname='SolarSystemYieldBarChart'
plt.savefig(os.path.join(PPoutpath,fname+'.png'))
plt.yscale('log')
plt.show(block=False)
fname='SolarSystemYieldBarChartLOG'
plt.savefig(os.path.join(PPoutpath,fname+'.png'))
