"""
Plotting planet population properties

Written By: Dean Keithly
2/1/2019
"""

try:
    import cPickle as pickle
except:
    import pickle
import os
if not 'DISPLAY' in os.environ.keys(): #Check environment for keys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
else:
    import matplotlib.pyplot as plt 
import numpy as np
from numpy import nan
import argparse
import json
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import astropy.units as u
import copy
import random
import datetime
import re
from EXOSIMS.util.vprint import vprint
from copy import deepcopy

from astropy.io import fits
import scipy.interpolate
import astropy.units as u
import numpy as np
from EXOSIMS.MissionSim import MissionSim
import numbers
from scipy import interpolate
from matplotlib import ticker, cm

import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo'))#HabExCompSpecPriors_HabEx_4m_TSDD_pop100DD_revisit_20180424'))#prefDistOBdursweep_WFIRSTcycle6core'))
filename = 'tmp.json'#'WFIRSTcycle6core_CKL2_PPKL2.json'#'HabEx_CKL2_PPSAG13.json'#'auto_2018_11_03_15_09__prefDistOBdursweep_WFIRSTcycle6core_9.json'#'./TestScripts/02_KnownRV_FAP=1_WFIRSTObs_staticEphem.json'#'Dean17Apr18RS05C01fZ01OB01PP01SU01.json'#'sS_SLSQP.json'#'sS_AYO4.json'#'sS_differentPopJTwin.json'#AYO4.json'
#filename = 'sS_intTime6_KeplerLike2.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile,nopar=True)
#sim.run_sim()

xnew = sim.SurveySimulation.Completeness.xnew
dMag = np.linspace(start=15.,stop=50.,num=200)
xmin = min(xnew)
xmax = max(xnew)
ymin = min(dMag)
ymax = max(dMag)
xlims = [xmin,sim.SurveySimulation.PlanetPopulation.rrange[1].to('AU').value]
ylims = [ymin,ymax]
f = list()
for k, dm in enumerate(dMag):
    f.append(sim.SurveySimulation.Completeness.EVPOCpdf(xnew,dm)[:,0])
f = np.asarray(f)
f[ 10**-5 > f] = np.nan

plt.close('all')
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
fig = plt.figure()
ax1 = plt.subplot(111)
CS = ax1.contour(xnew,dMag,f,15, extent=[xlims[0], xlims[1], ylims[0], ylims[1]], linewidths=0.5,colors='k')
CS = ax1.contourf(xnew,dMag,f,15, extent=[xlims[0], xlims[1], ylims[0], ylims[1]], cmap='jet', intepolation='nearest', locator=ticker.LogLocator())
ax1.set_xlim(xlims)
ax1.set_ylim(ylims)
#CS = plt.contour(X,Y,Z,15, extent=[xmin, xmax, ymin, ymax], linewidths=0.5,colors='k')
#CS = plt.contourf(X,Y,Z,15, extent=[xmin, xmax, ymin, ymax], cmap='jet', intepolation='nearest', locator=ticker.LogLocator())
cbar = fig.colorbar(CS)
plt.xlabel(r'$s$ (AU)',weight='bold')
plt.ylabel(r'$\Delta$mag',weight='bold')
#plt.cm.jet
#plt.scatter(rows,cols,marker='o',c='b',s=5)
plt.show(block=False)


#Redo with Garrett Completeness???
outspec = sim.SurveySimulation.genOutSpec()
outspec['modules']['Completeness'] = 'GarrettCompleteness'

sim2 = EXOSIMS.MissionSim.MissionSim(**outspec)
xnew = sim.SurveySimulation.Completeness.xnew
dMag = np.linspace(start=15.,stop=50.,num=200)
xmin = min(xnew)
xmax = max(xnew)
ymin = min(dMag)
ymax = max(dMag)
xlims = [xmin,sim.SurveySimulation.PlanetPopulation.rrange[1].to('AU').value]
ylims = [ymin,ymax]
f = list()
for k, dm in enumerate(dMag):
    f.append(sim.SurveySimulation.Completeness.EVPOCpdf(xnew,dm)[:,0])
f = np.asarray(f)
f[ 10**-5 > f] = np.nan

plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
fig = plt.figure()
ax1 = plt.subplot(111)
CS = ax1.contour(xnew,dMag,f,15, extent=[xlims[0], xlims[1], ylims[0], ylims[1]], linewidths=0.5,colors='k')
CS = ax1.contourf(xnew,dMag,f,15, extent=[xlims[0], xlims[1], ylims[0], ylims[1]], cmap='jet', intepolation='nearest', locator=ticker.LogLocator())
ax1.set_xlim(xlims)
ax1.set_ylim(ylims)
#CS = plt.contour(X,Y,Z,15, extent=[xmin, xmax, ymin, ymax], linewidths=0.5,colors='k')
#CS = plt.contourf(X,Y,Z,15, extent=[xmin, xmax, ymin, ymax], cmap='jet', intepolation='nearest', locator=ticker.LogLocator())
cbar = fig.colorbar(CS)
plt.xlabel(r'$s$ (AU)',weight='bold')
plt.ylabel(r'$\Delta$mag',weight='bold')
#plt.cm.jet
#plt.scatter(rows,cols,marker='o',c='b',s=5)
plt.show(block=False)