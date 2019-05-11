# -*- coding: utf-8 -*-
"""
Plotting utility for the production of Zodiacal Light Related Plots
Written by: Dean Keithly
Written on: 10/11/2018
"""

import os
from EXOSIMS.util.vprint import vprint
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

# class plotZodiacalLightFigures(object):
#     """Template format for adding singleRunPostProcessing to any plotting utility
#     singleRunPostProcessing method is a required method with the below inputs to work with runPostProcessing.py
#     """
#     _modtype = 'util'

#     def __init__(self, args=None):
#         vprint(args)
#         vprint('fakeSingleRunAnalysis done')
#         pass

#     def singleRunPostProcessing(self, PPoutpath=None, folder=None):
#         """This is called by runPostProcessing
#         Args:
#             PPoutpath (string) - output path to place data in
#             folder (string) - full filepath to folder containing runs
#         """
#         pass

PPoutpath = './'#'/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo/WFIRSTcycle6core_CKL2_PPKL2'
folder = '/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_405_19/WFIRSTcycle6core_CKL2_PPKL2'
#folder = '/home/dean/Documents/SIOSlab/EXOSIMSres/HabExCompSpecPriors_HabEx_4m_TSDD_pop100DD_revisit_20190203/HabEx_CSAG13_PPSAG13'

if not os.path.exists(folder):#Folder must exist
    raise ValueError('%s not found'%folder)
if not os.path.exists(PPoutpath):#PPoutpath must exist
    raise ValueError('%s not found'%PPoutpath) 
outspecfile = os.path.join(folder,'outspec.json')
if not os.path.exists(outspecfile):#outspec file not found
    raise ValueError('%s not found'%outspecfile) 



#Create Mission Object To Extract Some Plotting Limits
sim = EXOSIMS.MissionSim.MissionSim(outspecfile, nopar=True)
SS = sim.SurveySimulation
Obs = SS.Observatory
TK = SS.TimeKeeping
TL = SS.TargetList
OS = SS.OpticalSystem
ZL = SS.ZodiacalLight

sInds = np.where(sim.SurveySimulation.t0.value > 0.)[0] # Grab indices where time is greater than 0
#DELETE dt = 365.25/len(np.arange(1000))
time = np.linspace(start=0.,stop=365.25*TK.missionLife.value,num=1000*int(np.ceil(TK.missionLife.value)))#[j*dt for j in range(1000)]#Time since mission start
allModes = OS.observingModes
mode = filter(lambda mode: mode['detectionMode'] == True, allModes)[0]


kogoodStart = np.zeros([len(time),len(sInds)])
for i in np.arange(len(time)):
    kogoodStart[i,:] = Obs.keepout(TL, sInds, TK.currentTimeAbs+time[i]*u.d, False)
    kogoodStart[i,:] = (np.zeros(kogoodStart[i,:].shape[0])+1)*kogoodStart[i,:]
kogoodStart[kogoodStart==0] = np.nan

intmpfZ = np.zeros(len(sInds))
intmpfZ = ZL.fZ_startSaved[sInds]
tmpfZ = intmpfZ
for i in np.arange(np.ceil(TK.missionLife.value)-1):
    tmpfZ = np.concatenate((tmpfZ,intmpfZ), axis=1)
magfZ = -2.5*np.log10(tmpfZ)

magfZ2 = np.zeros([len(sInds),len(time)])
minmagfZ2 = np.zeros(len(sInds))
maxmagfZ2 = np.zeros(len(sInds))
for i in np.arange(len(sInds)):
    magfZ2[i,:] = magfZ[i,:]*kogoodStart[:,i]
    minmagfZ2[i] = min(magfZ2[i,magfZ2[i,:] > 0.])#calculate minimum value for all stars
    maxmagfZ2[i] = max(magfZ2[i,magfZ2[i,:] > 0.])#calculate maximum value for all stars


#### Plots The Distribution of maximum and minimum zodiacal light of each target
plt.close(9005)
figfZminmaxHist = plt.figure(9005)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold') 
out1 = plt.hist(minmagfZ2,label=r'$Z_{\mathrm{min}}$',color='b',alpha=0.5,bins=np.arange(np.min(minmagfZ2), np.max(minmagfZ2) + 0.1, 0.1))
out2 = plt.hist(maxmagfZ2,label=r'$Z_{\mathrm{max}}$',color='r',alpha=0.5,bins=np.arange(np.min(maxmagfZ2), np.max(maxmagfZ2) + 0.1, 0.1))
magfZ0 = -2.5*np.log10(ZL.fZ0.value)
maxCNT = np.max([np.max(out1[0]),np.max(out2[0])])
out3 = plt.plot([magfZ0,magfZ0],[0,1.1*maxCNT],color='k',label=r'$Z_0$')
out4 = plt.plot([-2.5*np.log10(np.mean(10**(np.asarray(minmagfZ2)/-2.5))),-2.5*np.log10(np.mean(10**(np.asarray(minmagfZ2)/-2.5)))],[0,1.1*maxCNT],color='b',label=r'$\mu_{fZ_{\mathrm{max}}}}$',linestyle='--')
out5 = plt.plot([-2.5*np.log10(np.mean(10**(np.asarray(maxmagfZ2)/-2.5))),-2.5*np.log10(np.mean(10**(np.asarray(maxmagfZ2)/-2.5)))],[0,1.1*maxCNT],color='r',label=r'$\mu_{fZ_{\mathrm{min}}}}$',linestyle='--')
#plt.title('Histogram of '+r'$magfZ_{min}$'+' and '+r'$magfZ_{max}$',weight='bold',fontsize=12)
plt.xlabel('Local Zodiacal Light in Magnitudes, Z',weight='bold',fontsize=12)
plt.ylabel('# of Targets',weight='bold',fontsize=12)

plt.ylim([0,1.1*maxCNT])
#plt.rc('axes',prop_cycle=(cycler('color',['red','blue','black','purple'])))
plt.legend()
plt.show(block=False)
#red_patch = matplotlib.mpatches.Patch(color='red', label=r'$magfZ){max}$')
#plt.legend(handles=[red_patch])

date = unicode(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'figfZminmaxHist_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))



COMP = SS.Completeness

WA = SS.WAint
_, Cbs, Csps = OS.Cp_Cb_Csp(TL, np.arange(TL.nStars), SS.valfZmin, ZL.fEZ0, 25.0, WA, SS.detmode)
comp0 = COMP.comp_per_intTime(SS.t0, TL, np.arange(TL.nStars), 
                    SS.valfZmin, ZL.fEZ0, SS.WAint, SS.detmode, C_b=Cbs, C_sp=Csps)#Integration time at the initially calculated t0


#### Plot Variation in fZ vs Time
plt.close(5656461)
figfZvsTime = plt.figure(5656461)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold') 
c2 = comp0[sInds]
c2maxInd = np.argmax(c2)
c2minInd = np.argmin(c2)

plt.plot(time,magfZ2[c2maxInd,:], label='max(c)')
plt.plot(time,magfZ2[c2minInd,:], label='min(c)')


plt.xlabel('Mission Elasped Time (d)', weight='bold')
plt.ylabel('Zodiacal Light in Magnitudes, Z', weight='bold')
plt.legend()
plt.show(block=False)
fname = 'figfZvsTime_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))


plt.xlim([0.,365.25*2.])
plt.show(block=False)
fname = 'figfZvsTime_ReasonableLimits_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))



#### Maximum Zodiacal Light Variation over 45 Days
boxesPerDay = 1000./365.25#box per day
numBoxes = np.floor(45*boxesPerDay)
delta = np.zeros([ZL.fZ_startSaved.shape[0],ZL.fZ_startSaved.shape[1]-123])
for i in np.arange(ZL.fZ_startSaved.shape[1]-123):
    delta[:,i] = abs(ZL.fZ_startSaved[:,i+123]-ZL.fZ_startSaved[:,i])
maxDeltafZ = np.zeros(ZL.fZ_startSaved.shape[0])
for j in np.arange(delta.shape[0]):
    maxDeltafZ[j] = np.max(delta[j,:])

#### Plot 45 day delta vs time
plt.close(32198652)
figMaxfZ45Days = plt.figure(32198652)
plt.plot(-2.5*np.log10(delta[sInds[c2maxInd],:]), label='max(c)', color='red')
plt.plot(-2.5*np.log10(delta[sInds[c2minInd],:]), label='min(c)', color='blue')

plt.legend()
plt.ylabel('Maximum Variation in Z over 45 Days', weight='bold')
plt.show(block=False)
fname = 'figMaxfZ45Days_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))


#### Plot Histogram of maximum 45 day deltas
plt.close(548962)
figHistMaxfZ45Days = plt.figure(548962)
plt.hist(maxDeltafZ, color='black')
plt.xlabel(r'Maximum Zodiacal Light $\Delta$ in Magnitudes, Z', weight='bold')
plt.show(block=False)
fname = 'figHistMaxfZ45Days_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))



#Log(Izod) vs wavelength semilog 1111111111111111111111111 #########################
#ZL.fZ(Obs, TL, [1], TK.currentTimeAbs, SS.detmode)
fignum = 56895113
plt.close(fignum)
fig2 = plt.figure(fignum)
plt.subplots_adjust(left=0.2)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold') 
#plt.title('Current Model Fit Produces This LogScale',weight='bold')
x = np.logspace(-0.69,2.14,num=100,base=10.0)#I create a nice range of test Lambdas to span the set
#y = ZL.logf(np.log10(x))#Calculate the logf for these
y = ZL.logf(np.log10(x))#Calculate the logf for these
plt.semilogx(x*1000., y, color='b',linestyle='--',label='quadratic interpolant')
#y2 = ZL.logf(np.log10(x))#Calculate the logf2 for these
#plt.semilogx(x, np.log10(y2), color='r',linestyle='--',label='cubic interpolant')
plt.semilogx([2.19*1000.,2.19*1000.],[min(y),max(y)],'ks-',label='U-Band')#Plot maximum lambda uses
plt.semilogx([0.365*1000.,0.365*1000.],[min(y),max(y)],'kd-',label='K-Band')#Plot minimum lambda uses
plt.semilogx(ZL.zodi_lam*1000.,np.log10(ZL.zodi_Blam),'kx',label='Leinert98 points')

plt.xlabel(r'Wavelength, $\lambda$ in nm',weight='bold')
plt.ylabel('Zodiacal Light Intensity Wavelength\nCorrection' + r'$f_\lambda(\lambda)$ in $W m^{-2} sr^{-1} \mu m^{-1}$', weight='bold')
#At lambda = 90deg or epsilon = 90deg (90deg from sun in ecliptic along ecliptic)
plt.legend()

date = unicode(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'logfZvsLamStarFit_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath,fname+'.png'))
plt.savefig(os.path.join(PPoutpath,fname+'.svg'))
plt.savefig(os.path.join(PPoutpath,fname+'.eps'))
plt.savefig(os.path.join(PPoutpath,fname+'.pdf'))
plt.show(block=False)
#########################################################################





#### WRITTEN DATA FILE
lines = list()
lines.append("mean Z_min (calculated as mean fZ_max) in 1/arcsec2: " + str(-2.5*np.log10(np.mean(10**(np.asarray(minmagfZ2)/-2.5)))))
lines.append("mean Z_max (calculated as mean fZ_min) in 1/arcsec2: " + str(-2.5*np.log10(np.mean(10**(np.asarray(maxmagfZ2)/-2.5)))))


#### Save Data File
fname = 'ZLDATA_' + folder.split('/')[-1] + '_' + date
with open(os.path.join(PPoutpath, fname + '.txt'), 'w') as g:
    g.write("\n".join(lines))
# end main