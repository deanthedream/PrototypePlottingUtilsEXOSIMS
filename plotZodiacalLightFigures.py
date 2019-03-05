# -*- coding: utf-8 -*-
"""
Template Plotting utility for post processing automation
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
folder = '/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo/WFIRSTcycle6core_CKL2_PPKL2'
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
out1 = plt.hist(minmagfZ2,label=r'$magfZ_{min}$',color='b',alpha=0.5,bins=np.arange(np.min(minmagfZ2), np.max(minmagfZ2) + 0.1, 0.1))
out2 = plt.hist(maxmagfZ2,label=r'$magfZ_{max}$',color='r',alpha=0.5,bins=np.arange(np.min(maxmagfZ2), np.max(maxmagfZ2) + 0.1, 0.1))
magfZ0 = -2.5*np.log10(ZL.fZ0.value)
maxCNT = np.max([np.max(out1[0]),np.max(out2[0])])
out3 = plt.plot([magfZ0,magfZ0],[0,1.1*maxCNT],color='k',label=r'$magfZ_0$')
out4 = plt.plot([np.mean(minmagfZ2),np.mean(minmagfZ2)],[0,1.1*maxCNT],color='b',label=r'$mean(magfZ_{min})$',linestyle='--')
out5 = plt.plot([np.mean(maxmagfZ2),np.mean(maxmagfZ2)],[0,1.1*maxCNT],color='r',label=r'$mean(magfZ_{max})$',linestyle='--')
plt.title('Histogram of '+r'$magfZ_{min}$'+' and '+r'$magfZ_{max}$',weight='bold',fontsize=12)
plt.xlabel('Zodiacal Light in Magnitudes',weight='bold',fontsize=12)
plt.ylabel('# of Targets',weight='bold',fontsize=12)

plt.ylim([0,1.1*maxCNT])
#plt.rc('axes',prop_cycle=(cycler('color',['red','blue','black','purple'])))
plt.legend()
plt.show(block=False)
#red_patch = matplotlib.mpatches.Patch(color='red', label=r'$magfZ){max}$')
#plt.legend(handles=[red_patch])
figfZminmaxHist.savefig(os.path.join(PPoutpath,'figfZminmaxHist'+'.png'))
figfZminmaxHist.savefig(os.path.join(PPoutpath,'figfZminmaxHist'+'.svg'))
figfZminmaxHist.savefig(os.path.join(PPoutpath,'figfZminmaxHist'+'.eps'))
figfZminmaxHist.savefig(os.path.join(PPoutpath,'figfZminmaxHist'+'.pdf'))



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
plt.ylabel('Zodiacal Light in Magnitudes', weight='bold')
plt.legend()
plt.show(block=False)
figfZvsTime.savefig(os.path.join(PPoutpath,'figfZvsTime'+'.pdf'))

plt.xlim([0.,365.25*2.])
plt.show(block=False)
figfZvsTime.savefig(os.path.join(PPoutpath,'figfZvsTime_ReasonableLimits'+'.pdf'))



#### Maximum Zodiacal Light Variation over 45 Days
boxesPerDay = 1000./365.25#box per day
numBoxes = np.floor(45*boxesPerDay)
delta = np.zeros([ZL.fZ_startSaved.shape[0],ZL.fZ_startSaved.shape[1]-45])
for i in np.arange(ZL.fZ_startSaved.shape[1]-45):
    delta[:,i] = abs(ZL.fZ_startSaved[:,i+45]-ZL.fZ_startSaved[:,i])
maxDeltafZ = np.zeros(ZL.fZ_startSaved.shape[0])
for j in np.arange(delta.shape[0]):
    maxDeltafZ[j] = np.max(delta[j,:])

#Plot 45 day delta vs time
plt.close(32198652)
figMaxfZ45Days = plt.figure(32198652)
plt.plot(-2.5*np.log10(delta[sInds[c2maxInd],:]), label='max(c)', color='red')
plt.plot(-2.5*np.log10(delta[sInds[c2minInd],:]), label='min(c)', color='blue')

plt.legend()
plt.ylabel('Maximum Variation in Zodiacal Light Magnitude over 45 Days')
plt.show(block=False)
figMaxfZ45Days.savefig(os.path.join(PPoutpath,'figMaxfZ45Days'+'.pdf'))

#Plot Histogram of maximum 45 day deltas
plt.close(548962)
figHistMaxfZ45Days = plt.figure(548962)
plt.hist(maxDeltafZ)
plt.xlabel(r'Maximum Zodiacal Light $\delta$ in Magnitudes')
plt.show(block=False)
figHistMaxfZ45Days.savefig(os.path.join(PPoutpath,'figHistMaxfZ45Days'+'.pdf'))


