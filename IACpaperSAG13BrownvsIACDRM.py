# A script to check integration time adjusted completeness against brown completeness

import random as myRand
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
try:
    import cPickle as pickle
except:
    import pickle
import os
import numpy as np
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
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import glob



def loadFiles(pklfile,outspecfile):
    """ loads pkl and outspec files
    Args:
        pklfile (string) - full filepath to pkl file to load
        outspecfile (string) - fille filepath to outspec.json file
    Return:
        DRM (dict) - a dict containing seed, DRM, system
        outspec (dict) - a dict containing input instructions
    """
    try:
        with open(pklfile, 'rb') as f:#load from cache
            DRM = pickle.load(f)
    except:
        print('Failed to open pklfile %s'%pklfile)
        pass
    try:
        with open(outspecfile, 'rb') as g:
            outspec = json.load(g)
    except:
        print('Failed to open outspecfile %s'%outspecfile)
        pass
    return DRM, outspec



#folder = '/home/dean/Documents/exosims/EXOSIMSres/HabEx_SAG13HabZone_6621/HabEx_SAG13HabZone_lam_lam/'
folder = '/home/dean/Documents/exosims/EXOSIMSres/HabExCSpPrior_52521/HabEx_CPFQL_PPPFQL/'

pklfiles_in_folder = [myFileName for myFileName in os.listdir(folder) if 'run' in myFileName and '.pkl' in myFileName]  # Get names of all pkl files in path
pklfname = np.random.choice(pklfiles_in_folder)
pklfile = os.path.join(folder,pklfname)
outspecfile = os.path.join(folder,'outspec.json')

import copy
DRM, outspec = loadFiles(pklfile, outspecfile)
outspec_saved = copy.deepcopy(outspec)


#Create Simulation Object
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=None, nopar=True, **outspec)
SS = sim.SurveySimulation
ZL = SS.ZodiacalLight
COMP = SS.Completeness
OS = SS.OpticalSystem
Obs = SS.Observatory
TL = SS.TargetList
TK = SS.TimeKeeping
print("Loaded Sim Object 1")

#Create Outspec2 with IAC
outspec2 = copy.deepcopy(outspec_saved)
outspec2["modules"]["Completeness"] = "IntegrationTimeAdjustedCompleteness"
del outspec2["cachefname"] #deleting the chacehfname so cached files are properly recalculated
sim2 = EXOSIMS.MissionSim.MissionSim(scriptfile=None, nopar=True, **outspec2)
print("Loaded Sim Object 2")

#scomp0
#t0

names1 = sim.TargetList.Name
names2 = sim2.TargetList.Name
nameIndsIn1and2 = list() #contains list of indicies in 1 that are also in 2
nameInd12 = list() #indicies of 2 where star in 1 is in 2
for i in np.arange(len(names1)):
    print(str(i) + " of " + str(len(names1)))
    if names1[i] in names2:
        nameIndsIn1and2.append(i)
        ind = np.where(names1[i] == names2)[0]
        nameInd12.append(ind)

nameIndsIn1and2 = np.asarray(nameIndsIn1and2).flatten()
nameInd12 = np.asarray(nameInd12).flatten()

SS = sim.SurveySimulation
SS2 = sim2.SurveySimulation

brownt0 = list()
#brownscomp0 = list()
for i in np.arange(len(nameIndsIn1and2)):
    #The Brown Completeness Integration Times
    brownt0.append(sim.SurveySimulation.t0[nameIndsIn1and2[i]])
    #brownscomp0.append(sim.SurveySimulation.scomp0[nameIndsIn1and2[i]])

    #sim2.SurveySimulation.t0[nameInd12[i]]
    #sim2.SurveySimulation.scomp0[nameInd12[i]]

#brownComps = list()
#for i in np.arange(len(nameIndsIn1and2)):
brownComps = SS.Completeness.comp_per_intTime(sim.SurveySimulation.t0, SS.TargetList, nameIndsIn1and2, SS.valfZmin, 
    SS.ZodiacalLight.fEZ0, SS.WAint, SS.detmode, TK=SS.TimeKeeping)

intTimes, sInds, fZ, fEZ, WA, smin, smax, dMag = sim2.SurveySimulation.Completeness.comps_input_reshape(sim.SurveySimulation.t0, sim2.SurveySimulation.TargetList, nameInd12, SS2.valfZmin[nameInd12],\
    SS2.ZodiacalLight.fEZ0, SS2.WAint[nameInd12], SS.detmode, C_b=None, C_sp=None, TK=sim2.SurveySimulation.TimeKeeping)

IACComps = SS2.Completeness.comp_calc(smin, smax, dMag,tmax=intTimes,starMass=sim2.SurveySimulation.TargetList.MsEst[nameInd12], IACbool=True)

#Calculate the Difference In Comps
diffInComps = np.sum(brownComps-IACComps)
print(diffInComps)

percentDiffInComps = (diffInComps/np.sum(IACComps))*100.
print(percentDiffInComps)



#### Print Out Table, Star Name, Star Dist, dMag Of Integration, Integration Time, fZ
lines = list()
lines.append("Name & Dist (pc) & Int. Time (d) & {\dmag} & $C_{\mathrm{brown}}$ & $C_{\mathrm{IAC}}$ \\\\")
for i in np.arange(len(nameInd12)):
    if sim.SurveySimulation.t0[i].value > 1e-10:
        lines.append(sim2.SurveySimulation.TargetList.Name[nameInd12[i]] + ' & ' + str(np.round(sim2.SurveySimulation.TargetList.dist[nameInd12[i]].value,2)) + ' & ' + \
            str(np.round(sim.SurveySimulation.t0[i].value,3)) + ' & ' + str(np.round(dMag[i],3)) + " & " + str(np.round(brownComps[i],3)) + " & " + str(np.round(IACComps[i],3)) + "\\\\")

#Creates the Table For the IAC Paper
for i in np.arange(len(lines)):
    print(lines[i])

distList = list()
for i in np.arange(len(nameInd12)):
    if sim.SurveySimulation.t0[i].value > 1e-10:
        distList.append(sim2.SurveySimulation.TargetList.dist[nameInd12[i]].value)





#### Calculating IAC vs Int. Time. ######################################
starName = 'HIP 37279'
starInd = np.where(sim2.SurveySimulation.TargetList.Name == starName)[0]
OS = sim2.OpticalSystem
TL = sim2.TargetList
COMP = sim2.Completeness


intTimes, sInds, fZ, fEZ, WA, smin, smax, dMag = sim2.SurveySimulation.Completeness.comps_input_reshape(sim.SurveySimulation.t0[starInd], sim2.SurveySimulation.TargetList, starInd, SS2.valfZmin[starInd],\
    SS2.ZodiacalLight.fEZ0, SS2.WAint[starInd], SS2.detmode, C_b=None, C_sp=None, TK=sim2.SurveySimulation.TimeKeeping)


dMags = list()
IACComps2 = list()
intTimes2 = np.logspace(start=-8.,stop=np.log10(90.),num=100)
for i in np.arange(len(intTimes2)):
    print("on run: " + str(i) + " out of " + str(300))
    #tmpdMag = OS.calc_dMag_per_intTime(intTimes2[i]*u.d, TL, starInd, SS2.valfZmin[starInd], ZL.fEZ0, SS2.WAint[starInd], SS2.detmode, C_b=None, C_sp=None, TK=None)
    #dMags.append(tmpdMag)
    #tCp, tCb, tCsp = OS.Cp_Cb_Csp(TL, starInd, SS2.valfZmin[starInd], ZL.fEZ0, tmpdMag, SS2.WAint[starInd], SS2.detmode)
    tmpintTimes, tmpsInds, tmpfZ, tmpfEZ, tmpWA, tmpsmin, tmpsmax, tmpdMag  = sim2.Completeness.comps_input_reshape(intTimes2[i]*u.d, TL, starInd, SS2.valfZmin[starInd], SS2.ZodiacalLight.fEZ0, SS2.WAint[starInd], SS2.detmode, C_b=None, C_sp=None, TK=None)
    dMags.append(tmpdMag)
    tmpIACComps = SS2.Completeness.comp_calc(tmpsmin, tmpsmax, tmpdMag,tmax=intTimes2[i],starMass=sim2.SurveySimulation.TargetList.MsEst[starInd], IACbool=True)
    IACComps2.append(tmpIACComps)

brownComps2 = list()
for i in np.arange(len(intTimes2)):
    tmpbrownComps = SS.Completeness.comp_per_intTime(intTimes2[i]*u.d, TL, starInd, SS2.valfZmin[starInd], 
        SS2.ZodiacalLight.fEZ0, SS2.WAint[starInd], SS2.detmode, TK=SS2.TimeKeeping)
    #tmpdMag2 = sim2.SurveySimulation.OpticalSystem.calc_dMag_per_intTime(intTimes2[i]*u.d, TL, starInd, SS2.valfZmin[starInd], SS2.ZodiacalLight.fEZ0, SS2.WAint[starInd], SS2.detmode, C_b=None, C_sp=None, TK=None)
    
    #tmpintTimes, tmpsInds, tmpfZ, tmpfEZ, tmpWA, tmpsmin, tmpsmax, tmpdMag2  = sim.Completeness.comps_input_reshape(intTimes2[i]*u.d, TL, starInd, SS2.valfZmin[starInd], SS2.ZodiacalLight.fEZ0, SS2.WAint[starInd], SS2.detmode, C_b=None, C_sp=None, TK=None)
    #tmpbrownComps = sim.SurveySimulation.Completeness.comp_calc(tmpsmin, tmpsmax, tmpdMag2)
    brownComps2.append(tmpbrownComps)


num=546843521843244
plt.close(num)
plt.figure(num=num)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
plt.plot(intTimes2,IACComps2,color='purple',label='IAC')
plt.plot(intTimes2,brownComps2,color='green',label='Brown')
#plt.xscale('log')
plt.ylabel('Integration Time Adjusted Completeness',weight='bold')
plt.xlabel('Integration Time (d)',weight='bold')
plt.show(block=False)

num=5468435218432445468
plt.close(num)
plt.figure(num=num)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
plt.plot(intTimes2,IACComps2,color='purple',label='This Work')
plt.plot(intTimes2,brownComps2,color='green',label='Brown')
plt.legend()
plt.xscale('log')
plt.ylabel('Integration Time Adjusted Completeness ' + starName,weight='bold')
plt.xlabel('Integration Time (d)',weight='bold')
plt.show(block=False)


plt.figure()
plt.plot(intTimes2,dMags)
plt.xscale('log')
plt.show(block=False)




#### Checling L and MsStar hists
from astropy.constants import iau2012 as const
usedStarInds = np.where(sim.SurveySimulation.t0.value > 1e-10)[0]
Ls = sim.SurveySimulation.TargetList.L[usedStarInds]
Ms = sim.SurveySimulation.TargetList.MsTrue[usedStarInds]


num=654684321
plt.figure(num=num)
plt.hist(Ls,bins=500)
plt.xlabel('L_Sun')
plt.show(block=False)

num=65468432111
plt.figure(num=num)
plt.hist(Ms.value,bins=50)
plt.xlabel('M_sun')
plt.show(block=False)


