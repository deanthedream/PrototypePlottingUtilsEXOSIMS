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



folder = '/home/dean/Documents/exosims/EXOSIMSres/HabEx_SAG13HabZone_6621/HabEx_SAG13HabZone_lam_lam/'

#if self.args == None: # Nothing was provided as input
# grab random pkl file from folder
# pklfiles_in_folder = [myFileName for myFileName in os.listdir(folder) if 'run' in myFileName and '.pkl' in myFileName]  # Get names of all pkl files in path
# pklfname = np.random.choice(pklfiles_in_folder)
# pklfile = os.path.join(folder,pklfname)
# elif 'pklfile' in self.args.keys(): # specific pklfile was provided for analysis
#     pklfile = self.args['pklfile']
#else: # grab random pkl file from folder
# pklfiles_in_folder = [myFileName for myFileName in os.listdir(folder) if 'run' in myFileName and '.pkl' in myFileName]  # Get names of all pkl files in path
# pklfname = np.random.choice(pklfiles_in_folder)
# pklfile = os.path.join(folder,pklfname)
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

for i in np.arange(len(lines)):
    print(lines[i])

