""" Calculate Star KeepOutMaps For Specific Stars
By: Dean Keithly
Date: 2/18/2021
"""
import os
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import numpy as np
import astropy.units as u
from astropy.time import Time

#### INPUTS ##########################
#(there are other ways to do this, but simulating a full mission simulation and generating a mission outspec is relatively easy)
starName = 'HIP 19855'
scriptfile = '/home/dean/Documents/exosims/Scripts/WFIRSTcycle6core_CKL2_PPKL2_modified.json'

#Create Mission Object To Extract Some Plotting Limits
sim = EXOSIMS.MissionSim.MissionSim(scriptfile, nopar=True)
obs, TL, TK = sim.Observatory, sim.TargetList, sim.TimeKeeping

#Get Star Name Ind
indWhereStarName = np.where(TL.Name == starName)[0]

#Generate Keepout over Time
koEvaltimes = np.arange(TK.missionStart.value, TK.missionStart.value+TK.missionLife.to('day').value,1) #2year mission, I guess
koEvaltimes = Time(koEvaltimes,format='mjd')

#initial arrays
koGood  = np.zeros([TL.nStars,len(koEvaltimes)])      #keeps track of when a star is in keepout or not (True = observable)
culprit = np.zeros([TL.nStars,len(koEvaltimes),11])   #keeps track of whose keepout the star is under

# choose observing modes selected for detection (default marked with a flag)
allModes = OS.observingModes

#Construct koangles
nSystems  = len(allModes)
systNames = np.unique([allModes[x]['syst']['name'] for x in np.arange(nSystems)])
systOrder = np.argsort(systNames)
koStr     = ["koAngles_Sun", "koAngles_Moon", "koAngles_Earth", "koAngles_Small"]
koangles  = np.zeros([len(systNames),4,2])
for x in systOrder:
    rel_mode = list(filter(lambda mode: mode['syst']['name'] == systNames[x], allModes))[0]
    koangles[x] = np.asarray([rel_mode['syst'][k] for k in koStr])

#Keepouts are calculated here
kogood = np.zeros([1,koEvaltimes.size])
culprit = np.zeros([1,koEvaltimes.size,12])
for t,date in enumerate(koEvaltimes):
    tmpkogood,r_body, r_targ, tmpculprit, koangleArray = obs.keepout(TL, [indWhereStarName,indWhereStarName], date, koangles, True)
    kogood[0,t] = tmpkogood[0,0,0] #reassign to boolean array of overall visibility
    culprit[0,t,:] = tmpculprit[0,0,0,:] #reassign to boolean array describing visibility of individual keepout perpetrators

#creating an array of visibility based on culprit
sunFault   = [bool(culprit[0,t,0]) for t in np.arange(len(koEvaltimes))]
earthFault = [bool(culprit[0,t,2]) for t in np.arange(len(koEvaltimes))]
moonFault  = [bool(culprit[0,t,1]) for t in np.arange(len(koEvaltimes))]
mercFault  = [bool(culprit[0,t,3]) for t in np.arange(len(koEvaltimes))]
venFault   = [bool(culprit[0,t,4]) for t in np.arange(len(koEvaltimes))]
marsFault  = [bool(culprit[0,t,5]) for t in np.arange(len(koEvaltimes))]
solarPanelFault  = [bool(culprit[0,t,11]) for t in np.arange(len(koEvaltimes))]
    
#### Outputs ############################################
#koEvalTimes - the times of the bin edges of the KeepOut Map koGood and culprit
#indWhereStarName - the ind of koGood[ind,:] indicating where starName occurs
#koGood - the array (nStars,len(koEvalTimes)) indicating whether the star with ind koGood[ind] is in keepout or not
#culprit  - the array (nStars,len(koEvalTimes),5) of integers indicating whether the star is in keepout by sun, earth, moon, mercury, venus, or mars
#sunFault - the array (nStars,len(koEvalTimes)) of booleans indicating whether the star is in keepout by the sun
#########################################################

