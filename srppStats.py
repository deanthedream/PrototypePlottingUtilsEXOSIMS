# Generate Statistics on the simulation


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
from EXOSIMS.util.read_ipcluster_ensemble import read_all

"""Generates a single yield histogram for the run_type
Args:
    PPoutpath (string) - output path to place data in
    folder (string) - full filepath to folder containing runs
"""
# #Get name of pkl file
# if isinstance(self.args,dict):
#     if 'file' in self.args.keys():
#         file = self.args['file']
# else:
#     file = self.pickPKL(folder)
# fullPathPKL = os.path.join(folder,file) # create full file path
# if not os.path.exists(fullPathPKL):
#     raise ValueError('%s not found'%fullPathPKL)

fullPathPKL = '/home/dean/Documents/SIOSlab/EXOSIMSres/HabExCompSpecPriors_HabEx_4m_TSDD_pop100DD_revisit_20180424/HabEx_CSAG13_PPSAG13/run94611245591.pkl'
folder = '/home/dean/Documents/SIOSlab/EXOSIMSres/HabExCompSpecPriors_HabEx_4m_TSDD_pop100DD_revisit_20180424/HabEx_CSAG13_PPSAG13'
#WFIRST pkl
#fullPathPKL = '/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3momaxC/WFIRSTcycle6core_CSAG13_PPSAG13/tmp/run56546329770.pkl'
#folder = '/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3momaxC/WFIRSTcycle6core_CSAG13_PPSAG13'




folder = '/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_40319_2/WFIRSTcycle6core_CKL2_PPKL2'
if not os.path.exists(folder):#Folder must exist
    raise ValueError('%s not found'%folder)
allres = read_all(folder)# contains all drm from all missions in folder. Length of number of pkl files in folder

#### Generate Ensemble Wide Properties
# For each parameter P, gather min, mean, and max
P = ['']



outspecPath = os.path.join(folder,'outspec.json')
try:
    with open(outspecPath, 'rb') as g:
        outspec = json.load(g)
except:
    vprint('Failed to open outspecfile %s'%outspecPath)
    pass
#Create Simulation Object
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=None, nopar=True, **outspec)
SS = sim.SurveySimulation
ZL = SS.ZodiacalLight
COMP = SS.Completeness
OS = SS.OpticalSystem
Obs = SS.Observatory
TL = SS.TargetList
TK = SS.TimeKeeping
totOH = Obs.settlingTime.value + OS.observingModes[0]['syst']['ohTime'].value



data = list()
for i in [0]:#np.arange(len(allres)):
    #allres[i]['systems'] # contains all information relating to planets and stars
    #allres[i]['DRM'] #has length number of detection observations
    dataElement = {}
    dataElement['det_times'] = [allres[i]['DRM'][j]['det_time'].value for j in np.arange(len(allres[i]['DRM']))]
    dataElement['anydet'] = [np.any(allres[i]['DRM'][j]['det_status'] == 1) for j in np.arange(len(allres[i]['DRM']))] #were there any detections in this observations
    dataElement['madedet_times'] = [allres[i]['DRM'][j]['det_time'].value for j in np.arange(len(allres[i]['DRM'])) if dataElement['anydet'][i] == True]
    dataElement['numDetsPerTarget'] = [(allres[i]['DRM'][j]['det_status'] == 1).tolist().count(True) for j in np.arange(len(allres[i]['DRM']))] # number of detections per target
    dataElement['star_inds'] = [allres[i]['DRM'][j]['star_ind'] for j in np.arange(len(allres[i]['DRM'])) if dataElement['anydet'][i] == True] # All star indices
    dataElement['plan_inds'] = [[allres[i]['DRM'][j]['plan_inds'][ii] for ii in np.arange(len(allres[i]['DRM'][j])) if allres[i]['DRM'][j]['det_status'][ii] == 1]\
                                    for j in np.arange(len(allres[i]['DRM']))] # double iterator. Iterates over planets around star ii. Iterates over targets stars j
                                    # keeps planet inds of positive detections
    dataElement['star_indsCorrePlanInds'] = [[allres[i]['DRM'][j]['star_ind'] for ii in np.arange(len(allres[i]['DRM'][j])) if allres[i]['DRM'][j]['det_status'][ii] == 1]\
                                    for j in np.arange(len(allres[i]['DRM']))] # double iterator. Iterates over planets around star ii. Iterates over targets stars j
                                    # keeps stars associated with planet inds of positive detections
    #for j in np.arange(len(allres[i]['DRM'])):# iterate over all observations in DRM   
    #allres[i]['DRM'][j]
    data.append(dataElement)




#Load pkl and outspec files
try:
    with open(fullPathPKL, 'rb') as f:#load from cache
        DRM = pickle.load(f)
except:
    vprint('Failed to open fullPathPKL %s'%fullPathPKL)
    pass

#lines.append('comp min: ' + str(min(comp[t_dets.value>1e-10])) + '\n')
#lines.append('comp max: ' + str(max(t_dets[t_dets.value>1e-10])) + '\n')
#lines.append('comp mean: ' + str(mean(t_dets[t_dets.value>1e-10])) + '\n')

#### Extract Data from PKL file
det_times = [DRM['DRM'][i]['det_time'].value for i in np.arange(len(DRM['DRM']))]
anydet = [np.any(DRM['DRM'][i]['det_status'] == 1) for i in np.arange(len(DRM['DRM']))] #were there any detections in this observations


madedet_times = [DRM['DRM'][i]['det_time'].value for i in np.arange(len(DRM['DRM'])) if anydet[i] == True]
numDetsPerTarget = [(DRM['DRM'][i]['det_status'] == 1).tolist().count(True) for i in np.arange(len(DRM['DRM']))]
star_inds = [DRM['DRM'][i]['star_ind'] for i in np.arange(len(DRM['DRM'])) if anydet[i] == True]

plan_inds = []
star_indsCorrePlanInds = []
earthLike_sInds = []
for i in np.arange(len(DRM['DRM'])):#iterates over stars observed
    for ii in np.arange(len(DRM['DRM'][i]['plan_inds'])):# iterates over planets around a star
        if DRM['DRM'][i]['det_status'][ii] == 1:
            plan_inds.append(DRM['DRM'][i]['plan_inds'][ii])
            star_indsCorrePlanInds.append(DRM['DRM'][i]['star_ind'])
for i in np.arange(len(plan_inds)):
    pInd = plan_inds[i]
    Rp = DRM['systems']['Rp'][pInd].value
    starind = int(star_indsCorrePlanInds[i])
    sma = DRM['systems']['a'][pInd].value
    L_star = TL.L[starind] # grab star luminosity
    L_plan = L_star/sma**2. # adjust star luminosity by distance^2 in AU
    if (Rp >= 0.90 and Rp <= 1.4) and (L_plan >= 0.3586 and L_plan <= 1.1080):
        earthLike_sInds.append(starind)
earthLike_sDistances = [TL.dist[sInd].value for sInd in earthLike_sInds]

star_distances = [TL.dist[sInd].value for sInd in star_inds]


lines = []
lines.append('planned tau min (d): ' + str(min(det_times)) + '\n')
lines.append('planned tau max (d): ' + str(max(det_times)) + '\n')
lines.append('planned tau mean (d): ' + str(np.mean(det_times)) + '\n')
lines.append('planned sum(tau) (d): ' + str(sum(det_times)) + '\n')
lines.append('planned sum(tau+OHTime) (d): ' + str(len(det_times)*(totOH)+sum(det_times)) + '\n')
lines.append('planned num Star Observations: ' + str(len(det_times)) + '\n')
lines.append('obs tau min (d): ' + str(min(madedet_times)) + '\n')
lines.append('obs tau max (d): ' + str(max(madedet_times)) + '\n')
lines.append('obs tau mean (d): ' + str(np.mean(madedet_times)) + '\n')
lines.append('obs sum(tau+OHTime) (d): ' + str(len(madedet_times)*totOH+sum(madedet_times)) + '\n')
lines.append('obs num Stars with Detections: ' + str(len(madedet_times)) + '\n')
lines.append('obs sum Detections: ' + str(sum(numDetsPerTarget)) + '\n')
lines.append('obs min num Detections: ' + str(min(numDetsPerTarget)) + '\n')
lines.append('obs max num Detections: ' + str(max(numDetsPerTarget)) + '\n')
numDetsPerTarget_whereDet = [numDetsPerTarget[i] for i in np.arange(len(DRM['DRM'])) if anydet[i] == True]
lines.append('obs mean num Detections: ' + str(np.mean(numDetsPerTarget_whereDet)) + '\n')
lines.append('obs distance closest star with detection (pc): ' + str(min(star_distances)) + '\n')
lines.append('obs distance closest star with Earthlike detection (pc): ' + str(min(earthLike_sDistances)) + '\n')
lines.append('obs distance furthest star with Earthlike detection (pc): ' + str(max(earthLike_sDistances)) + '\n')



#### Calculate Maximum Theoretical Completeness ###########################################################
lines.append('#### Maximum Theoretical Completeness #############################################################################')
sInds = np.arange(TL.nStars)
intTimes = (np.zeros(TL.nStars) + 1.0e10)*u.d
mode = filter(lambda mode: mode['detectionMode'] == True, OS.observingModes)[0]
dMag_max = TL.OpticalSystem.calc_dMag_per_intTime(intTimes,TL,sInds,SS.valfZmin,ZL.fEZ0,OS.WA0,mode,C_b,C_sp)

Observable_dMagsMax = dMag_max[np.where(SS.t0.value>1e-10)[0]]
min_dMag_lim = np.min(Observable_dMagsMax)
max_dMag_lim = np.max(Observable_dMagsMax)
###########################################################################################################







# #IF SurveySimulation module is SLSQPScheduler
# initt0 = None
# comp0 = None
# if 'SLSQPScheduler' in outspec['modules']['SurveySimulation']:
#     #Extract Initial det_time and scomp0
#     initt0 = sim.SurveySimulation.t0#These are the optmial times generated by SLSQP
#     numObs0 = initt0[initt0.value>1e-10].shape[0]
#     timeConservationCheck = numObs0*(outspec['settlingTime'] + outspec['starlightSuppressionSystems'][0]['ohTime'].value) + sum(initt0).value # This assumes a specific instrument for ohTime
#     #assert abs(timeConservationCheck-outspec['missionLife']*outspec['missionPortion']*365.25) < 0.1, 'total instrument time not consistent with initial calculation'
#     if not abs(timeConservationCheck-outspec['missionLife']*outspec['missionPortion']*365.25) < 0.1:
#         vprint('total instrument time used is not within total allowed time with 0.1d')
#     assert abs(timeConservationCheck-outspec['missionLife']*outspec['missionPortion']*365.25) < 0.5, 'total instrument time not consistent with initial calculation'
#     #THIS IS JUST SUMCOMP initscomp0 = sim.SurveySimulation.scomp0

#     _, Cbs, Csps = OS.Cp_Cb_Csp(TL, range(TL.nStars), ZL.fZ0, ZL.fEZ0, 25.0, SS.WAint, SS.detmode)

#     #find baseline solution with dMagLim-based integration times
#     #self.vprint('Finding baseline fixed-time optimal target set.')
#     # t0 = OS.calc_intTime(TL, range(TL.nStars),  
#     #         ZL.fZ0, ZL.fEZ0, SS.dMagint, SS.WAint, SS.detmode)
#     comp0 = COMP.comp_per_intTime(initt0, TL, range(TL.nStars), 
#             ZL.fZ0, ZL.fEZ0, SS.WAint, SS.detmode, C_b=Cbs, C_sp=Csps)#Integration time at the initially calculated t0
#     sumComp0 = sum(comp0)

#     #Plot t0 vs c0
#     plt.figure()
#     plt.rc('axes',linewidth=2)
#     plt.rc('lines',linewidth=2)
#     #rcParams['axes.linewidth']=2
#     plt.rc('font',weight='bold')
#     #scatter(initt0.value, comp0, label='SLSQP $C_0$ ALL')
#     plt.scatter(initt0[initt0.value > 1e-10].value, comp0[initt0.value > 1e-10], label=r'SLSQP $C_0$, $\sum C_0$' + "=%0.2f"%sumComp0, alpha=0.5, color='blue')


#     #This is a calculation check to ensure the targets at less than 1e-10 d are trash
#     sIndsLT1us = np.arange(TL.nStars)[initt0.value < 1e-10]
#     t0LT1us = initt0[initt0.value < 1e-10].value + 0.1
#     comp02 = COMP.comp_per_intTime(t0LT1us*u.d, TL, sIndsLT1us.tolist(), 
#             ZL.fZ0, ZL.fEZ0, SS.WAint[sIndsLT1us], SS.detmode, C_b=Cbs[sIndsLT1us], C_sp=Csps[sIndsLT1us])

#     #Overwrite DRM with DRM just calculated
#     res = sim.run_sim()
#     DRM['DRM'] = sim.SurveySimulation.DRM



# #extract mission information from DRM
# arrival_times = [DRM['DRM'][i]['arrival_time'].value for i in np.arange(len(DRM['DRM']))]
# star_inds = [DRM['DRM'][i]['star_ind'] for i in np.arange(len(DRM['DRM']))]
# sumOHTIME = outspec['settlingTime'] + outspec['starlightSuppressionSystems'][0]['ohTime'].value
# raw_det_time = [DRM['DRM'][i]['det_time'].value for i in np.arange(len(DRM['DRM']))]#DOES NOT INCLUDE overhead time
# det_times = [DRM['DRM'][i]['det_time'].value+sumOHTIME for i in np.arange(len(DRM['DRM']))]#includes overhead time
# det_timesROUNDED = [round(DRM['DRM'][i]['det_time'].value+sumOHTIME,1) for i in np.arange(len(DRM['DRM']))]
# ObsNums = [DRM['DRM'][i]['ObsNum'] for i in np.arange(len(DRM['DRM']))]
# y_vals = np.zeros(len(det_times)).tolist()
# char_times = [DRM['DRM'][i]['char_time'].value*(1.+outspec['charMargin'])+sumOHTIME for i in np.arange(len(DRM['DRM']))]
# OBdurations = np.asarray(outspec['OBendTimes'])-np.asarray(outspec['OBstartTimes'])
# #sumOHTIME = [1 for i in np.arange(len(DRM['DRM']))]
# vprint(sum(det_times))
# vprint(sum(char_times))



# #calculate completeness at the time of each star observation
# slewTimes = np.zeros(len(star_inds))
# fZ = ZL.fZ(Obs, TL, star_inds, TK.missionStart + (arrival_times + slewTimes)*u.d, SS.detmode)
# comps = COMP.comp_per_intTime(raw_det_time*u.d, TL, star_inds, fZ, 
#         ZL.fEZ0, SS.WAint[star_inds], SS.detmode)
# sumComps = sum(comps)


# if not plt.get_fignums(): # there is no figure open
#     plt.figure()
# plt.rc('axes',linewidth=2)
# plt.rc('lines',linewidth=2)
# #rcParams['axes.linewidth']=2
# plt.rc('font',weight='bold')
# plt.scatter(raw_det_time, comps, label=r'SLSQP $C_{t_{Obs}}$, $\sum C_{t_{Obs}}$' + "=%0.2f"%sumComps, alpha=0.5, color='black')
# plt.xlim([0, 1.1*max(raw_det_time)])
# plt.ylim([0, 1.1*max(comps)])
# plt.xlabel(r'Integration Time, $\tau_i$, in (days)',weight='bold')
# plt.ylabel(r'Target Completeness, $C_i$',weight='bold')
# legend_properties = {'weight':'bold'}
# plt.legend(prop=legend_properties)

# #Done plotting Comp vs intTime of Observations
# #fullPathPKL.split('/')[-2]
# date = unicode(datetime.datetime.now())
# date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
# fname = 'C0vsT0andCvsT_' + folder.split('/')[-1] + '_' + date
# plt.savefig(os.path.join(PPoutpath, fname + '.png'))
# plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
# plt.savefig(os.path.join(PPoutpath, fname + '.eps'))


# #Manually Calculate the difference to veryify all det_times are the same
# tmpdiff = np.asarray(initt0[star_inds]) - np.asarray(raw_det_time)
# vprint(max(tmpdiff))

# vprint(-2.5*np.log10(ZL.fZ0.value)) # This is 23
# vprint(-2.5*np.log10(np.mean(fZ).value))


# ##################################
# ####  Print out Observation Stats
# lines = []
# lines.append('tau min: ' + str(min(t_dets[t_dets.value>1e-10])) + '\n')
# lines.append('tau max: ' + str(max(t_dets[t_dets.value>1e-10])) + '\n')
# lines.append('tau mean: ' + str(mean(t_dets[t_dets.value>1e-10])) + '\n')
# lines.append('sum(tau+OHTime): ' + str(len(t_dets[t_dets.value>1e-10])*1.+sum(t_dets.value)) + '\n')
# lines.append('comp min: ' + str(min(comp[t_dets.value>1e-10])) + '\n')
# lines.append('comp max: ' + str(max(t_dets[t_dets.value>1e-10])) + '\n')
# lines.append('comp mean: ' + str(mean(t_dets[t_dets.value>1e-10])) + '\n')
# lines2 = []

# sIndsObs = sInds[t_dets.value>1e-10]

# # 'BC',
# #  'BV',
# #  'Binary_Cut',
# #  'Bmag',
# #  'Hmag',
# #  'Imag',
# #  'Jmag',
# #  'Kmag',
# #  'L',
# #  'MV',
# #  'MsEst',
# #  'MsTrue',
# #  'Name',
# #  'Rmag',
# #  'Spec',
# #  'Umag',
# #  'Vmag',
# # 'dist',
# #  'starMag',
# #  'starprop',
# #  'starprop_static',
# #  'staticStars',
# #  'stellarTeff',
# #  'stellar_mass',

# #         dist# = data['st_dist'].data*u.pc
# #         parx# = self.dist.to('mas', equivalencies=u.parallax())
# #         coords# = SkyCoord(ra=data['ra']*u.deg, dec=data['dec']*u.deg,
# #               #  distance=self.dist)
# #         pmra# = data['st_pmra'].data*u.mas/u.yr
# #         pmdec# = data['st_pmdec'].data*u.mas/u.yr
# #         L# = data['st_lbol'].data
        
# #         # list of non-astropy attributes
# #         Name# = data['hip_name']
# #         Spec# = data['st_spttype']
# #         Vmag# = data['st_vmag']
# #         Jmag# = data['st_j2m']
# #         Hmag# = data['st_h2m']
# #         BV# = data['st_bmv']
# #         Bmag# = self.Vmag + data['st_bmv']
# #         Kmag# = self.Vmag - data['st_vmk']
# #         BC# = -self.Vmag + data['st_mbol']
# #         MV# = self.Vmag - 5*(np.log10(self.dist.to('pc').value) - 1)
# #         stellar_diameters# = data['st_rad']*2.*R_sun # stellar_diameters in solar diameters
# #         Binary_Cut#

# # #sim.SurveySimulation.TargetList

# header = ['sInd','dist', 'parx', 'coords', 'pmra', 'pmdec', 'L', 'Name',\
#     'Spec', 'Vmag', 'Jmag', 'Hmag', 'BV', 'Bmag', 'Kmag', 'BC', 'MV',\
#     'stellar_diameters', 'Binary_Cut']


# ##Redo
# lines.append('Name,sInd,dist,parx,coords,pmra,pmdec,L,Spec,Vmag,Jmag,Hmag,BV,Bmag,Kmag,Bmag,BC,MV,stellar_diameters\n')
# lines2.append('Name & sInd & dist & parx & coords & pmra & pmdec & L & Spec & Vmag & Jmag & Hmag & BV & Bmag & Kmag & Bmag & BC & MV & stellar_diameters \\')
# for sInd in sIndsObs:
#     #csv file
#     lineC = [str(sim.SurveySimulation.TargetList.Name[sInd]) ,\
#     str(sInd) ,\
#     str(sim.SurveySimulation.TargetList.dist[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.parx[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.coords[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.pmra[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.pmdec[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.L[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.Spec[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.Vmag[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.Jmag[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.Hmag[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.BV[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.Bmag[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.Kmag[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.Bmag[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.BC[sInd]) ,\
#     str(sim.SurveySimulation.TargetList.MV[sInd])]# ,\
#     #str(sim.SurveySimulation.TargetList.stellar_diameters[sInd])]
    
#     #csv
#     lines.append(lineC.join(','))
#     lines[-1]  = lines[-1] + '\n'

#     #Latex Table
#     lines2.append(lineC.join(' & '))
#     lines2[-1] = lines2[-1] + ' \\ \n'
