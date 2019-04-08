import sys, os.path, EXOSIMS, EXOSIMS.MissionSim, json
import numpy as np
import copy
from copy import deepcopy
import astropy.units as u
# folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_40319_2'))#EXOSIMS/EXOSIMS/Scripts'))#EXOSIMS/EXOSIMS/Scripts'))
# #filename = 'HabEx_4m_TSDD_pop100DD_revisit_20180424.json'
# filename = 'WFIRSTcycle6core_CKL2_PPKL2.json'#'HabEx_4m_TSDD_pop100DD_revisit_20180424.json'#'WFIRSTcycle6core.json'#'Dean3June18RS26CXXfZ01OB66PP01SU01.json'#'Dean1June18RS26CXXfZ01OB56PP01SU01.json'#'./TestScripts/04_KeplerLike_Occulter_linearJScheduler.json'#'Dean13May18RS09CXXfZ01OB01PP03SU01.json'#'sS_AYO7.json'#'ICDcontents.json'###'sS_protoTimeKeeping.json'#'sS_AYO3.json'#sS_SLSQPstatic_parallel_ensembleJTWIN.json'#'sS_JTwin.json'#'sS_AYO4.json'#'sS_AYO3.json'
# #filename = 'sS_intTime6_KeplerLike2.json'
# #folder = '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/Scripts/TestScripts/'
# #filename='01_all_defaults.json'
# scriptfile = os.path.join(folder,filename)
# sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
# #Note no completness specs in SAG13 SAG13

# folder = os.path.normpath(os.path.expandvars('$HOME/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_40319_2\
# /WFIRSTcycle6core_CKL2_PPKL2/'))#WFIRSTcycle6core_CSAG13_PPSAG13/'))BAD_USEDPROTO_WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo\
# outspecPath = os.path.join(folder,'outspec.json')
# with open(outspecPath, 'rb') as g:
#     outspec = json.load(g)
# sim = EXOSIMS.MissionSim.MissionSim(scriptfile=None,nopar=True,**outspec)


import json
scriptfile = '/home/dean/Documents/exosims/Scripts/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_40319_2/WFIRSTcycle6core_CKL2_PPKL2.json'
with open(scriptfile, 'rb') as f:
    a = json.load(f)
    #scripta = f.read()
#a = json.loads(scripta)
with open('/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_40319_2/WFIRSTcycle6core_CKL2_PPKL2/outspec.json', 'rb') as f:
    b = json.load(f)
    #scriptb = f.read()
#b = json.loads(scriptb)
from jsondiff import diff
diff(a,b)


for key in a.keys():
    print(a[key])
    print(b[key])
print('*****************************************')
for i in np.arange(len(a['starlightSuppressionSystems'])):
    for key in a['starlightSuppressionSystems'][i].keys():
        print(a['starlightSuppressionSystems'][i][key])
        print(b['starlightSuppressionSystems'][i][key])
print('*****************************************')
for i in np.arange(len(a['scienceInstruments'])):
    for key in a['scienceInstruments'][i].keys():
        print(a['scienceInstruments'][i][key])
        print(b['scienceInstruments'][i][key])

b2 = copy.deepcopy(b)
# sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
# sim2 = EXOSIMS.MissionSim.MissionSim(scriptfile=None,nopar=True,**b)
# out = sim.genOutSpec()
# out2 = sim2.genOutSpec()
# set1 = set(out.items())
# set2 = set(out.items())

# for key in out.keys():
#     print(out[key])
#     print(out2[key])
# print('*****************************************')
# for i in np.arange(len(out['starlightSuppressionSystems'])):
#     for key in out['starlightSuppressionSystems'][i].keys():
#         print(out['starlightSuppressionSystems'][i][key])
#         print(out2['starlightSuppressionSystems'][i][key])
# print('*****************************************')
# for i in np.arange(len(out['scienceInstruments'])):
#     for key in out['scienceInstruments'][i].keys():
#         print(out['scienceInstruments'][i][key])
#         print(out2['scienceInstruments'][i][key])



# sim.run_sim()
# initt0 = sim.SurveySimulation.t0


# SS = sim.SurveySimulation
# ZL = SS.ZodiacalLight
# COMP = SS.Completeness
# OS = SS.OpticalSystem
# Obs = SS.Observatory
# TL = SS.TargetList
# TK = SS.TimeKeeping

# _, Cbs, Csps = OS.Cp_Cb_Csp(TL, range(TL.nStars), ZL.fZ0, ZL.fEZ0, 25.0, SS.WAint, SS.detmode)

# #find baseline solution with dMagLim-based integration times
# #self.vprint('Finding baseline fixed-time optimal target set.')
# # t0 = OS.calc_intTime(TL, range(TL.nStars),  
# #         ZL.fZ0, ZL.fEZ0, SS.dMagint, SS.WAint, SS.detmode)
# comp0 = COMP.comp_per_intTime(initt0, TL, range(TL.nStars), 
#         ZL.fZ0, ZL.fEZ0, SS.WAint, SS.detmode, C_b=Cbs, C_sp=Csps)


# COMP.xnew


# AAS run SAG13 SAG13 intTimes
# <Quantity 
# [0.        , 0.09846247, 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.22837258,
#            0.        , 0.        , 0.1255496 , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.32602634, 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.21689025, 0.        , 0.        , 0.        ,
#            0.        , 0.28595387, 0.30284413, 0.16834581, 0.3221259 ,
#            0.        , 0.        , 0.        , 0.16456935, 0.11705623,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.0857372 , 0.        , 0.        , 0.        , 0.        ,
#            0.27886695, 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.21803448,
#            0.24902319, 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.21318869, 0.        , 0.        , 0.        , 0.        ,
#            0.30387184, 0.        , 0.27701325, 0.23008331, 0.        ,
#            0.18408011, 0.23340966, 0.        , 0.        , 0.16945636,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.05090361, 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.14757401, 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.26597054, 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.17715346, 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.18719908, 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.01905503,
#            0.1577811 , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.04283256,
#            0.        , 0.        , 0.        , 0.05897702, 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.12453089, 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.1442378 , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.14687245, 0.        , 0.31132041, 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.34380867, 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.20880048,
#            0.27629866, 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.11354317, 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.30729374, 0.28077932,
#            0.        , 0.09413918, 0.17692941, 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.21235771, 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.22915012, 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.22927449, 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.12225273, 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.11805468, 0.        , 0.        ,
#            0.        , 0.08808796, 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.21297312, 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.14788923, 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.09733857,
#            0.        , 0.        , 0.        , 0.        , 0.24331853,
#            0.        , 0.        , 0.        , 0.12623979, 0.19751527,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.32688213, 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.09581371, 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.1624829 , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.14918224, 0.        ,
#            0.        , 0.        , 0.        , 0.0368401 , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.16119859, 0.        , 0.26334719,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.05062694, 0.        , 0.        , 0.18419505, 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.1711181 , 0.        , 0.        ,
#            0.        , 0.32525347, 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.1616438 , 0.10490874, 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.106705  , 0.        , 0.        , 0.        ,
#            0.22821691, 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.1286106 ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.19134825, 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.22635116, 0.        , 0.        , 0.        ,
#            0.        , 0.3425973 , 0.        , 0.06257459, 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.        , 0.        , 0.        ,
#            0.        , 0.        , 0.14742928, 0.        , 0.        ,
#            0.22041959, 0.        , 0.        , 0.        , 0.        ,
#            0.        ] 
#d>



# out = sim.genOutSpec()
# print(out['modules']['StarCatalog'])
# print(sim.TargetList.keepStarCatalog)
# sim.reset_sim()
# out = sim.genOutSpec()
# print(out['modules']['StarCatalog'])
# print(sim.TargetList.keepStarCatalog)


#FOR TESTING RESET ISSUES############################
# import numpy as np
# for i in range(1,100):
#     sim.run_sim()
#     sim.reset_sim()
#     print 'Print Current Abs Time: ' + str(sim.TimeKeeping.currentTimeAbs)
#     print 'AbsTimefZmin: ' + str(min(sim.SurveySimulation.absTimefZmin))
#     #sInds = np.arange(sim.TargetList.nStars)
#     #tmp, sim.SurveySimulation.absTimefZmin = sim.ZodiacalLight.calcfZmin(sInds, sim.Observatory, sim.TargetList, sim.TimeKeeping, sim.SurveySimulation.mode, sim.SurveySimulation.cachefname) # find fZmin to use in intTimeFilter
#     print 'AbsTimefZmin: ' + str(min(sim.SurveySimulation.absTimefZmin))

    

# #Sum total mission time...########
# import numpy as np
# import astropy.units as u
# DRM = sim.SurveySimulation.DRM

# det_times = list()
# eOt = list()
# arrival_times = list()
# for i in np.arange(len(DRM)):
#     det_times.append(DRM[i]['det_time'].value)
#     eOt.append(DRM[i]['exoplanetObsTime'].value)
#     arrival_times.append(DRM[i]['arrival_time'].value)

# sum(det_times) + len(DRM)
# ####################

# from pylab import *
# try:
#     plt.close('all')
# except:
#     pass
# fig = figure(num=1)
# plot(arrival_times[1:], eOt[:-1])
# axis('equal')
# ylabel('exoplanetObsTime')
# xlabel('arrivalTime')
# plt.show(block=False)


# ###CHECK IF ANY CHARACTERIZATIONS WERE MADE
# chars = [x for x in DRM if x['char_time'].value > 0]
# print chars







# arrival_times = [DRM[i]['arrival_time'].value for i in np.arange(len(DRM))]
# sumOHTIME = 1
# det_times = [DRM[i]['det_time'].value+sumOHTIME for i in np.arange(len(DRM))]
# det_timesROUNDED = [round(DRM[i]['det_time'].value+sumOHTIME,1) for i in np.arange(len(DRM))]
# ObsNums = [DRM[i]['ObsNum'] for i in np.arange(len(DRM))]
# y_vals = np.zeros(len(det_times)).tolist()
# char_times = [DRM[i]['char_time'].value*(1+sim.SurveySimulation.charMargin)+sumOHTIME for i in np.arange(len(DRM))]
# OBdurations = np.asarray(sim.TimeKeeping.OBendTimes-sim.TimeKeeping.OBstartTimes)
# #sumOHTIME = [1 for i in np.arange(len(DRM))]
# print(sum(det_times))
# print(sum(char_times))


# #Check if plotting font #########################################################
# tmpfig = plt.figure(figsize=(30,3.5),num=0)
# ax = tmpfig.add_subplot(111)
# t = ax.text(0, 0, "Obs#   ,  d", ha='center',va='center',rotation='vertical', fontsize=8)
# r = tmpfig.canvas.get_renderer()
# bb = t.get_window_extent(renderer=r)
# Obstxtwidth = bb.width#Width of text
# Obstxtheight = bb.height#height of text
# FIGwidth, FIGheight = tmpfig.get_size_inches()*tmpfig.dpi
# plt.show(block=False)
# plt.close()
# daysperpixelapprox = max(arrival_times)/FIGwidth#approximate #days per pixel
# if mean(det_times)*0.8/daysperpixelapprox > Obstxtwidth:
#     ObstextBool = True
# else:
#     ObstextBool = False

# tmpfig = plt.figure(figsize=(30,3.5),num=0)
# ax = tmpfig.add_subplot(111)
# t = ax.text(0, 0, "OB#  , dur.=    d", ha='center',va='center',rotation='horizontal', fontsize=12)
# r = tmpfig.canvas.get_renderer()
# bb = t.get_window_extent(renderer=r)
# OBtxtwidth = bb.width#Width of text
# OBtxtheight = bb.height#height of text
# FIGwidth, FIGheight = tmpfig.get_size_inches()*tmpfig.dpi
# plt.show(block=False)
# plt.close()
# if mean(OBdurations)*0.8/daysperpixelapprox > OBtxtwidth:
#     OBtextBool = True
# else:
#     OBtextBool = False
# #################################################################################



# colors = 'rb'#'rgbwmc'
# patch_handles = []
# fig = plt.figure(figsize=(30,3.5),num=1)

# # Plot All Detection Observations
# ind = 0
# obs = 0
# for (det_time, l, char_time) in zip(det_times, ObsNums, char_times):
#     #print det_time, l
#     patch_handles.append(ax.barh(0, det_time, align='center', left=arrival_times[ind],
#         color=colors[int(obs) % len(colors)]))
#     if not char_time == 0:
#         ax.barh(0, char_time, align='center', left=arrival_times[ind]+det_time,color=(255/255.,69/255.,0/255.))
#     ind += 1
#     obs += 1
#     patch = patch_handles[-1][0] 
#     bl = patch.get_xy()
#     x = 0.5*patch.get_width() + bl[0]
#     y = 0.5*patch.get_height() + bl[1]
#     plt.rc('axes',linewidth=2)
#     plt.rc('lines',linewidth=2)
#     rcParams['axes.linewidth']=2
#     rc('font',weight='bold')
#     if ObstextBool: 
#         ax.text(x, y, "Obs#%d, %dd" % (l,det_time), ha='center',va='center',rotation='vertical', fontsize=8)

# # Plot Observation Blocks
# patch_handles2 = []
# for (OBnum, OBdur, OBstart) in zip(xrange(len(OBdurations)), OBdurations, np.asarray(sim.TimeKeeping.OBstartTimes)):
#     patch_handles2.append(ax.barh(1, OBdur, align='center', left=OBstart, hatch='//',linewidth=2.0, edgecolor='black'))
#     patch = patch_handles2[-1][0] 
#     bl = patch.get_xy()
#     x = 0.5*patch.get_width() + bl[0]
#     y = 0.5*patch.get_height() + bl[1]
#     if OBtextBool:
#         ax.text(x, y, "OB#%d, dur.= %dd" % (OBnum,OBdur), ha='center',va='center',rotation='horizontal',fontsize=12)

# # Plot Asthetics
# y_pos = np.arange(2)#Number of xticks to have
# plt.rc('axes',linewidth=2)
# plt.rc('lines',linewidth=2)
# rcParams['axes.linewidth']=2
# rc('font',weight='bold') 
# ax.set_yticks(y_pos)
# ax.set_yticklabels(('Obs','OB'),fontsize=12)
# ax.set_xlabel('Current Normalized Time (days)', weight='bold',fontsize=12)
# #title('Mission Timeline for runName: ' + dirs[cnt] + '\nand pkl file: ' + pklfname[cnt], weight='bold',fontsize=12)
# plt.tight_layout()
# plt.show(block=False)
# #savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'Timeline' + '.png')
# #savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'Timeline' + '.svg')
# #savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'Timeline' + '.eps')




#### Calculate minimim Observable Separation on closes star
# minObservableSeparation = np.min(sim.TargetList.dist).to('AU').value*np.tan(sim.OpticalSystem.IWA.to('rad').value) # In AU using tan(theta) = S/dist
# minSMAObservable = minObservableSeparation/(1.+np.max(sim.PlanetPopulation.erange)) # From Vallado, Rp = SMA*(1+e), min SMA in AU

#out['keepStarCatalog'] = True
b2['keepStarCatalog'] = True
TL = EXOSIMS.MissionSim.get_module('TargetList')(**b2)
TL.catalog_atts.remove('MsEst')
TL.catalog_atts.remove('MsTrue')
TL.catalog_atts.remove('comp0')
TL.catalog_atts.remove('tint0')
#Bring all attributes from Star Catalog to TL as in populate_target_list
#print(saltyburrito)
def moveAtts(TL):
    SC = TL.StarCatalog
    for att in TL.catalog_atts:
        if type(getattr(SC, att)) == np.ma.core.MaskedArray:
            setattr(TL, att, getattr(SC, att).filled(fill_value=float('nan')))
        else:
            setattr(TL, att, getattr(SC, att))
    return TL
TL = moveAtts(TL)

import re
#### subM filter
preSubM_filter = len(TL.Spec)
specregex = re.compile('([OBAFGKMLTY])*')
spect = np.full(TL.Spec.size, '')
for j,s in enumerate(TL.Spec):
     m = specregex.match(s)
     if m:
         spect[j] = m.groups()[0]

i = np.where((spect != 'L') & (spect != 'T') & (spect != 'Y'))[0]
postSubM_filter = len(i)
###########

#### main_sequence_filter #######################
premain_sequence_filter = len(TL.BV)
i1 = np.where((TL.BV < 0.74) & (TL.MV < 6.*TL.BV + 1.8))[0]
i2 = np.where((TL.BV >= 0.74) & (TL.BV < 1.37) & \
        (TL.MV < 4.3*TL.BV + 3.05))[0]
i3 = np.where((TL.BV >= 1.37) & (TL.MV < 18.*TL.BV - 15.7))[0]
i4 = np.where((TL.BV < 0.87) & (TL.MV > -8.*(TL.BV - 1.35)**2. + 7.01))[0]
i5 = np.where((TL.BV >= 0.87) & (TL.BV < 1.45) & \
        (TL.MV < 5.*TL.BV + 0.81))[0]
i6 = np.where((TL.BV >= 1.45) & (TL.MV > 18.*TL.BV - 18.04))[0]
ia = np.append(np.append(i1, i2), i3)
ib = np.append(np.append(i4, i5), i6)
i = np.intersect1d(np.unique(ia), np.unique(ib))
postmain_sequence_filter = len(i)
#################################################

#### fgk filter #################################
prefgk_filter = len(TL.Spec)
spec = np.array(list(map(str, TL.Spec)))
iF = np.where(np.core.defchararray.startswith(spec, 'F'))[0]
iG = np.where(np.core.defchararray.startswith(spec, 'G'))[0]
iK = np.where(np.core.defchararray.startswith(spec, 'K'))[0]
i = np.append(np.append(iF, iG), iK)
i = np.unique(i)
postfgk_filter = len(i)
#################################################

#### vis_mag_filter #############################
# previs_mag_filter = len(TL.Vmag)
# i = np.where(TL.Vmag < Vmagcrit)[0]
# postvis_magfilter = len(i)
#################################################

#### outside_IWA_filter #########################
preoutside_IWA_filter = len(TL.L)
PPop = TL.PlanetPopulation
OS = TL.OpticalSystem

s = np.tan(OS.IWA)*TL.dist
L = np.sqrt(TL.L) if PPop.scaleOrbits else 1.
i = np.where(s < L*np.max(PPop.rrange))[0]
postoutside_IWA_filter = len(i)
#################################################

from EXOSIMS.util.deltaMag import deltaMag
#### max_dmag_filter ############################
premax_dmag_filter = len(TL.dist)
PPop = TL.PlanetPopulation
PPMod = TL.PlanetPhysicalModel
Comp = TL.Completeness

# s and beta arrays
s = np.tan(TL.OpticalSystem.WA0)*TL.dist
if PPop.scaleOrbits:
    s /= np.sqrt(TL.L)
beta = np.array([1.10472881476178]*len(s))*u.rad

# fix out of range values
below = np.where(s < np.min(PPop.rrange)*np.sin(beta))[0]
above = np.where(s > np.max(PPop.rrange)*np.sin(beta))[0]
s[below] = np.sin(beta[below])*np.min(PPop.rrange)
beta[above] = np.arcsin(s[above]/np.max(PPop.rrange))

# calculate delta mag
p = np.max(PPop.prange)
Rp = np.max(PPop.Rprange)
d = s/np.sin(beta)
Phi = PPMod.calc_Phi(beta)
i = np.where(deltaMag(p, Rp, d, Phi) < Comp.dMagLim)[0]
postmax_dmag_filter = len(i)
#################################################

#### int_cutoff_filter ##########################
nanVmagInds = np.argwhere(np.isnan(TL.Vmag))
nanBVInds = np.argwhere(np.isnan(TL.BV))
sInds = list(np.arange(len(TL.Vmag)))
sInds  = np.asarray([ind for ind in sInds if not ind in nanVmagInds and not ind in nanBVInds])
#DELETE mV = TL.starMag(sInds,565.0*u.nm)
#DELETE Cp, Cb, Csp = OS.Cp_Cb_Csp(TL, sInds, )
mode = list(filter(lambda mode: mode['detectionMode'] == True, TL.OpticalSystem.observingModes))[0]
fZ = 0./u.arcsec**2
fEZ = 0./u.arcsec**2
dMag = TL.OpticalSystem.dMag0
WA = TL.OpticalSystem.WA0
minintTime = TL.OpticalSystem.calc_intTime(TL, sInds, fZ, fEZ, dMag, WA, mode)
preint_cutoff_filter = len(sInds)
i = np.where(minintTime < TL.OpticalSystem.intCutoff)[0]
postint_cutoff_filter = len(i)
#################################################

#### completeness_filter ########################
comp0 = TL.Completeness.target_completeness(TL)
precompleteness_filter = len(comp0)
i = np.where(comp0 >= TL.Completeness.minComp)[0]
postcompleteness_filter = len(i)
#################################################

#### life_expectancy_filter #####################
prelife_expectancy_filter = len(TL.BV)
i = np.where(TL.BV > 0.3)[0]
postlife_expectancy_filter = len(i)
#################################################

#### binary_filter ##############################
prebinary_filter = len(TL.Binary_Cut)
i = np.where(TL.Binary_Cut == False)[0]
postbinary_filter = len(i)
#################################################

#### nan_filter #################################
# filter out nan values in numerical attributes
tmp = list()
for att in TL.catalog_atts:
    if getattr(TL, att).shape[0] == 0:
        pass
    elif (type(getattr(TL, att)[0]) == str) or (type(getattr(TL, att)[0]) == bytes):
        # FIXME: intent here unclear: 
        #   note float('nan') is an IEEE NaN, getattr(.) is a str, and != on NaNs is special
        i = np.where(getattr(TL, att) != float('nan'))[0]
        tmp.append(i)
        #TL.revise_lists(i)
    # exclude non-numerical types
    elif type(getattr(TL, att)[0]) not in (np.unicode_, np.string_, np.bool_, bytes):
        if att == 'coords':
            i1 = np.where(~np.isnan(TL.coords.ra.to('deg').value))[0]
            i2 = np.where(~np.isnan(TL.coords.dec.to('deg').value))[0]
            i = np.intersect1d(i1,i2)
            tmp.append(i)
        else:
            i = np.where(~np.isnan(getattr(TL, att)))[0]
            tmp.append(i)
tmpl = [len(tmp[i]) for j in np.arange(len(tmp))]
postnan_filter = np.sum(2396-np.asarray(tmpl))
#################################################



