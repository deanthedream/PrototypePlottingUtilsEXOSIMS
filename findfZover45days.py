import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/EXOSIMS/EXOSIMS/Scripts'))
filename = 'sS_AYO4.json'#'sS_protoTimeKeeping.json'#'sS_AYO3.json'#sS_SLSQPstatic_parallel_ensembleJTWIN.json'#'sS_JTwin.json'#'sS_AYO4.json'#'sS_AYO3.json'
#filename = 'sS_intTime6_KeplerLike2.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)

boxesPerDay = 1000./365.25#box per day
import numpy as np
numBoxes = np.floor(45*boxesPerDay)
delta = np.zeros([sim.SurveySimulation.fZ_startSaved.shape[0],sim.SurveySimulation.fZ_startSaved.shape[1]-45])
for i in np.arange(sim.SurveySimulation.fZ_startSaved.shape[1]-45):
    delta[:,i] = abs(sim.SurveySimulation.fZ_startSaved[:,i+45]-sim.SurveySimulation.fZ_startSaved[:,i])
maxDeltafZ = np.zeros(sim.SurveySimulation.fZ_startSaved.shape[0])
for j in np.arange(delta.shape[0]):
    maxDeltafZ[j] = np.max(delta[j,:])

sim.SurveySimulation.ZodiacalLight.fEZ0




#sim.run_sim()

#sim.SurveySimulation.DRM

#print(saltyburrito)
