import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/EXOSIMS/EXOSIMS/Scripts'))
filename = 'sS_AYO6.json'#'sS_protoTimeKeeping.json'#'sS_AYO3.json'#sS_SLSQPstatic_parallel_ensembleJTWIN.json'#'sS_JTwin.json'#'sS_AYO4.json'#'sS_AYO3.json'
#filename = 'sS_intTime6_KeplerLike2.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)

#sim.run_sim()

sim.SurveySimulation.DRM

from pylab import *
import numpy as np
beta = np.arange(0,90)*1.0
lonMin = np.zeros(beta.shape[0])
for i in np.arange(beta.shape[0]):
    lonMin[i] = sim.ZodiacalLight.Lon_at_model11LambdaPartialZero(beta[i])

fig = figure(1)
#plot(beta,lonMin)
plot(lonMin,beta)
xlabel('lonMin')
ylabel('Beta')
xlim(0,180)
ylim(0,90)
show(block=False)
fig.savefig('/home/dean/Documents/SIOSlab/fZminModelBetavsLon'+'.png')

## Load Stark stuff for fZ calculation Look at starkAYO target List

folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/EXOSIMS/EXOSIMS/Scripts'))
filename = 'sS_AYO5.json'#'sS_protoTimeKeeping.json'#'sS_AYO3.json'#sS_SLSQPstatic_parallel_ensembleJTWIN.json'#'sS_JTwin.json'#'sS_AYO4.json'#'sS_AYO3.json'
#filename = 'sS_intTime6_KeplerLike2.json'
scriptfile = os.path.join(folder,filename)
sim2 = EXOSIMS.MissionSim.MissionSim(scriptfile)

ZL = sim2.ZodiacalLight
TL = sim2.TargetList
TK = sim2.TimeKeeping
Obs = sim2.Observatory
hashname = sim2.SurveySimulation.cachefname
sInds = np.arange(TL.nStars)
mode = sim2.SurveySimulation.mode

fZmin, absTimefZmin = sim2.ZodiacalLight.calcfZmin(sInds, Obs, TL, TK, mode, hashname)

# #dec = np.zeros(len(imat2))
# #for i in np.arange(len(imat2)):
from astropy.coordinates import SkyCoord
tmpdec = np.zeros(absTimefZmin.shape[0])
ra = np.zeros(absTimefZmin.shape[0])
r_obs = np.zeros((absTimefZmin.shape[0],3))
ra_obs = np.zeros(absTimefZmin.shape[0])
tmpdec_obs = np.zeros(absTimefZmin.shape[0])
for i in np.arange(absTimefZmin.shape[0]):
    r_targ = TL.starprop(sInds[i].astype(int),absTimefZmin[i],False)
    c = SkyCoord(r_targ[:,0],r_targ[:,1],r_targ[:,2],representation='cartesian')
    c.representation = 'spherical'
    tmpdec[i] = c.dec.value
    ra[i] = c.ra.value

    r_obs[i] = Obs.orbit(absTimefZmin[i])[0] # Sun to Spacecraft vector in HEE
    c2 = SkyCoord(r_obs[i][0],r_obs[i][1],r_obs[i][2],representation='cartesian')
    c2.representation = 'spherical'
    tmpdec_obs[i] = c2.dec.value
    ra_obs[i] = c2.ra.value
dec = abs(TL.coords.dec.value)

fig2 = figure(2)
#plot(beta,lonMin)
#WANT 
scatter((ra-ra_obs),dec)
plot(lonMin-180,beta,color='r')
xlabel('Heliocentric Earth Ecliptic ra_star - ra_Obs (deg)')
ylabel('dec_star (deg)')
#xlim(0,180)
#ylim(0,90)
show(block=False)
fig2.savefig('/home/dean/Documents/SIOSlab/fZminCalculatedBetavsLon'+'.png')

