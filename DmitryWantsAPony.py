#DmitryWantsAPony.py
# Detectability of 47 UMa c

import os
import EXOSIMS.missionSim
import numpy.random as rand

folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/PrototypePlottingUtilsEXOSIMS'))
filename = DmitryWantsAPony.json
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)

PPop = sim.PlanetPopulation

#From Dmitry's links
period = 2391 #days +100 -87
sma = 3.6 #+/-0.1
e = 0.098 #+0.047 -0.096
#Time periastron passage (days) 2452441 +628-825
#Longitude of Periastron (deg) 295 +114-160
mass = 0.54 #+.066 -.073 in jupiter mass Msin(i)

#Host Star Aliases
#47 UMa     2MASS J10592802+4025485     BD+41 2147  Chalawan    GJ 407  HD 95128    HIP 53721   HR 4277     IRAS 10566+4041     SAO 43557   TYC 3009-02703-1    WISE J105927.66+402549.4
Bmag = 5.66 #(mag)
radius = 1.23 #r sun
d = 13.802083302115193#distance (pc) ±0.028708172014593
star_mass = 1.03 #0.05


#### Randomly Generate 47 UMa c planet parameters
n = 10**3
inc, W, w = PPop.gen_angles(n,None)
inc = inc.to('rad').value
inc[np.where(inc>np.pi/2)[0]] = np.pi - inc[np.where(inc>np.pi/2)[0]]
W = W.to('rad').value
w = w.to('rad').value
a, e, p, Rp = PPop.gen_plan_params(n)
a = a.to('AU').value

a = (3.7-3.5)*rand.random(n)+3.5 #uniform random
e = (0.145-0.002)*rand.random(n)+0.02 #uniform random
Msini = (0.606-0.467)*rand.random(n)+0.467
pmass = Msini/np.sin(inc)
#TODO CHECK FOR INF/TOO LARGE


