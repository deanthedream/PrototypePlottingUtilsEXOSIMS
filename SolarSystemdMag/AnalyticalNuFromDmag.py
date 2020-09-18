# Written by Dean Keithly
import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as random
import time
from astropy import constants as const
import astropy.units as u

#### PLOT BOOL
plotBool = True
if plotBool == True:
    from plotProjectedEllipse import *

#### Randomly Generate Orbits
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))
filename = 'HabEx_CKL2_PPKL2.json'
filename = 'WFIRSTcycle6core.json'
filename = 'HabEx_CSAG13_PPSAG13_compSubtype.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
PPop = sim.PlanetPopulation
comp = sim.Completeness
TL = sim.TargetList
n = 10**5 #Dean's nice computer can go up to 10**8 what can atuin go up to?
inc, W, w = PPop.gen_angles(n,None)
W = W.to('rad').value
w = w.to('rad').value
#w correction caused in smin smax calcs
wReplacementInds = np.where(np.abs(w-1.5*np.pi)<1e-4)[0]
w[wReplacementInds] = w[wReplacementInds] - 0.001
wReplacementInds = np.where(np.abs(w-0.5*np.pi)<1e-4)[0]
w[wReplacementInds] = w[wReplacementInds] - 0.001
del wReplacementInds
inc = inc.to('rad').value
inc[np.where(inc>np.pi/2)[0]] = np.pi - inc[np.where(inc>np.pi/2)[0]]
sma, e, p, Rp = PPop.gen_plan_params(n)
sma = sma.to('AU').value





# The following are coefficients to A*x**4. + B*x**3. + C*x**2. + D*x + E = 0
# where x = np.cos(v)
# A = e**4.*lhs**2. - e**2.*lhs*np.sin(i)**2.*np.sin(omega)**2./2. + e**2.*lhs*np.sin(i)**2.*np.cos(omega)**2./2. + np.sin(i)**4.*np.sin(omega)**4./16. + np.sin(i)**4.*np.sin(omega)**2.*np.cos(omega)**2./8. + np.sin(i)**4.*np.cos(omega)**4./16.
# B = 4.*e**3.*lhs**2. - e**2.*lhs*np.sin(i)*np.sin(omega) - e*lhs*np.sin(i)**2.*np.sin(omega)**2. + e*lhs*np.sin(i)**2.*np.cos(omega)**2. + np.sin(i)**3.*np.sin(omega)**3./4. + np.sin(i)**3.*np.sin(omega)*np.cos(omega)**2./4.
# C = 6*e**2.*lhs**2. - e**2.*lhs*np.sin(i)**2.*np.cos(omega)**2./2. - e**2.*lhs/2. - 2.*e*lhs*np.sin(i)*np.sin(omega) - lhs*np.sin(i)**2.*np.sin(omega)**2./2. + lhs*np.sin(i)**2.*np.cos(omega)**2./2. - np.sin(i)**4.*np.sin(omega)**2.*np.cos(omega)**2./8. - np.sin(i)**4.*np.cos(omega)**4./8. + 3.*np.sin(i)**2.*np.sin(omega)**2./8. + np.sin(i)**2.*np.cos(omega)**2./8.
# D = 4.*e*lhs**2. - e*lhs*np.sin(i)**2.*np.cos(omega)**2. - e*lhs - lhs*np.sin(i)*np.sin(omega) - np.sin(i)**3.*np.sin(omega)*np.cos(omega)**2./4. + np.sin(i)*np.sin(omega)/4.
# E = lhs**2. - lhs*np.sin(i)**2.*np.cos(omega)**2./2. - lhs/2. + np.sin(i)**4.*np.cos(omega)**4./16. - np.sin(i)**2.*np.cos(omega)**2./8. + 1/16.

A = (4.*e**3.*lhs**2. - e**2.*lhs*np.sin(i)*np.sin(omega) - e*lhs*np.sin(i)**2.*np.sin(omega)**2. + e*lhs*np.sin(i)**2.*np.cos(omega)**2. + np.sin(i)**3.*np.sin(omega)**3./4. + np.sin(i)**3.*np.sin(omega)*np.cos(omega)**2./4.)/(e**4.*lhs**2. - e**2.*lhs*np.sin(i)**2.*np.sin(omega)**2./2. + e**2.*lhs*np.sin(i)**2.*np.cos(omega)**2./2. + np.sin(i)**4.*np.sin(omega)**4./16. + np.sin(i)**4.*np.sin(omega)**2.*np.cos(omega)**2./8. + np.sin(i)**4.*np.cos(omega)**4./16.)
B = (6.*e**2.*lhs**2. - e**2.*lhs*np.sin(i)**2.*np.cos(omega)**2./2. - e**2.*lhs/2. - 2.*e*lhs*np.sin(i)*np.sin(omega) - lhs*np.sin(i)**2.*np.sin(omega)**2./2. + lhs*np.sin(i)**2.*np.cos(omega)**2./2. - np.sin(i)**4.*np.sin(omega)**2.*np.cos(omega)**2./8. - np.sin(i)**4.*np.cos(omega)**4./8. + 3.*np.sin(i)**2.*np.sin(omega)**2./8. + np.sin(i)**2.*np.cos(omega)**2./8.)/(e**4.*lhs**2. - e**2.*lhs*np.sin(i)**2.*np.sin(omega)**2./2. + e**2.*lhs*np.sin(i)**2.*np.cos(omega)**2./2. + np.sin(i)**4.*np.sin(omega)**4./16. + np.sin(i)**4.*np.sin(omega)**2.*np.cos(omega)**2./8. + np.sin(i)**4.*np.cos(omega)**4./16.)
C = (4.*e*lhs**2. - e*lhs*np.sin(i)**2.*np.cos(omega)**2. - e*lhs - lhs*np.sin(i)*np.sin(omega) - np.sin(i)**3.*np.sin(omega)*np.cos(omega)**2./4. + np.sin(i)*np.sin(omega)/4.)/(e**4.*lhs**2. - e**2.*lhs*np.sin(i)**2.*np.sin(omega)**2./2. + e**2.*lhs*np.sin(i)**2.*np.cos(omega)**2./2. + np.sin(i)**4.*np.sin(omega)**4./16. + np.sin(i)**4.*np.sin(omega)**2.*np.cos(omega)**2./8. + np.sin(i)**4.*np.cos(omega)**4./16.)
D = (lhs**2. - lhs*np.sin(i)**2.*np.cos(omega)**2./2. - lhs/2. + np.sin(i)**4.*np.cos(omega)**4./16. - np.sin(i)**2.*np.cos(omega)**2./8. + 1./16.)/(e**4.*lhs**2. - e**2.*lhs*np.sin(i)**2.*np.sin(omega)**2./2. + e**2.*lhs*np.sin(i)**2.*np.cos(omega)**2./2. + np.sin(i)**4.*np.sin(omega)**4./16. + np.sin(i)**4.*np.sin(omega)**2.*np.cos(omega)**2./8. + np.sin(i)**4.*np.cos(omega)**4./16.)


xreal, delta, P, D2, R, delta_0 = quarticSolutions_ellipse_to_Quarticipynb(A, B, C, D)

#Convert x to nu
nus = np.arccos(xreal)

