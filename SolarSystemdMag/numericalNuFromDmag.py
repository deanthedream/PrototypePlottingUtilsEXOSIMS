# numericalNuFromDmag

import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as random
import time
from astropy import constants as const
import astropy.units as u
import itertools
import os
try:
    import cPickle as pickle
except:
    import pickle
from EXOSIMS.util.deltaMag import deltaMag
import time

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
n = 10**5. #Dean's nice computer can go up to 10**8 what can atuin go up to?
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
inc[np.where(inc>np.pi/2.)[0]] = np.pi - inc[np.where(inc>np.pi/2.)[0]]
sma, e, p, Rp = PPop.gen_plan_params(n)
sma = sma.to('AU').value

a = sma*u.AU
dmag = 25. #specify the dmag to calculate for

#This is a solver for nu from dmag assuming a quasi-lambert phase function fullEqnX from AnalyticalNuFromDmag3.ipynb
#Generate terms
lhs = 10.**(-0.4*dmag)*(1.-e**2.)**2.*(a.to('AU')/Rp.to('AU')).decompose().value**2/p


tstart_cos = time.time()
#These are the coefficients starting at cos(nu)^8 
A = e**4.*np.sin(inc)**4.*np.sin(w)**4./16. + e**4.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./8. + e**4.*np.sin(inc)**4.*np.cos(w)**4./16.
B = e**4.*np.sin(inc)**3.*np.sin(w)**3./4. + e**4.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./4. + e**3.*np.sin(inc)**4.*np.sin(w)**4./4. + e**3.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. + e**3.*np.sin(inc)**4.*np.cos(w)**4./4.
C = -e**4.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./8. - e**4.*np.sin(inc)**4.*np.cos(w)**4./8. + 3.*e**4.*np.sin(inc)**2.*np.sin(w)**2./8. + e**4.*np.sin(inc)**2.*np.cos(w)**2./8. + e**3.*np.sin(inc)**3.*np.sin(w)**3. + e**3.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + 3.*e**2.*np.sin(inc)**4.*np.sin(w)**4./8. + 3.*e**2.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./4. + 3.*e**2.*np.sin(inc)**4.*np.cos(w)**4./8.
D = -e**4.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./4. + e**4.*np.sin(inc)*np.sin(w)/4. - e**3.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. - e**3.*np.sin(inc)**4.*np.cos(w)**4./2. + 3.*e**3.*np.sin(inc)**2.*np.sin(w)**2./2. + e**3.*np.sin(inc)**2.*np.cos(w)**2./2. + 3.*e**2.*np.sin(inc)**3.*np.sin(w)**3./2. + 3.*e**2.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./2. + e*np.sin(inc)**4.*np.sin(w)**4./4. + e*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. + e*np.sin(inc)**4.*np.cos(w)**4./4.
E = e**4.*np.sin(inc)**4.*np.cos(w)**4./16. - e**4.*np.sin(inc)**2.*np.cos(w)**2./8. + e**4./16. - e**3.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + e**3.*np.sin(inc)*np.sin(w) - e**2.*lhs*np.sin(inc)**2.*np.sin(w)**2./2. + e**2.*lhs*np.sin(inc)**2.*np.cos(w)**2./2. - 3.*e**2.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./4. - 3.*e**2.*np.sin(inc)**4.*np.cos(w)**4./4. + 9*e**2.*np.sin(inc)**2.*np.sin(w)**2./4. + 3.*e**2.*np.sin(inc)**2.*np.cos(w)**2./4. + e*np.sin(inc)**3.*np.sin(w)**3. + e*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + np.sin(inc)**4.*np.sin(w)**4./16. + np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./8. + np.sin(inc)**4.*np.cos(w)**4./16.
F = e**3.*np.sin(inc)**4.*np.cos(w)**4./4. - e**3.*np.sin(inc)**2.*np.cos(w)**2./2. + e**3./4. - e**2.*lhs*np.sin(inc)*np.sin(w) - 3.*e**2.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./2. + 3.*e**2.*np.sin(inc)*np.sin(w)/2. - e*lhs*np.sin(inc)**2.*np.sin(w)**2. + e*lhs*np.sin(inc)**2.*np.cos(w)**2. - e*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. - e*np.sin(inc)**4.*np.cos(w)**4./2. + 3.*e*np.sin(inc)**2.*np.sin(w)**2./2. + e*np.sin(inc)**2.*np.cos(w)**2./2. + np.sin(inc)**3.*np.sin(w)**3./4. + np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./4.
G = -e**2.*lhs*np.sin(inc)**2.*np.cos(w)**2./2. - e**2.*lhs/2. + 3.*e**2.*np.sin(inc)**4.*np.cos(w)**4./8. - 3.*e**2.*np.sin(inc)**2.*np.cos(w)**2./4. + 3.*e**2./8. - 2*e*lhs*np.sin(inc)*np.sin(w) - e*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + e*np.sin(inc)*np.sin(w) - lhs*np.sin(inc)**2.*np.sin(w)**2./2. + lhs*np.sin(inc)**2.*np.cos(w)**2./2. - np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./8. - np.sin(inc)**4.*np.cos(w)**4./8. + 3.*np.sin(inc)**2.*np.sin(w)**2./8. + np.sin(inc)**2.*np.cos(w)**2./8.
H = -e*lhs*np.sin(inc)**2.*np.cos(w)**2. - e*lhs + e*np.sin(inc)**4.*np.cos(w)**4./4. - e*np.sin(inc)**2.*np.cos(w)**2./2. + e/4. - lhs*np.sin(inc)*np.sin(w) - np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./4. + np.sin(inc)*np.sin(w)/4.
I = lhs**2. - lhs*np.sin(inc)**2.*np.cos(w)**2./2. - lhs/2. + np.sin(inc)**4.*np.cos(w)**4./16. - np.sin(inc)**2.*np.cos(w)**2./8. + 1/16.
coeffs = np.asarray([A,B,C,D,E,F,G,H,I])

#solve for x in the polynomial (where x=cos(nu))
out = list()
for i in np.arange(coeffs.shape[1]):
    out.append(np.roots(coeffs[:,i])) # this is x)
out = np.asarray(out)

#Throw out roots not in correct bounds
inBoundsBools = (np.abs(out.imag) <= 1e-7)*(out.real >= -1.)*(out.real <= 1.) #the out2 solutions that are inside of the desired bounds
outBoundsBools = np.logical_not(inBoundsBools) # the out2 solutions that are inside the desired bounds
outReal = out
outReal[outBoundsBools] = out[outBoundsBools]*np.nan
#For arccos in 0-pi
nuReal = np.ones(outReal.shape)*np.nan
nuReal[inBoundsBools] = np.arccos(outReal[inBoundsBools]) #calculate arccos, there are 2 potential solutions... need to calculate both
gPhi = (1.+np.sin(np.tile(inc,(8,1)).T)*np.sin(nuReal+np.tile(w,(8,1)).T))**2./4 #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
gd = np.tile(a.to('AU'),(8,1)).T*(1.-np.tile(e,(8,1)).T**2.)/(np.tile(e,(8,1)).T*np.cos(nuReal)+1.)
gdmags = deltaMag(np.tile(p,(8,1)).T,np.tile(Rp.to('AU'),(8,1)).T,gd,gPhi) #calculate dmag of the specified x-value
#For arccos in pi-2pi 
nuReal2 = np.ones(outReal.shape)*np.nan
nuReal2[inBoundsBools] = 2.*np.pi - np.arccos(outReal[inBoundsBools])
gPhi2 = (1.+np.sin(np.tile(inc,(8,1)).T)*np.sin(nuReal2+np.tile(w,(8,1)).T))**2./4 #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
gd2 = np.tile(a.to('AU'),(8,1)).T*(1.-np.tile(e,(8,1)).T**2.)/(np.tile(e,(8,1)).T*np.cos(nuReal2)+1.)
gdmags2 = deltaMag(np.tile(p,(8,1)).T,np.tile(Rp.to('AU'),(8,1)).T,gd2,gPhi2) #calculate dmag of the specified x-value

#Evaluate which solutions are good and which aren't
correctValBoolean1 = np.abs(gdmags - dmag) < 1e-2 #Values of nuReal which yield the desired dmag
correctValBoolean2 = np.abs(gdmags2 - dmag) < 1e-2 #values of nuReal2 which yield the desired dmag
bothBools = correctValBoolean1*correctValBoolean2 #values of nuReal 
np.abs(gdmags[bothBools] - gdmags2[bothBools])
print(np.sum(bothBools))
#histogram check for number of solutions. We should see either 0, 2, or 4
vals1Hist = np.histogram(np.sum(correctValBoolean1,axis=1),bins=[-0.1,0.9,1.9,2.9,3.9,4.9,5.9])
vals2Hist = np.histogram(np.sum(correctValBoolean2,axis=1),bins=[-0.1,0.9,1.9,2.9,3.9,4.9,5.9])
#Take nuReal1, and nuReal2 where not in both Bools

#Combine the two sets of solutions
nusCombined = np.zeros(nuReal.shape)
nusCombined = nuReal*np.logical_xor(correctValBoolean1,bothBools) + nuReal2*np.logical_xor(correctValBoolean2,bothBools) + nuReal*bothBools #these are the nus where intersections occur

#Combine and verify the two sets of dmags resulting from the solutions
gdmagsCombined = np.zeros(gdmags.shape)
#gdmagsCombined = gdmags*correctValBoolean1 + gdmags2*correctValBoolean2#np.logical_xor(correctValBoolean2,bothBools)
gdmagsCombined = gdmags*np.logical_xor(correctValBoolean1,bothBools) + gdmags2*np.logical_xor(correctValBoolean2,bothBools) + gdmags*bothBools

sumNumSol = np.sum(np.logical_xor(correctValBoolean1,bothBools)) + np.sum(np.logical_xor(correctValBoolean2,bothBools)) + np.sum(bothBools)
numSols = np.sum(np.logical_xor(correctValBoolean1,bothBools) + np.logical_xor(correctValBoolean2,bothBools) + bothBools,axis=1)
numSolHist = np.histogram(numSols,bins=[-0.1,0.9,1.9,2.9,3.9,4.9])
np.sum(np.histogram(numSols,bins=[-0.1,0.9,1.9,2.9,3.9,4.9])[0])
tstop_cos = time.time()

plt.figure(num=9000)
#plt.hist(gdmags.flatten(),alpha=0.3,color='blue',bins=50)
#plt.hist(gdmags2.flatten(),alpha=0.3,color='cyan',bins=50)
plt.hist(gdmagsCombined.flatten(),alpha=0.3,color='red',bins=50)
plt.yscale('log')
plt.show(block=False)
####################################################################


####Timing test
ttstart = time.time()
np.roots(coeffs[:,i])
ttstop = time.time()


####################################################################
# #Using the derivative dFullEqnX from AnalyticalNuFromDmag3.ipynb
# #This solve for where the derivatives are 0 (a max, min, local max, local min)
# A2 = e**4.*np.sin(inc)**4.*np.sin(w)**4./2. + e**4.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2. + e**4.*np.sin(inc)**4.*np.cos(w)**4./2.
# B2 = 7.*e**4.*np.sin(inc)**3.*np.sin(w)**3/4. + 7.*e**4.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./4. + 7.*e**3.*np.sin(inc)**4.*np.sin(w)**4./4. + 7.*e**3.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. + 7.*e**3.*np.sin(inc)**4.*np.cos(w)**4./4.
# C2 = -3.*e**4.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./4. - 3.*e**4.*np.sin(inc)**4.*np.cos(w)**4./4. + 9*e**4.*np.sin(inc)**2.*np.sin(w)**2./4. + 3.*e**4.*np.sin(inc)**2.*np.cos(w)**2./4. + 6*e**3.*np.sin(inc)**3.*np.sin(w)**3 + 6*e**3.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + 9*e**2.*np.sin(inc)**4.*np.sin(w)**4./4. + 9*e**2.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. + 9*e**2.*np.sin(inc)**4.*np.cos(w)**4./4.
# D2 = -5.*e**4.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./4. + 5.*e**4.*np.sin(inc)*np.sin(w)/4. - 5.*e**3.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. - 5.*e**3.*np.sin(inc)**4.*np.cos(w)**4./2. + 15.*e**3.*np.sin(inc)**2.*np.sin(w)**2./2. + 5.*e**3.*np.sin(inc)**2.*np.cos(w)**2./2. + 15.*e**2.*np.sin(inc)**3.*np.sin(w)**3/2. + 15.*e**2.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./2. + 5.*e*np.sin(inc)**4.*np.sin(w)**4./4. + 5.*e*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. + 5.*e*np.sin(inc)**4.*np.cos(w)**4./4.
# E2 = e**4.*np.sin(inc)**4.*np.cos(w)**4./4. - e**4.*np.sin(inc)**2.*np.cos(w)**2./2. + e**4./4. - 4*e**3.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + 4*e**3.*np.sin(inc)*np.sin(w) - 2*e**2.*lhs*np.sin(inc)**2.*np.sin(w)**2. + 2*e**2.*lhs*np.sin(inc)**2.*np.cos(w)**2. - 3.*e**2.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2. - 3.*e**2.*np.sin(inc)**4.*np.cos(w)**4. + 9*e**2.*np.sin(inc)**2.*np.sin(w)**2. + 3.*e**2.*np.sin(inc)**2.*np.cos(w)**2. + 4*e*np.sin(inc)**3.*np.sin(w)**3 + 4*e*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + np.sin(inc)**4.*np.sin(w)**4./4. + np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. + np.sin(inc)**4.*np.cos(w)**4./4.
# F2 = 3.*e**3.*np.sin(inc)**4.*np.cos(w)**4./4. - 3.*e**3.*np.sin(inc)**2.*np.cos(w)**2./2. + 3.*e**3/4. - 3.*e**2.*lhs*np.sin(inc)*np.sin(w) - 9*e**2.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./2. + 9*e**2.*np.sin(inc)*np.sin(w)/2. - 3.*e*lhs*np.sin(inc)**2.*np.sin(w)**2. + 3.*e*lhs*np.sin(inc)**2.*np.cos(w)**2. - 3.*e*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. - 3.*e*np.sin(inc)**4.*np.cos(w)**4./2. + 9*e*np.sin(inc)**2.*np.sin(w)**2./2. + 3.*e*np.sin(inc)**2.*np.cos(w)**2./2. + 3.*np.sin(inc)**3.*np.sin(w)**3/4. + 3.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./4.
# G2 = -e**2.*lhs*np.sin(inc)**2.*np.cos(w)**2. - e**2.*lhs + 3.*e**2.*np.sin(inc)**4.*np.cos(w)**4./4. - 3.*e**2.*np.sin(inc)**2.*np.cos(w)**2./2. + 3.*e**2./4. - 4*e*lhs*np.sin(inc)*np.sin(w) - 2*e*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + 2*e*np.sin(inc)*np.sin(w) - lhs*np.sin(inc)**2.*np.sin(w)**2. + lhs*np.sin(inc)**2.*np.cos(w)**2. - np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./4. - np.sin(inc)**4.*np.cos(w)**4./4. + 3.*np.sin(inc)**2.*np.sin(w)**2./4. + np.sin(inc)**2.*np.cos(w)**2./4.
# H2 = -e*lhs*np.sin(inc)**2.*np.cos(w)**2. - e*lhs + e*np.sin(inc)**4.*np.cos(w)**4./4. - e*np.sin(inc)**2.*np.cos(w)**2./2. + e/4. - lhs*np.sin(inc)*np.sin(w) - np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./4. + np.sin(inc)*np.sin(w)/4.

# coeffs2 = np.asarray([A2,B2,C2,D2,E2,F2,G2,H2])

# out2 = list()
# for i in np.arange(coeffs2.shape[1]):
#     out2.append(np.roots(coeffs2[:,i])) # this is x)
# out2 = np.asarray(out2)


# #get real solutions within viable range
# inBoundsBools = (np.abs(out2.imag) <= 1e-7)*(out2.real >= -1.)*(out2.real <= 1.) #the out2 solutions that are inside of the desired bounds
# outBoundsBools = np.logical_not(inBoundsBools) # the out2 solutions that are inside the desired bounds
# out2Real = out2
# out2Real[outBoundsBools] = out2[outBoundsBools]*np.nan
# nu2Real = np.ones(out2Real.shape)*np.nan
# nu2Real[inBoundsBools] = np.arccos(out2Real[inBoundsBools]) #calculate arccos, there are 2 potential solutions... need to calculate both
# tPhi = (1.+np.sin(np.tile(inc,(7,1)).T)*np.sin(nu2Real+np.tile(w,(7,1)).T))**2./4 #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
# td = np.tile(a.to('AU'),(7,1)).T*(1.-np.tile(e,(7,1)).T**2.)/(np.tile(e,(7,1)).T*np.cos(nu2Real)+1.)
# tdmags = deltaMag(np.tile(p,(7,1)).T,np.tile(Rp.to('AU'),(7,1)).T,td,tPhi) #calculate dmag of the specified x-value

# #DELETEout2RealInds = np.asarray([np.where((out2[i].imag <= 1e-7)*(out2[i].real >= -1.)*(out2[i].real <= 1.))[0] for i in np.arange(out2.shape[0])]) #Find the terms with 0 imaginary parts within viable range
# #DELETEout2Real = np.asarray([out2[i,out2RealInds[i]].real for i in np.arange(out2.shape[0])]) #Take only the real parts of those terms
# #out2RealInRange = np.asarray([np.where((out2Real[i] >= -1)*(out2Real[i] <= 1))[0] for i in np.arange(out2.shape[0])]) #find inds of terms within viable range
# #nuout2 = np.arccos(tmp)
# #inherently, 0 and pi are always 0 to the above expression

# #np.sum(out2imag == 0,axis=1)
############################################################################






#### Lets do a simple model fit of 
def sinBhaskara_0_pi(x):
    """From https://en.wikipedia.org/wiki/Bhaskara_I%27s_sine_approximation_formula
    With Equation Written in AnalyticalNuFromDmag3.ipynb
    Valid in Range 0-pi
    Args:
        x (numpy array):
            input angle in radians
    Returns:
        (numpy array):
            sinusoid function output
    """
    #return (16.*x*np.pi-16.*x**2.)/(5.*np.pi**2. -4.*np.pi*x+4.*x**2.)
    return (16.*np.pi*x-16.*x**2.)/(5.*np.pi**2.-4.*np.pi*x+4.*x**2.)

def sinBhaskara_pi_2pi(x):
    """From https://en.wikipedia.org/wiki/Bhaskara_I%27s_sine_approximation_formula
    With Equation Written in AnalyticalNuFromDmag3.ipynb
    Valid in Range pi-2pi
    Args:
        x (numpy array):
            input angle in radians
    Returns:
        (numpy array):
            sinusoid function output
    """
    return (32.*np.pi**2. - 48.*np.pi*x + 16.*x**2.)/(13.*np.pi**2. - 12.*np.pi*x + 4.*x**2.)

# Fitting a polynomial to ##C6 was removed for consistency with ipynb
trigFuncs = [lambda x: np.sin(x)**2*np.cos(x)**2, #c0
lambda x: np.sin(x)*np.cos(x)**3, #c1
lambda x: np.cos(x)**4, #c2
lambda x: np.sin(x)*np.cos(x)**2, #c3
lambda x: np.cos(x)**3, #c4
lambda x: np.sin(x)**2*np.cos(x), #c5
lambda x: np.sin(x)*np.cos(x), #c7
lambda x: np.sin(x)**2, #c8
lambda x: np.cos(x)**2, #C9
lambda x: np.sin(x), #c10
lambda x: np.cos(x)] #c11

diffTrigFuncs = [lambda x: -2.*np.sin(x)**3.*np.cos(x) + 2.*np.sin(x)*np.cos(x)**3., #0 #OK
lambda x: -3.*np.sin(x)**2.*np.cos(x)**2. + np.cos(x)**4., #1
lambda x: -4.*np.sin(x)*np.cos(x)**3., #2
lambda x: -2.*np.sin(x)**2.*np.cos(x) + np.cos(x)**3., #3
lambda x: -3.*np.sin(x)*np.cos(x)**2., #4
lambda x: -np.sin(x)**3. + 2.*np.sin(x)*np.cos(x)**2., #5 #OK
lambda x: -np.sin(x)**2 + np.cos(x)**2, #7 #OK
lambda x: 2.*np.sin(x)*np.cos(x), #8 OK 
lambda x: -2.*np.sin(x)*np.cos(x), #9 #OK 
lambda x: np.cos(x), #10 #OK
lambda x: -np.sin(x)] #11 #OK

#### Nice nu ranges for each function KEEP until done
# xtmp = np.linspace(start=0, stop=2.*np.pi, num=200)
# ytmps = list()
# dytmps = list()
# for i in np.arange(len(trigFuncs)):
#     ytmp = trigFuncs[i](xtmp)
#     ytmps.append(ytmp)
#     plt.figure(num=i)
#     plt.subplot(211)
#     plt.plot(xtmp,ytmp,color='blue')
#     plt.xlim([0.,2.*np.pi])
#     plt.subplot(212)
#     dytmp = diffTrigFuncs[i](xtmp)
#     dytmps.append(dytmp)
#     plt.plot(xtmp,dytmp,color='blue')
#     plt.plot([0.,2.*np.pi],[0.,0.],color='black')
#     plt.xlim([0.,2.*np.pi])
#     plt.show(block=False)


#The zeros to the diffTrigFuncs expressions
zeros = [[0., -3.*np.pi/4., -np.pi/2., -np.pi/4., np.pi/4., np.pi/2., 3.*np.pi/4.],
 [-5.*np.pi/6., -np.pi/2., -np.pi/6., np.pi/6., np.pi/2., 5.*np.pi/6.],
 [0., np.pi/2., np.pi, 3.*np.pi/2.],
 [-np.pi/2.,  np.pi/2.,  -2.*np.arctan(np.sqrt(5. - 2.*np.sqrt(6.))),  2.*np.arctan(np.sqrt(5. - 2.*np.sqrt(6.))),  -2.*np.arctan(np.sqrt(2.*np.sqrt(6.) + 5.)),  2.*np.arctan(np.sqrt(2.*np.sqrt(6.) + 5.))],
 [0., np.pi/2., np.pi, 3.*np.pi/2.],
 [0.,  -2.*np.arctan(np.sqrt(2. - np.sqrt(3.))),  2.*np.arctan(np.sqrt(2. - np.sqrt(3.))),  -2.*np.arctan(np.sqrt(np.sqrt(3.) + 2.)),  2.*np.arctan(np.sqrt(np.sqrt(3.) + 2.))],
 [-np.pi/2.,  np.pi/2.,  -2.*np.arctan(np.sqrt(5. - 2.*np.sqrt(6.))),  2.*np.arctan(np.sqrt(5. - 2.*np.sqrt(6.))),  -2.*np.arctan(np.sqrt(2.*np.sqrt(6.) + 5.)),  2.*np.arctan(np.sqrt(2.*np.sqrt(6.) + 5.))],
 [-np.pi/4., np.pi/4.],
 [0., np.pi/2., np.pi, 3.*np.pi/2.],
 [0., np.pi/2., np.pi, 3.*np.pi/2.],
 [np.pi/2., 3.*np.pi/2.],
 [0., np.pi]]
zeros = [np.concatenate((np.asarray(zeros[i]),np.asarray(zeros[i])+np.pi)).tolist() for i in np.arange(len(zeros))]
start_stop_points = zeros

#Here we produce flattenedZeros2 which contains a set of unique angles in radians where the trigFuncs have min/max/saddle points
flattenedZeros = list()
for i in np.arange(len(zeros)):
    flattenedZeros = flattenedZeros + zeros[i]
flattenedZeros = np.asarray(flattenedZeros)
flattenedZeros[flattenedZeros < 0] = flattenedZeros[flattenedZeros < 0] + 2.*np.pi
flattenedZeros = np.unique(flattenedZeros)
flattenedZeros2 = list()
for i in np.arange(len(flattenedZeros)-1):
    if np.abs(flattenedZeros[i]-flattenedZeros[i+1]) > 1e-10:
        flattenedZeros2.append(flattenedZeros[i])
flattenedZeros2 = np.asarray(flattenedZeros2) #works okay. A few bits with bad gaps
tmp = flattenedZeros2[:-1][np.diff(flattenedZeros2)/2. > 0.1] + (np.diff(flattenedZeros2)/2.)[np.diff(flattenedZeros2)/2. > 0.1]
flattenedZeros3 = np.sort(np.concatenate((flattenedZeros2,tmp)))
####


def polyFunc(x,nus):
    return x[0]*nus**4.+x[1]*nus**3.+x[2]*nus**2.+x[3]*nus+ x[4]

def diffPolyFunc(x,nus):
    return 4.*x[0]*nus**3.+3.*x[1]*nus**2.+2.*x[2]*nus+x[3]    

def polyCoeffsError(x,trigFunc,nus):
    polyFuncVals = polyFunc(x,nus)
    trigFuncVals = trigFunc(nus)
    return np.sum((polyFuncVals-trigFuncVals)**2.)

# #### Calculate InterpolationGrid
# #Get initial memory usage
cachedir = './'
dfilename = 'dmagTrigParams' + '.pkl'
path = os.path.join(cachedir, dfilename)
#OLDOLD start_stop_points = np.pi/8.*np.arange(int((2.*np.pi)/(np.pi/8.))) #older set of points to fit
#OLDstart_stop_points = np.pi/180.*360./15.*np.arange(int(360./15.))
start_stop_points = flattenedZeros3
nuss = np.asarray([np.linspace(start=start_stop_points[i],stop=start_stop_points[i+1],num=100,endpoint=True) for i in np.arange(len(start_stop_points)-1)])

# if the 2D completeness update array exists as a .dcomp file load it
if os.path.exists(path):
    print('Loading cached analytical dmag trig params from "%s".' % path)
    try:
        with open(path, "rb") as ff:
            outDict = pickle.load(ff)
    except UnicodeDecodeError:
        with open(path, "rb") as ff:
            outDict = pickle.load(ff,encoding='latin1')
    print(' analytical dmag trig params loaded from cache.')
else:
    from scipy.optimize import minimize
    x0 = np.asarray([0.,0.,0.,0.,0.])
    outDict = dict()
    outDict['opt'] = dict()
    outDict['nus'] = dict()
    outDict['coeffs'] = dict()
    outDict['trigFunVals'] = dict()
    outDict['polyFunVals'] = dict()
    outDict['error'] = dict()
    for i,j in itertools.product(np.arange(len(trigFuncs)),np.arange(nuss.shape[0])):
        #for i in np.arange(len(trigFuncs)):
        #    for j in np.arange(len(start_stop_points[i])-1):
        #DELETEnus = np.linspace(start=start_stop_points[i][j],stop=start_stop_points[i][j+1],num=100,endpoint=True)
        outDict['opt'][(i,j)] = minimize(polyCoeffsError, x0,args=(trigFuncs[i],nuss[j]),tol=1e-8,constraints=[{'type':'eq','fun':lambda x: trigFuncs[i](nuss[j][0])-polyFunc(x,nuss[j][0])},
            {'type':'eq','fun':lambda x: trigFuncs[i](nuss[j][-1])-polyFunc(x,nuss[j][-1])},{'type':'eq','fun':lambda x: diffTrigFuncs[i](nuss[j][0])-diffPolyFunc(x,nuss[j][0])},{'type':'eq','fun':lambda x: diffTrigFuncs[i](nuss[j][-1])-diffPolyFunc(x,nuss[j][-1])}])
        outDict['nus'][(i,j)] = nuss[j]
        outDict['coeffs'][(i,j)] = outDict['opt'][(i,j)].x
        outDict['trigFunVals'][(i,j)] = trigFuncs[i](outDict['nus'][(i,j)])
        outDict['polyFunVals'][(i,j)] = polyFunc(outDict['coeffs'][(i,j)],outDict['nus'][(i,j)])
        outDict['error'][(i,j)] = outDict['opt'][(i,j)].fun
        if outDict['error'][(i,j)] > 1e-4:
            plt.figure()
            plt.plot(outDict['nus'][(i,j)],outDict['trigFunVals'][(i,j)],color='blue')
            plt.plot(outDict['nus'][(i,j)],outDict['polyFunVals'][(i,j)],color='red')
            plt.title('i: ' + str(i) + ' j: ' + str(j) + ' error: ' + str(outDict['error'][(i,j)]))
            plt.show(block=False)
        print('i: ' + str(i) + ' j: ' + str(j) + ' error: ' + str(outDict['opt'][(i,j)].fun))

    # store dynamic completeness array as .dcomp file
    with open(path, 'wb') as ff:
        pickle.dump(outDict, ff)



#Next, I need to correlate the coefficients to c10l0 from coefficients
#input i, w, e properly

tstart_quart = time.time()
quartList = list()
for i in np.arange(len(nuss)):
    start0 = time.time()
    quartDict = dict()
    (c_0l0, c_0l1, c_0l2, c_0l3, c_0l4) = outDict['coeffs'][(0,i)]
    (c_1l0, c_1l1, c_1l2, c_1l3, c_1l4) = outDict['coeffs'][(1,i)]
    (c_2l0, c_2l1, c_2l2, c_2l3, c_2l4) = outDict['coeffs'][(2,i)]
    (c_3l0, c_3l1, c_3l2, c_3l3, c_3l4) = outDict['coeffs'][(3,i)]
    (c_4l0, c_4l1, c_4l2, c_4l3, c_4l4) = outDict['coeffs'][(4,i)]
    (c_5l0, c_5l1, c_5l2, c_5l3, c_5l4) = outDict['coeffs'][(5,i)]
    (c_6l0, c_6l1, c_6l2, c_6l3, c_6l4) = outDict['coeffs'][(6,i)]
    (c_7l0, c_7l1, c_7l2, c_7l3, c_7l4) = outDict['coeffs'][(7,i)]
    (c_8l0, c_8l1, c_8l2, c_8l3, c_8l4) = outDict['coeffs'][(8,i)]
    (c_9l0, c_9l1, c_9l2, c_9l3, c_9l4) = outDict['coeffs'][(9,i)]
    (c_10l0, c_10l1, c_10l2, c_10l3, c_10l4) = outDict['coeffs'][(10,i)]
    #Quartic Expression Coefficients from AnalyticalNuFromDmag3.ipynb
    E = 0.25*c_0l0*e**2*np.sin(inc)**2*np.cos(w)**2 + 0.5*c_10l0*e + 0.5*c_10l0*np.sin(inc)*np.sin(w) + 0.5*c_1l0*e**2*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.25*c_2l0*e**2*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_3l0*e**2*np.sin(inc)*np.cos(w) + c_3l0*e*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.5*c_4l0*e**2*np.sin(inc)*np.sin(w) + 0.5*c_4l0*e*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_5l0*e*np.sin(inc)**2*np.cos(w)**2 + c_6l0*e*np.sin(inc)*np.cos(w) + 0.5*c_6l0*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.25*c_7l0*np.sin(inc)**2*np.cos(w)**2 + 0.25*c_8l0*e**2 + c_8l0*e*np.sin(inc)*np.sin(w) + 0.25*c_8l0*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_9l0*np.sin(inc)*np.cos(w) #x**4 coeff that is used to depreciate into the others
    A = (0.25*c_0l1*e**2*np.sin(inc)**2*np.cos(w)**2 + 0.5*c_10l1*e + 0.5*c_10l1*np.sin(inc)*np.sin(w) + 0.5*c_1l1*e**2*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.25*c_2l1*e**2*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_3l1*e**2*np.sin(inc)*np.cos(w) + c_3l1*e*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.5*c_4l1*e**2*np.sin(inc)*np.sin(w) + 0.5*c_4l1*e*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_5l1*e*np.sin(inc)**2*np.cos(w)**2 + c_6l1*e*np.sin(inc)*np.cos(w) + 0.5*c_6l1*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.25*c_7l1*np.sin(inc)**2*np.cos(w)**2 + 0.25*c_8l1*e**2 + c_8l1*e*np.sin(inc)*np.sin(w) + 0.25*c_8l1*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_9l1*np.sin(inc)*np.cos(w))/E
    B = (0.25*c_0l2*e**2*np.sin(inc)**2*np.cos(w)**2 + 0.5*c_10l2*e + 0.5*c_10l2*np.sin(inc)*np.sin(w) + 0.5*c_1l2*e**2*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.25*c_2l2*e**2*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_3l2*e**2*np.sin(inc)*np.cos(w) + c_3l2*e*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.5*c_4l2*e**2*np.sin(inc)*np.sin(w) + 0.5*c_4l2*e*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_5l2*e*np.sin(inc)**2*np.cos(w)**2 + c_6l2*e*np.sin(inc)*np.cos(w) + 0.5*c_6l2*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.25*c_7l2*np.sin(inc)**2*np.cos(w)**2 + 0.25*c_8l2*e**2 + c_8l2*e*np.sin(inc)*np.sin(w) + 0.25*c_8l2*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_9l2*np.sin(inc)*np.cos(w))/E
    C = (0.25*c_0l3*e**2*np.sin(inc)**2*np.cos(w)**2 + 0.5*c_10l3*e + 0.5*c_10l3*np.sin(inc)*np.sin(w) + 0.5*c_1l3*e**2*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.25*c_2l3*e**2*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_3l3*e**2*np.sin(inc)*np.cos(w) + c_3l3*e*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.5*c_4l3*e**2*np.sin(inc)*np.sin(w) + 0.5*c_4l3*e*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_5l3*e*np.sin(inc)**2*np.cos(w)**2 + c_6l3*e*np.sin(inc)*np.cos(w) + 0.5*c_6l3*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.25*c_7l3*np.sin(inc)**2*np.cos(w)**2 + 0.25*c_8l3*e**2 + c_8l3*e*np.sin(inc)*np.sin(w) + 0.25*c_8l3*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_9l3*np.sin(inc)*np.cos(w))/E
    D = (0.25*c_0l4*e**2*np.sin(inc)**2*np.cos(w)**2 + 0.5*c_10l4*e + 0.5*c_10l4*np.sin(inc)*np.sin(w) + 0.5*c_1l4*e**2*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.25*c_2l4*e**2*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_3l4*e**2*np.sin(inc)*np.cos(w) + c_3l4*e*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.5*c_4l4*e**2*np.sin(inc)*np.sin(w) + 0.5*c_4l4*e*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_5l4*e*np.sin(inc)**2*np.cos(w)**2 + c_6l4*e*np.sin(inc)*np.cos(w) + 0.5*c_6l4*np.sin(inc)**2*np.sin(w)*np.cos(w) + 0.25*c_7l4*np.sin(inc)**2*np.cos(w)**2 + 0.25*c_8l4*e**2 + c_8l4*e*np.sin(inc)*np.sin(w) + 0.25*c_8l4*np.sin(inc)**2*np.sin(w)**2 + 0.5*c_9l4*np.sin(inc)*np.cos(w) + 0.25 - lhs)/E
    quartDict['A'] = A
    quartDict['B'] = B
    quartDict['C'] = C
    quartDict['D'] = D
    stop0 = time.time()

    #Solve Quartics
    startquart = time.time()
    xreal, delta, P, D2, R, delta_0 = quarticSolutions_ellipse_to_Quarticipynb(A.astype('complex128'), B, C, D)
    stopquart = time.time()

    start1 = time.time()
    #Create Boolean Array of solutions that are viable
    xrealIsViable = np.logical_and(np.logical_and((np.abs(xreal.imag) < 2.*1e-2),(xreal.real >= nuss[i][0])),(xreal.real <= nuss[i][-1])) #creates 2d array of booleans indicating if imaginary component is small and sol is within viable range
    xrealNotViable = np.logical_not(xrealIsViable)#np.logical_or(np.logical_or((np.abs(xreal.imag) > 1e-10),(np.abs(xreal.real) < nuss[i][0])), (np.abs(xreal.real) > nuss[i][-1])) #creates 2d array of booleans indicating if imaginary component is large or sol is within viable range
    xhasViable = np.any(xrealIsViable,axis=1) #checks if planet has a 'real' solution
    xhasrealInds = np.where(xhasViable == True)[0] #find planets that have at least 1 real solution
    #DELETE xreal2 = xreal + np.nan*xrealIsViable #replace all bad solutions with nan unnecessary???
    stop1 = time.time()

    start2 = time.time()
    xsols = xreal[xrealIsViable] #the viable solutions
    xInds = np.tile(np.arange(len(e)),(4,1)).T[xrealIsViable] #the index of each planet that has solutions
    quartDict['xsols'] = xsols #the true anomaly of each intersection
    quartDict['xInds'] = xInds #the planet ind of each xsols

    quartDict['xsols2'] = xreal #Turn all non-viable solutions into nan
    quartDict['xsols2'][xrealNotViable] = xreal[xrealNotViable]*np.nan #Turn all non-viable solutions into nan
    #quartDict['beta'] = np.arccos(np.sin(np.tile(inc,(4,1)).T)*np.sin(quartDict['xsols2']+np.tile(w,(4,1)).T)) #calculate phase angle
    #quartDict['Phi'] = np.cos(quartDict['beta']/2.)**4. #calculate phase function value
    quartDict['Phi'] = (1.+np.sin(np.tile(inc,(4,1)).T)*np.sin(quartDict['xsols2']+np.tile(w,(4,1)).T))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
    quartDict['d'] = np.tile(a.to('AU'),(4,1)).T*(1.-np.tile(e,(4,1)).T**2.)/(np.tile(e,(4,1)).T*np.cos(quartDict['xsols2'])+1.)
    quartDict['dmags'] = deltaMag(np.tile(p,(4,1)).T,np.tile(Rp.to('AU'),(4,1)).T,quartDict['d'],quartDict['Phi']).real #calculate dmag of the specified x-value

    quartDict['dmagErrors'] = np.abs(quartDict['dmags']-dmag) #calculate error from desired value
    stop2 = time.time()

    # #Plot the resulting dmags
    # plt.figure(num=i)
    # #plt.hist(quartDict['dmagErrors'].flatten(),color='black')
    # plt.hist(quartDict['dmags'].flatten(),color='black',bins=200)
    # plt.yscale('log')
    # plt.title(str(i))
    # plt.show(block=False)

    #Keep for Nostalgia
    quartDict['xreal'] = xreal #These should be raw quartic solutions
    #DELETEquartDict['xhasrealInds']= xhasrealInds #indicies of planets that have a viable solution in this range

    #Remove Solutions that do not produce the desired dmag (they are obviously wrong)
    quartDict['xrealFiltered'] = np.ones(xreal.shape)*np.nan
    quartDict['quartDictBoolean'] = quartDict['dmagErrors'] < 5.*1e-2
    quartDict['xrealFiltered'][quartDict['quartDictBoolean']] = quartDict['xsols2'][quartDict['quartDictBoolean']].real

    quartDict['numSolsPerPlanet'] = np.sum(quartDict['quartDictBoolean'],axis=1) #calculates the number of intersection solutions each planet has

    #Reducing 4 solutions down into 

    #save to quartList
    quartList.append(quartDict)

    #DELETEprint(saltyburrito)
tstop_quart = time.time()

#TOTAL numSols from cos
print('Total Num Sols Comparison')
print(np.sum(numSols))

#Combine the viable xreal solutions
totalNumSols = 0
for i in np.arange(len(nuss)):
    totalNumSols = totalNumSols + np.sum(quartList[i]['quartDictBoolean'])
print(totalNumSols)

#### Need to compare solutions between methods.
#Are all planets with 0 solutions
print('Num Sols Per Planet Comparison')
numSolsPerPlanetCosine = np.sum(~np.isnan(nusCombined),axis=1)
print(numSolsPerPlanetCosine)
numSolsPerPlanetApprox = np.zeros(len(quartList[0]['numSolsPerPlanet']),dtype=int)
for i in np.arange(len(nuss)):
    numSolsPerPlanetApprox = numSolsPerPlanetApprox + quartList[i]['numSolsPerPlanet']
print(numSolsPerPlanetApprox)

#Look at planets that have no solutions
print('Num Planets with 0 Solutions')
print(np.sum(numSolsPerPlanetCosine == 0))
print(np.sum(numSolsPerPlanetApprox == 0))
print(np.sum((numSolsPerPlanetCosine == 0)*(numSolsPerPlanetApprox == 0)))
print('Num Planets with 1 Solutions')
print(np.sum(numSolsPerPlanetCosine == 1))
print(np.sum(numSolsPerPlanetApprox == 1))
print(np.sum((numSolsPerPlanetCosine == 1)*(numSolsPerPlanetApprox == 1)))
print('Num Planets with 2 Solutions')
print(np.sum(numSolsPerPlanetCosine == 2))
print(np.sum(numSolsPerPlanetApprox == 2))
print(np.sum((numSolsPerPlanetCosine == 2)*(numSolsPerPlanetApprox == 2)))
print('Num Planets with 3 Solutions')
print(np.sum(numSolsPerPlanetCosine == 3))
print(np.sum(numSolsPerPlanetApprox == 3))
print(np.sum((numSolsPerPlanetCosine == 3)*(numSolsPerPlanetApprox == 3)))
print('Num Planets with 4 Solutions')
print(np.sum(numSolsPerPlanetCosine == 4))
print(np.sum(numSolsPerPlanetApprox == 4))
print(np.sum((numSolsPerPlanetCosine == 4)*(numSolsPerPlanetApprox == 4)))
print('Num Planets with 5 Solutions')
print(np.sum(numSolsPerPlanetCosine == 5))
print(np.sum(numSolsPerPlanetApprox == 5))
print(np.sum((numSolsPerPlanetCosine == 5)*(numSolsPerPlanetApprox == 5)))

#Iterate over each nu range and see how many (if any) solutions exists
#for each nu solution create array
#create associated array with nu-ranges
alldmags = np.array([])
for i in np.arange(len(quartList)):
    alldmags = np.concatenate((alldmags,quartList[i]['dmags'].flatten()))
plt.figure(num=10000)
plt.hist(alldmags,color='blue',bins=200)
plt.yscale('log')
plt.title(str(i))
plt.show(block=False)

#Timing
print('Timing')
print(tstop_cos-tstart_cos)
print(tstop_quart-tstart_quart)

#other timing
print('other Timing')
print(ttstop-ttstart) #timing for a single np roots call
print(stop0-start0) #timing for dict generation and coeff extraction
print(stopquart-startquart) #timing for the whole quartic process
print(stop1-start1) #timing for some np.where bs
print(stop2-start2) #timing for the dmag calcs

