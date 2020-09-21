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
i = inc
dmag = 25.

#This is a solver for nu from dmag assuming a quasi-lambert phase function fullEqnX from AnalyticalNuFromDmag3.ipynb
#Generate terms
lhs = (10.**(-0.4*dmag)*a**2.*(e**2. + 1.)**2./(Rp.to('AU')**2.*p)).decompose().value

#These are the coefficients starting at cos(nu)^8 
A = e**4.*np.sin(i)**4.*np.sin(w)**4./16. + e**4.*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./8. + e**4.*np.sin(i)**4.*np.cos(w)**4./16.
B = e**4.*np.sin(i)**3.*np.sin(w)**3./4. + e**4.*np.sin(i)**3.*np.sin(w)*np.cos(w)**2./4. + e**3.*np.sin(i)**4.*np.sin(w)**4./4. + e**3.*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./2. + e**3.*np.sin(i)**4.*np.cos(w)**4./4.
C = -e**4.*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./8. - e**4.*np.sin(i)**4.*np.cos(w)**4./8. + 3.*e**4.*np.sin(i)**2.*np.sin(w)**2./8. + e**4.*np.sin(i)**2.*np.cos(w)**2./8. + e**3.*np.sin(i)**3.*np.sin(w)**3. + e**3.*np.sin(i)**3.*np.sin(w)*np.cos(w)**2. + 3.*e**2.*np.sin(i)**4.*np.sin(w)**4./8. + 3.*e**2.*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./4. + 3.*e**2.*np.sin(i)**4.*np.cos(w)**4./8.
D = -e**4.*np.sin(i)**3.*np.sin(w)*np.cos(w)**2./4. + e**4.*np.sin(i)*np.sin(w)/4. - e**3.*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./2. - e**3.*np.sin(i)**4.*np.cos(w)**4./2. + 3.*e**3.*np.sin(i)**2.*np.sin(w)**2./2. + e**3.*np.sin(i)**2.*np.cos(w)**2./2. + 3.*e**2.*np.sin(i)**3.*np.sin(w)**3./2. + 3.*e**2.*np.sin(i)**3.*np.sin(w)*np.cos(w)**2./2. + e*np.sin(i)**4.*np.sin(w)**4./4. + e*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./2. + e*np.sin(i)**4.*np.cos(w)**4./4.
E = e**4.*np.sin(i)**4.*np.cos(w)**4./16. - e**4.*np.sin(i)**2.*np.cos(w)**2./8. + e**4./16. - e**3.*np.sin(i)**3.*np.sin(w)*np.cos(w)**2. + e**3.*np.sin(i)*np.sin(w) - e**2.*lhs*np.sin(i)**2.*np.sin(w)**2./2. + e**2.*lhs*np.sin(i)**2.*np.cos(w)**2./2. - 3.*e**2.*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./4. - 3.*e**2.*np.sin(i)**4.*np.cos(w)**4./4. + 9*e**2.*np.sin(i)**2.*np.sin(w)**2./4. + 3.*e**2.*np.sin(i)**2.*np.cos(w)**2./4. + e*np.sin(i)**3.*np.sin(w)**3. + e*np.sin(i)**3.*np.sin(w)*np.cos(w)**2. + np.sin(i)**4.*np.sin(w)**4./16. + np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./8. + np.sin(i)**4.*np.cos(w)**4./16.
F = e**3.*np.sin(i)**4.*np.cos(w)**4./4. - e**3.*np.sin(i)**2.*np.cos(w)**2./2. + e**3./4. - e**2.*lhs*np.sin(i)*np.sin(w) - 3.*e**2.*np.sin(i)**3.*np.sin(w)*np.cos(w)**2./2. + 3.*e**2.*np.sin(i)*np.sin(w)/2. - e*lhs*np.sin(i)**2.*np.sin(w)**2. + e*lhs*np.sin(i)**2.*np.cos(w)**2. - e*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./2. - e*np.sin(i)**4.*np.cos(w)**4./2. + 3.*e*np.sin(i)**2.*np.sin(w)**2./2. + e*np.sin(i)**2.*np.cos(w)**2./2. + np.sin(i)**3.*np.sin(w)**3./4. + np.sin(i)**3.*np.sin(w)*np.cos(w)**2./4.
G = -e**2.*lhs*np.sin(i)**2.*np.cos(w)**2./2. - e**2.*lhs/2. + 3.*e**2.*np.sin(i)**4.*np.cos(w)**4./8. - 3.*e**2.*np.sin(i)**2.*np.cos(w)**2./4. + 3.*e**2./8. - 2*e*lhs*np.sin(i)*np.sin(w) - e*np.sin(i)**3.*np.sin(w)*np.cos(w)**2. + e*np.sin(i)*np.sin(w) - lhs*np.sin(i)**2.*np.sin(w)**2./2. + lhs*np.sin(i)**2.*np.cos(w)**2./2. - np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./8. - np.sin(i)**4.*np.cos(w)**4./8. + 3.*np.sin(i)**2.*np.sin(w)**2./8. + np.sin(i)**2.*np.cos(w)**2./8.
H = -e*lhs*np.sin(i)**2.*np.cos(w)**2. - e*lhs + e*np.sin(i)**4.*np.cos(w)**4./4. - e*np.sin(i)**2.*np.cos(w)**2./2. + e/4. - lhs*np.sin(i)*np.sin(w) - np.sin(i)**3.*np.sin(w)*np.cos(w)**2./4. + np.sin(i)*np.sin(w)/4.
I = lhs**2. - lhs*np.sin(i)**2.*np.cos(w)**2./2. - lhs/2. + np.sin(i)**4.*np.cos(w)**4./16. - np.sin(i)**2.*np.cos(w)**2./8. + 1/16.

coeffs = np.asarray([A,B,C,D,E,F,G,H,I])

out = list()
#nuList = list()
for i in np.arange(coeffs.shape[1]):
    out.append(np.roots(coeffs[:,i])) # this is x)
#nuList.append(np.arccos(out[i]))


#Using the derivative dFullEqnX from AnalyticalNuFromDmag3.ipynb
#This solve for where the derivatives are 0 (a max, min, local max, local min)
A2 = e**4.*np.sin(i)**4.*np.sin(w)**4./2. + e**4.*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2. + e**4.*np.sin(i)**4.*np.cos(w)**4./2.
B2 = 7.*e**4.*np.sin(i)**3.*np.sin(w)**3/4. + 7.*e**4.*np.sin(i)**3.*np.sin(w)*np.cos(w)**2./4. + 7.*e**3.*np.sin(i)**4.*np.sin(w)**4./4. + 7.*e**3.*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./2. + 7.*e**3.*np.sin(i)**4.*np.cos(w)**4./4.
C2 = -3.*e**4.*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./4. - 3.*e**4.*np.sin(i)**4.*np.cos(w)**4./4. + 9*e**4.*np.sin(i)**2.*np.sin(w)**2./4. + 3.*e**4.*np.sin(i)**2.*np.cos(w)**2./4. + 6*e**3.*np.sin(i)**3.*np.sin(w)**3 + 6*e**3.*np.sin(i)**3.*np.sin(w)*np.cos(w)**2. + 9*e**2.*np.sin(i)**4.*np.sin(w)**4./4. + 9*e**2.*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./2. + 9*e**2.*np.sin(i)**4.*np.cos(w)**4./4.
D2 = -5.*e**4.*np.sin(i)**3.*np.sin(w)*np.cos(w)**2./4. + 5.*e**4.*np.sin(i)*np.sin(w)/4. - 5.*e**3.*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./2. - 5.*e**3.*np.sin(i)**4.*np.cos(w)**4./2. + 15.*e**3.*np.sin(i)**2.*np.sin(w)**2./2. + 5.*e**3.*np.sin(i)**2.*np.cos(w)**2./2. + 15.*e**2.*np.sin(i)**3.*np.sin(w)**3/2. + 15.*e**2.*np.sin(i)**3.*np.sin(w)*np.cos(w)**2./2. + 5.*e*np.sin(i)**4.*np.sin(w)**4./4. + 5.*e*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./2. + 5.*e*np.sin(i)**4.*np.cos(w)**4./4.
E2 = e**4.*np.sin(i)**4.*np.cos(w)**4./4. - e**4.*np.sin(i)**2.*np.cos(w)**2./2. + e**4./4. - 4*e**3.*np.sin(i)**3.*np.sin(w)*np.cos(w)**2. + 4*e**3.*np.sin(i)*np.sin(w) - 2*e**2.*lhs*np.sin(i)**2.*np.sin(w)**2. + 2*e**2.*lhs*np.sin(i)**2.*np.cos(w)**2. - 3.*e**2.*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2. - 3.*e**2.*np.sin(i)**4.*np.cos(w)**4. + 9*e**2.*np.sin(i)**2.*np.sin(w)**2. + 3.*e**2.*np.sin(i)**2.*np.cos(w)**2. + 4*e*np.sin(i)**3.*np.sin(w)**3 + 4*e*np.sin(i)**3.*np.sin(w)*np.cos(w)**2. + np.sin(i)**4.*np.sin(w)**4./4. + np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./2. + np.sin(i)**4.*np.cos(w)**4./4.
F2 = 3.*e**3.*np.sin(i)**4.*np.cos(w)**4./4. - 3.*e**3.*np.sin(i)**2.*np.cos(w)**2./2. + 3.*e**3/4. - 3.*e**2.*lhs*np.sin(i)*np.sin(w) - 9*e**2.*np.sin(i)**3.*np.sin(w)*np.cos(w)**2./2. + 9*e**2.*np.sin(i)*np.sin(w)/2. - 3.*e*lhs*np.sin(i)**2.*np.sin(w)**2. + 3.*e*lhs*np.sin(i)**2.*np.cos(w)**2. - 3.*e*np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./2. - 3.*e*np.sin(i)**4.*np.cos(w)**4./2. + 9*e*np.sin(i)**2.*np.sin(w)**2./2. + 3.*e*np.sin(i)**2.*np.cos(w)**2./2. + 3.*np.sin(i)**3.*np.sin(w)**3/4. + 3.*np.sin(i)**3.*np.sin(w)*np.cos(w)**2./4.
G2 = -e**2.*lhs*np.sin(i)**2.*np.cos(w)**2. - e**2.*lhs + 3.*e**2.*np.sin(i)**4.*np.cos(w)**4./4. - 3.*e**2.*np.sin(i)**2.*np.cos(w)**2./2. + 3.*e**2./4. - 4*e*lhs*np.sin(i)*np.sin(w) - 2*e*np.sin(i)**3.*np.sin(w)*np.cos(w)**2. + 2*e*np.sin(i)*np.sin(w) - lhs*np.sin(i)**2.*np.sin(w)**2. + lhs*np.sin(i)**2.*np.cos(w)**2. - np.sin(i)**4.*np.sin(w)**2.*np.cos(w)**2./4. - np.sin(i)**4.*np.cos(w)**4./4. + 3.*np.sin(i)**2.*np.sin(w)**2./4. + np.sin(i)**2.*np.cos(w)**2./4.
H2 = -e*lhs*np.sin(i)**2.*np.cos(w)**2. - e*lhs + e*np.sin(i)**4.*np.cos(w)**4./4. - e*np.sin(i)**2.*np.cos(w)**2./2. + e/4. - lhs*np.sin(i)*np.sin(w) - np.sin(i)**3.*np.sin(w)*np.cos(w)**2./4. + np.sin(i)*np.sin(w)/4.

coeffs2 = np.asarray([A2,B2,C2,D2,E2,F2,G2,H2])

out2 = list()
#nuList2 = list()
for i in np.arange(coeffs2.shape[1]):
    out2.append(np.roots(coeffs2[:,i])) # this is x)
    #nuList2.append(np.arccos(out2[i]))


out2 = np.asarray(out2)


#get real solutions within viable range
out2RealInds = np.asarray([np.where((out2[i].imag <= 10**-7)*(out2[i].real >= -1.)*(out2[i].real <= 1.))[0] for i in np.arange(out2.shape[0])]) #Find the terms with 0 imaginary parts within viable range
out2Real = np.asarray([out2[i,out2RealInds[i]].real for i in np.arange(out2.shape[0])]) #Take only the real parts of those terms
#out2RealInRange = np.asarray([np.where((out2Real[i] >= -1)*(out2Real[i] <= 1))[0] for i in np.arange(out2.shape[0])]) #find inds of terms within viable range
#nuout2 = np.arccos(tmp)
#inherently, 0 and pi are always 0 to the above expression

#np.sum(out2imag == 0,axis=1)



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



# Fitting a polynomial to 

trigFuncs = [lambda x: np.sin(x)**2*np.cos(x)**2,
lambda x: np.sin(x)*np.cos(x)**3,
lambda x: np.cos(x)**4,
lambda x: np.sin(x)*np.cos(x)**2,
lambda x: np.cos(x)**3,
lambda x: np.sin(x)**2*np.cos(x),
lambda x: np.sin(x)*np.cos(x)**2]

diffTrigFuncs = [lambda x: -2.*np.sin(x)**3.*np.cos(x) + 2.*np.sin(x)*np.cos(x)**3.,
lambda x: -3.*np.sin(x)**2.*np.cos(x)**2. + np.cos(x)**4.,
lambda x: -4.*np.sin(x)*np.cos(x)**3.,
lambda x: -2.*np.sin(x)**2.*np.cos(x) + np.cos(x)**3.,
lambda x: -3.*np.sin(x)*np.cos(x)**2.,
lambda x: -np.sin(x)**3. + 2.*np.sin(x)*np.cos(x)**2.,
lambda x: -2.*np.sin(x)**2.*np.cos(x) + np.cos(x)**3.]

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
    start_stop_points = np.pi/8.*np.arange(int((2.*np.pi)/(np.pi/8.)))
    nuss = [np.linspace(start=start_stop_points[i],stop=start_stop_points[i+1],num=50) for i in np.arange(len(start_stop_points)-1)]

    x0 = np.asarray([0.,0.,0.,0.,0.])
    outDict = dict()
    outDict['opt'] = dict()
    outDict['nus'] = dict()
    outDict['coeffs'] = dict()
    outDict['trigFunVals'] = dict()
    outDict['polyFunVals'] = dict()
    outDict['error'] = dict()
    for i,j in itertools.product(np.arange(len(trigFuncs)),np.arange(len(nuss))):
        outDict['opt'][(i,j)] = minimize(polyCoeffsError, x0,args=(trigFuncs[i],nuss[j]),tol=1e-6,constraints=[{'type':'eq','fun':lambda x: trigFuncs[i](nuss[j][0])-polyFunc(x,nuss[j][0])},
            {'type':'eq','fun':lambda x: trigFuncs[i](nuss[j][-1])-polyFunc(x,nuss[j][-1])},{'type':'eq','fun':lambda x: diffTrigFuncs[i](nuss[j][0])-diffPolyFunc(x,nuss[j][0])},{'type':'eq','fun':lambda x: diffTrigFuncs[i](nuss[j][-1])-diffPolyFunc(x,nuss[j][-1])}])
        outDict['nus'][(i,j)] = nuss[j]
        outDict['coeffs'][(i,j)] = outDict['opt'][(i,j)].x
        outDict['trigFunVals'][(i,j)] = trigFuncs[i](outDict['nus'][(i,j)])
        outDict['polyFunVals'][(i,j)] = polyFunc(outDict['coeffs'][(i,j)],outDict['nus'][(i,j)])
        outDict['error'][(i,j)] = outDict['opt'][(i,j)].fun
        # if i*j %2 != 0:
        # plt.figure()
        # plt.plot(outDict['nus'][(i,j)],outDict['trigFunVals'][(i,j)],color='blue')
        # plt.plot(outDict['nus'][(i,j)],outDict['polyFunVals'][(i,j)],color='red')
        # plt.title('i: ' + str(i) + ' j: ' + str(j) + ' error: ' + str(outDict['error'][(i,j)]))
        # plt.show(block=False)
        print('i: ' + str(i) + ' j: ' + str(j) + ' error: ' + str(outDict['opt'][(i,j)].fun))

        # store dynamic completeness array as .dcomp file
        with open(path, 'wb') as ff:
            pickle.dump(outDict, ff)


