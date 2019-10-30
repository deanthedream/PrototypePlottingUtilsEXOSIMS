# Calculate Indentical dMag of planets

import matplotlib.pyplot as plt
import numpy as np
#DELETE import matplotlib.dates as mdates
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import numpy as np
from EXOSIMS.util.deltaMag import *
#DELETimport EXOSIMS.PlanetPhysicalModel as PPM#calc_Phi
from scipy.optimize import fsolve
import astropy.units as u

folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_40419/'))#/WFIRSTCompSpecPriors_WFIRSTcycle6core'))#HabExCompSpecPriors_HabEx_4m_TSDD_pop100DD_revisit_20180424'))#prefDistOBdursweep_WFIRSTcycle6core'))
filename = 'WFIRSTcycle6core_CKL2_PPKL2.json'#'HabEx_CKL2_PPSAG13.json'#'auto_2018_11_03_15_09__prefDistOBdursweep_WFIRSTcycle6core_9.json'#'./TestScripts/02_KnownRV_FAP=1_WFIRSTObs_staticEphem.json'#'Dean17Apr18RS05C01fZ01OB01PP01SU01.json'#'sS_SLSQP.json'#'sS_AYO4.json'#'sS_differentPopJTwin.json'#AYO4.json'
#filename = 'sS_intTime6_KeplerLike2.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile,nopar=True)
#sim.run_sim()

def phiD(dMag, Rp_max, p, s):
    """ Calculates a set of Phi and D combinations to achieve the selected dMag
    Args:
        dMag (float) - planet-star difference in magnitudes
        Rp_max (float) - maximum planet size considered
        p (float) - 
        s (float) - observed planet-star separation
    Returns:
        phi (numpy array) - planet phase function
        d (numpy array) - planet-star distance
    """
    #         o   y
    #         |
    #         |
    # *-------o   x == s
    #Assume circular orbits
    d_min = s
    #d_max = inf
    phi_d2 = 10.**(-dMag/2.5)/p/Rp**2. #equals phi/d**2
    #solve this function for beta
    out = fsolve(phi_d2*s**2.- calc_Phi(beta)/(1.+np.tan(beta)**2.),np.pi/4.)
    print(saltyburrito)
    return phi, d


def calc_Phi(beta):
    """Calculate the phase function. Prototype method uses the Lambert phase 
    function from Sobolev 1975.
    From sim.PlanetPhysicalModel.calc_Phi
    
    Args:
        beta (astropy Quantity array):
            Planet phase angles at which the phase function is to be calculated,
            in units of rad
            
    Returns:
        Phi (ndarray):
            Planet phase function
    
    """
    
    beta = beta.to('rad').value
    Phi = (np.sin(beta) + (np.pi - beta)*np.cos(beta))/np.pi
    
    return Phi

#### Plot Phase function value
plt.close(11112233)
plt.figure(num=11112233)
betas = np.linspace(start=0.,stop=np.pi,num=100,endpoint=True)*u.rad
plt.plot(betas,calc_Phi(betas),color='blue')
plt.show(block=False)

# def calc_beta(Phi):
#     Phi*np.pi = np.sin(beta) + (np.pi - beta)*np.cos(beta)
#     return beta

#### Planet Properties #####################################
R_venus = 6051.8*1000.*u.m #m
a_venus = 108.21*10.**9.*u.m #m
p_venus = 0.689
R_earth = 6371.0*1000.*u.m #m
a_earth = 149.60*10.**9.*u.m #m
p_earth = 0.434
R_mars = 3389.92*1000.*u.m #m
a_mars = 227.92*10.**9.*u.m #m
p_mars = 0.150 
R_jupiter = 69911.*1000.*u.m #m
a_jupiter = 778.57*10.**9.*u.m #m
p_jupiter = 0.538
R_saturn = 58232.*1000.*u.m #m
a_saturn = 1433.53*10.**9.*u.m #m
p_saturn = 0.499

R_s = [R_venus,R_earth,R_mars,R_jupiter,R_saturn]
a_s = [a_venus,a_earth,a_mars,a_jupiter,a_saturn]
p_s = [p_venus,p_earth,p_mars,p_jupiter,p_saturn]

#### Maximum Solar System Planet dMags
dMag_max = [deltaMag(p_s[i],R_s[i],a_s[i],1.) for i in np.arange(len(p_s))]

#### Create list of Solar System planet properties
#deltaMag.phiD(25.,)



