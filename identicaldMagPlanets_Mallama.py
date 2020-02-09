# Calculate Indentical dMag of planets

import matplotlib.pyplot as plt
import numpy as np
#DELETE import matplotlib.dates as mdates
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import numpy as np
from EXOSIMS.util.deltaMag import *
#DELETimport EXOSIMS.PlanetPhysicalModel as PPM#calc_Phi
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import astropy.units as u
from scipy.interpolate import interp1d, PchipInterpolator
from matplotlib import colors
import datetime
import re
from scipy.misc import derivative
from time import time

folder = './'
PPoutpath = './'


#### Planet Properties #####################################
planProp = dict() #all in units of meters
planProp['mercury'] = {'R':2439.7*1000.,'a':57.91*10.**9.,'p':0.142}
planProp['venus'] = {'R':6051.8*1000.,'a':108.21*10.**9.,'p':0.689}
planProp['earth'] = {'R':6371.0*1000.,'a':149.60*10.**9.,'p':0.434}
planProp['mars'] = {'R':3389.92*1000.,'a':227.92*10.**9.,'p':0.150}
planProp['jupiter'] = {'R':69911.*1000.,'a':778.57*10.**9.,'p':0.538}
planProp['saturn'] = {'R':58232.*1000.,'a':1433.53*10.**9.,'p':0.499}
planProp['uranus'] = {'R':25362.*1000.,'a':2872.46*10.**9.,'p':0.488}
planProp['neptune'] = {'R':24622.*1000.,'a':4495.*10.**9.,'p':0.442}

def alpha_ap_D(D,a_p): #OK
    """ Given some Angle between r_earth/sun and r_planet/sun
    Args:
        D (float) - angle between r_earth/sun and r_planet/sun from 0 to 180 deg
        a_p (float) - planet star semi-major axis
    Return:
        alpha (float) - phase angle
    """
    a_earth = 1. #AU
    x_p = a_p*np.cos(D)
    y_p = a_p*np.sin(D)
    alpha = np.zeros(len(D))

    #ang2 extends beyone a_earth
    x_p_GT_a_earth_Inds = np.where(x_p > a_earth)[0]
    #if x_p > a_earth:
    ang1 = np.arctan2(x_p[x_p_GT_a_earth_Inds], y_p[x_p_GT_a_earth_Inds])
    ang2 = np.arctan2(np.abs(x_p[x_p_GT_a_earth_Inds]-a_earth),y_p[x_p_GT_a_earth_Inds]) #from a_earth and a_p aligned up to p directly above.
    alpha[x_p_GT_a_earth_Inds] = ang1 - ang2
    #elif x_p < a_earth and x_p > 0.:
    x_p_GTa_earth_GT0_Inds = np.where((x_p < a_earth)*(x_p > 0.))[0]
    ang1 = np.arctan2(x_p[x_p_GTa_earth_GT0_Inds], y_p[x_p_GTa_earth_GT0_Inds])
    ang2 = np.arctan2(a_earth - x_p[x_p_GTa_earth_GT0_Inds],y_p[x_p_GTa_earth_GT0_Inds])
    alpha[x_p_GTa_earth_GT0_Inds] = ang1 + ang2
    #else: #x_p < 0
    x_p_LT0_Inds = np.where(x_p < 0.)[0]
    ang1 = np.arctan2(np.abs(x_p[x_p_LT0_Inds]) + a_earth, y_p[x_p_LT0_Inds])
    ang2 = np.arctan2(np.abs(x_p[x_p_LT0_Inds]), y_p[x_p_LT0_Inds])
    alpha[x_p_LT0_Inds] = ang1 - ang2
    return alpha

def d_ap_D(D,a_p): #OK
    """ Calculate planet-Earth distance d given D
    D is the angle between r_k/star and r_earth/star
    """
    a_earth = 1. #AU
    d = np.sqrt(a_p**2. + a_earth**2. - 2.*a_p*a_earth*np.cos(D))
    return d

def alpha_crit_fromEarth(a_p): #OK
    """ Calculate the maximum phase angle for an object 
    Args:
        a_p (float) - planet semi-major axis in AU
    Results:
        alpha_max (float) - maximum alpha in radians
    """
    a_earth = 1. #in AU
    if a_p > a_earth:
        alpha_max = np.arcsin(a_earth/a_p) #occurs at quadrature
    else: #if a_p < a_earth:
        alpha_max = np.pi #0 deg when opposite side of sta180 deg on same side of star
    return alpha_max

def D_alphacrit(a_p): #OK
    """ This assumes alpha_crit occurs when planet is at quadrature (same x location but different y)
    Returns:
        Dcrit (float) - the D angle where theta crit occurs in radians
    """
    Dcrit = np.pi - np.pi/2. - alpha_crit_fromEarth(a_p)
    return Dcrit

def d_crit_ap(a_p): #OK
    """ We know this because at quadrature, alpha is at its maximum
    Additionally, we assume this maximum occurs when the planets for a right triangle
    """
    a_earth = 1 #AU
    dcrit = np.sqrt(a_p**2. - a_earth**2.)
    return dcrit

Ds = np.linspace(start=0.,stop=np.pi,num=100)
alphas1 = alpha_ap_D(Ds,0.7)
alphas2 = alpha_ap_D(Ds,1.5)
ds1 = d_ap_D(Ds,0.7)
ds2 = d_ap_D(Ds,1.5)
plt.close(99)
plt.figure(num=99)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
plt.plot(1.5*np.cos(Ds),1.5*np.sin(Ds),color='blue')
plt.show(block=False)
plt.close(199)
plt.figure(num=199)
plt.plot(Ds,alphas1,color='red')
plt.plot(Ds,alphas2,color='green')
plt.xlabel('Earth-Sun-Planet Angle, ' + r'$D$' + ' in rad',weight='bold')
plt.ylabel('Sun-Planet-Earth Phase Angle, ' + r'$\alpha$' + ' in rad',weight='bold')
plt.show(block=False)
plt.close(299)
plt.figure(num=299)
plt.plot(Ds,ds1,color='red')
plt.plot(Ds,ds2,color='green')
plt.xlabel('Earth-Sun-Planet Angle, ' + r'$D$' + ' in rad',weight='bold')
plt.ylabel('Earth-Planet distance, ' + r'$d$' + ' in AU',weight='bold')
plt.show(block=False)


def calc_Vmag(solarFlux, bulk_albedo, r_sun_sc, nhat, A_aperture, r_gs_sc, vegaFlux):
    """ The fully parameterized equation for visual magnitude
    Args:
        solarFlux (float) - solar flux in W/m^2
        bulk_albedo (float) - Bulk Albedo of spacecraft surface
        r_sun_sc (numpy array) - 3 component vector of sun from spacecraft
        nhat (numpy array) - 3 component unit vector describing spacecraft surface normal
        A_aperture (float) - Area of telescope aperture in m^2
        r_gs_sc (numpy array) - 3 component vector of ground station w.r.t SC
        vegaFlux (float) - Flux From Vega in W/m^2
    """
    numerator = solarFlux*(1.-bulk_albedo)*\
        (no.dot(r_sun_sc,nhat)/np.linalg.norm(r_sun_sc))*\
        (A_aperture/np.linalg.norm(r_gs_sc)**3.)*\
        (np.dot(r_gs_sc,nhat))
    denominator = 2.*np.pi*vegaFlux
    Vmag = -2.5*np.log10(numerator/denominator)
    return Vmag

# def calc_elongation_alpha_ap(alpha,a_p):
#     """ Calculates elongation given phase angle and planet semi-major axis
#     https://en.wikipedia.org/wiki/Elongation_(astronomy)
#     """
#     return elongation


def d_planet_earth_D(D,a_p):
    """ Assuming circular orbits for the Earth and a general planet p, the earth-planet distances 
        is directly calculable from phase angle
    Args:
        D (float) - angular position of planet around sun relative to Earth position ranging from 0 to pi in deg
        a_p (float) - planet semi-major axis in AU
    Returns:
        d (float) - Earth to planet distance at given phase angle in AU
    """
    a_earth = 1. #in AU
    d = np.sqrt(a_p**2. + a_earth**2. - 2.*a_p*a_earth*np.cos(D))
    return d

def d_planet_earth_alpha(alpha,a_p):
    """ Assuming circular orbits for the Earth and a general planet p, the earth-planet distances 
        is directly calculable from phase angle
    Args:
        alpha (float) - angle formed between sun-planet and planet-Earth vectors ranging from 0 to 180 in deg
        a_p (float) - planet semi-major axis in AU
    Returns:
        d (float) - Earth to planet distance at given phase angle in AU
    """
    a_earth = 1. #in AU

    #For the given a_p, check all alphas are possible (Not above alpha_crit)
    alpha_crit = alpha_crit_fromEarth(a_p)*180./np.pi #in deg
    assert np.all(alpha <= alpha_crit), "an alpha is above the maximum possible alpha"

    #Nominally a_p < a_earth
    elongation1 = np.arcsin(a_p*np.sin(alpha*np.pi/180.)/a_earth) #using law of sines
    D1 = np.pi - alpha*np.pi/180. - elongation1 #all sides add to pi
    d1 = np.sqrt(a_p**2. + a_earth**2. - 2.*a_p*a_earth*np.cos(D1))
    d2 = d1
    if a_p > a_earth:
        #There will be two distances satisfying this solution
        elongation2 = np.pi - np.arcsin(a_p*np.sin(alpha*np.pi/180.)/a_earth) #using law of sines
        D2 = np.pi - alpha*np.pi/180. - elongation2 #all sides add to pi
        d2 = np.sqrt(a_p**2. + a_earth**2. - 2.*a_p*a_earth*np.cos(D2))

    #inds = np.arange(len(alpha))
    #zeroInds = np.where(alpha == 0.)[0]
    #inner[zeroInds] = 0.
    # if inner > 1.: #Angle is actually complement
    #     inner = a_p*np.sin(np.pi - alpha*np.pi/180.)/a_earth
    #assert np.all(a_p*np.sin(alpha*np.pi/180.)/a_earth >= -1.), 'arcsin below range'
    #assert np.all(a_p*np.sin(alpha*np.pi/180.)/a_earth <= 1.), 'arcsin above range'
    #d = np.sqrt(a_p**2. + a_earth**2. - 2.*a_p*a_earth*np.cos( 180.*np.pi/180. - alpha*np.pi/180. - np.arcsin(inner)))
    return d1, d2

def elongation_max(a_p):
    """
    Return:
        elongation (float) - maximum elongation in radians
    """
    a_earth = 1. #AU
    if a_p > a_earth:
        elongation = np.pi
    else:
        elongation = np.arctan2(a_p,a_earth)

    return elongation

def d_D_test():
    return d

def d_alpha_test(alpha,a_p):
    a_earth = 1.
    d = np.sqrt(a_p**2. + a_earth**2. + 2.*a_p*a_earth*(np.cos(alpha)*np.sqrt(1.-a_p**2./a_earth**2.*np.sin(alpha)**2.) - a_p/a_earth*np.sin(alpha)**2.))
    return d

alpha = np.linspace(start=0., stop=alpha_crit_fromEarth(1.5),num=30)
d = d_alpha_test(alpha,1.5)

plt.close(999)
plt.figure(num=999)
plt.plot(1.+d*np.cos(alpha),d*np.sin(alpha))
plt.show(block=False)

#### Functions for calculating dmag given s,ds,a_p,phaseFunc,dmagatsmax
def separation_from_alpha_ap(alpha,a_p):
    s = a_p*np.sin(alpha)
    return s

def ds_by_dalpha(alpha,a_p):
    """calculates ds given alpha
    Args:
        alpha (float) - in radians
        a_p (float) - in AU
    """
    ds_dalpha = a_p*np.cos(alpha)
    return ds_dalpha

def alpha_from_dmagapseparationdmagatsmax(sep,a_p,dmag,dmagatsmax):
    """ Inverts separation_from_alpha_ap to calculate alpha from given observation
    Args:
        sep (float) - observed planet-star separation in AU
        a_p (float) - planet-semi-major axis in AU
        dmag (float) - observed dmag
        dmagatsmax (float) - the dmag at smax
    Returns:
        alpha (float) - phase angle in radians
    """
    if np.all(dmag >= dmagatsmax): # on poor side of phase curve
        #assert np.all(np.abs(sep/a_p) <= 1.), 'Invalid sep or a_p'
        alpha = np.zeros(len(sep)) + np.pi/2.
        inds = np.where(sep < a_p)[0]
        alpha[inds] = np.pi - np.arcsin(sep[inds]/a_p)
    else: #dmag < d,agatsmax
        #assert np.all(np.abs(sep/a_p) <= 1.), 'Invalid sep or a_p'
        alpha = np.zeros(len(sep)) + np.pi/2.
        inds = np.where(sep < a_p)[0]
        alpha[inds] = np.arcsin(sep[inds]/a_p)
    return alpha

def dalpha_given_ds_alpha(alpha,a_p,ds):
    """ Calculates 
    """
    dalpha = ds/(a_p*np.cos(alpha))
    return dalpha

def calc_dPhi(phaseFunc,a_p,s,ds,dmag,dmagatsmax):
    """ Calculated dPhi given an observation and phase function
    """
    alpha = alpha_from_dmagapseparationdmagatsmax(s,a_p,dmag,dmagatsmax)
    dalpha = dalpha_given_ds_alpha(alpha,a_p,ds)
    dPhi = derivative(phaseFunc,x0=alpha,dx=dalpha)
    return dPhi

def calc_ddmag(phaseFunc,a_p,separation,ds,dmag,dmagatsmax):
    """ Calculates ddmag given the above parameters
    """
    alpha = alpha_from_dmagapseparationdmagatsmax(separation,a_p,dmag,dmagatsmax)
    dPhi = calc_dPhi(phaseFunc,a_p,separation,ds,dmag,dmagatsmax)
    ddmag = -2.5*dPhi/(phaseFunc(alpha)*np.log(10))
    return ddmag


#### Flux Radios
def fluxRatio_fromVmag(Vmag):
    """Calculates The Flux Ratio from a given Vmag
    """
    fluxRatio = 10.**(-0.4*Vmag)
    return fluxRatio
def planetFlux_fromFluxRatio(fluxRatio):
    """ Calculated planet Flux from fluxRatio
    """
    vegaFlux = 6.958*10.**(-35.) # $W/m^2
    pFlux = fluxRatio*vegaFlux
    return pFlux

def phi_lambert(alpha):
    """ Lambert phase function as presented in Garrett2016
    Args:
        alpha (float) - phase angle in radians
    Returns:
        Phi (float) - phase function value between 0 and 1
    """
    phi = (np.sin(alpha) + (np.pi-alpha)*np.cos(alpha))/np.pi
    return phi

def transitionStart(x,a,b):
    """ Smoothly transition from one 0 to 1
    Args:
        x (float) - in deg input value in deg
        a (float) - transition midpoint in deg
    """
    s = 0.5+0.5*np.tanh((x-a)/b)
    return s
def transitionEnd(x,a,b):
    """ Smoothly transition from one 1 to 0
    Smaller b is sharper step
    a is midpoint, s(a)=0.5
    Args:
        x (float) - in deg input value in deg
        a (float) - transition midpoint in deg
    """
    s = 0.5-0.5*np.tanh((x-a)/b)
    return s

#### Planet Visual Magnitudes From Mallama 2018 #################################################################################
#Mercury
#r distance of planet from sun
#d distance of planet from Earth
#V = 5.*np.log10(r*d) - 0.613 + 6.3280e-02*alpha - 1.6336e-03*alpha**2. + 3.3644e-05*alpha**3. - 3.4265e-07*alpha**4. + 1.6893e-09*alpha**5. - 3.0334e-12*alpha**6.
#planProp['mercury'] = {'numV':1,'alpha_mins':[0.],'alpha_maxs':[180.],'V':[5.*np.log10(r*d) - 0.613 + 6.3280e-02*alpha - 1.6336e-03*alpha**2. + 3.3644e-05*alpha**3. - 3.4265e-07*alpha**4. + 1.6893e-09*alpha**5. - 3.0334e-12*alpha**6.]}
def V_magMercury(alpha,a_p,d):
    """ Valid from 0 to 180 deg
    """
    V = 5.*np.log10(a_p*d) - 0.613 + 6.3280e-02*alpha - 1.6336e-03*alpha**2. + 3.3644e-05*alpha**3. - 3.4265e-07*alpha**4. + 1.6893e-09*alpha**5. - 3.0334e-12*alpha**6.
    return V
def phase_Mercury(alpha):
    """ Valid from 0 to 180 deg
    """
    phase = 10.**(-0.4*(6.3280e-02*alpha - 1.6336e-03*alpha**2. + 3.3644e-05*alpha**3. - 3.4265e-07*alpha**4. + 1.6893e-09*alpha**5. - 3.0334e-12*alpha**6.))
    return phase
planProp['mercury']['num_Vmag_models'] = 1
planProp['mercury']['earth_Vmag_model'] = 0
planProp['mercury']['Vmag'] = [V_magMercury]
planProp['mercury']['phaseFunc'] = [phase_Mercury]
planProp['mercury']['alphas_min'] = [0.]
planProp['mercury']['alphas_max'] = [180.]
planProp['mercury']['phaseFuncMelded'] = phase_Mercury

#Venus
#0<alpha<163.7
#V = 5.*np.log10(r*d) - 4.384 - 1.044e-03*alpha + 3.687e-04*alpha**2. - 2.814e-06*alpha**3. + 8.938e-09*alpha**4.
#163.7<alpha<179
#V = 5.*np.log10(r*d) + 236.05828 - 2.81914e-00*alpha + 8.39034e-03*alpha**2.
#planProp['venus'] = {'numV':2,'alpha_mins':[0.,163.7],'alpha_maxs':[163.7,179.],'V':[5.*np.log10(r*d) - 4.384 - 1.044e-03*alpha + 3.687e-04*alpha**2. - 2.814e-06*alpha**3. + 8.938e-09*alpha**4.,\
#    5.*np.log10(r*d) + 236.05828 - 2.81914e-00*alpha + 8.39034e-03*alpha**2.]}
def V_magVenus_1(alpha, a_p, d):
    """ Valid from 0 to 163.7 deg
    """
    V = 5.*np.log10(a_p*d) - 4.384 - 1.044e-03*alpha + 3.687e-04*alpha**2. - 2.814e-06*alpha**3. + 8.938e-09*alpha**4.
    return V
def V_magVenus_2(alpha, a_p, d):
    """ Valid from 163.7 to 179 deg
    """
    V = 5.*np.log10(a_p*d) + 236.05828 - 2.81914e-00*alpha + 8.39034e-03*alpha**2.
    return V
def phase_Venus_1(alpha):
    """ Valid from 0 to 163.7 deg
    """
    phase = 10.**(-0.4*(- 1.044e-03*alpha + 3.687e-04*alpha**2. - 2.814e-06*alpha**3. + 8.938e-09*alpha**4.))
    return phase
def phase_Venus_2(alpha):
    """ Valid from 163.7 to 179 deg
    """
    phase = 10.**(-0.4*( - 2.81914e-00*alpha + 8.39034e-03*alpha**2.))
    #1 Scale Properly
    h1 = phase_Venus_1(163.7) - 0. #Total height desired over range
    h2 = 10.**(-0.4*( - 2.81914e-00*163.7 + 8.39034e-03*163.7**2.)) - 10.**(-0.4*( - 2.81914e-00*179. + 8.39034e-03*179.**2.))
    phase = phase * h1/h2 #Scale so height is proper
    #2 Lateral movement to make two functions line up
    difference = phase_Venus_1(163.7) - h1/h2*(10.**(-0.4*( - 2.81914e-00*163.7 + 8.39034e-03*163.7**2.)))
    phase = phase + difference

    # + 
    #-(- 2.81914e-00*163.7 + 8.39034e-03*163.7**2.)
    # - 1.
    return phase
def phase_Venus_melded(alpha):
    phase = transitionEnd(alpha,163.7,5.)*phase_Venus_1(alpha) + \
        transitionStart(alpha,163.7,5.)*transitionEnd(alpha,179.,0.5)*phase_Venus_2(alpha) + \
        transitionStart(alpha,179.,0.5)*phi_lambert(alpha*np.pi/180.)
    return phase
planProp['venus']['num_Vmag_models'] = 2
planProp['venus']['earth_Vmag_model'] = 0
planProp['venus']['Vmag'] = [V_magVenus_1,V_magVenus_2]
planProp['venus']['phaseFunc'] = [phase_Venus_1,phase_Venus_2]
planProp['venus']['alphas_min'] = [0.,163.7]
planProp['venus']['alphas_max'] = [163.7,179.]
planProp['venus']['phaseFuncMelded'] = phase_Venus_melded

#Earth
#V = 5.*np.log10(r*d) - 3.99 - 1.060e-3*alpha + 2.054e-4*alpha**2.
#planProp['earth'] = {'numV':1,'alpha_mins':[0.],'alpha_maxs':[180.],'V':[5.*np.log10(r*d) - 3.99 - 1.060e-3*alpha + 2.054e-4*alpha**2.]}
def V_magEarth(alpha,a_p,d):
    """ Valid from 0 to 180 deg
    """
    V = 5.*np.log10(a_p*d) - 3.99 - 1.060e-3*alpha + 2.054e-4*alpha**2.
    return V
def phase_Earth(alpha):
    """ Valid from 0 to 180 deg
    """
    phase = 10.**(-0.4*(- 1.060e-3*alpha + 2.054e-4*alpha**2.))
    return phase
planProp['earth']['num_Vmag_models'] = 1
planProp['earth']['earth_Vmag_model'] = 0
planProp['earth']['Vmag'] = [V_magEarth]
planProp['earth']['phaseFunc'] = [phase_Earth]
planProp['earth']['alphas_min'] = [0.]
planProp['earth']['alphas_max'] = [180.]
planProp['earth']['phaseFuncMelded'] = phase_Earth

#Mars
#alpha<=50
#V = 5.*np.log10(r*d) - 1.601 + 0.02267*alpha - 0.0001302*alpha**2.+ L(λe) + L(LS)
#alpha > 50 approximated by average dimming magnitudes for mercury and Earth
#V = 5.*np.log10(r*d) - 0.367 - 0.02573*alpha + 0.0003445*alpha**2. + L(λe) + L(Ls)
#Find L
#planProp['mars'] = {'numV':2,'alpha_mins':[0.,50.],'alpha_maxs':[50.,180.],'V':[5.*np.log10(r*d) - 1.601 + 0.02267*alpha - 0.0001302*alpha**2.+ L(λe) + L(LS),\
#    5.*np.log10(r*d) - 0.367 - 0.02573*alpha + 0.0003445*alpha**2. + L(λe) + L(Ls)]}
def V_magMars_1(alpha,a_p,d):
    """ Valid from 0 to 50 deg
    """
    V = 5.*np.log10(a_p*d) - 1.601 + 0.02267*alpha - 0.0001302*alpha**2.+ 0. + 0.#L(λe) + L(LS)
    return V
def V_magMars_2(alpha,a_p,d):
    """ Valid from 50 to 180 deg
    """
    V = 5.*np.log10(a_p*d) - 0.367 - 0.02573*alpha + 0.0003445*alpha**2. + 0. + 0. #L(λe) + L(Ls)
    return V
def phase_Mars_1(alpha):
    """ Valid from 0 to 50 deg
    """
    phase = 10.**(-0.4*(0.02267*alpha - 0.0001302*alpha**2.+ 0. + 0.))#L(λe) + L(LS)
    return phase
def phase_Mars_2(alpha):
    """ Valid from 50 to 180 deg
    """
    phase = phase_Mars_1(50.)/10.**(-0.4*(- 0.02573*50. + 0.0003445*50.**2.)) * 10.**(-0.4*(- 0.02573*alpha + 0.0003445*alpha**2. + 0. + 0.)) #L(λe) + L(Ls)
    return phase
def phase_Mars_melded(alpha):
    phase = transitionEnd(alpha,50.,5.)*phase_Mars_1(alpha) + \
        transitionStart(alpha,50.,5.)*phase_Mars_2(alpha)
    return phase
planProp['mars']['num_Vmag_models'] = 2
planProp['mars']['earth_Vmag_model'] = 0
planProp['mars']['Vmag'] = [V_magMars_1,V_magMars_2]
planProp['mars']['phaseFunc'] = [phase_Mars_1,phase_Mars_2]
planProp['mars']['alphas_min'] = [0.,50.]
planProp['mars']['alphas_max'] = [50.,180.]
planProp['mars']['phaseFuncMelded'] = phase_Mars_melded

#Jupiter
#alpha<12
# V = 5.*np.log10(r*d) - 9.395 - 3.7e-04*alpha + 6.16e-04*alpha**2.
#12<alpha<130
# V = 5.*np.log10(r*d) - 9.428 - 2.5*np.log10(1.0 - 1.507*(alpha/180.)\
#     - 0.363*(alpha/180.)**2. - 0.062*(alpha/180.)**3.\
#     + 2.809*(alpha/180.)**4. - 1.876*(alpha/180.)**5.)
#no data beyond 130 deg
#planProp['jupiter'] = {'numV':2,'alpha_mins':[0.,12.],'alpha_maxs':[12.,130.],'V':[5.*np.log10(r*d) - 9.395 - 3.7e-04*alpha + 6.16e-04*alpha**2.,\
#    5.*np.log10(r*d) - 9.428 - 2.5*np.log10(1.0 - 1.507*(alpha/180.) - 0.363*(alpha/180.)**2. - 0.062*(alpha/180.)**3.+ 2.809*(alpha/180.)**4. - 1.876*(alpha/180.)**5.)]}

def V_magJupiter_1(alpha,a_p,d):
    """ Valid from 0 to 12 deg
    """
    V = 5.*np.log10(a_p*d) - 9.395 - 3.7e-04*alpha + 6.16e-04*alpha**2.
    return V
def V_magJupiter_2(alpha,a_p,d):
    """ Valid from 12 to 130 deg
    """
    V = 5.*np.log10(a_p*d) - 9.428 - 2.5*np.log10(1.0 - 1.507*(alpha/180.) - 0.363*(alpha/180.)**2. - 0.062*(alpha/180.)**3.+ 2.809*(alpha/180.)**4. - 1.876*(alpha/180.)**5.)
    return V
def phase_Jupiter_1(alpha):
    """ Valid from 0 to 12 deg
    """
    phase = 10.**(-0.4*(- 3.7e-04*alpha + 6.16e-04*alpha**2.))
    return phase
def phase_Jupiter_2(alpha):
    """ Valid from 12 to 130 deg
    """
    inds = np.where(alpha > 180.)[0]
    alpha[inds] = [180.]*len(inds)
    assert np.all((1.0 - 1.507*(alpha/180.) - 0.363*(alpha/180.)**2. - 0.062*(alpha/180.)**3.+ 2.809*(alpha/180.)**4. - 1.876*(alpha/180.)**5.) >= 0.), "error in alpha input"
    difference = phase_Jupiter_1(12.) - 10.**(-0.4*(- 2.5*np.log10(1.0 - 1.507*(12./180.) - 0.363*(12./180.)**2. - 0.062*(12./180.)**3.+ 2.809*(12./180.)**4. - 1.876*(12./180.)**5.)))
    phase = difference + 10.**(-0.4*(- 2.5*np.log10(1.0 - 1.507*(alpha/180.) - 0.363*(alpha/180.)**2. - 0.062*(alpha/180.)**3.+ 2.809*(alpha/180.)**4. - 1.876*(alpha/180.)**5.)))
    return phase
def phase_Jupiter_melded(alpha):
    phase = transitionEnd(alpha,12.,5.)*phase_Jupiter_1(alpha) + \
        transitionStart(alpha,12.,5.)*transitionEnd(alpha,130.,5.)*phase_Jupiter_2(alpha) + \
        transitionStart(alpha,130.,5.)*phi_lambert(alpha*np.pi/180.)
    return phase
planProp['jupiter']['num_Vmag_models'] = 2
planProp['jupiter']['earth_Vmag_model'] = 0
planProp['jupiter']['Vmag'] = [V_magJupiter_1,V_magJupiter_2]
planProp['jupiter']['phaseFunc'] = [phase_Jupiter_1,phase_Jupiter_2]
planProp['jupiter']['alphas_min'] = [0.,12.]
planProp['jupiter']['alphas_max'] = [12.,130.]
planProp['jupiter']['phaseFuncMelded'] = phase_Jupiter_melded

#Saturn
#V = 5.*np.log10(r*d) - 8.914 - 1.825*np.sin(beta) + 0.026*alpha \
#    - 0.378*np.sin(beta)*np.exp(-2.25*alpha)

#6<alpha<150 this approximates the globe of saturn only, not the rings
#V = 5.*np.log10(r*d) - 8.94 + 2.446e-4*alpha \
#    + 2.672e-4*alpha**2. - 1.505e-6*alpha**3. + 4.767e-9*alpha**4.

#Not enough data to include saturn's rings for alpha>6.5
def V_magSaturn_1(alpha,a_p,d,beta=0.):
    """ Valid alpha from 0 to 6.5 deg
    Valid beta from 0 to 27
    Globe and Rings
    beta in deg
    """
    V = 5.*np.log10(a_p*d) - 8.914-1.825*np.sin(beta*np.pi/180.) + 0.026*alpha - 0.378*np.sin(beta*np.pi/180.)*np.exp(-2.25*alpha)
    return V
def V_magSaturn_2(alpha,a_p,d):
    """ Valid alpha from 0 to 6.5 deg
    Saturn Globe Only Earth Observations
    """
    V = 5.*np.log10(a_p*d) - 8.95 - 3.7e-04*alpha +6.16e-04*alpha**2.
    return V
def V_magSaturn_3(alpha,a_p,d):
    """ Valid alpha from 6 to 150. deg
    Saturn Globe Only Pioneer Observations
    """
    V = 5.*np.log10(a_p*d) - 8.94 + 2.446e-4*alpha + 2.672e-4*alpha**2. - 1.505e-6*alpha**3. + 4.767e-9*alpha**2.
    return V
def phase_Saturn_1(alpha,beta=0.):
    """ Valid alpha from 0 to 6.5 deg
    Valid beta from 0 to 27
    Globe and Rings
    beta in deg
    """
    phase = 10.**(-0.4*(-1.825*np.sin(beta*np.pi/180.) + 0.026*alpha - 0.378*np.sin(beta*np.pi/180.)*np.exp(-2.25*alpha)))
    return phase
def phase_Saturn_2(alpha):
    """ Valid alpha from 0 to 6.5 deg
    Saturn Globe Only Earth Observations
    """
    phase = 10.**(-0.4*(- 3.7e-04*alpha +6.16e-04*alpha**2.))
    return phase
def phase_Saturn_3(alpha):
    """ Valid alpha from 6 to 150. deg
    Saturn Globe Only Pioneer Observations
    """
    difference = phase_Saturn_2(6.5) - 10.**(-0.4*(2.446e-4*6.5 + 2.672e-4*6.5**2. - 1.505e-6*6.5**3. + 4.767e-9*6.5**2.))
    phase = difference + 10.**(-0.4*(2.446e-4*alpha + 2.672e-4*alpha**2. - 1.505e-6*alpha**3. + 4.767e-9*alpha**2.))
    return phase
def phase_Saturn_melded(alpha):
    phase = transitionEnd(alpha,6.5,5.)*phase_Saturn_2(alpha) + \
                transitionStart(alpha,6.5,5.)*transitionEnd(alpha,150.,5.)*phase_Saturn_3(alpha)  + \
                transitionStart(alpha,150.,5.)*phi_lambert(alpha*np.pi/180.)
    return phase
planProp['saturn']['num_Vmag_models'] = 3
planProp['saturn']['earth_Vmag_model'] = 1
planProp['saturn']['Vmag'] = [V_magSaturn_1,V_magSaturn_2,V_magSaturn_3]
planProp['saturn']['phaseFunc'] = [phase_Saturn_1,phase_Saturn_2,phase_Saturn_3]
planProp['saturn']['alphas_min'] = [0.,0.,6.5]
planProp['saturn']['alphas_max'] = [6.5,6.5,150.]
planProp['saturn']['phaseFuncMelded'] = phase_Saturn_melded

#Uranus
#f = 0.0022927 #flattening of the planet
#phi = #planetocentric latitude
#Phi ranges from -82 to 82
#phi_prime = np.arctan2(np.tan(phi),(1.-f)**2.) #planetographic latitude
#V = 5.*np.log10(r*d) - 7.110 - 8.4E-04*phi + 6.587E-3*alpha + 1.045E-4*alpha**2.
def phiprime_phi(phi):
    """ Valid for phi from -82 to 82 deg
    Returns:
        phiprime (float) - in deg
    """
    f = 0.0022927
    phiprime = np.arctan2(np.tan(phi*np.pi/180.),(1.-f)**2.)*180./np.pi
    return phiprime
def V_magUranus(alpha,a_p,d,phi=-82.):
    """ Valid for alpha 0 to 154 deg
    phi in deg
    """
    V = 5.*np.log10(a_p*d) - 7.110 - 8.4e-04*phiprime_phi(phi) + 6.587e-3*alpha + 1.045e-4*alpha**2.
    return V
def phase_Uranus(alpha,phi=-82.):
    """ Valid for alpha 0 to 154 deg
    phi in deg
    """
    phase = 10.**(-0.4*(- 8.4e-04*phiprime_phi(phi) + 6.587e-3*alpha + 1.045e-4*alpha**2.))
    return phase
def phase_Uranus_melded(alpha):
    phase = transitionEnd(alpha,154.,5.)*phase_Uranus(alpha) + \
        transitionStart(alpha,154.,5.)*phi_lambert(alpha*np.pi/180.)
    return phase
planProp['uranus']['num_Vmag_models'] = 1
planProp['uranus']['earth_Vmag_model'] = 0
planProp['uranus']['Vmag'] = [V_magUranus]
planProp['uranus']['phaseFunc'] = [phase_Uranus]
planProp['uranus']['alphas_min'] = [0.]
planProp['uranus']['alphas_max'] = [154.]
planProp['uranus']['phaseFuncMelded'] = phase_Uranus_melded


#Neptune
def V_magNeptune(alpha,a_p,d):
    """ Valid for alpha 0 to 133.14 deg
    """
    V = 5.*np.log10(a_p*d) - 7.00 + 7.944e-3*alpha + 9.617e-5*alpha**2.
    return V
def phase_Neptune(alpha):
    """ Valid for alpha 0 to 133.14 deg
    """
    phase = 10.**(-0.4*(7.944e-3*alpha + 9.617e-5*alpha**2.))
    return phase
def phase_Neptune_melded(alpha):
    phase = transitionEnd(alpha,133.14,5.)*phase_Neptune(alpha) + \
        transitionStart(alpha,133.14,5.)*phi_lambert(alpha*np.pi/180.)
    return phase
planProp['neptune']['num_Vmag_models'] = 1
planProp['neptune']['earth_Vmag_model'] = 0
planProp['neptune']['Vmag'] = [V_magNeptune]
planProp['neptune']['phaseFunc'] = [phase_Neptune]
planProp['neptune']['alphas_min'] = [0.]
planProp['neptune']['alphas_max'] = [133.14]
planProp['neptune']['phaseFuncMelded'] = phase_Neptune_melded


planets=['mercury','venus','earth','mars','jupiter','saturn','uranus','neptune']
pColors = [colors.to_rgba('grey'),colors.to_rgba('gold'),colors.to_rgba('blue'),colors.to_rgba('red'),\
    colors.to_rgba('orange'),colors.to_rgba('goldenrod'),colors.to_rgba('darkblue'),colors.to_rgba('deepskyblue')]
#### Possible From Earth Alphas Range
for i in np.arange(len(planets)):
    #### From Earth
    planProp[planets[i]]['alpha_max_fromearth'] = alpha_crit_fromEarth(planProp[planets[i]]['a']*u.m.to('AU')) #in rad
    earthViewAlpha = np.min([planProp[planets[i]]['alphas_max'][planProp[planets[i]]['earth_Vmag_model']], planProp[planets[i]]['alpha_max_fromearth']*180./np.pi ])
    #print(earthViewAlpha)
    planProp[planets[i]]['alphas_max_fromearth'] = np.linspace(start=0.,stop=earthViewAlpha,num=100,endpoint=True) #in deg
    d1, d2 = d_planet_earth_alpha(planProp[planets[i]]['alphas_max_fromearth'],planProp[planets[i]]['a']*u.m.to('AU'))
    planProp[planets[i]]['distances_fromearth'] = [d1,d2]
    planProp[planets[i]]['Vmags_fromearth'] = [planProp[planets[i]]['Vmag'][planProp[planets[i]]['earth_Vmag_model']](planProp[planets[i]]['alphas_max_fromearth'],planProp[planets[i]]['a']*u.m.to('AU'),d1),\
                                                planProp[planets[i]]['Vmag'][planProp[planets[i]]['earth_Vmag_model']](planProp[planets[i]]['alphas_max_fromearth'],planProp[planets[i]]['a']*u.m.to('AU'),d2)]
    planProp[planets[i]]['expVmags_fromearth'] = [10.**(planProp[planets[i]]['Vmags_fromearth'][0]),10.**(planProp[planets[i]]['Vmags_fromearth'][1])]
    planProp[planets[i]]['normalizedPhaseFunction_fromearth'] = [planProp[planets[i]]['expVmags_fromearth'][0]/np.max(planProp[planets[i]]['expVmags_fromearth'][0]),\
                                                planProp[planets[i]]['expVmags_fromearth'][0]/np.max(planProp[planets[i]]['expVmags_fromearth'][0])]

    #### Largest Phase
    planProp[planets[i]]['phaseFuncValues'] = planProp[planets[i]]['phaseFunc'][planProp[planets[i]]['earth_Vmag_model']](planProp[planets[i]]['alphas_max_fromearth'])

    #### All alphas
    planProp[planets[i]]['alphas'] = [np.linspace(start=planProp[planets[i]]['alphas_min'][j], stop=planProp[planets[i]]['alphas_max'][j],num=100) for j in np.arange(len(planProp[planets[i]]['alphas_max']))]

    planProp[planets[i]]['planet_name'] = planets[i]
    planProp[planets[i]]['planet_labelcolors'] = pColors[i]
    planProp[planets[i]]['pFluxs_FromEarth'] = [planetFlux_fromFluxRatio(fluxRatio_fromVmag(planProp[planets[i]]['Vmags_fromearth'][0])),\
                                                planetFlux_fromFluxRatio(fluxRatio_fromVmag(planProp[planets[i]]['Vmags_fromearth'][1]))]


#### A plot over the ranges a planet is visible from Earth
plt.close(10)
fig10 = plt.figure(num=10)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
for i in np.arange(len(planets)):
    plt.plot(planProp[planets[i]]['alphas_max_fromearth'],planProp[planets[i]]['Vmags_fromearth'][0],color=planProp[planets[i]]['planet_labelcolors'],label=planProp[planets[i]]['planet_name'])
    plt.plot(planProp[planets[i]]['alphas_max_fromearth'],planProp[planets[i]]['Vmags_fromearth'][1],color=planProp[planets[i]]['planet_labelcolors'],label=planProp[planets[i]]['planet_name'],linestyle='--')
plt.xlim([0.,180.])
plt.ylim([-35.,10.])
plt.ylabel('Visual Apparent Magnitude', weight='bold')
plt.xlabel('Phase Angle in deg', weight='bold')
plt.legend()
plt.show(block=False)

#### Plot Fluxes from Earth
plt.close(11)
fig11 = plt.figure(num=11)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
for i in np.arange(len(planets)):
    plt.semilogy(planProp[planets[i]]['alphas_max_fromearth'],planProp[planets[i]]['pFluxs_FromEarth'][0],color=planProp[planets[i]]['planet_labelcolors'],label=planProp[planets[i]]['planet_name'])
    plt.semilogy(planProp[planets[i]]['alphas_max_fromearth'],planProp[planets[i]]['pFluxs_FromEarth'][1],color=planProp[planets[i]]['planet_labelcolors'],label=planProp[planets[i]]['planet_name'],linestyle='--')
plt.xlim([0.,180.])
# plt.ylim([-35.,10.])
plt.ylabel('Planet Flux', weight='bold')
plt.xlabel('Phase Angle in deg', weight='bold')
plt.legend()
plt.show(block=False)

plt.close(12)
fig12 = plt.figure(num=12)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
for i in np.arange(len(planets)):
    plt.semilogy(planProp[planets[i]]['alphas_max_fromearth'],planProp[planets[i]]['pFluxs_FromEarth'][0]/planProp[planets[i]]['distances_fromearth'][0]**2.,color=planProp[planets[i]]['planet_labelcolors'],label=planProp[planets[i]]['planet_name'])
    plt.semilogy(planProp[planets[i]]['alphas_max_fromearth'],planProp[planets[i]]['pFluxs_FromEarth'][1]/planProp[planets[i]]['distances_fromearth'][1]**2.,color=planProp[planets[i]]['planet_labelcolors'],label=planProp[planets[i]]['planet_name'],linestyle='--')
plt.xlim([0.,180.])
# plt.ylim([-35.,10.])
plt.ylabel('Planet Flux/distance from earth squared', weight='bold')
plt.xlabel('Phase Angle in deg', weight='bold')
plt.legend()
plt.show(block=False)


plt.close(13)
fig13 = plt.figure(num=13)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
for i in np.arange(len(planets)):
    plt.plot(planProp[planets[i]]['alphas_max_fromearth'],planProp[planets[i]]['normalizedPhaseFunction_fromearth'][0],color=planProp[planets[i]]['planet_labelcolors'],label=planProp[planets[i]]['planet_name'])
    #DELETEplt.plot(planProp[planets[i]]['alphas_max_fromearth'],planProp[planets[i]]['normalizedPhaseFunction_fromearth'][1],color=planProp[planets[i]]['planet_labelcolors'],label=planProp[planets[i]]['planet_name'],linestyle='--')
plt.xlim([0.,180.])
# plt.ylim([-35.,10.])
plt.ylabel('Phase Function From Earth?', weight='bold')
plt.xlabel('Phase Angle in deg', weight='bold')
plt.legend()
plt.show(block=False)

#### Phase Function over Range Visible From Earth
plt.close(14)
fig14 = plt.figure(num=14)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
for i in np.arange(len(planets)):
    plt.plot(planProp[planets[i]]['alphas_max_fromearth'],planProp[planets[i]]['phaseFuncValues'],color=planProp[planets[i]]['planet_labelcolors'],label=planProp[planets[i]]['planet_name'])
plt.xlim([0.,180.])
# plt.ylim([-35.,10.])
plt.ylabel('Phase Function Mallama alphas', weight='bold')
plt.xlabel('Phase Angle in deg', weight='bold')
plt.legend()
plt.show(block=False)

#### All Phase Functions
plt.close(15)
fig16 = plt.figure(num=15)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
for i in np.arange(len(planets)):
    for jj in np.arange(len(planProp[planets[i]]['alphas'])):
        plt.plot(planProp[planets[i]]['alphas'][jj],planProp[planets[i]]['phaseFunc'][jj](planProp[planets[i]]['alphas'][jj]),color=planProp[planets[i]]['planet_labelcolors'],label=planProp[planets[i]]['planet_name'])
plt.xlim([0.,180.])
plt.ylim([0.,1.0])
plt.ylabel('Phase Function Mallama alphas ALL', weight='bold')
plt.xlabel('Phase Angle in deg', weight='bold')
plt.legend()
plt.show(block=False)

#### All Phase Functions MELDED
plt.close(17)
fig17 = plt.figure(num=17)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
for i in np.arange(len(planets)):
    plt.plot(np.linspace(start=0.,stop=180.,num=180),planProp[planets[i]]['phaseFuncMelded'](np.linspace(start=0.,stop=180.,num=180)),color=planProp[planets[i]]['planet_labelcolors'],label=planProp[planets[i]]['planet_name'])
plt.xlim([0.,180.])
plt.ylim([0.,1.0])
plt.ylabel('Phase Function Mallama alphas Melded', weight='bold')
plt.xlabel('Phase Angle in deg', weight='bold')
plt.legend()
plt.show(block=False)



#### Calculate dMag vs s plots
uncertainty_dmag = 0.01 #HabEx requirement is 1%
uncertainty_s = 5.*u.mas.to('rad')*10.*u.pc.to('AU')
alphas = np.linspace(start=0.,stop=180.,num=1200,endpoint=True)
plt.close(66)
fig66 = plt.figure(num=66)
for i in np.arange(len(planets)):
    planProp[planets[i]]['dmag'] = deltaMag(planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m, planProp[planets[i]]['a']*u.m,\
            planProp[planets[i]]['phaseFuncMelded'](alphas))
    planProp[planets[i]]['s'] = separation_from_alpha_ap(alphas*np.pi/180.,planProp[planets[i]]['a']*u.m).to('AU').value

    tmpColor = list(planProp[planets[i]]['planet_labelcolors'])
    tmpColor[3] = 0.3
    tmpColor = tuple(tmpColor)
    #Find Index where smax occurs
    smaxInd = np.argmax(planProp[planets[i]]['s'])
    dmag_at_smax = planProp[planets[i]]['dmag'][smaxInd] #dmag value at smax
    indsLTdmag_at_smax = np.arange(smaxInd+1) #all inds where s is less than smax from 0 to dmag at smax
    indsGTdmag_at_smax = np.arange(len(planProp[planets[i]]['s']) - (smaxInd+1)) + (len(planProp[planets[i]]['s']) - (smaxInd+1)) #all inds where s is less than smax from dmag at smax to the max dmag

    #Below dmag at smax Lower
    tmps1 = planProp[planets[i]]['s'][indsLTdmag_at_smax]+uncertainty_s
    tmpdmag1 = deltaMag(planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m, planProp[planets[i]]['a']*u.m,\
            planProp[planets[i]]['phaseFuncMelded'](180./np.pi*alpha_from_dmagapseparationdmagatsmax(tmps1,planProp[planets[i]]['a']*u.m.to('AU'),[0.9*dmag_at_smax]*len(tmps1),dmag_at_smax)))
    dmagReplacementInds1 = np.where(tmps1>planProp[planets[i]]['s'][smaxInd])[0]
    tmpdmag1[dmagReplacementInds1] = [dmag_at_smax]*len(dmagReplacementInds1)
    plt.fill_between(planProp[planets[i]]['s'][indsLTdmag_at_smax]+uncertainty_s,(1.-uncertainty_dmag)*planProp[planets[i]]['dmag'][indsLTdmag_at_smax],tmpdmag1,color=tmpColor,edgecolor=tmpColor)#,alpha=0.3,linewidth=0.0)
    
    #Below dmag at smax Upper
    #1. Calculate slope of line
    indsLTdmag_at_smax2 = np.arange(smaxInd+1) #all inds where s is less than smax from 0 to dmag at smax
    ddmag2 = calc_ddmag(planProp[planets[i]]['phaseFuncMelded'],planProp[planets[i]]['a']*u.m.to('AU'),planProp[planets[i]]['s'][indsLTdmag_at_smax2],uncertainty_s,planProp[planets[i]]['dmag'][indsLTdmag_at_smax2],dmag_at_smax)
    tmpAlphas = alpha_from_dmagapseparationdmagatsmax(planProp[planets[i]]['s'][indsLTdmag_at_smax2],planProp[planets[i]]['a']*u.m.to('AU'),planProp[planets[i]]['dmag'][indsLTdmag_at_smax2],dmag_at_smax)
    tmpds = ds_by_dalpha(tmpAlphas,planProp[planets[i]]['a']*u.m.to('AU'))
    tmpThetas = np.arctan2(ddmag2,tmpds)
    #2. find upper limit of the lower portion as x,y coordinates by
    xs = planProp[planets[i]]['s'][indsLTdmag_at_smax2] - uncertainty_s*np.sin(tmpAlphas)
    ys = planProp[planets[i]]['dmag'][indsLTdmag_at_smax2]*(1. + uncertainty_dmag*np.cos(tmpAlphas))
    tmpdmag2 = deltaMag(planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m, planProp[planets[i]]['a']*u.m,\
            planProp[planets[i]]['phaseFuncMelded'](180./np.pi*alpha_from_dmagapseparationdmagatsmax(xs,planProp[planets[i]]['a']*u.m.to('AU'),[0.9*dmag_at_smax]*len(tmps1),dmag_at_smax)))
    plt.fill_between(xs,tmpdmag2,ys,color=tmpColor,edgecolor=tmpColor)#,alpha=0.3,linewidth=0.0)


    #inside Phase Curve bounded by upper and lower
    xs2 = np.linspace(start=np.max(planProp[planets[i]]['s'][indsLTdmag_at_smax2] - uncertainty_s*np.sin(tmpAlphas)), stop=np.max(planProp[planets[i]]['s'][indsLTdmag_at_smax2]))
    ys2_lower = deltaMag(planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m, planProp[planets[i]]['a']*u.m,\
            planProp[planets[i]]['phaseFuncMelded'](180./np.pi*alpha_from_dmagapseparationdmagatsmax(xs2,planProp[planets[i]]['a']*u.m.to('AU'),[0.9*dmag_at_smax]*len(tmps1),dmag_at_smax)))
    ys2_upper = deltaMag(planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m, planProp[planets[i]]['a']*u.m,\
            planProp[planets[i]]['phaseFuncMelded'](180./np.pi*alpha_from_dmagapseparationdmagatsmax(xs2,planProp[planets[i]]['a']*u.m.to('AU'),[1.1*dmag_at_smax]*len(tmps1),dmag_at_smax)))
    plt.fill_between(xs2,ys2_lower,ys2_upper,color=tmpColor,edgecolor=tmpColor)#,alpha=0.3,)#linewidth=0.0)

    #Above dmag at smax lower
    #1. Calculate slope of line
    indsLTdmag_at_smax2 = np.arange(smaxInd+1) #all inds where s is less than smax from 0 to dmag at smax
    ddmag3 = calc_ddmag(planProp[planets[i]]['phaseFuncMelded'],planProp[planets[i]]['a']*u.m.to('AU'),planProp[planets[i]]['s'][indsGTdmag_at_smax],uncertainty_s,planProp[planets[i]]['dmag'][indsGTdmag_at_smax],dmag_at_smax)
    tmpAlphas2 = alpha_from_dmagapseparationdmagatsmax(planProp[planets[i]]['s'][indsGTdmag_at_smax],planProp[planets[i]]['a']*u.m.to('AU'),planProp[planets[i]]['dmag'][indsGTdmag_at_smax],dmag_at_smax)
    tmpds2 = ds_by_dalpha(tmpAlphas2,planProp[planets[i]]['a']*u.m.to('AU'))
    tmpThetas2 = np.arctan2(ddmag3,tmpds2)
    #2. find upper limit of the lower portion as x,y coordinates by
    xs3 = planProp[planets[i]]['s'][indsGTdmag_at_smax] - uncertainty_s*np.sin(tmpAlphas2)
    ys3 = planProp[planets[i]]['dmag'][indsGTdmag_at_smax]*(1. + uncertainty_dmag*np.cos(tmpAlphas2))
    tmpdmag3 = deltaMag(planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m, planProp[planets[i]]['a']*u.m,\
            planProp[planets[i]]['phaseFuncMelded'](180./np.pi*alpha_from_dmagapseparationdmagatsmax(xs3,planProp[planets[i]]['a']*u.m.to('AU'),[1.1*dmag_at_smax]*len(tmps1),dmag_at_smax)))
    plt.fill_between(xs3,ys3,tmpdmag3,color=tmpColor,edgecolor=tmpColor)#,alpha=0.3,linewidth=0.0)

    #Above dmag at smax upper
    indsGTdmag_at_smax2 = np.arange(smaxInd+1) #all inds where s is less than smax from 0 to dmag at smax
    ddmag4 = calc_ddmag(planProp[planets[i]]['phaseFuncMelded'],planProp[planets[i]]['a']*u.m.to('AU'),planProp[planets[i]]['s'][indsGTdmag_at_smax],uncertainty_s,planProp[planets[i]]['dmag'][indsGTdmag_at_smax],dmag_at_smax)
    tmpAlphas4 = alpha_from_dmagapseparationdmagatsmax(planProp[planets[i]]['s'][indsGTdmag_at_smax],planProp[planets[i]]['a']*u.m.to('AU'),planProp[planets[i]]['dmag'][indsGTdmag_at_smax],dmag_at_smax)
    tmpds4 = ds_by_dalpha(tmpAlphas4,planProp[planets[i]]['a']*u.m.to('AU'))
    tmpThetas4 = np.arctan2(ddmag4,tmpds4)
    #2. find upper limit of the lower portion as x,y coordinates by
    xs4 = planProp[planets[i]]['s'][indsGTdmag_at_smax] + uncertainty_s*np.sin(tmpAlphas4)
    ys4 = planProp[planets[i]]['dmag'][indsGTdmag_at_smax]*(1. - uncertainty_dmag*np.cos(tmpAlphas4))
    tmpdmag4 = deltaMag(planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m, planProp[planets[i]]['a']*u.m,\
            planProp[planets[i]]['phaseFuncMelded'](180./np.pi*alpha_from_dmagapseparationdmagatsmax(xs4,planProp[planets[i]]['a']*u.m.to('AU'),[1.1*dmag_at_smax]*len(tmps1),dmag_at_smax)))
    plt.fill_between(xs4,tmpdmag4,ys4,color=tmpColor,edgecolor=tmpColor)#,alpha=0.3,linewidth=0.0)

    #Plot Central Line
    plt.plot(planProp[planets[i]]['s'],planProp[planets[i]]['dmag'],color=planProp[planets[i]]['planet_labelcolors'],label=planProp[planets[i]]['planet_name'].capitalize())


#ADD SMIN FOR TELESCOPE
IWA_HabEx = 0.045*u.arcsec #taken from a Habex Script in units of mas
smin_telescope = IWA_HabEx.to('rad').value*10.*u.pc.to('AU') #IWA for HabEx 45 mas observing target at 10 pc
plt.plot([smin_telescope,smin_telescope],[10.,70.],color='black',linestyle='-')

plt.text(7,19.5,'Credit: Dean Keithly',fontsize='small',fontweight='normal')
plt.text(1.05*smin_telescope,42,'IWA at\n10 pc',fontsize='medium',fontweight='bold',rotation=90)
plt.xlim([1e-1,32.])
plt.ylim([19.,46.])
plt.xscale('log')
plt.ylabel('Planet-Star ' + r'$\Delta \mathrm{mag}$', weight='bold')
plt.xlabel('Projected Planet-Star Separation, ' + r'$s$,' +' in AU', weight='bold')
plt.legend()
plt.show(block=False)
#Save Plots
# Save to a File
date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'dMagvsS_solarSystem' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)




#Saturn geometric functions
"""The light reflected by Saturn and Saturn's rings is a combination of geometric functions.
The central body is a sphere with, under some viewing angles, some portion of the body obstructed by Saturn's rings.
The rings are effectively a flat disk with, under some viewing angles, some portion of the rings obstructed by Saturn.
"""
# rings = dict()
# rings['D'] = {'widths':[66000.*1000.,74000.*1000.],'opticalDepth':10e-3,'albedo':}
# rings['C'] = {'widths':[74490.*1000.,91983.*1000.],'opticalDepth':0.1,'albedo':0.2}
# rings['B'] = {'widths':[91983.*1000.,117516.*1000.],'opticalDepth':2.75,'albedo':0.5}
# rings['CassiniDivision'] = {'widths':[117516.*1000.,122053.*1000.],'opticalDepth':0.15,'albedo':0.2}
# rings['A'] = {'widths':[122053.*1000.,136774.*1000.],'opticalDepth':0.5,'albedo':0.5}
# rings['F'] = {'widths':[140200.*1000.,140250.*1000.],'opticalDepth':0.5,'albedo':}
# rings['G'] = {'widths':[166000.*1000.,173000.*1000.],'opticalDepth':10e-6,'albedo':}
# rings['E'] = {'widths':[180000.*1000.,450000.*1000.],'opticalDepth':10e-5,'albedo':}
#https://www.aanda.org/articles/aa/pdf/2015/11/aa26673-15.pdf
#https://iopscience.iop.org/article/10.1086/426050/pdf




##########################################################################
from eqnsEXOSIMS2020 import *
import sympy as sp
v1 = sp.Symbol('v1', real=True, positive=True)
v2 = sp.Symbol('v2', real=True, positive=True)
i=0
j=1
print('Simplify')
eqnDmag1LHS = eqnDmag.subs(Phi,symbolicPhases[i]).subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(R,planProp[planets[i]]['R']*u.m.to('earthRad')).subs(p,planProp[planets[i]]['p'])#.subs(w,90.).subs(W,0).subs(e,0)
tic = time()
eqnDmag1LHS = sp.simplify(eqnDmag1LHS) 
print(time()-tic)
eqnDmag1RHS = eqnDmag.subs(Phi,symbolicPhases[j]).subs(a,planProp[planets[j]]['a']*u.m.to('AU')).subs(R,planProp[planets[j]]['R']*u.m.to('earthRad')).subs(p,planProp[planets[j]]['p'])#.subs(w,90.).subs(W,0).subs(e,0)
tic = time()
eqnDmag1RHS = sp.simplify(eqnDmag1RHS)
print(time()-tic)
eqnS1LHS = eqnS.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(w,90.).subs(W,0).subs(e,0).subs(v,v1)
eqnS1RHS = eqnS.subs(a,planProp[planets[j]]['a']*u.m.to('AU')).subs(w,90.).subs(W,0).subs(e,0).subs(v,v2)

eqnAlpha1LHS = eqnAlpha1.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(w,90.).subs(W,0).subs(e,0).subs(v,v1)
eqnAlpha1RHS = eqnAlpha1.subs(a,planProp[planets[j]]['a']*u.m.to('AU')).subs(w,90.).subs(W,0).subs(e,0).subs(v,v2)
eqnAlpha2LHS = eqnAlpha2.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(w,90.).subs(W,0).subs(e,0).subs(v,v1)
eqnAlpha2RHS = eqnAlpha2.subs(a,planProp[planets[j]]['a']*u.m.to('AU')).subs(w,90.).subs(W,0).subs(e,0).subs(v,v2)


#### Separation vs a,v,inc
print('separation vs a,v,inc')
s_anuinc = sp.simplify(eqnS.subs(W,0).subs(w,90.).subs(e,0))
# #### Plot s vs nu, inc
# nu_range  = np.linspace(start=0.,stop=360.,num=180)
# inc_range = np.linspace(start=0.,stop=90.,num=90)
# s_anuincVals = np.zeros((len(nu_range),len(inc_range)))
# for i in np.arange(len(nu_range)):
#     for j in np.arange(len(inc_range)):
#         s_anuincVals[i,j] = s_anuinc.subs(a,1.).subs(v,nu_range[i]).subs(inc,inc_range[j])
# plt.figure(num=888777666)
# plt.contourf(nu_range,inc_range,s_anuincVals.T,cmap='bwr',levels=100)
# cbar = plt.colorbar()
# cbar.set_label('Planet-Star Separation')
# plt.xlabel('nu')
# plt.ylabel('inc')
# plt.show(block=False)
# ##########################

#### Minimum Possible Separation of 2 planets
#Minimum and Maximum $s$ of Coincidence
def rangeSep_2planets(a_s,a_l,inc):
    """ Finds the minimum and maximum planet-star separation achievable by both planets
    Args:
        inc (float) - inclination in degrees
    """
    s_min = a_l*np.cos(inc*np.pi/180.) #smallest planet-star separation of the a_larger planet
    s_max = a_s #largest planet-star separation of the a_smaller planet
    return s_min, s_max
############################

#### Phase Angle vs nu, inc
print('Phase Angle vs nu, inc')
alpha_nuinc = sp.simplify(eqnAlpha1.subs(W,0).subs(w,90.).subs(e,0))
# #### Plot s vs nu, inc
# nu_range  = np.linspace(start=0.,stop=360.,num=180)
# inc_range = np.linspace(start=0.,stop=90.,num=90)
# alpha_nuincVals = np.zeros((len(nu_range),len(inc_range)))
# for i in np.arange(len(nu_range)):
#     for j in np.arange(len(inc_range)):
#         #DELETEif nu_range[i] <= 90.:
#         alpha_nuincVals[i,j] = alpha_nuinc.subs(v,nu_range[i]).subs(inc,inc_range[j]).evalf()
#         #DELETE else:
#         #     alpha_nuincVals[i,j] = alpha_nuinc2.subs(v,nu_range[i]).subs(inc,inc_range[j]).evalf()
# plt.figure(num=888777666555)
# plt.contourf(nu_range,inc_range,alpha_nuincVals.T,cmap='bwr',levels=100)
# cbar2 = plt.colorbar()
# cbar2.set_label('Planet Phase Angle')
# plt.xlabel('nu')
# plt.ylabel('inc')
# plt.show(block=False)
# ###########################



#### Calculate max of parameters both planets could share (s,dmag coincidence) ###################################
i=0
j=1
#### a_max
def a_Lmaller_Larger(planProp,i,j):
    a_smaller = planProp[planets[i]]['a']*u.m.to('AU')
    if a_smaller < planProp[planets[j]]['a']*u.m.to('AU'):
        a_larger = planProp[planets[j]]['a']*u.m.to('AU')
        ind_smaller = i #The naturally smaller brightness planet
        ind_larger = j #The naturally larger brightness planet
    else:
        a_larger = a_smaller
        a_smaller = planProp[planets[j]]['a']*u.m.to('AU')
        ind_smaller = j
        ind_larger = i
    return a_smaller, a_larger, ind_smaller, ind_larger
a_smaller, a_larger, ind_smaller, ind_larger = a_Lmaller_Larger(planProp,i,j)
#### Smax
#Find out which SMA is smaller and pick that as smax
def s_smaller_larger_max(planProp,i,j):
    s_smaller = planProp[planets[i]]['a']*u.m.to('AU')
    if s_smaller < planProp[planets[j]]['a']*u.m.to('AU'):
        s_larger = planProp[planets[j]]['a']*u.m.to('AU')
    else:
        s_larger = s_smaller
        s_smaller = planProp[planets[j]]['a']*u.m.to('AU')
    s_max = s_smaller
    return s_smaller, s_larger, s_max
s_smaller, s_larger, s_max = s_smaller_larger_max(planProp,i,j)
#### Maximum alpha ranges for smax (based on smax)
def alpha_MinMaxranges(s_max,a_larger):
    alpha_min_smaller = 0.
    alpha_max_smaller = 180.
    alpha_min_fullphase_larger = 0.
    alpha_max_fullphase_larger = np.arcsin(s_max/a_larger)*180./np.pi
    alpha_min_crescent_larger = np.arcsin(s_max/a_larger)*180./np.pi+90.
    alpha_max_crescent_larger = 180.
    return alpha_min_smaller, alpha_max_smaller, alpha_min_fullphase_larger, alpha_max_fullphase_larger, alpha_min_crescent_larger, alpha_max_crescent_larger
alpha_min_smaller, alpha_max_smaller, alpha_min_fullphase_larger, alpha_max_fullphase_larger, alpha_min_crescent_larger, alpha_max_crescent_larger = alpha_MinMaxranges(s_max,a_larger)
#### dmag range fullphase/crescant phase
dmag_min_smaller = eqnDmag.subs(R,planProp[planets[ind_smaller]]['R']*u.m.to('m')).subs(p,planProp[planets[ind_smaller]]['p']).subs(a,planProp[planets[ind_smaller]]['a']*u.m.to('m')).subs(Phi,symbolicPhases[ind_smaller]).subs(alpha,alpha_min_smaller)
dmag_max_smaller = eqnDmag.subs(R,planProp[planets[ind_smaller]]['R']*u.m.to('m')).subs(p,planProp[planets[ind_smaller]]['p']).subs(a,planProp[planets[ind_smaller]]['a']*u.m.to('m')).subs(Phi,symbolicPhases[ind_smaller]).subs(alpha,alpha_max_smaller)
dmag_min_fullphase_larger = eqnDmag.subs(R,planProp[planets[ind_larger]]['R']*u.m.to('m')).subs(p,planProp[planets[ind_larger]]['p']).subs(a,planProp[planets[ind_larger]]['a']*u.m.to('m')).subs(Phi,symbolicPhases[ind_larger]).subs(alpha,alpha_min_fullphase_larger)
dmag_max_fullphase_larger = eqnDmag.subs(R,planProp[planets[ind_larger]]['R']*u.m.to('m')).subs(p,planProp[planets[ind_larger]]['p']).subs(a,planProp[planets[ind_larger]]['a']*u.m.to('m')).subs(Phi,symbolicPhases[ind_larger]).subs(alpha,alpha_max_fullphase_larger)
dmag_min_crescent_larger = eqnDmag.subs(R,planProp[planets[ind_larger]]['R']*u.m.to('m')).subs(p,planProp[planets[ind_larger]]['p']).subs(a,planProp[planets[ind_larger]]['a']*u.m.to('m')).subs(Phi,symbolicPhases[ind_larger]).subs(alpha,alpha_min_crescent_larger)
dmag_max_crescent_larger = eqnDmag.subs(R,planProp[planets[ind_larger]]['R']*u.m.to('m')).subs(p,planProp[planets[ind_larger]]['p']).subs(a,planProp[planets[ind_larger]]['a']*u.m.to('m')).subs(Phi,symbolicPhases[ind_larger]).subs(alpha,alpha_max_crescent_larger)
#if dmag_max_fullphase_larger < dmag_min_fullphase_smaller:
    #If the larger planet at it's dimmest in the full phase portion is brighter than the maximum brightness of the smaller planet
    #then no intersection could occur on the maximum phase side
#else:
    # check flux ratio between planets in this region of common separations

#### Calculate Nominal Flux Ratios 
#[ps(Rs/as)^2]/[pL(RL/aL)^2]
print('Calculating Nominal Flux Ratios')
alpha_smaller = sp.Symbol('alpha_smaller', real=True, positive=True)
alpha_larger = sp.Symbol('alpha_larger', real=True, positive=True)
fluxRatioPLANET = eqnDmagInside.subs(R,planProp[planets[ind_smaller]]['R']*u.m.to('m')).subs(p,planProp[planets[ind_smaller]]['p']).subs(a,planProp[planets[ind_smaller]]['a']*u.m.to('m')).subs(Phi,1.) / \
                eqnDmagInside.subs(R,planProp[planets[ind_larger]]['R']*u.m.to('m')).subs(p,planProp[planets[ind_larger]]['p']).subs(a,planProp[planets[ind_larger]]['a']*u.m.to('m')).subs(Phi,1.)
fluxRatioPHASE = symbolicPhases[ind_larger].subs(alpha,alpha_larger)/symbolicPhases[ind_smaller].subs(alpha,alpha_smaller)
from scipy.optimize import fsolve
from matplotlib import ticker
def errorFluxRatio(x):
    a_s = x[0]
    a_l = x[1]
    error = fluxRatioPLANET - fluxRatioPHASE.subs(alpha_smaller,a_s).subs(alpha_larger,a_l)
    return [error,error]
def errorFluxRatio2(x):
    a_s = x[0]
    a_l = x[1]
    error = np.abs(fluxRatioPLANET.evalf() - fluxRatioPHASE.subs(alpha_smaller,a_s).subs(alpha_larger,a_l).evalf())
    return error
x0 = np.asarray([45.,135.])
out = fsolve(func=errorFluxRatio,x0=x0)
out2 = minimize(fun=errorFluxRatio2,x0=x0,bounds=[(alpha_min_smaller,alpha_max_smaller),(alpha_min_crescent_larger,alpha_max_crescent_larger)], \
                )
##############

#### Verifying symbolic phase functions
#some functions don't have 0 at 180 deg phase
# print('Verifying Symbolic Phase Functions')
# alpha_range = np.linspace(start=0.,stop=180.,num=180)
# PHPHvals = list()
# for i in np.arange(len(symbolicPhases)):
#     tmp = list()
#     for j in np.arange(len(alpha_range)):
#         tmp.append(symbolicPhases[i].subs(alpha,alpha_range[j]))
#     PHPHvals.append(tmp)
# plt.figure(num=10000123123)
# for i in np.arange(len(symbolicPhases)):
#     plt.plot(alpha_range,PHPHvals[i],label=str(i))
# plt.xlabel('alpha')
# plt.legend()
# plt.show(block=False)
##############################

#### verifying the a solution exists somewhere in the entire alpha range
print('Verifying a Solution exists somewhere in the entire alpha range')
alpha1_range = np.linspace(start=0.,stop=180.,num=180)
alpha2_range = np.linspace(start=alpha_min_fullphase_larger,stop=alpha_max_fullphase_larger,num=90)
alpha3_range = np.linspace(start=alpha_min_crescent_larger,stop=alpha_max_crescent_larger,num=30)
FRgrid = np.zeros((len(alpha1_range),len(alpha2_range)+len(alpha3_range)))
tic = time()
for i in np.arange(len(alpha1_range)):
    for j in np.arange(len(alpha2_range)):
        FRgrid[i,j] = fluxRatioPHASE.subs(alpha_smaller,alpha1_range[i]).subs(alpha_larger,alpha2_range[j])
    for j in np.arange(len(alpha3_range)):
        FRgrid[i,j+len(alpha2_range)-1] = fluxRatioPHASE.subs(alpha_smaller,alpha1_range[i]).subs(alpha_larger,alpha3_range[j])
    print('Verify Soln: ' + str(i))
print(time()-tic)

plt.figure(num=97987987)
tmp = FRgrid.copy()
#tmp[tmp > 20.] = np.nan
plt.contourf(alpha1_range,list(alpha2_range)+list(alpha3_range),tmp.T, locator=ticker.LogLocator(), levels=[10**i for i in np.linspace(-5,5,num=11)])
plt.plot([0.,180.],[alpha_max_fullphase_larger,alpha_max_fullphase_larger],color='black')
plt.plot([0.,180.],[alpha_min_crescent_larger,alpha_min_crescent_larger],color='black')
cbar3 = plt.colorbar()
plt.xlabel('alpha1')
plt.ylabel('alpha2,3')
plt.show(block=False)
plt.figure(num=979879872)
tmp = FRgrid.copy()
tmp[tmp > 1.] = np.nan
plt.contourf(alpha1_range,list(alpha2_range)+list(alpha3_range),tmp.T, levels=100)# locator=ticker.LogLocator(), levels=[10**i for i in np.linspace(-5,5,num=11)])
plt.plot([0.,180.],[alpha_max_fullphase_larger,alpha_max_fullphase_larger],color='black')
plt.plot([0.,180.],[alpha_min_crescent_larger,alpha_min_crescent_larger],color='black')
cbar3 = plt.colorbar()
plt.xlabel('alpha1')
plt.ylabel('alpha2,3')

plt.scatter(out2.x[0],out2.x[1],marker='x', color='k')


plt.show(block=False)

#out = sp.solvers.solve(fluxRatioPLANET - fluxRatioPHASE, alpha_smaller)
####
#### Minimization for alpha1, alpha2, inc
print('Minimizing1')
i_crit = np.arccos(a_smaller/a_larger)*180./np.pi
def funcMaxInc(x):
    tv1 = x[0]
    tv2 = x[1]
    tinc = x[2]
    return -tinc
def con_sep(x):
    tv1 = x[0]
    tv2 = x[1]
    tinc = x[2]
    error = eqnS1LHS.subs(v1,tv1).subs(inc,tinc).evalf() - eqnS1RHS.subs(v2,tv2).subs(inc,tinc).evalf()
    return error
def con_dmag(x):
    tv1 = x[0]
    tv2 = x[1]
    tinc = x[2]
    error = eqnDmag1LHS.subs(alpha,eqnAlpha2LHS).subs(v1,tv1).subs(inc,tinc).evalf() - eqnDmag1RHS.subs(alpha,eqnAlpha2RHS).subs(v2,tv2).subs(inc,tinc).evalf()
    return error
x0 = np.asarray([1.,1.,0.])
out = minimize(funcMaxInc, x0, method='SLSQP', bounds=[(0.,180.),(0.,180.),(0.,i_crit)], constraints=[{'type':'eq','fun':con_sep},{'type':'eq','fun':con_dmag}], options={'disp':True,})#'eps':1.})#constraints=[con1])
####
print('Minimizing2')
i_crit = np.arccos(a_smaller/a_larger)*180./np.pi
def funcMaxInc(x):
    tv1 = x[0]
    tv2 = x[1]
    tinc = x[2]
    return -tinc
def con_sep1(x,sep):
    tv1 = x[0]
    tv2 = x[1]
    tinc = x[2]
    error = eqnS1LHS.subs(v1,tv1).subs(inc,tinc).evalf() - sep#eqnS1RHS.subs(v2,tv2).subs(inc,tinc).evalf()
    return error
def con_sep2(x,sep):
    tv1 = x[0]
    tv2 = x[1]
    tinc = x[2]
    error = sep - eqnS1RHS.subs(v2,tv2).subs(inc,tinc).evalf()
    return error
def con_dmag(x):
    tv1 = x[0]
    tv2 = x[1]
    tinc = x[2]
    error = eqnDmag1LHS.subs(alpha,eqnAlpha2LHS).subs(v1,tv1).subs(inc,tinc).evalf() - eqnDmag1RHS.subs(alpha,eqnAlpha2RHS).subs(v2,tv2).subs(inc,tinc).evalf()
    return error
s_range = np.linspace(start=0.,stop=a_smaller,num=int(np.ceil((a_smaller-0.)/0.1)))#int(np.ceil((a_smaller-0.)*5./180.)))
outList = list()
for i in np.arange(len(s_range)):
    s_i = s_range[i]
    x0 = np.asarray([1.,1.,0.])
    out = minimize(funcMaxInc, x0, method='SLSQP', bounds=[(0.,180.),(0.,180.),(0.,i_crit)], constraints=[{'type':'eq','fun':con_sep1,'args':(s_i,)},{'type':'eq','fun':con_sep2,'args':(s_i,)},{'type':'eq','fun':con_dmag}], options={'disp':True,})#'eps':1.})#constraints=[con1])
    outList.append(out)
####
print('Minimizing3')
i_crit = np.arccos(a_smaller/a_larger)*180./np.pi
def funcMaxInc(x):
    tv1 = x[0]
    tv2 = x[1]
    tinc = x[2]
    return np.abs(eqnDmag1LHS.subs(alpha,eqnAlpha2LHS).subs(v1,tv1).subs(inc,tinc).evalf() - eqnDmag1RHS.subs(alpha,eqnAlpha2RHS).subs(v2,tv2).subs(inc,tinc).evalf())
# s_range = np.linspace(start=0.,stop=a_smaller,num=int(np.ceil((a_smaller-0.)/0.1)))#int(np.ceil((a_smaller-0.)*5./180.)))
# outList = list()
# for i in np.arange(len(s_range)):
#     s_i = s_range[i]
x0 = np.asarray([1.,1.,0.])
out = minimize(funcMaxInc, x0, method='SLSQP', bounds=[(0.,180.),(0.,180.),(0.,i_crit)], constraints=[{'type':'eq','fun':con_sep}], options={'disp':True,})#'eps':1.})#constraints=[con1])
#outList.append(out)
#### SUCCESS
print('Minimizing4')
i_crit = np.arccos(a_smaller/a_larger)*180./np.pi
def funcMaxInc(x):
    talpha1 = x[0]
    talpha2 = x[1]
    return np.abs(eqnDmag1LHS.subs(alpha,talpha1).evalf() - eqnDmag1RHS.subs(alpha,talpha2).evalf())
def con_sepAlpha(x):
    talpha1 = x[0]
    talpha2 = x[1]
    error = eqnSAlpha.subs(a,planProp[planets[ind_smaller]]['a']*u.m.to('AU')).subs(alpha,talpha1).evalf() - eqnSAlpha.subs(a,planProp[planets[ind_larger]]['a']*u.m.to('AU')).subs(alpha,talpha2).evalf()
    return error
# s_range = np.linspace(start=0.,stop=a_smaller,num=int(np.ceil((a_smaller-0.)/0.1)))#int(np.ceil((a_smaller-0.)*5./180.)))
# outList = list()
# for i in np.arange(len(s_range)):
#     s_i = s_range[i]
x0 = np.asarray([1.,1.])
out = minimize(funcMaxInc, x0, method='SLSQP', bounds=[(0.,180.),(alpha_min_crescent_larger,180.)], constraints=[{'type':'eq','fun':con_sepAlpha}], options={'disp':True,})#'eps':1.})#constraints=[con1])
#outList.append(out)
print(eqnSAlpha.subs(a,planProp[planets[ind_smaller]]['a']*u.m.to('AU')).subs(alpha,out.x[0]))
print(eqnSAlpha.subs(a,planProp[planets[ind_larger]]['a']*u.m.to('AU')).subs(alpha,out.x[1]).evalf())
print(eqnDmag1LHS.subs(alpha,out.x[0]).evalf())
print(eqnDmag1RHS.subs(alpha,out.x[1]).evalf())

####SUCCESS!!! We get inclinations of the solved system by imposing them as constraints
print('Minimizing5')
i_crit = np.arccos(a_smaller/a_larger)*180./np.pi
def funcMaxInc(x):
    talpha1 = x[0]
    talpha2 = x[1]
    return np.abs(eqnDmag1LHS.subs(alpha,talpha1).evalf() - eqnDmag1RHS.subs(alpha,talpha2).evalf())
def con_sepAlpha(x):
    talpha1 = x[0]
    talpha2 = x[1]
    error = eqnSAlpha.subs(a,planProp[planets[ind_smaller]]['a']*u.m.to('AU')).subs(alpha,talpha1).evalf() - eqnSAlpha.subs(a,planProp[planets[ind_larger]]['a']*u.m.to('AU')).subs(alpha,talpha2).evalf()
    return error
#CREATE BOUNDS AND USE THOSE BOUNDS TO DEFINE THE RANGE FOR ALPHA1 AND ALPHA2
inc_range = np.linspace(start=0.,stop=90.,num=45)
outList = list()
dmagErrorList = list()
for i in np.arange(len(inc_range)):
    min_alpha = inc_range[i]
    x0 = np.asarray([1.,1.])
    if alpha_min_crescent_larger > 180.-min_alpha:
        continue
    out = minimize(funcMaxInc, x0, method='SLSQP', bounds=[(0.+min_alpha,180.-min_alpha),(alpha_min_crescent_larger,180.-min_alpha)], constraints=[{'type':'eq','fun':con_sepAlpha}], options={'disp':True,})#'eps':1.})#constraints=[con1])
    outList.append(out)
    dmagErrorList.append(np.abs(eqnDmag1RHS.subs(alpha,out.x[1]).evalf() - eqnDmag1LHS.subs(alpha,out.x[0]).evalf()))
successList = [outList[i].success for i in np.arange(len(outList))]
# print(eqnSAlpha.subs(a,planProp[planets[ind_smaller]]['a']*u.m.to('AU')).subs(alpha,out.x[0]))
# print(eqnSAlpha.subs(a,planProp[planets[ind_larger]]['a']*u.m.to('AU')).subs(alpha,out.x[1]).evalf())
# print(eqnDmag1LHS.subs(alpha,out.x[0]).evalf())
# print(eqnDmag1RHS.subs(alpha,out.x[1]).evalf())

#### Finding Minimum Inclination For Overlaps
print('Max Inclination Ranges')
#Creates List of all planet-planet comparisons to do
planInds = np.arange(8)
planIndPairs = list()
tmp = [[planIndPairs.append([i,j]) for j in np.arange(7+1) if not i == j and j > i] for i in planInds]
#Find all inc. alpha1, alpha2 intersections
incDict = {}
for pair_k in np.arange(len(planIndPairs)):
    i = planIndPairs[pair_k][0]
    j = planIndPairs[pair_k][1]
    print("i: " + str(i) + " j: " + str(j))
    a_smaller, a_larger, ind_smaller, ind_larger  = a_Lmaller_Larger(planProp,i,j)
    s_smaller, s_larger, s_max = s_smaller_larger_max(planProp,i,j)
    alpha_min_smaller, alpha_max_smaller, alpha_min_fullphase_larger, alpha_max_fullphase_larger, alpha_min_crescent_larger, alpha_max_crescent_larger = alpha_MinMaxranges(s_max,a_larger)
    i_crit = np.arccos(a_smaller/a_larger)*180./np.pi
    incDict[ind_smaller,ind_larger] = {}
    incDict[ind_smaller,ind_larger]['opt1'] = {}
    incDict[ind_smaller,ind_larger]['opt2'] = {}
    eqnDmagLHS = eqnDmag.subs(Phi,symbolicPhases[ind_smaller]).subs(a,planProp[planets[ind_smaller]]['a']*u.m.to('AU')).subs(R,planProp[planets[ind_smaller]]['R']*u.m.to('earthRad')).subs(p,planProp[planets[ind_smaller]]['p'])
    eqnDmagRHS = eqnDmag.subs(Phi,symbolicPhases[ind_larger]).subs(a,planProp[planets[ind_larger]]['a']*u.m.to('AU')).subs(R,planProp[planets[ind_larger]]['R']*u.m.to('earthRad')).subs(p,planProp[planets[ind_larger]]['p'])
    def funcMaxInc(x):
        talpha1 = x[0]
        talpha2 = x[1]
        return np.abs(eqnDmagLHS.subs(alpha,talpha1).evalf() - eqnDmagRHS.subs(alpha,talpha2).evalf())
    def con_sepAlpha(x):
        talpha1 = x[0]
        talpha2 = x[1]
        error = eqnSAlpha.subs(a,planProp[planets[ind_smaller]]['a']*u.m.to('AU')).subs(alpha,talpha1).evalf() - eqnSAlpha.subs(a,planProp[planets[ind_larger]]['a']*u.m.to('AU')).subs(alpha,talpha2).evalf()
        return error
    #CREATE BOUNDS AND USE THOSE BOUNDS TO DEFINE THE RANGE FOR ALPHA1 AND ALPHA2
    inc_range = np.linspace(start=0.,stop=90.,num=45)
    outList1 = list()
    outList2 = list()
    dmagErrorList = list()
    continueOpt1 = True #Boolean indicating if last opt failed
    continueOpt2 = True #Boolean indicating if last opt failed
    opt1Incs = list()
    opt2Incs = list()
    for i in np.arange(len(inc_range)):
        min_alpha = inc_range[i]
        if alpha_min_crescent_larger > 180.-min_alpha:
            continue
        else:
            if not continueOpt1:
                continue
            #Find Intersection (brighter of smaller, dimmer of larger)
            x0 = np.asarray([(90.+min_alpha)/2.,(alpha_min_crescent_larger+180.-min_alpha)/2.])
            out = minimize(funcMaxInc, x0, method='SLSQP', bounds=[(0.+min_alpha,180.-min_alpha),(alpha_min_crescent_larger,180.-min_alpha)], constraints=[{'type':'eq','fun':con_sepAlpha}], options={'disp':True,})
            outList1.append(out)
            dmagErrorList.append(np.abs(eqnDmag1RHS.subs(alpha,out.x[1]).evalf() - eqnDmag1LHS.subs(alpha,out.x[0]).evalf()))
            if out.success == False:#If we did not successfully converge, do not run this opt again
                continueOpt1 = False
        if alpha_max_fullphase_larger < min_alpha:
            continue
        else:
            if not continueOpt2:
                continue
            #Find Intersection (brighter of smaller, brighter of larger)
            x0 = np.asarray([(90.+min_alpha)/2.,(min_alpha+alpha_max_fullphase_larger)/2.])
            out = minimize(funcMaxInc, x0, method='SLSQP', bounds=[(0.+min_alpha,180.-min_alpha),(0.+min_alpha,alpha_max_fullphase_larger)], constraints=[{'type':'eq','fun':con_sepAlpha}], options={'disp':True,})
            outList2.append(out)
            dmagErrorList.append(np.abs(eqnDmag1RHS.subs(alpha,out.x[1]).evalf() - eqnDmag1LHS.subs(alpha,out.x[0]).evalf()))
            if out.success == False: #If we did not successfully converge, do not run this opt again
                continueOpt1 = False
    incDict[ind_smaller,ind_larger]['opt1']['incs'] = opt1Incs
    incDict[ind_smaller,ind_larger]['opt2']['incs'] = opt2Incs
    incDict[ind_smaller,ind_larger]['opt1']['outList'] = outList1
    incDict[ind_smaller,ind_larger]['opt2']['outList'] = outList2
    successList = [outList[i].success for i in np.arange(len(outList))]
######################################################################





#### Inclination min
#Find the bounding inclinations which cause intersection
i_max = 90.
#i_min = 


#out = sp.solvers.solve((eqnDmag1RHS-eqnDmag1LHS,eqnS1RHS-eqnS1LHS),[v1,v2,inc], force=True, manual=True, set=True)
#out = sp.solvers.solve((eqnDmag1RHS-eqnDmag1LHS,eqnS1RHS-eqnS1LHS),[v1,v2,inc], force=True, manual=True, set=True)


#### Plotting dmag vs S to see if it looks right
vvv = np.linspace(start=0.,stop=180.,num=181)
# sss = [eqnS1LHS.subs(inc,90.).subs(alpha,vvv[ind]).evalf() for ind in np.arange(len(vvv))] 
# mmm = [eqnDmag1LHS.subs(inc,90.).subs(v1,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
# sss1 = [eqnS1LHS.subs(inc,45.).subs(v1,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
# mmm1 = [eqnDmag1LHS.subs(inc,45.).subs(v1,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
# sss2 = [eqnS1LHS.subs(inc,20.).subs(v1,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
# mmm2 = [eqnDmag1LHS.subs(inc,20.).subs(v1,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
# sss3 = [eqnS1LHS.subs(inc,0.).subs(v1,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
# mmm3 = [eqnDmag1LHS.subs(inc,0.).subs(v1,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
# Rsss = [eqnS1RHS.subs(inc,90.).subs(v2,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
# Rmmm = [eqnDmag1RHS.subs(inc,90.).subs(v2,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
# Rsss1 = [eqnS1RHS.subs(inc,45.).subs(v2,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
# Rmmm1 = [eqnDmag1RHS.subs(inc,45.).subs(v2,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
# Rsss2 = [eqnS1RHS.subs(inc,20.).subs(v2,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
# Rmmm2 = [eqnDmag1RHS.subs(inc,20.).subs(v2,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
# Rsss3 = [eqnS1RHS.subs(inc,0.).subs(v2,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
# Rmmm3 = [eqnDmag1RHS.subs(inc,0.).subs(v2,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
sss = [eqnS1LHS.subs(v1,vvv[ind]).subs(inc,90.).evalf() for ind in np.arange(len(vvv))] 
mmm = [eqnDmag1LHS.subs(alpha,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
sss1 = [eqnS1LHS.subs(v1,vvv[ind]).subs(inc,90.).evalf() for ind in np.arange(len(vvv))]
mmm1 = [eqnDmag1LHS.subs(alpha,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
sss2 = [eqnS1LHS.subs(v1,vvv[ind]).subs(inc,90.).evalf() for ind in np.arange(len(vvv))]
mmm2 = [eqnDmag1LHS.subs(alpha,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
sss3 = [eqnS1LHS.subs(v1,vvv[ind]).subs(inc,90.).evalf() for ind in np.arange(len(vvv))]
mmm3 = [eqnDmag1LHS.subs(alpha,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
Rsss = [eqnS1RHS.subs(v2,vvv[ind]).subs(inc,90.).evalf() for ind in np.arange(len(vvv))]
Rmmm = [eqnDmag1RHS.subs(alpha,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
Rsss1 = [eqnS1RHS.subs(v2,vvv[ind]).subs(inc,90.).evalf() for ind in np.arange(len(vvv))]
Rmmm1 = [eqnDmag1RHS.subs(alpha,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
Rsss2 = [eqnS1RHS.subs(v2,vvv[ind]).subs(inc,90.).evalf() for ind in np.arange(len(vvv))]
Rmmm2 = [eqnDmag1RHS.subs(alpha,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
Rsss3 = [eqnS1RHS.subs(v2,vvv[ind]).subs(inc,90.).evalf() for ind in np.arange(len(vvv))]
Rmmm3 = [eqnDmag1RHS.subs(alpha,vvv[ind]).evalf() for ind in np.arange(len(vvv))]
plt.close(10920198230)
plt.figure(num=10920198230)
plt.plot(sss,mmm,color=(1.,0.,1.-1.))
plt.plot(sss1,mmm1,color=(0.5,0.,1.-0.5))
plt.plot(sss2,mmm2,color=(0.2,0.,1.-0.2))
plt.plot(sss3,mmm3,color=(0.,0.,1.-0.))
plt.plot(Rsss,Rmmm,color=(1.,0.5,1.-1.),linestyle='--')
plt.plot(Rsss1,Rmmm1,color=(0.5,0.5,1.-0.5),linestyle='--')
plt.plot(Rsss2,Rmmm2,color=(0.2,0.5,1.-0.2),linestyle='--')
plt.plot(Rsss3,Rmmm3,color=(0.,0.5,1.-0.),linestyle='--')
plt.show(block=False)
####

#v1 and v2 as functions of inclination FROM SEPARATION
incFunc_s = sp.solvers.solve(eqnS1LHS-eqnS1RHS,inc)

#v1 vs v2 function
# tic = time()
# tmp = sp.simplify(eqnDmag1RHS - eqnDmag1LHS)
# toc = time()
# print('simplify time: ' + str(toc-tic))
#dmagEquiv = [tmp.subs(inc,incFunc_s[i]) for i in np.arange(len(incFunc_s))]
#TAKESTOOLONGdmagEquiv = [sp.simplify(dmagEquiv[i]) for i in np.arange(len(incFunc_s))]
#out = sp.solvers.solve(dmagEquiv[0],v1)

#v1 and v2 as functions of inclination from dmag
#incFunc_dmag = sp.solvers.solve(1.-eqnDmag1RHS/eqnDmag1LHS,inc)


print('Minimizing2')
def func(x,inc_val):
    tv1 = x[0]
    tv2 = x[1]
    equivalency = eqnDmag1RHS.subs(inc,inc_val).subs(v1,tv1).subs(v2,tv2) - eqnDmag1LHS.subs(inc,inc_val).subs(v1,tv1).subs(v2,tv2)
    out = equivalency.evalf(10)
    print("out: " + str(np.abs(out)) + "  tv1: " + str(tv1) + "  tv2: " + str(tv2))
    if not out.is_real:#out == complex:
        return 100
    else:
        return np.abs(out)

print(func([45.,55.],90.))

#### Equivalent Separation Constraint##################################################################################
def sepCon(x,inc_val):
    tv1 = x[0]
    tv2 = x[1]
    #error = incFunc_s.subs(v1,tv1).subs(v2,tv2) - inc_val
    error = eqnS1RHS.subs(v1,tv1).subs(v2,tv2).subs(inc,inc_val) - eqnS1LHS.subs(v1,tv1).subs(v2,tv2).subs(inc,inc_val)
    return error.evalf()
#######################################################################################################################

inc_val = 90.
#### Setup Jacobians#######################################################
#objective function jacobian
funcJac = [sp.diff(eqnDmag1RHS-eqnDmag1LHS,v1),sp.diff(eqnDmag1RHS-eqnDmag1LHS,v2)]
#separation constraint jacobian
sepConJac = [sp.diff(eqnS1RHS-eqnS1LHS,v1).subs(inc,inc_val), sp.diff(eqnS1RHS-eqnS1LHS,v2).subs(inc,inc_val)]
###########################################################################
con2 = NonlinearConstraint(lambda y: sepCon(y,inc_val),lb=0.,ub=0., jac=lambda x: sepConJac[0].subs(v1,x))
con1 = LinearConstraint(np.asarray([[1.,0.],[0.,1.]]),np.asarray([0.,0.]),np.asarray([180.,180.]))
x0 = np.asarray([45.,55.])
out = minimize(func, x0, args=(inc_val,), method='SLSQP', bounds=[(0.,180.),(0.,180.)], constraints=[{'type':'eq','fun':sepCon, 'args':(inc_val,)}], options={'disp':True,})#'eps':1.})#constraints=[con1])

print(eqnDmag1RHS.subs(inc,inc_val).subs(v1,out.x[0]).subs(v2,out.x[1]).evalf())
print(eqnDmag1LHS.subs(inc,inc_val).subs(v1,out.x[0]).subs(v2,out.x[1]).evalf())
print(eqnS1RHS.subs(v1,out.x[0]).subs(v2,out.x[1]).subs(inc,inc_val))
print(eqnS1RHS.subs(v1,out.x[0]).subs(v2,out.x[1]).subs(inc,inc_val))


plt.close(299)
plt.close(199)
plt.close(11)
plt.close(12)
plt.close(99)
plt.close(999)


#### Testing eqnLAMBERT
vvv = np.linspace(start=1.,stop=359.,num=180)
out = [eqnLAMBERT.subs(alpha,vvv[i]) for i in np.arange(len(vvv))]
plt.figure(num=9878907896)
plt.plot(vvv,out)
plt.show(block=False)

