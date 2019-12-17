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
        alpha (float) - phase angle
    Returns:
        Phi (float) - phase function value between 0 and 1
    """
    phi = (np.sin(alpha) + (np.pi-alpha)*np.cos(alpha))/np.pi
    return phi

def transitionStart(x,a,b):
    """ Smoothly transition from one 0 to 1
    """
    s = 0.5+0.5*np.tanh((x-a)/b)
    return s
def transitionEnd(x,a,b):
    """ Smoothly transition from one 1 to 0
    Smaller b is sharper step
    a is midpoint, s(a)=0.5
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
planProp['mercury']['num_Vmag_models'] = 1
planProp['mercury']['earth_Vmag_model'] = 0
planProp['mercury']['Vmag'] = [V_magMercury]
planProp['mercury']['alphas_min'] = [0.]
planProp['mercury']['alphas_max'] = [180.]

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
planProp['venus']['num_Vmag_models'] = 2
planProp['venus']['earth_Vmag_model'] = 0
planProp['venus']['Vmag'] = [V_magVenus_1,V_magVenus_2]
planProp['venus']['alphas_min'] = [0.,163.7]
planProp['venus']['alphas_max'] = [163.7,179.]

#Earth
#V = 5.*np.log10(r*d) - 3.99 - 1.060e-3*alpha + 2.054e-4*alpha**2.
#planProp['earth'] = {'numV':1,'alpha_mins':[0.],'alpha_maxs':[180.],'V':[5.*np.log10(r*d) - 3.99 - 1.060e-3*alpha + 2.054e-4*alpha**2.]}
def V_magEarth(alpha,a_p,d):
    """ Valid from 0 to 180 deg
    """
    V = 5.*np.log10(a_p*d) - 3.99 - 1.060e-3*alpha + 2.054e-4*alpha**2.
    return V
planProp['earth']['num_Vmag_models'] = 1
planProp['earth']['earth_Vmag_model'] = 0
planProp['earth']['Vmag'] = [V_magEarth]
planProp['earth']['alphas_min'] = [0.]
planProp['earth']['alphas_max'] = [180.]

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
planProp['mars']['num_Vmag_models'] = 2
planProp['mars']['earth_Vmag_model'] = 0
planProp['mars']['Vmag'] = [V_magMars_1,V_magMars_2]
planProp['mars']['alphas_min'] = [0.,50.]
planProp['mars']['alphas_max'] = [50.,180.]

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
planProp['jupiter']['num_Vmag_models'] = 2
planProp['jupiter']['earth_Vmag_model'] = 0
planProp['jupiter']['Vmag'] = [V_magJupiter_1,V_magJupiter_2]
planProp['jupiter']['alphas_min'] = [0.,12.]
planProp['jupiter']['alphas_max'] = [12.,130.]

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
planProp['saturn']['num_Vmag_models'] = 3
planProp['saturn']['earth_Vmag_model'] = 1
planProp['saturn']['Vmag'] = [V_magSaturn_1,V_magSaturn_2,V_magSaturn_3]
planProp['saturn']['alphas_min'] = [0.,0.,6.5]
planProp['saturn']['alphas_max'] = [6.5,6.5,150.]

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
planProp['uranus']['num_Vmag_models'] = 1
planProp['uranus']['earth_Vmag_model'] = 0
planProp['uranus']['Vmag'] = [V_magUranus]
planProp['uranus']['alphas_min'] = [0.]
planProp['uranus']['alphas_max'] = [154.]


#Neptune
def V_magNeptune(alpha,a_p,d):
    """ Valid for alpha 0 to 133.14 deg
    """
    V = 5.*np.log10(a_p*d) - 7.00 + 7.944e-3*alpha + 9.617e-5*alpha**2.
    return V
planProp['neptune']['num_Vmag_models'] = 1
planProp['neptune']['earth_Vmag_model'] = 0
planProp['neptune']['Vmag'] = [V_magNeptune]
planProp['neptune']['alphas_min'] = [0.]
planProp['neptune']['alphas_max'] = [133.14]


planets=['mercury','venus','earth','mars','jupiter','saturn','uranus','saturn']
pColors = ['grey','gold','blue','red','orange','goldrod','darkblue','turquoise']
#### Possible From Earth Alphas Range
for i in np.arange(len(planets)):
    planProp[planets[i]]['alpha_max_fromearth'] = alpha_crit_fromEarth(planProp[planets[i]]['a']*u.m.to('AU')) #in rad
    earthViewAlpha = np.min([planProp[planets[i]]['alphas_max'][planProp[planets[i]]['earth_Vmag_model']], planProp[planets[i]]['alpha_max_fromearth']*180./np.pi ])
    print(earthViewAlpha)
    planProp[planets[i]]['alphas_max_fromearth'] = np.linspace(start=0.,stop=earthViewAlpha,num=100.,endpoint=True) #in deg
    d1, d2 = d_planet_earth_alpha(planProp[planets[i]]['alphas_max_fromearth'],planProp[planets[i]]['a']*u.m.to('AU'))
    planProp[planets[i]]['Vmags_fromearth'] = [planProp[planets[i]]['Vmag'][planProp[planets[i]]['earth_Vmag_model']](planProp[planets[i]]['alphas_max_fromearth'],planProp['mercury']['a']*u.m.to('AU'),d1),\
                                                planProp[planets[i]]['Vmag'][planProp[planets[i]]['earth_Vmag_model']](planProp[planets[i]]['alphas_max_fromearth'],planProp['mercury']['a']*u.m.to('AU'),d2)]
    
    planProp[planets[i]]['planet_name'] = planets[i]
    planProp[planets[i]]['planet_labelcolors'] = pColors[i]
    pFlux_FromEarth = planetFlux_fromFluxRatio(fluxRatio_fromVmag(planProp[planets[i]]['Vmags_fromearth']))

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


#### Full Range of Vmag equations in Mallama
alphas_mercury = np.linspace(start=0.,stop=180.,num=np.ceil(180./3.),endpoint=True)
alphas_venus1 = np.linspace(start=0.,stop=163.7,num=np.ceil(163.7/3.),endpoint=True)
alphas_venus2 = np.linspace(start=163.7,stop=179.,num=np.ceil((179.-163.7)/3.),endpoint=True)
alphas_earth = np.linspace(start=0.,stop=180.,num=np.ceil(180./3.),endpoint=True)
alphas_mars1 = np.linspace(start=0.,stop=50.,num=np.ceil(50./3.),endpoint=True)
alphas_mars2 = np.linspace(start=50.,stop=180.,num=np.ceil((180.-50.)/3.),endpoint=True)
alphas_jupiter1 = np.linspace(start=0.,stop=12.,num=np.ceil(12./3.),endpoint=True)
alphas_jupiter2 = np.linspace(start=12.,stop=130.,num=np.ceil((130.-12.)/3.),endpoint=True)
alphas_saturn1 = np.linspace(start=0.,stop=6.5,num=np.ceil(6.5/3.),endpoint=True)
betas_saturn1 = np.linspace(start=0.,stop=27.,num=np.ceil(27./3.),endpoint=True)
alphas_saturn2 = np.linspace(start=0.,stop=6.5,num=np.ceil(6.5/3.),endpoint=True)
alphas_saturn3 = np.linspace(start=0.,stop=6.5,num=np.ceil(6.5/3.),endpoint=True)
alphas_saturn4 = np.linspace(start=6.,stop=150.,num=np.ceil((150.-6.5)/3.),endpoint=True)
alphas_uranus = np.linspace(start=0.,stop=154.,num=np.ceil(154./3.),endpoint=True)
phis_uranus = np.linspace(start=-82,stop=82.,num=np.ceil((82+82)/3.),endpoint=True)
alphas_neptune = np.linspace(start=0.,stop=133.14,num=np.ceil(133.14/3.),endpoint=True)
V_magsMercury = V_magMercury(alphas_mercury,planProp['mercury']['a']*u.m.to('AU'))
V_magsVenus_1 = V_magVenus_1(alphas_venus1,planProp['venus']['a']*u.m.to('AU'))
V_magsVenus_2 = V_magVenus_2(alphas_venus2,planProp['venus']['a']*u.m.to('AU'))
V_magsEarth = V_magEarth(alphas_earth,planProp['earth']['a']*u.m.to('AU'))
V_magsMars_1 = V_magMars_1(alphas_mars1,planProp['mars']['a']*u.m.to('AU'))
V_magsMars_2 = V_magMars_2(alphas_mars2,planProp['mars']['a']*u.m.to('AU'))
V_magsJupiter_1 = V_magJupiter_1(alphas_jupiter1,planProp['jupiter']['a']*u.m.to('AU'))
V_magsJupiter_2 = V_magJupiter_2(alphas_jupiter2,planProp['jupiter']['a']*u.m.to('AU'))
V_magsSaturn_1 = V_magSaturn_1(alphas_saturn1,planProp['saturn']['a']*u.m.to('AU'),beta=0.)
V_magsSaturn_2 = V_magSaturn_1(alphas_saturn2,planProp['saturn']['a']*u.m.to('AU'),beta=27.)
V_magsSaturn_3 = V_magSaturn_2(alphas_saturn3,planProp['saturn']['a']*u.m.to('AU'))
V_magsSaturn_4 = V_magSaturn_3(alphas_saturn4,planProp['saturn']['a']*u.m.to('AU'))
V_magsUranus_1 = V_magUranus(alphas_uranus,planProp['uranus']['a']*u.m.to('AU'),phi=-82.)
V_magsUranus_2 = V_magUranus(alphas_uranus,planProp['uranus']['a']*u.m.to('AU'),phi=0.)
V_magsUranus_3 = V_magUranus(alphas_uranus,planProp['uranus']['a']*u.m.to('AU'),phi=82.)
V_magsNeptune = V_magNeptune(alphas_neptune,planProp['neptune']['a']*u.m.to('AU'))

#### Plot Raw plane Visual Apparent Magnitudes ######################################################################
plt.close(1)
fig1 = plt.figure(num=1)
plt.plot(alphas_mercury,V_magsMercury,color='gray')
plt.plot(alphas_venus1,V_magsVenus_1,color='yellow', marker='x')
plt.plot(alphas_venus2,V_magsVenus_2,color='yellow',linestyle='--')
plt.plot(alphas_earth,V_magsEarth,color='blue')
plt.plot(alphas_mars1,V_magsMars_1,color='red', marker='x')
plt.plot(alphas_mars2,V_magsMars_2,color='red',linestyle='--')
plt.plot(alphas_jupiter1,V_magsJupiter_1,color='orange', marker='x')
plt.plot(alphas_jupiter2,V_magsJupiter_2,color='orange', linestyle='--')
plt.plot(alphas_saturn1,V_magsSaturn_1,color='gold', marker='x')
plt.plot(alphas_saturn2,V_magsSaturn_2,color='gold',linestyle='--')
plt.plot(alphas_saturn3,V_magsSaturn_3,color='gold',linestyle='.')
plt.plot(alphas_saturn4,V_magsSaturn_4,color='gold',linestyle='.-')
plt.plot(alphas_uranus,V_magsUranus_1,color='blue', marker='x')
plt.plot(alphas_uranus,V_magsUranus_2,color='blue', linestyle='--')
plt.plot(alphas_uranus,V_magsUranus_3,color='blue',linestyle='.')
plt.plot(alphas_neptune,V_magsNeptune,color='cyan')
plt.show(block=False)
######################################################################################################################





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




