# Calculate Indentical dMag of planets

import matplotlib.pyplot as plt
import numpy as np
#DELETE import matplotlib.dates as mdates
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
#DELETEfrom EXOSIMS.util.deltaMag import *
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
from mallama2018PlanetProperties import *

folder = './'
PPoutpath = './'




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

# inc_val = 90.
# #### Setup Jacobians#######################################################
# #objective function jacobian
# funcJac = [sp.diff(eqnDmag1RHS-eqnDmag1LHS,v1),sp.diff(eqnDmag1RHS-eqnDmag1LHS,v2)]
# #separation constraint jacobian
# sepConJac = [sp.diff(eqnS1RHS-eqnS1LHS,v1).subs(inc,inc_val), sp.diff(eqnS1RHS-eqnS1LHS,v2).subs(inc,inc_val)]
# ###########################################################################
# con2 = NonlinearConstraint(lambda y: sepCon(y,inc_val),lb=0.,ub=0., jac=lambda x: sepConJac[0].subs(v1,x))
# con1 = LinearConstraint(np.asarray([[1.,0.],[0.,1.]]),np.asarray([0.,0.]),np.asarray([180.,180.]))
# x0 = np.asarray([45.,55.])
# out = minimize(func, x0, args=(inc_val,), method='SLSQP', bounds=[(0.,180.),(0.,180.)], constraints=[{'type':'eq','fun':sepCon, 'args':(inc_val,)}], options={'disp':True,})#'eps':1.})#constraints=[con1])

# print(eqnDmag1RHS.subs(inc,inc_val).subs(v1,out.x[0]).subs(v2,out.x[1]).evalf())
# print(eqnDmag1LHS.subs(inc,inc_val).subs(v1,out.x[0]).subs(v2,out.x[1]).evalf())
# print(eqnS1RHS.subs(v1,out.x[0]).subs(v2,out.x[1]).subs(inc,inc_val))
# print(eqnS1RHS.subs(v1,out.x[0]).subs(v2,out.x[1]).subs(inc,inc_val))


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

