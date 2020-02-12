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
#DELETEfrom matplotlib import colors
import datetime
import re
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

# #KEEP#### verifying the a solution exists somewhere in the entire alpha range
# print('Verifying a Solution exists somewhere in the entire alpha range')
# alpha1_range = np.linspace(start=0.,stop=180.,num=180)
# alpha2_range = np.linspace(start=alpha_min_fullphase_larger,stop=alpha_max_fullphase_larger,num=90)
# alpha3_range = np.linspace(start=alpha_min_crescent_larger,stop=alpha_max_crescent_larger,num=30)
# FRgrid = np.zeros((len(alpha1_range),len(alpha2_range)+len(alpha3_range)))
# tic = time()
# for i in np.arange(len(alpha1_range)):
#     for j in np.arange(len(alpha2_range)):
#         FRgrid[i,j] = fluxRatioPHASE.subs(alpha_smaller,alpha1_range[i]).subs(alpha_larger,alpha2_range[j])
#     for j in np.arange(len(alpha3_range)):
#         FRgrid[i,j+len(alpha2_range)-1] = fluxRatioPHASE.subs(alpha_smaller,alpha1_range[i]).subs(alpha_larger,alpha3_range[j])
#     print('Verify Soln: ' + str(i))
# print(time()-tic)

# plt.figure(num=97987987)
# tmp = FRgrid.copy()
# #tmp[tmp > 20.] = np.nan
# plt.contourf(alpha1_range,list(alpha2_range)+list(alpha3_range),tmp.T, locator=ticker.LogLocator(), levels=[10**i for i in np.linspace(-5,5,num=11)])
# plt.plot([0.,180.],[alpha_max_fullphase_larger,alpha_max_fullphase_larger],color='black')
# plt.plot([0.,180.],[alpha_min_crescent_larger,alpha_min_crescent_larger],color='black')
# cbar3 = plt.colorbar()
# plt.xlabel('alpha1')
# plt.ylabel('alpha2,3')
# plt.show(block=False)
# plt.figure(num=979879872)
# tmp = FRgrid.copy()
# tmp[tmp > 1.] = np.nan
# plt.contourf(alpha1_range,list(alpha2_range)+list(alpha3_range),tmp.T, levels=100)# locator=ticker.LogLocator(), levels=[10**i for i in np.linspace(-5,5,num=11)])
# plt.plot([0.,180.],[alpha_max_fullphase_larger,alpha_max_fullphase_larger],color='black')
# plt.plot([0.,180.],[alpha_min_crescent_larger,alpha_min_crescent_larger],color='black')
# cbar3 = plt.colorbar()
# plt.xlabel('alpha1')
# plt.ylabel('alpha2,3')

# plt.scatter(out2.x[0],out2.x[1],marker='x', color='k')
# plt.show(block=False)

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
    #LARGER REFERS TO LARGER SMA
    a_smaller, a_larger, ind_smaller, ind_larger  = a_Lmaller_Larger(planProp,i,j)
    s_smaller, s_larger, s_max = s_smaller_larger_max(planProp,i,j)
    alpha_min_smaller, alpha_max_smaller, alpha_min_fullphase_larger, alpha_max_fullphase_larger, alpha_min_crescent_larger, alpha_max_crescent_larger = alpha_MinMaxranges(s_max,a_larger)
    i_crit = np.arccos(a_smaller/a_larger)*180./np.pi
    incDict[ind_smaller,ind_larger] = {}
    incDict[ind_smaller,ind_larger]['opt1'] = {}
    incDict[ind_smaller,ind_larger]['opt2'] = {}
    incDict[ind_smaller,ind_larger]['opt3'] = {}
    incDict[ind_smaller,ind_larger]['opt4'] = {}
    eqnDmagLHS = eqnDmag.subs(Phi,symbolicPhases[ind_smaller]).subs(a,planProp[planets[ind_smaller]]['a']).subs(R,planProp[planets[ind_smaller]]['R']).subs(p,planProp[planets[ind_smaller]]['p'])
    eqnDmagRHS = eqnDmag.subs(Phi,symbolicPhases[ind_larger]).subs(a,planProp[planets[ind_larger]]['a']).subs(R,planProp[planets[ind_larger]]['R']).subs(p,planProp[planets[ind_larger]]['p'])
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
    inc_range = np.linspace(start=0.,stop=90.,num=360)#Was 45
    outList1 = list()
    outList2 = list()
    outList3 = list()
    outList4 = list()
    dmagErrorList = list()
    continueOpt1 = True #Boolean indicating if last opt failed
    continueOpt2 = True #Boolean indicating if last opt failed
    continueOpt3 = True #Boolean indicating if last opt failed
    continueOpt4 = True #Boolean indicating if last opt failed
    opt1Incs = list()
    opt2Incs = list()
    opt3Incs = list()
    opt4Incs = list()
    incDict[ind_smaller,ind_larger]['opt1']['success'] = list()
    incDict[ind_smaller,ind_larger]['opt2']['success'] = list()
    incDict[ind_smaller,ind_larger]['opt3']['success'] = list()
    incDict[ind_smaller,ind_larger]['opt4']['success'] = list()
    incDict[ind_smaller,ind_larger]['opt1']['v1'] = list()
    incDict[ind_smaller,ind_larger]['opt1']['v2'] = list()
    incDict[ind_smaller,ind_larger]['opt2']['v1'] = list()
    incDict[ind_smaller,ind_larger]['opt2']['v2'] = list()
    incDict[ind_smaller,ind_larger]['opt3']['v1'] = list()
    incDict[ind_smaller,ind_larger]['opt3']['v2'] = list()
    incDict[ind_smaller,ind_larger]['opt4']['v1'] = list()
    incDict[ind_smaller,ind_larger]['opt4']['v2'] = list()
    incDict[ind_smaller,ind_larger]['opt1']['fun'] = list()
    incDict[ind_smaller,ind_larger]['opt2']['fun'] = list()
    incDict[ind_smaller,ind_larger]['opt3']['fun'] = list()
    incDict[ind_smaller,ind_larger]['opt4']['fun'] = list()
    incDict[ind_smaller,ind_larger]['opt1']['funZERO'] = list()
    incDict[ind_smaller,ind_larger]['opt2']['funZERO'] = list()
    incDict[ind_smaller,ind_larger]['opt3']['funZERO'] = list()
    incDict[ind_smaller,ind_larger]['opt4']['funZERO'] = list()
    for i in np.arange(len(inc_range)):
        #Opt1 Start Small: Full Phase Side, Large: Dim Side
        min_alpha = inc_range[i]
        if alpha_min_crescent_larger > 180.-min_alpha:
            continue
        else:
            if not continueOpt1:
                continue
            #Find Intersection (brighter of smaller, dimmer of larger)
            x0 = np.asarray([(90.+min_alpha)/2.,alpha_min_crescent_larger*0.3+(180.-min_alpha)*0.7])
            out = minimize(funcMaxInc, x0, method='SLSQP', bounds=[(0.+min_alpha,180.-min_alpha),(alpha_min_crescent_larger,180.-min_alpha)], constraints=[{'type':'eq','fun':con_sepAlpha}], options={'disp':True,})
            outList1.append(out)
            opt1Incs.append(inc_range[i])
            dmagErrorList.append(np.abs(eqnDmag1RHS.subs(alpha,out.x[1]).evalf() - eqnDmag1LHS.subs(alpha,out.x[0]).evalf()))
            if out.success == False:# or out.fun > 0.1:#If we did not successfully converge, do not run this opt again
                continueOpt1 = False
            incDict[ind_smaller,ind_larger]['opt1']['success'].append(out.success)
            incDict[ind_smaller,ind_larger]['opt1']['v1'].append(out.x[0])
            incDict[ind_smaller,ind_larger]['opt1']['v2'].append(out.x[1])
            incDict[ind_smaller,ind_larger]['opt1']['fun'].append(out.fun)
            incDict[ind_smaller,ind_larger]['opt1']['funZERO'].append(out.fun < 0.1)
            incDict[ind_smaller,ind_larger]['opt1']['inc_range'] = opt1Incs
        #Opt2 Start Small: Full Phase Side, Large: Full Phase Side
        if alpha_max_fullphase_larger < min_alpha:
            continue
        else:
            if not continueOpt2:
                continue
            #Find Intersection (brighter of smaller, brighter of larger)
            x0 = np.asarray([(90.+min_alpha)/2.,(min_alpha+alpha_max_fullphase_larger)/2.])
            out = minimize(funcMaxInc, x0, method='SLSQP', bounds=[(0.+min_alpha,180.-min_alpha),(0.+min_alpha,alpha_max_fullphase_larger)], constraints=[{'type':'eq','fun':con_sepAlpha}], options={'disp':True,})
            outList2.append(out)
            opt2Incs.append(inc_range[i])
            dmagErrorList.append(np.abs(eqnDmag1RHS.subs(alpha,out.x[1]).evalf() - eqnDmag1LHS.subs(alpha,out.x[0]).evalf()))
            if out.success == False:# or out.fun > 0.1: #If we did not successfully converge, do not run this opt again
                continueOpt2 = False
            incDict[ind_smaller,ind_larger]['opt2']['success'].append(out.success)
            incDict[ind_smaller,ind_larger]['opt2']['v1'].append(out.x[0])
            incDict[ind_smaller,ind_larger]['opt2']['v2'].append(out.x[1])
            incDict[ind_smaller,ind_larger]['opt2']['fun'].append(out.fun)
            incDict[ind_smaller,ind_larger]['opt2']['funZERO'].append(out.fun < 1e-5) #The solution is optimal
            incDict[ind_smaller,ind_larger]['opt2']['inc_range'] = opt2Incs
        #Opt3 Start Small: Dim Side, Large: Dim Side
        min_alpha = inc_range[i]
        if alpha_min_crescent_larger > 180.-min_alpha:
            continue
        else:
            if not continueOpt3:
                continue
            #Find Intersection (brighter of smaller, dimmer of larger)
            x0 = np.asarray([90.*0.3+(180.-min_alpha)*0.7,alpha_min_crescent_larger*0.3+(180.-min_alpha)*0.7])
            out = minimize(funcMaxInc, x0, method='SLSQP', bounds=[(0.+min_alpha,180.-min_alpha),(alpha_min_crescent_larger,180.-min_alpha)], constraints=[{'type':'eq','fun':con_sepAlpha}], options={'disp':True,})
            outList3.append(out)
            opt3Incs.append(inc_range[i])
            dmagErrorList.append(np.abs(eqnDmag1RHS.subs(alpha,out.x[1]).evalf() - eqnDmag1LHS.subs(alpha,out.x[0]).evalf()))
            if out.success == False:# or out.fun > 0.1:#If we did not successfully converge, do not run this opt again
                continueOpt3 = False
            incDict[ind_smaller,ind_larger]['opt3']['success'].append(out.success)
            incDict[ind_smaller,ind_larger]['opt3']['v1'].append(out.x[0])
            incDict[ind_smaller,ind_larger]['opt3']['v2'].append(out.x[1])
            incDict[ind_smaller,ind_larger]['opt3']['fun'].append(out.fun)
            incDict[ind_smaller,ind_larger]['opt3']['funZERO'].append(out.fun < 0.1)
            incDict[ind_smaller,ind_larger]['opt3']['inc_range'] = opt3Incs
        #Opt4 Start Small: Dim Side, Large: Full Phase Side
        if alpha_max_fullphase_larger < min_alpha:
            continue
        else:
            if not continueOpt4:
                continue
            #Find Intersection (brighter of smaller, brighter of larger)
            x0 = np.asarray([90.*0.3+(180.-min_alpha)*0.7,(min_alpha+alpha_max_fullphase_larger)/2.])
            out = minimize(funcMaxInc, x0, method='SLSQP', bounds=[(0.+min_alpha,180.-min_alpha),(0.+min_alpha,alpha_max_fullphase_larger)], constraints=[{'type':'eq','fun':con_sepAlpha}], options={'disp':True,})
            outList4.append(out)
            opt4Incs.append(inc_range[i])
            dmagErrorList.append(np.abs(eqnDmag1RHS.subs(alpha,out.x[1]).evalf() - eqnDmag1LHS.subs(alpha,out.x[0]).evalf()))
            if out.success == False:# or out.fun > 0.1: #If we did not successfully converge, do not run this opt again
                continueOpt4 = False
            incDict[ind_smaller,ind_larger]['opt4']['success'].append(out.success)
            incDict[ind_smaller,ind_larger]['opt4']['v1'].append(out.x[0])
            incDict[ind_smaller,ind_larger]['opt4']['v2'].append(out.x[1])
            incDict[ind_smaller,ind_larger]['opt4']['fun'].append(out.fun)
            incDict[ind_smaller,ind_larger]['opt4']['funZERO'].append(out.fun < 1e-5) #The solution is optimal
            incDict[ind_smaller,ind_larger]['opt4']['inc_range'] = opt4Incs

    incDict[ind_smaller,ind_larger]['opt1']['incs'] = opt1Incs
    incDict[ind_smaller,ind_larger]['opt2']['incs'] = opt2Incs
    incDict[ind_smaller,ind_larger]['opt3']['incs'] = opt3Incs
    incDict[ind_smaller,ind_larger]['opt4']['incs'] = opt4Incs
    incDict[ind_smaller,ind_larger]['opt1']['outList'] = outList1
    incDict[ind_smaller,ind_larger]['opt2']['outList'] = outList2
    incDict[ind_smaller,ind_larger]['opt3']['outList'] = outList3
    incDict[ind_smaller,ind_larger]['opt4']['outList'] = outList4

    #Pick which side was smaller, Opt1 or Opt2
    minOpt1 = np.min(incDict[ind_smaller,ind_larger]['opt1']['fun'])
    minOpt2 = np.min(incDict[ind_smaller,ind_larger]['opt2']['fun'])
    minOpt3 = np.min(incDict[ind_smaller,ind_larger]['opt3']['fun'])
    minOpt4 = np.min(incDict[ind_smaller,ind_larger]['opt4']['fun'])
    tmp = [minOpt1, minOpt2, minOpt3, minOpt4]
    ind_minSide = np.argmin(tmp)
    optNum = 'opt' + str(ind_minSide+1)
    tmp.pop(ind_minSide)
    ind_minSide2 = np.argmin(tmp)
    nonOptNum = 'opt' + str(ind_minSide2+1)
    # if ind_minSide == 0:
    #     optNum = 'opt1'
    #     nonOptNum = 'opt2'
    # elif ind_minSide == 1:
    #     optNum = 'opt2'
    #     nonOptNum = 'opt1'
    incDict[ind_smaller,ind_larger]['optNum'] = optNum #which side has fun closest to 0
    incDict[ind_smaller,ind_larger]['optNum_isOpt'] = np.asarray(incDict[ind_smaller,ind_larger][incDict[ind_smaller,ind_larger]['optNum']]['fun']) < 0.1#1e-5 #true if close to optimal
    # Does the other side have dmag-s coincidence?
    incDict[ind_smaller,ind_larger]['nonOptNum'] = nonOptNum #the keyName of the nonOptNum side 
    incDict[ind_smaller,ind_larger]['nonOptNum_isOpt'] = np.asarray(incDict[ind_smaller,ind_larger][incDict[ind_smaller,ind_larger]['nonOptNum']]['fun']) < 0.1#1e-1 #checks to see if a solution is second intersection (mars-jupiter)
    incDict[ind_smaller,ind_larger]['isNonOptNum_Opt'] = np.any(incDict[ind_smaller,ind_larger]['nonOptNum_isOpt']) #checks to see if a second intersection occurs (mars-jupiter)
    #What is the maximum inclination where a solution still occurs
    tmp = [i for i, val in enumerate(incDict[ind_smaller,ind_larger]['optNum_isOpt']) if val]
    if not tmp == []:
        incDict[ind_smaller,ind_larger]['maxIncInd_Opt'] = int(np.max(tmp))
    if incDict[ind_smaller,ind_larger]['isNonOptNum_Opt']:
        incDict[ind_smaller,ind_larger]['maxIncInd_NonOpt'] = int(np.max([i for i, val in enumerate(incDict[ind_smaller,ind_larger]['nonOptNum_isOpt']) if val])) #finds index of second solution point
######################################################################

#Analysis of Optimization Runs
# assert len(incDict[0,1]['opt1']['outList']) > 1, "there was no success" #There must be at least 1 unsuccessful optimization attempt
# successfulOut = incDict[0,1]['opt1']['outList'][-2] #second from last is the successful one closest to limit
# successfulv1 = successfulOut.x[0]
# successfulv2 = successfulOut.x[1]
# successfulInc = incDict[0,1]['opt1']['incs'][-2]

#### Craft Intersection Table
#ADD DMAG AND S
line1 = r"(Y,$i$,$\alpha_{smaller}$,$\alpha_{larger}$) & $\mercury$ & $\venus$ & $\earth$ & $\mars$ & $\jupiter$ & $\saturn$ & $\uranus$ & $\neptune$"
ind_smaller = 0
eqnDmagLHS = eqnDmag.subs(Phi,symbolicPhases[ind_smaller]).subs(a,planProp[planets[ind_smaller]]['a']).subs(R,planProp[planets[ind_smaller]]['R']).subs(p,planProp[planets[ind_smaller]]['p'])
i01 =  incDict[0,1][incDict[0,1]['optNum']]['inc_range'][incDict[0,1]['maxIncInd_Opt']]
as01 = incDict[0,1][incDict[0,1]['optNum']]['v1'][incDict[0,1]['maxIncInd_Opt']]
al01 = incDict[0,1][incDict[0,1]['optNum']]['v2'][incDict[0,1]['maxIncInd_Opt']]
dmag01 = float(eqnDmagLHS.subs(alpha,as01).evalf())
s01 = float(eqnSAlpha.subs(a,planProp[planets[0]]['a']*u.m.to('AU')).subs(alpha,as01).evalf())
i02 =  incDict[0,2][incDict[0,2]['optNum']]['inc_range'][incDict[0,2]['maxIncInd_Opt']]
as02 = incDict[0,2][incDict[0,2]['optNum']]['v1'][incDict[0,2]['maxIncInd_Opt']]
al02 = incDict[0,2][incDict[0,2]['optNum']]['v2'][incDict[0,2]['maxIncInd_Opt']]
dmag02 = float(eqnDmagLHS.subs(alpha,as02).evalf())
s02 = float(eqnSAlpha.subs(a,planProp[planets[0]]['a']*u.m.to('AU')).subs(alpha,as02).evalf())
i03 =  incDict[0,3][incDict[0,3]['optNum']]['inc_range'][incDict[0,3]['maxIncInd_Opt']]
as03 = incDict[0,3][incDict[0,3]['optNum']]['v1'][incDict[0,3]['maxIncInd_Opt']]
al03 = incDict[0,3][incDict[0,3]['optNum']]['v2'][incDict[0,3]['maxIncInd_Opt']]
dmag03 = float(eqnDmagLHS.subs(alpha,as03).evalf())
s03 = float(eqnSAlpha.subs(a,planProp[planets[0]]['a']*u.m.to('AU')).subs(alpha,as03).evalf())
i06 =  incDict[0,6][incDict[0,6]['optNum']]['inc_range'][incDict[0,6]['maxIncInd_Opt']]
as06 = incDict[0,6][incDict[0,6]['optNum']]['v1'][incDict[0,6]['maxIncInd_Opt']]
al06 = incDict[0,6][incDict[0,6]['optNum']]['v2'][incDict[0,6]['maxIncInd_Opt']]
dmag06 = float(eqnDmagLHS.subs(alpha,as06).evalf())
s06 = float(eqnSAlpha.subs(a,planProp[planets[0]]['a']*u.m.to('AU')).subs(alpha,as06).evalf())
i07 =  incDict[0,7][incDict[0,7]['optNum']]['inc_range'][incDict[0,7]['maxIncInd_Opt']]
as07 = incDict[0,7][incDict[0,7]['optNum']]['v1'][incDict[0,7]['maxIncInd_Opt']]
al07 = incDict[0,7][incDict[0,7]['optNum']]['v2'][incDict[0,7]['maxIncInd_Opt']]
dmag07 = float(eqnDmagLHS.subs(alpha,as07).evalf())
s07 = float(eqnSAlpha.subs(a,planProp[planets[0]]['a']*u.m.to('AU')).subs(alpha,as07).evalf())
line2 = r"$\mercury$    & \cellcolor{black} & (Y,"+str(np.round(i01,2))+","+str(np.round(as01,2))+","+str(np.round(al01,2))+","+str(np.round(dmag01,2))+","+str(np.round(s01,2))+")  & (Y,"+str(np.round(i02,2))+","+str(np.round(as02,2))+","+str(np.round(al02,2))+","+str(np.round(dmag02,2))+","+str(np.round(s02,2))+")"\
        +"  & (Y,"+str(np.round(i03,2))+","+str(np.round(as03,2))+","+str(np.round(al03,2))+","+str(np.round(dmag03,2))+","+str(np.round(s03,2))+")  & (N,) & (N,) & (Y,"+str(np.round(i06,2))+","+str(np.round(as06,2))+","+str(np.round(al06,2))+","+str(np.round(dmag06,2))+","+str(np.round(s06,2))+")"\
        +"  & (Y,"+str(np.round(i07,2))+","+str(np.round(as07,2))+","+str(np.round(al07,2))+","+str(np.round(dmag07,2))+","+str(np.round(s07,2))+")\\"
ind_smaller = 1
eqnDmagLHS = eqnDmag.subs(Phi,symbolicPhases[ind_smaller]).subs(a,planProp[planets[ind_smaller]]['a']).subs(R,planProp[planets[ind_smaller]]['R']).subs(p,planProp[planets[ind_smaller]]['p'])
i12 =  incDict[1,2][incDict[1,2]['optNum']]['inc_range'][incDict[1,2]['maxIncInd_Opt']]
as12 = incDict[1,2][incDict[1,2]['optNum']]['v1'][incDict[1,2]['maxIncInd_Opt']]
al12 = incDict[1,2][incDict[1,2]['optNum']]['v2'][incDict[1,2]['maxIncInd_Opt']]
dmag12 = float(eqnDmagLHS.subs(alpha,as12).evalf())
s12 = float(eqnSAlpha.subs(a,planProp[planets[1]]['a']*u.m.to('AU')).subs(alpha,as12).evalf())
i15 =  incDict[1,5][incDict[1,5]['optNum']]['inc_range'][incDict[1,5]['maxIncInd_Opt']]
as15 = incDict[1,5][incDict[1,5]['optNum']]['v1'][incDict[1,5]['maxIncInd_Opt']]
al15 = incDict[1,5][incDict[1,5]['optNum']]['v2'][incDict[1,5]['maxIncInd_Opt']]
dmag15 = float(eqnDmagLHS.subs(alpha,as15).evalf())
s15 = float(eqnSAlpha.subs(a,planProp[planets[1]]['a']*u.m.to('AU')).subs(alpha,as15).evalf())
line3 = r"$\venus$      & & \cellcolor{black} & (Y,"+str(np.round(i12,2))+","+str(np.round(as12,2))+","+str(np.round(al12,2))+","+str(np.round(dmag12,2))+","+str(np.round(s12,2))+")  & (N,) & (N,)  & (Y,"+str(np.round(i15,2))+","+str(np.round(as15,2))+","+str(np.round(al15,2))+","+str(np.round(dmag15,2))+","+str(np.round(s15,2))+") & (N,) & (N,)\\"
i23 =  incDict[2,3][incDict[2,3]['optNum']]['inc_range'][incDict[2,3]['maxIncInd_Opt']]
as23 = incDict[2,3][incDict[2,3]['optNum']]['v1'][incDict[2,3]['maxIncInd_Opt']]
al23 = incDict[2,3][incDict[2,3]['optNum']]['v2'][incDict[2,3]['maxIncInd_Opt']]
dmag23 = float(eqnDmagLHS.subs(alpha,as23).evalf())
s23 = float(eqnSAlpha.subs(a,planProp[planets[2]]['a']*u.m.to('AU')).subs(alpha,as23).evalf())
i25 =  incDict[2,5][incDict[2,5]['optNum']]['inc_range'][incDict[2,5]['maxIncInd_Opt']]
as25 = incDict[2,5][incDict[2,5]['optNum']]['v1'][incDict[2,5]['maxIncInd_Opt']]
al25 = incDict[2,5][incDict[2,5]['optNum']]['v2'][incDict[2,5]['maxIncInd_Opt']]
dmag25 = float(eqnDmagLHS.subs(alpha,as25).evalf())
s25 = float(eqnSAlpha.subs(a,planProp[planets[2]]['a']*u.m.to('AU')).subs(alpha,as25).evalf())
i26 =  incDict[2,6][incDict[2,6]['optNum']]['inc_range'][incDict[2,6]['maxIncInd_Opt']]
as26 = incDict[2,6][incDict[2,6]['optNum']]['v1'][incDict[2,6]['maxIncInd_Opt']]
al26 = incDict[2,6][incDict[2,6]['optNum']]['v2'][incDict[2,6]['maxIncInd_Opt']]
dmag26 = float(eqnDmagLHS.subs(alpha,as26).evalf())
s26 = float(eqnSAlpha.subs(a,planProp[planets[2]]['a']*u.m.to('AU')).subs(alpha,as26).evalf())
i27 =  incDict[2,7][incDict[2,7]['optNum']]['inc_range'][incDict[2,7]['maxIncInd_Opt']]
as27 = incDict[2,7][incDict[2,7]['optNum']]['v1'][incDict[2,7]['maxIncInd_Opt']]
al27 = incDict[2,7][incDict[2,7]['optNum']]['v2'][incDict[2,7]['maxIncInd_Opt']]
dmag27 = float(eqnDmagLHS.subs(alpha,as27).evalf())
s27 = float(eqnSAlpha.subs(a,planProp[planets[2]]['a']*u.m.to('AU')).subs(alpha,as27).evalf())
line4 = r"$\earth$      & &  & \cellcolor{black} & (Y,"+str(np.round(i23,2))+","+str(np.round(as23,2))+","+str(np.round(al23,2))+","+str(np.round(dmag23,2))+","+str(np.round(s23,2))+")"\
        +"  & (N,) & (Y,"+str(np.round(i25,2))+","+str(np.round(as25,2))+","+str(np.round(al25,2))+","+str(np.round(dmag25,2))+","+str(np.round(s25,2))+")  & (Y,"+str(np.round(i26,2))+","+str(np.round(as26,2))+\
        ","+str(np.round(al26,2))+","+str(np.round(dmag26,2))+","+str(np.round(s26,2))+")  & (Y,"+str(np.round(i27,2))+","+str(np.round(as27,2))+","+str(np.round(al27,2))+","+str(np.round(dmag27,2))+","+str(np.round(s27,2))+")\\"
ind_smaller = 3
eqnDmagLHS = eqnDmag.subs(Phi,symbolicPhases[ind_smaller]).subs(a,planProp[planets[ind_smaller]]['a']).subs(R,planProp[planets[ind_smaller]]['R']).subs(p,planProp[planets[ind_smaller]]['p'])
i34 =  incDict[3,4][incDict[3,4]['optNum']]['inc_range'][incDict[3,4]['maxIncInd_Opt']]
as34 = incDict[3,4][incDict[3,4]['optNum']]['v1'][incDict[3,4]['maxIncInd_Opt']]
al34 = incDict[3,4][incDict[3,4]['optNum']]['v2'][incDict[3,4]['maxIncInd_Opt']]
dmag34 = float(eqnDmagLHS.subs(alpha,as34).evalf())
s34 = float(eqnSAlpha.subs(a,planProp[planets[3]]['a']*u.m.to('AU')).subs(alpha,as34).evalf())
i36 =  incDict[3,6][incDict[3,6]['optNum']]['inc_range'][incDict[3,6]['maxIncInd_Opt']]
as36 = incDict[3,6][incDict[3,6]['optNum']]['v1'][incDict[3,6]['maxIncInd_Opt']]
al36 = incDict[3,6][incDict[3,6]['optNum']]['v2'][incDict[3,6]['maxIncInd_Opt']]
dmag36 = float(eqnDmagLHS.subs(alpha,as36).evalf())
s36 = float(eqnSAlpha.subs(a,planProp[planets[3]]['a']*u.m.to('AU')).subs(alpha,as36).evalf())
line5 = r"$\mars$       & & & & \cellcolor{black} & (Y,"+str(np.round(i34,2))+","+str(np.round(as34,2))+","+str(np.round(al34,2))+","+str(np.round(dmag34,2))+","+str(np.round(s34,2))+")"\
        +"  & (N,) & (Y,"+str(np.round(i36,2))+","+str(np.round(as36,2))+","+str(np.round(al36,2))+","+str(np.round(dmag36,2))+","+str(np.round(s36,2))+")  & (N,)\\"
#FIX Should be other solution
i34 =  incDict[3,4][incDict[3,4]['optNum']]['inc_range'][incDict[3,4]['maxIncInd_Opt']]
as34 = incDict[3,4][incDict[3,4]['optNum']]['v1'][incDict[3,4]['maxIncInd_Opt']]
al34 = incDict[3,4][incDict[3,4]['optNum']]['v2'][incDict[3,4]['maxIncInd_Opt']]
dmag34 = float(eqnDmagLHS.subs(alpha,al34).evalf())
s34 = float(eqnSAlpha.subs(a,planProp[planets[3]]['a']*u.m.to('AU')).subs(alpha,as34).evalf())
ind_smaller = 4
eqnDmagLHS = eqnDmag.subs(Phi,symbolicPhases[ind_smaller]).subs(a,planProp[planets[ind_smaller]]['a']).subs(R,planProp[planets[ind_smaller]]['R']).subs(p,planProp[planets[ind_smaller]]['p'])
i45 =  incDict[4,5][incDict[4,5]['optNum']]['inc_range'][incDict[4,5]['maxIncInd_Opt']]
as45 = incDict[4,5][incDict[4,5]['optNum']]['v1'][incDict[4,5]['maxIncInd_Opt']]
al45 = incDict[4,5][incDict[4,5]['optNum']]['v2'][incDict[4,5]['maxIncInd_Opt']]
dmag45 = float(eqnDmagLHS.subs(alpha,as45).evalf())
s45 = float(eqnSAlpha.subs(a,planProp[planets[4]]['a']*u.m.to('AU')).subs(alpha,as45).evalf())
i46 =  incDict[4,6][incDict[4,6]['optNum']]['inc_range'][incDict[4,6]['maxIncInd_Opt']]
as46 = incDict[4,6][incDict[4,6]['optNum']]['v1'][incDict[4,6]['maxIncInd_Opt']]
al46 = incDict[4,6][incDict[4,6]['optNum']]['v2'][incDict[4,6]['maxIncInd_Opt']]
dmag46 = float(eqnDmagLHS.subs(alpha,as46).evalf())
s46 = float(eqnSAlpha.subs(a,planProp[planets[4]]['a']*u.m.to('AU')).subs(alpha,as46).evalf())
i47 =  incDict[4,7][incDict[4,7]['optNum']]['inc_range'][incDict[4,7]['maxIncInd_Opt']]
as47 = incDict[4,7][incDict[4,7]['optNum']]['v1'][incDict[4,7]['maxIncInd_Opt']]
al47 = incDict[4,7][incDict[4,7]['optNum']]['v2'][incDict[4,7]['maxIncInd_Opt']]
dmag47 = float(eqnDmagLHS.subs(alpha,as47).evalf())
s47 = float(eqnSAlpha.subs(a,planProp[planets[4]]['a']*u.m.to('AU')).subs(alpha,as47).evalf())
line6 = r"$\jupiter$    & & & & (Y,"+str(np.round(i34,2))+","+str(np.round(as34,2))+","+str(np.round(al34,2))+","+str(np.round(dmag34,2))+","+str(np.round(s34,2))+r") & \cellcolor{black} & (Y,"+str(np.round(i45,2))+","+str(np.round(as45,2))+","+str(np.round(al45,2))+","+str(np.round(dmag45,2))+","+str(np.round(s45,2))+")"\
        +"  & (Y,"+str(np.round(i46,2))+","+str(np.round(as46,2))+","+str(np.round(al46,2))+","+str(np.round(dmag46,2))+","+str(np.round(s46,2))+")  & (Y,"+str(np.round(i47,2))+","+str(np.round(as47,2))+","+str(np.round(al47,2))+","+str(np.round(dmag47,2))+","+str(np.round(s47,2))+")\\"
ind_smaller = 5
eqnDmagLHS = eqnDmag.subs(Phi,symbolicPhases[ind_smaller]).subs(a,planProp[planets[ind_smaller]]['a']).subs(R,planProp[planets[ind_smaller]]['R']).subs(p,planProp[planets[ind_smaller]]['p'])
i56 =  incDict[5,6][incDict[5,6]['optNum']]['inc_range'][incDict[5,6]['maxIncInd_Opt']]
as56 = incDict[5,6][incDict[5,6]['optNum']]['v1'][incDict[5,6]['maxIncInd_Opt']]
al56 = incDict[5,6][incDict[5,6]['optNum']]['v2'][incDict[5,6]['maxIncInd_Opt']]
dmag56 = float(eqnDmagLHS.subs(alpha,as56).evalf())
s56 = float(eqnSAlpha.subs(a,planProp[planets[5]]['a']*u.m.to('AU')).subs(alpha,as56).evalf())
i57 =  incDict[5,7][incDict[5,7]['optNum']]['inc_range'][incDict[5,7]['maxIncInd_Opt']]
as57 = incDict[5,7][incDict[5,7]['optNum']]['v1'][incDict[5,7]['maxIncInd_Opt']]
al57 = incDict[5,7][incDict[5,7]['optNum']]['v2'][incDict[5,7]['maxIncInd_Opt']]
dmag57 = float(eqnDmagLHS.subs(alpha,as57).evalf())
s57 = float(eqnSAlpha.subs(a,planProp[planets[5]]['a']*u.m.to('AU')).subs(alpha,as57).evalf())
line7 = r"$\saturn$     & & & & & & \cellcolor{black} & (Y,"+str(np.round(i56,2))+","+str(np.round(as56,2))+","+str(np.round(al56,2))+","+str(np.round(dmag56,2))+","+str(np.round(s56,2))+")"\
        +"  & (Y,"+str(np.round(i57,2))+","+str(np.round(as57,2))+","+str(np.round(al57,2))+","+str(np.round(dmag57,2))+","+str(np.round(s57,2))+")\\"
ind_smaller = 6
eqnDmagLHS = eqnDmag.subs(Phi,symbolicPhases[ind_smaller]).subs(a,planProp[planets[ind_smaller]]['a']).subs(R,planProp[planets[ind_smaller]]['R']).subs(p,planProp[planets[ind_smaller]]['p'])
i67 =  incDict[6,7][incDict[6,7]['optNum']]['inc_range'][incDict[6,7]['maxIncInd_Opt']]
as67 = incDict[6,7][incDict[6,7]['optNum']]['v1'][incDict[6,7]['maxIncInd_Opt']]
al67 = incDict[6,7][incDict[6,7]['optNum']]['v2'][incDict[6,7]['maxIncInd_Opt']]
dmag67 = float(eqnDmagLHS.subs(alpha,as67).evalf())
s67 = float(eqnSAlpha.subs(a,planProp[planets[6]]['a']*u.m.to('AU')).subs(alpha,as67).evalf())
line8 = r"$\uranus$     & & & & & & & \cellcolor{black} & (Y,"+str(np.round(i67,2))+","+str(np.round(as67,2))+","+str(np.round(al67,2))+","+str(np.round(dmag67,2))+","+str(np.round(s67,2))+")\\"
line9 = r"$\neptune$    & & & & & & & & \cellcolor{black}\\"
print(line1)
print(line2)
print(line3)
print(line4)
print(line5)
print(line6)
print(line7)
print(line8)
print(line9)









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

