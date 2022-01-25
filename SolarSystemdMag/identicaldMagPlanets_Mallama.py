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
plt.gcf().canvas.draw()
plt.close(199)
plt.figure(num=199)
plt.plot(Ds,alphas1,color='red')
plt.plot(Ds,alphas2,color='green')
plt.xlabel('Earth-Sun-Planet Angle, ' + r'$D$' + ' in rad',weight='bold')
plt.ylabel('Sun-Planet-Earth Phase Angle, ' + r'$\alpha$' + ' in rad',weight='bold')
plt.show(block=False)
plt.gcf().canvas.draw()
plt.close(299)
plt.figure(num=299)
plt.plot(Ds,ds1,color='red')
plt.plot(Ds,ds2,color='green')
plt.xlabel('Earth-Sun-Planet Angle, ' + r'$D$' + ' in rad',weight='bold')
plt.ylabel('Earth-Planet distance, ' + r'$d$' + ' in AU',weight='bold')
plt.show(block=False)
plt.gcf().canvas.draw()


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
plt.gcf().canvas.draw()




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
line1 = r"(Y, $i$,$\alpha_{smaller}$, $\alpha_{larger}$) & $\mercury$ & $\venus$ & $\earth$ & $\mars$ & $\jupiter$ & $\saturn$ & $\uranus$ & $\neptune$"
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
#i43 =  incDict[3,4][incDict[3,4]['nonOptNum']]['inc_range'][incDict[3,4]['maxIncInd_NonOpt']]
#as43 = incDict[3,4][incDict[3,4]['nonOptNum']]['v1'][incDict[3,4]['maxIncInd_NonOpt']]
#al43 = incDict[3,4][incDict[3,4]['nonOptNum']]['v2'][incDict[3,4]['maxIncInd_NonOpt']]
i43 = incDict[3,4]['opt3']['inc_range'][12]
as43 = incDict[3,4]['opt3']['v1'][12]
al43 = incDict[3,4]['opt3']['v2'][12]
dmag43 = float(eqnDmagLHS.subs(alpha,al43).evalf())
s43 = float(eqnSAlpha.subs(a,planProp[planets[3]]['a']*u.m.to('AU')).subs(alpha,as43).evalf())
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
line6 = r"$\jupiter$    & & & & (Y,"+str(np.round(i43,2))+","+str(np.round(as43,2))+","+str(np.round(al43,2))+","+str(np.round(dmag43,2))+","+str(np.round(s43,2))+r") & \cellcolor{black} & (Y,"+str(np.round(i45,2))+","+str(np.round(as45,2))+","+str(np.round(al45,2))+","+str(np.round(dmag45,2))+","+str(np.round(s45,2))+")"\
        +"  & (Y,"+str(np.round(i46,2))+","+str(np.round(as46,2))+","+str(np.round(al46,2))+","+str(np.round(dmag46,2))+","+str(np.round(s46,2))+")  & (Y,"+str(np.round(i47,2))+","+str(np.round(as47,2))+","+str(np.round(al47,2))+","+str(np.round(dmag47,2))+","+str(np.round(s47,2))+")\\"
i54 = 0.75#incDict[4,5]['opt3']['inc_range'][12]
as54 = 178.55#incDict[4,5]['opt3']['v1'][12]
al54 = 179.20#incDict[4,5]['opt3']['v2'][12]
eqnDmagLHSs = eqnDmag.subs(Phi,symbolicPhases[4]).subs(a,planProp[planets[4]]['a']).subs(R,planProp[planets[4]]['R']).subs(p,planProp[planets[4]]['p'])  
eqnDmagLHSl = eqnDmag.subs(Phi,symbolicPhases[5]).subs(a,planProp[planets[5]]['a']).subs(R,planProp[planets[5]]['R']).subs(p,planProp[planets[5]]['p'])
dmag54 = float(eqnDmagLHSs.subs(alpha,as54).evalf())
dmag54_2 = float(eqnDmagLHSl.subs(alpha,al54).evalf())
s54 = float(eqnSAlpha.subs(a,planProp[planets[4]]['a']*u.m.to('AU')).subs(alpha,as54).evalf())
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



#### Table 2 #############################################################################
lines = list()
for k in [(0,1),(0,2),(0,3),(0,6),(0,7),(1,2),(1,5),(2,3),(2,5),(2,6),(2,7),(3,4),(3,6),(3,7),(4,5),(4,6),(4,7),(5,6),(5,7),(6,7)]: #iterate over planet pairs
    i = k[0]
    j = k[1]
    eqnDmagLHS = eqnDmag.subs(Phi,symbolicPhases[i]).subs(a,planProp[planets[i]]['a']).subs(R,planProp[planets[i]]['R']).subs(p,planProp[planets[i]]['p'])
    if i==3 and j==4:
        ik =  incDict[i,j][incDict[i,j]['optNum']]['inc_range'][incDict[i,j]['maxIncInd_Opt']]
        ask = incDict[i,j][incDict[i,j]['optNum']]['v1'][incDict[i,j]['maxIncInd_Opt']]
        alk = incDict[i,j][incDict[i,j]['optNum']]['v2'][incDict[i,j]['maxIncInd_Opt']]
        dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])
        lines.append(planets[i] + "-" + planets[j] + " & " + str(np.round(dmagk,2)) + " & " + str(np.round(sk,2)) + " & " + str(np.round(ask,2)) + " & " + str(np.round(alk,2)) + " & " + str(np.round(ik2,2)) + "\\\\")

        # i43 = incDict[3,4]['opt3']['inc_range'][12]
        # as43 = incDict[3,4]['opt3']['v1'][12]
        # al43 = incDict[3,4]['opt3']['v2'][12]
        # dmag43 = float(eqnDmagLHS.subs(alpha,al43).evalf())
        # s43 = float(eqnSAlpha.subs(a,planProp[planets[3]]['a']*u.m.to('AU')).subs(alpha,as43).evalf())
        # ind_smaller = 4
        # eqnDmagLHS = eqnDmag.subs(Phi,symbolicPhases[ind_smaller]).subs(a,planProp[planets[ind_smaller]]['a']).subs(R,planProp[planets[ind_smaller]]['R']).subs(p,planProp[planets[ind_smaller]]['p'])

        ik =  incDict[i,j][incDict[i,j]['optNum']]['inc_range'][12]
        ask = incDict[i,j][incDict[i,j]['optNum']]['v1'][12]
        alk = incDict[i,j][incDict[i,j]['optNum']]['v2'][12]
        dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])
        lines.append(planets[i] + "-" + planets[j] + " & " + str(np.round(dmagk,2)) + " & " + str(np.round(sk,2)) + " & " + str(np.round(ask,2)) + " & " + str(np.round(alk,2)) + " & " + str(np.round(ik2,2)) + "\\\\")
    elif i==4 and j==5:
        ik =  incDict[i,j][incDict[i,j]['optNum']]['inc_range'][incDict[i,j]['maxIncInd_Opt']]
        ask = incDict[i,j][incDict[i,j]['optNum']]['v1'][incDict[i,j]['maxIncInd_Opt']]
        alk = incDict[i,j][incDict[i,j]['optNum']]['v2'][incDict[i,j]['maxIncInd_Opt']]
        dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])
        lines.append(planets[i] + "-" + planets[j] + " & " + str(np.round(dmagk,2)) + " & " + str(np.round(sk,2)) + " & " + str(np.round(ask,2)) + " & " + str(np.round(alk,2)) + " & " + str(np.round(ik2,2)) + "\\\\")

        # i54 = 0.75#incDict[4,5]['opt3']['inc_range'][12]
        # as54 = 178.55#incDict[4,5]['opt3']['v1'][12]
        # al54 = 179.20#incDict[4,5]['opt3']['v2'][12]
        # eqnDmagLHSs = eqnDmag.subs(Phi,symbolicPhases[4]).subs(a,planProp[planets[4]]['a']).subs(R,planProp[planets[4]]['R']).subs(p,planProp[planets[4]]['p'])  
        # eqnDmagLHSl = eqnDmag.subs(Phi,symbolicPhases[5]).subs(a,planProp[planets[5]]['a']).subs(R,planProp[planets[5]]['R']).subs(p,planProp[planets[5]]['p'])
        # dmag54 = float(eqnDmagLHSs.subs(alpha,as54).evalf())
        # dmag54_2 = float(eqnDmagLHSl.subs(alpha,al54).evalf())
        # s54 = float(eqnSAlpha.subs(a,planProp[planets[4]]['a']*u.m.to('AU')).subs(alpha,as54).evalf())
        # ind_smaller = 5
        # eqnDmagLHS = eqnDmag.subs(Phi,symbolicPhases[ind_smaller]).subs(a,planProp[planets[ind_smaller]]['a']).subs(R,planProp[planets[ind_smaller]]['R']).subs(p,planProp[planets[ind_smaller]]['p'])

        ik =  0.75 #incDict[i,j][incDict[i,j]['optNum']]['inc_range'][incDict[i,j]['maxIncInd_Opt']]
        ask = 178.55 #incDict[i,j][incDict[i,j]['optNum']]['v1'][incDict[i,j]['maxIncInd_Opt']]
        alk = 179.20 #incDict[i,j][incDict[i,j]['optNum']]['v2'][incDict[i,j]['maxIncInd_Opt']]
        dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])
        lines.append(planets[i] + "-" + planets[j] + " & " + str(np.round(dmagk,2)) + " & " + str(np.round(sk,2)) + " & " + str(np.round(ask,2)) + " & " + str(np.round(alk,2)) + " & " + str(np.round(ik2,2)) + "\\\\")
    else:

        ik =  incDict[i,j][incDict[i,j]['optNum']]['inc_range'][incDict[i,j]['maxIncInd_Opt']]
        ask = incDict[i,j][incDict[i,j]['optNum']]['v1'][incDict[i,j]['maxIncInd_Opt']]
        alk = incDict[i,j][incDict[i,j]['optNum']]['v2'][incDict[i,j]['maxIncInd_Opt']]
        dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])
        lines.append(planets[i] + "-" + planets[j] + " & " + str(np.round(dmagk,2)) + " & " + str(np.round(sk,2)) + " & " + str(np.round(ask,2)) + " & " + str(np.round(alk,2)) + " & " + str(np.round(ik2,2)) + "\\\\")
print('******************Lines for Table 2 in Paper ***************************')
print(lines)
for line in lines:
    print(line)
print('******************Lines for Table 2 in Paper ***************************')
##########################################################################################








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
plt.gcf().canvas.draw()
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
plt.gcf().canvas.draw()


IWA2=0.150*u.arcsec #Suggested by dmitry as analahous to WFIRST
plotDmagvss(planProp,planets,uncertainty_dmag,uncertainty_s,IWA_HabEx=IWA_HabEx,IWA2=IWA2,inclination=0., folder='./', PPoutpath='./')
plotDmagvss(planProp,planets,uncertainty_dmag,uncertainty_s,IWA_HabEx=IWA_HabEx,IWA2=IWA2,inclination=0.5, folder='./', PPoutpath='./')
plotDmagvss(planProp,planets,uncertainty_dmag,uncertainty_s,IWA_HabEx=IWA_HabEx,IWA2=IWA2,inclination=0.85, folder='./', PPoutpath='./')
plotDmagvss(planProp,planets,uncertainty_dmag,uncertainty_s,IWA_HabEx=IWA_HabEx,IWA2=IWA2,inclination=25.3, folder='./', PPoutpath='./')





#### Calculate Phi Inverse Function for each melded phase function

table2Data = [['Mercury-Venus' , 25.41 , 0.26 , 42.18 , 158.94 , 21.06], #mercury all dmag Venus s-3sigma s-2sigma s-1sigma dmag-1sigma dmag-2sigma dmag-3sigma
['Mercury-Earth' , 27.72 , 0.36 , 111.84 , 158.94 , 21.06], #Mercury all dmag Earth all dmag
['Mercury-Mars' , 26.57 , 0.38 , 81.16 , 14.54 , 14.54], #Mercury all dmag Mars all s
['Mercury-Uranus' , 26.13 , 0.36 , 67.06 , 1.06 , 1.06], #Mercury all dmag Uranus all s
['Mercury-Neptune' , 27.2 , 0.38 , 99.27 , 0.73 , 0.73], #Mercury all dmag Neptune all s
['Venus-Earth' , 23.15 , 0.72 , 92.46 , 46.27 , 46.27], # Venus all dmag Earth all s
['Venus-Saturn' , 22.72 , 0.69 , 73.37 , 4.15 , 4.15], # Venus all dmag Saturn all s
['Earth-Mars' , 26.6 , 0.65 , 139.34 , 25.32 , 25.32], #Earth all dmag Mars all s
['Earth-Saturn' , 22.8 , 0.29 , 17.06 , 1.75 , 1.75], # Earth all s Saturn all s
['Earth-Uranus' , 26.13 , 0.76 , 130.77 , 2.26 , 2.26], #Earth all dmag Uranus all s
['Earth-Neptune' , 27.12 , 0.53 , 148.27 , 1.0 , 1.0], #Earth all dmag Neptune all s
['Mars-Jupiter (1)' , 27.48 , 1.48 , 76.61 , 163.45 , 16.55], #Jupiter: left s-3sigma, s-2sigma, s-1sigma  right dmag-1sigma, dmag-2sigma, s+3sigma, Mars: 
['Mars-Jupiter (2)' , 32.98 , 0.27 , 169.67 , 176.99 , 3.01], #intersections, all dmag intersections
['Mars-Uranus' , 26.2 , 0.0 , 0.0 , 0.0 , 0.0], #Both all s
['Mars-Neptune', 27.22 , 1.38 , 65.07 , 2.64 , 2.64],
['Jupiter-Saturn' , 22.83 , 4.73 , 114.64 , 29.58 , 29.58], #Both All s
['Jupiter-Uranus' , 26.1 , 2.18 , 155.24 , 6.52 , 6.52], #Both all s
['Jupiter-Neptune' , 27.19 , 1.58 , 162.36 , 3.01 , 3.01], #Jupiter all s Neptune: all s
['Saturn-Uranus' , 26.19 , 5.47 , 145.21 , 16.55 , 16.55], #Both all s
['Saturn-Neptune' , 27.27 , 4.35 , 153.02 , 8.32 , 8.32], #Both all s
['Uranus-Neptune' , 27.66 , 19.16 , 93.89 , 39.61 , 39.61]] #Uranus s-3sigma s-2sigma s-1sigma dmag-1sigma s-2sigma s-3sigma Neptune all s
# mercury -0
# venus -1
# earth -2
# mars -3
# jupiter -4
# saturn -5
# uranus -6
# neptune -7

#Create a dictionary of the intersection types
#encodes what line each sigma intersects [[6 smaller sma planet intersections],[6 larger sma planet intersections]]
#each pair blongs to a sigma. the pair is ordered from smallest beta to largest beta (counter-clockwise) [[1sigma 1sigma],[2sigma 2sigma],[3sigma 3sigma]]
#encoding reads as s or d for separation or dmag and + or - for lower or upper and a or b indicates beta is (approximately) above or below 90deg. example 'd-a' means the intersection is dmag-uncertainty_dmag and 90deg<beta<180deg
sigmaIntTypeDict = dict() #[smaller planet 3sigma,]
sigmaIntTypeDict[(0,1)] = [[['d-b','d+b'],['d-b','d+b'],['d-b','d+b']],[['d-a','s-a'],['d-a','s-a'],['d-a','s-a']]] #OK
sigmaIntTypeDict[(0,2)] = [[['d-a','d+a'],['d-a','d+a'],['d-a','d+a']],[['d-a','d+a'],['d-a','d+a'],['d-a','d+a']]] #OK
sigmaIntTypeDict[(0,3)] = [[['d-b','d+b'],['d-b','d+b'],['d-b','d+a']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK
sigmaIntTypeDict[(0,6)] = [[['d-b','d+b'],['d-b','d+b'],['d-b','d+b']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK
sigmaIntTypeDict[(0,7)] = [[['d-a','d+a'],['d-a','d+a'],['d-b','d+a']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK

sigmaIntTypeDict[(1,2)] = [[['d-b','d+a'],['d-b','d+a'],['d-b','d+a']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK
sigmaIntTypeDict[(1,5)] = [[['d-b','d+b'],['d-b','d+b'],['d-b','d+a']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK

sigmaIntTypeDict[(2,3)] = [[['d-a','d+a'],['d-a','d+a'],['d-a','d+a']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK
sigmaIntTypeDict[(2,5)] = [[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK
sigmaIntTypeDict[(2,6)] = [[['d-a','d+a'],['d-a','d+a'],['d-a','d+a']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK
sigmaIntTypeDict[(2,7)] = [[['d-a','d+a'],['d-a','d+a'],['d-a','d+a']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK

sigmaIntTypeDict[(3,4)] = [[[['d-b','d+b'],['s-b','d+b'],['s-b','d+b']],[['s+a','s-a'],['s+a','s-a'],['s+a','s-a']]], # JUPITER 0 OK
    [[['d-a','d+a'],['d-a','d+a'],['d-a','d+a']],[['d-a','d+a'],['d-a','d+a'],['d-a','d+a']]]] #the high dmag one #OK JUPITER 1
sigmaIntTypeDict[(3,6)] = [[['s+b','s+b'],['s+b','s+b'],['s+b','s+b']],[['s+b','s+b'],['s+b','s+b'],['s+b','s+b']]] #OK
sigmaIntTypeDict[(3,7)] = [[['s-b','s+b'],['s-b','s+b'],['s-b','d+b']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK

sigmaIntTypeDict[(4,5)] = [[['s+a','s-a'],['s+a','s-a'],['s+a','s-a']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK
sigmaIntTypeDict[(4,6)] = [[['s+a','s-a'],['s+a','s-a'],['s+a','s-a']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK
sigmaIntTypeDict[(4,7)] = [[['s+a','s-a'],['s+a','s-a'],['s+a','s-a']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK

sigmaIntTypeDict[(5,6)] = [[['s+a','s-a'],['s+a','s-a'],['s+a','s-a']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK
sigmaIntTypeDict[(5,7)] = [[['s+a','s-a'],['s+a','s-a'],['s+a','s-a']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK

sigmaIntTypeDict[(6,7)] = [[['d-b','s-a'],['s-b','s-a'],['s-b','s-a']],[['s-b','s+b'],['s-b','s+b'],['s-b','s+b']]] #OK


#Define Phase Function Inverse For Eacg Planet
from scipy.interpolate import PchipInterpolator
import scipy.integrate as integrate
for i in np.arange(len(planets)):
    betas = np.linspace(start=0.,stop=180.,num=1000,endpoint=True)
    if i == 1: #VENUS MUST BE DONE PIECEWISE
        #need to find the betas of local extrema for Venus phase function.... sigh
        from scipy.signal import argrelextrema
        betas = np.linspace(start=0.,stop=180.,num=10**6,endpoint=True) #in deg
        indlmax = argrelextrema(planProp[planets[1]]['phaseFuncMelded'](betas), np.greater)[0]# for local maxima
        dmaglmax = planProp[planets[1]]['phaseFuncMelded'](betas[indlmax[1]])
        indlmin = argrelextrema(planProp[planets[1]]['phaseFuncMelded'](betas), np.less)[0] # for local minima
        dmaglmin = planProp[planets[1]]['phaseFuncMelded'](betas[indlmin[0]])

        #Create 3 independent beta ranges
        beta0 = np.linspace(start=betas[indlmax[0]],stop=betas[indlmin[0]-1],num=2000) # 0->localmin
        beta1 = np.linspace(start=betas[indlmin[0]+1],stop=betas[indlmax[1]-1],num=2000) # localmin->localmax
        beta2 = np.linspace(start=betas[indlmax[1]+1],stop=betas[indlmin[1]],num=2000) # localmax -> pi
        Phis0 = planProp[planets[i]]['phaseFuncMelded'](beta0)
        Phis1 = planProp[planets[i]]['phaseFuncMelded'](beta1)
        Phis2 = planProp[planets[i]]['phaseFuncMelded'](beta2)

        #DELETE
        # plt.figure(53548431)
        # plt.plot(beta0,-Phis0)
        # plt.show(block=False)
        # plt.figure(53548431)
        # plt.plot(beta2,-Phis2)
        # plt.show(block=False)

        planProp[planets[i]]['phaseFuncMeldedInverse'] = [PchipInterpolator(-Phis0,beta0),\
            PchipInterpolator(Phis1,beta1),\
            PchipInterpolator(-Phis2,beta2)] #the -Phis ensure the function monotonically increases NOTE THAT PHIS1 MUST BE THE OPPOSITE SIGN
        planProp[planets[i]]['threedmagIntLimits'] = [dmaglmin,dmaglmax]
        planProp[planets[i]]['threebetaLimits'] = [betas[indlmin[1]],betas[indlmax[1]]]

    else:
        #Phis = self.calc_Phi(betas,np.asarray([])) #TODO: Redefine for compatability with whichPlanetPhaseFunction Input realSolarSystemPhaseFunc
        Phis = planProp[planets[i]]['phaseFuncMelded'](betas)
        indlmax = argrelextrema(Phis, np.greater)[0]# for local maxima
        if len(indlmax) > 0:
            #DELETE
            # plt.figure(53548431)
            # plt.plot(betas[indlmax[0]:],-Phis[indlmax[0]:])#[indlmax[0]:])
            # plt.title(str(i))
            # plt.show(block=False)
            planProp[planets[i]]['phaseFuncMeldedInverse'] = PchipInterpolator(-Phis[indlmax[0]:],betas[indlmax[0]:]) #the -Phis ensure the function monotonically increases
        else:
            planProp[planets[i]]['phaseFuncMeldedInverse'] = PchipInterpolator(-Phis,betas) #the -Phis ensure the function monotonically increases










num=8888888883333
plt.close(num)
fig67 = plt.figure(num=num)

#Plot Central Line
betas = np.linspace(start=0.,stop=180.,num=1000)
for i in np.arange(len(planets)):
    dmags69 = deltaMag(planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m, planProp[planets[i]]['a']*u.m, planProp[planets[i]]['phaseFuncMelded'](betas))
    seps69 = separation_from_alpha_ap(betas*np.pi/180.,planProp[planets[i]]['a']*u.m).to('AU').value
    plt.plot(seps69,dmags69,color=planProp[planets[i]]['planet_labelcolors'],label=planProp[planets[i]]['planet_name'].capitalize())

#Plot all intersections
for k in [(0,1),(0,2),(0,3),(0,6),(0,7),(1,2),(1,5),(2,3),(2,5),(2,6),(2,7),(3,4),(3,6),(3,7),(4,5),(4,6),(4,7),(5,6),(5,7),(6,7)]: #iterate over planet pairs
    i = k[0]
    j = k[1]
    eqnDmagLHS = eqnDmag.subs(Phi,symbolicPhases[i]).subs(a,planProp[planets[i]]['a']).subs(R,planProp[planets[i]]['R']).subs(p,planProp[planets[i]]['p'])
    if i==3 and j==4:
        ik =  incDict[i,j][incDict[i,j]['optNum']]['inc_range'][incDict[i,j]['maxIncInd_Opt']]
        ask = incDict[i,j][incDict[i,j]['optNum']]['v1'][incDict[i,j]['maxIncInd_Opt']]
        alk = incDict[i,j][incDict[i,j]['optNum']]['v2'][incDict[i,j]['maxIncInd_Opt']]
        dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])

        plt.scatter(sk,dmagk,color='black')
        plt.plot([sk-uncertainty_s,sk-uncertainty_s,sk+uncertainty_s,sk+uncertainty_s,sk-uncertainty_s],\
            [dmagk+uncertainty_dmag,dmagk-uncertainty_dmag,dmagk-uncertainty_dmag,dmagk+uncertainty_dmag,dmagk+uncertainty_dmag],color='darkblue',linewidth=0.5)
        plt.plot([sk-2.*uncertainty_s,sk-2.*uncertainty_s,sk+2.*uncertainty_s,sk+2.*uncertainty_s,sk-2.*uncertainty_s],\
            [dmagk+2.*uncertainty_dmag,dmagk-2.*uncertainty_dmag,dmagk-2.*uncertainty_dmag,dmagk+2.*uncertainty_dmag,dmagk+2.*uncertainty_dmag],color='blue',linewidth=0.5)
        plt.plot([sk-3.*uncertainty_s,sk-3.*uncertainty_s,sk+3.*uncertainty_s,sk+3.*uncertainty_s,sk-3.*uncertainty_s],\
            [dmagk+3.*uncertainty_dmag,dmagk-3.*uncertainty_dmag,dmagk-3.*uncertainty_dmag,dmagk+3.*uncertainty_dmag,dmagk+3.*uncertainty_dmag],color='skyblue',linewidth=0.5)

        ik =  incDict[i,j][incDict[i,j]['optNum']]['inc_range'][12]
        ask = incDict[i,j][incDict[i,j]['optNum']]['v1'][12]
        alk = incDict[i,j][incDict[i,j]['optNum']]['v2'][12]
        dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])
    elif i==4 and j==5:
        ik =  incDict[i,j][incDict[i,j]['optNum']]['inc_range'][incDict[i,j]['maxIncInd_Opt']]
        ask = 118.238
        alk = 28.585
        # ask = incDict[i,j][incDict[i,j]['optNum']]['v1'][incDict[i,j]['maxIncInd_Opt']]
        # alk = incDict[i,j][incDict[i,j]['optNum']]['v2'][incDict[i,j]['maxIncInd_Opt']]
        dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])

        # ik =  0.75 #incDict[i,j][incDict[i,j]['optNum']]['inc_range'][incDict[i,j]['maxIncInd_Opt']]
        
        # #ask = 178.55 #incDict[i,j][incDict[i,j]['optNum']]['v1'][incDict[i,j]['maxIncInd_Opt']]
        # #alk = 179.20 #incDict[i,j][incDict[i,j]['optNum']]['v2'][incDict[i,j]['maxIncInd_Opt']]
        # ik =  incDict[i,j][incDict[i,j]['optNum']]['inc_range'][incDict[i,j]['maxIncInd_Opt']]
        # ask = incDict[i,j][incDict[i,j]['optNum']]['v1'][incDict[i,j]['maxIncInd_Opt']]
        # alk = incDict[i,j][incDict[i,j]['optNum']]['v2'][incDict[i,j]['maxIncInd_Opt']]
        dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])
    else:
        ik =  incDict[i,j][incDict[i,j]['optNum']]['inc_range'][incDict[i,j]['maxIncInd_Opt']]
        ask = incDict[i,j][incDict[i,j]['optNum']]['v1'][incDict[i,j]['maxIncInd_Opt']]
        alk = incDict[i,j][incDict[i,j]['optNum']]['v2'][incDict[i,j]['maxIncInd_Opt']]
        dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])

    plt.scatter(sk,dmagk,color='black')
    plt.plot([sk-uncertainty_s,sk-uncertainty_s,sk+uncertainty_s,sk+uncertainty_s,sk-uncertainty_s],\
        [dmagk+uncertainty_dmag,dmagk-uncertainty_dmag,dmagk-uncertainty_dmag,dmagk+uncertainty_dmag,dmagk+uncertainty_dmag],color='darkblue',linewidth=0.5)
    plt.plot([sk-2.*uncertainty_s,sk-2.*uncertainty_s,sk+2.*uncertainty_s,sk+2.*uncertainty_s,sk-2.*uncertainty_s],\
        [dmagk+2.*uncertainty_dmag,dmagk-2.*uncertainty_dmag,dmagk-2.*uncertainty_dmag,dmagk+2.*uncertainty_dmag,dmagk+2.*uncertainty_dmag],color='blue',linewidth=0.5)
    plt.plot([sk-3.*uncertainty_s,sk-3.*uncertainty_s,sk+3.*uncertainty_s,sk+3.*uncertainty_s,sk-3.*uncertainty_s],\
        [dmagk+3.*uncertainty_dmag,dmagk-3.*uncertainty_dmag,dmagk-3.*uncertainty_dmag,dmagk+3.*uncertainty_dmag,dmagk+3.*uncertainty_dmag],color='skyblue',linewidth=0.5)

#plt.scatter(planProp[planets[i]]['s'],planProp[planets[i]]['dmag'],color='black',s=16,zorder=50) #Used for checking spacing
plt.xlim([1e-1,0.5])
plt.xscale('log')
plt.gcf().canvas.draw()
plt.show(block=False)
plt.gcf().canvas.draw()

#ADD SMIN FOR TELESCOPE
smin_telescope = IWA_HabEx.to('rad').value*10.*u.pc.to('AU') #IWA for HabEx 45 mas observing target at 10 pc
plt.plot([smin_telescope,smin_telescope],[10.,70.],color='black',linestyle='-')
smin_telescope2 = IWA2.to('rad').value*10.*u.pc.to('AU') #IWA for HabEx 45 mas observing target at 10 pc
plt.plot([smin_telescope2,smin_telescope2],[10.,70.],color='black',linestyle='-')

#plt.text(7,19.5,'Credit: Dean Keithly',fontsize='small',fontweight='normal')
plt.text(1.05*smin_telescope,41, str(int(IWA_HabEx.value*1000)) + ' mas\nat 10 pc',fontsize='medium',fontweight='bold',rotation=90)
plt.text(1.05*smin_telescope2,41, str(int(IWA2.value*1000)) + ' mas\nat 10 pc',fontsize='medium',fontweight='bold',rotation=90)
plt.xlim([1e-1,32.])
plt.ylim([19.,46.])
plt.ylabel('Planet-Star ' + r'$\Delta \mathrm{mag}$', weight='bold')
plt.xlabel('Projected Planet-Star Separation, ' + r'$s$,' +' in AU', weight='bold')
plt.legend(loc=1)
#plt.title('Inclination: ' + str(np.round(90-inclination,1)) + r'$^\circ$' ,weight='bold')
plt.gcf().canvas.draw()
plt.show(block=False)
plt.gcf().canvas.draw()
#print('Done with planet: ' + str(planets[i]))












#### Calculate Probability
from astropy import constants as const
#from exodetbox.trueAnomalyFromEccentricAnomaly import trueAnomalyFromEccentricAnomaly
#from exodetbox.projectedEllipse import *
for k in [(0,1),(0,2),(0,3),(0,6),(0,7),(1,2),(1,5),(2,3),(2,5),(2,6),(2,7),(3,4),(3,6),(3,7),(4,5),(4,6),(4,7),(5,6),(5,7),(6,7)]: #iterate over planet pairs
    i = k[0]
    j = k[1]
    eqnDmagLHS = eqnDmag.subs(Phi,symbolicPhases[i]).subs(a,planProp[planets[i]]['a']).subs(R,planProp[planets[i]]['R']).subs(p,planProp[planets[i]]['p'])
    if i==3 and j==4:
        ik =  incDict[i,j][incDict[i,j]['optNum']]['inc_range'][incDict[i,j]['maxIncInd_Opt']]
        ask = incDict[i,j][incDict[i,j]['optNum']]['v1'][incDict[i,j]['maxIncInd_Opt']]
        alk = incDict[i,j][incDict[i,j]['optNum']]['v2'][incDict[i,j]['maxIncInd_Opt']]
        dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])
        sk_34 = sk
        dmagk_34 = dmagk


        ik =  incDict[i,j][incDict[i,j]['optNum']]['inc_range'][12]
        ask = incDict[i,j][incDict[i,j]['optNum']]['v1'][12]
        alk = incDict[i,j][incDict[i,j]['optNum']]['v2'][12]
        dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])
    elif i==4 and j==5:
        # ik =  incDict[i,j][incDict[i,j]['optNum']]['inc_range'][incDict[i,j]['maxIncInd_Opt']]
        # ask = incDict[i,j][incDict[i,j]['optNum']]['v1'][incDict[i,j]['maxIncInd_Opt']]
        # alk = incDict[i,j][incDict[i,j]['optNum']]['v2'][incDict[i,j]['maxIncInd_Opt']]
        # dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        # sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        # ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])

        ask = 118.238
        alk = 28.585
        # ask = incDict[i,j][incDict[i,j]['optNum']]['v1'][incDict[i,j]['maxIncInd_Opt']]
        # alk = incDict[i,j][incDict[i,j]['optNum']]['v2'][incDict[i,j]['maxIncInd_Opt']]
        dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])
    else:
        ik =  incDict[i,j][incDict[i,j]['optNum']]['inc_range'][incDict[i,j]['maxIncInd_Opt']]
        ask = incDict[i,j][incDict[i,j]['optNum']]['v1'][incDict[i,j]['maxIncInd_Opt']]
        alk = incDict[i,j][incDict[i,j]['optNum']]['v2'][incDict[i,j]['maxIncInd_Opt']]
        dmagk = float(eqnDmagLHS.subs(alpha,ask).evalf())
        sk = float(eqnSAlpha.subs(a,planProp[planets[i]]['a']*u.m.to('AU')).subs(alpha,ask).evalf())
        ik2 = np.min([np.min([ask,180.-ask]),np.min([alk,180.-alk])])

    
    #ik2
    #sk
    #dmagk

    ####
    #[(i,j) determines planet pair][0 smaller or 1 larger sma planet][clockwise intersection start 0 or stop 1]
    intbounds = sigmaIntTypeDict[(i,j)] #The types of intersections the planet makes with the various sigma bounds
    intTypes = intbounds[0][0]

    def limitingIncFuncs(intTypes, sk, dmagk, uncertainty_dmag, uncertainty_s, sma, p, R, pm, PhiInv):
        limitingIncs = list()
        for intType in intTypes:
            if '+' in intType:
                pm = 1.
            else: #'-' in intType
                pm = -1.

            if 's' in intType: #Calculate the time of this orbit intersection using the separation form
                #it shouldn't matter whether beta is 0<beta<pi/2 or pi/2<beta<pi
                inc_tmp = np.arcsin(np.cos(np.arcsin((sk+pm*uncertainty_s)/(sma)))) #*180./np.pi
                # if 'a' in intType: #the intersection occurs where beta>90deg
                #     #will need to substitute in np.pi - beta in the relevant bit of code
                #     inc_tmp = np.arcsin(np.cos(np.arcsin((sk+pm*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU')))))    
                #     #beta = np.pi-np.arcsin((sk + pm*uncertainty_s)/sma)
                #     #ti = (np.pi - np.arcsin(np.cos( beta )/np.sin(inc2)))*periods/(2.*np.pi)
                # else: #'b' in intType
                #     #Use the normal form, 0<beta<np.pi/2
            else: #'d' is in the 
                inc_tmp = np.arcsin(np.cos( np.pi/180.*PhiInv(- sma**2.* 10**(-(dmagk + pm*uncertainty_dmag)/2.5) /(p * R**2.)) ) ) #*180./np.pi
                #beta = PhiInv(- sma**2.* 10**(-(dmagk + pm*uncertainty_dmag)/2.5) /(p * R**2.))
                #DO MODIFICATIONS HERE TO INCLINATION CALCULATION
            limitingIncs.append(inc_tmp)
        return np.asarray(limitingIncs)
        
    if i == 1:
        limitingIncsi = limitingIncFuncs(intTypes, sk, dmagk, uncertainty_dmag, uncertainty_s, planProp[planets[i]]['a']*u.m.to('AU'), planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m.to('AU'), pm, planProp[planets[i]]['phaseFuncMeldedInverse'][0])
    else:
        limitingIncsi = limitingIncFuncs(intTypes, sk, dmagk, uncertainty_dmag, uncertainty_s, planProp[planets[i]]['a']*u.m.to('AU'), planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m.to('AU'), pm, planProp[planets[i]]['phaseFuncMeldedInverse'])

    if j == 1:
        limitingIncsj = limitingIncFuncs(intTypes, sk, dmagk, uncertainty_dmag, uncertainty_s, planProp[planets[j]]['a']*u.m.to('AU'), planProp[planets[j]]['p'], planProp[planets[j]]['R']*u.m.to('AU'), pm, planProp[planets[j]]['phaseFuncMeldedInverse'][0])
    else:
        limitingIncsj = limitingIncFuncs(intTypes, sk, dmagk, uncertainty_dmag, uncertainty_s, planProp[planets[j]]['a']*u.m.to('AU'), planProp[planets[j]]['p'], planProp[planets[j]]['R']*u.m.to('AU'), pm, planProp[planets[j]]['phaseFuncMeldedInverse'])


    #THIS PLOT VERIFIES LIMITINGINCS ARE CALCULATED CORRECTLY
    nus = np.linspace(start=0.,stop=2.*np.pi,num=200)
    limitingBeta0 = np.arccos(np.sin(limitingIncsi[0])*np.sin(nus))
    limitingBeta1 = np.arccos(np.sin(limitingIncsi[1])*np.sin(nus))
    #betas0 = np.linspace(start=np.min(limitingBeta0,limitingBeta1),stop=np.max(limitingBeta0,limitingBeta1),num=300)
    # seps0 = planProp[planets[i]]['a']*u.m.to('AU')*np.sin(limitingBeta0)
    # dmags0 = -2.5*np.log10(planProp[planets[i]]['p']*(planProp[planets[i]]['R']*u.m.to('AU'))**2./(planProp[planets[i]]['a']*u.m.to('AU'))**2.*planProp[planets[i]]['phaseFuncMelded'](limitingBeta0))
    seps0 = separation_from_alpha_ap(limitingBeta0,planProp[planets[i]]['a']*u.m).to('AU').value#planProp[planets[i]]['a']*u.m.to('AU')*np.sin(limitingBeta1)
    dmags0 = deltaMag(planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m, planProp[planets[i]]['a']*u.m, planProp[planets[i]]['phaseFuncMelded'](limitingBeta0*180./np.pi))#-2.5*np.log10(planProp[planets[i]]['p']*(planProp[planets[i]]['R']*u.m.to('AU'))**2./(planProp[planets[i]]['a']*u.m.to('AU'))**2.*planProp[planets[i]]['phaseFuncMelded'](limitingBeta1))
    seps1 = separation_from_alpha_ap(limitingBeta1,planProp[planets[i]]['a']*u.m).to('AU').value#planProp[planets[i]]['a']*u.m.to('AU')*np.sin(limitingBeta1)
    dmags1 = deltaMag(planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m, planProp[planets[i]]['a']*u.m, planProp[planets[i]]['phaseFuncMelded'](limitingBeta1*180./np.pi))#-2.5*np.log10(planProp[planets[i]]['p']*(planProp[planets[i]]['R']*u.m.to('AU'))**2./(planProp[planets[i]]['a']*u.m.to('AU'))**2.*planProp[planets[i]]['phaseFuncMelded'](limitingBeta1))
    plt.figure(54354832131)
    plt.plot(seps0,dmags0)
    plt.plot(seps1,dmags1)
    plt.scatter(sk,dmagk)
    indOfMin = np.argmin(np.abs(np.pi/2.-nus))
    plt.scatter(seps0[indOfMin],dmags0[indOfMin],color='red')
    plt.plot([sk-uncertainty_s,sk-uncertainty_s,sk+uncertainty_s,sk+uncertainty_s,sk-uncertainty_s],[dmagk+uncertainty_dmag,dmagk-uncertainty_dmag,dmagk-uncertainty_dmag,dmagk+uncertainty_dmag,dmagk+uncertainty_dmag],color='darkblue')
    plt.show(block=False)

    #tmp_inc = np.arcsin(np.cos( np.arcsin(sk/(planProp[planets[i]]['a']*u.m.to('AU'))) ))
    # there is an obvious issue with using the ik and ik2 as inclinations in beta=np.arccos(np.sin(i)), the core of all of these calculations for computing the limiting inclinations
    # currently, I think I need to use arccos. but this doesnt make ligical sense to me. At the end of the day, we need to convert from the system inclination in table2 to the phase angle in table 2

    # plt.figure(2124411122223354)
    # y = np.linspace(start=0.,stop=np.pi,num=100)
    # plt.plot(y,np.pi - np.arcsin(np.cos(y)))
    # plt.show(block=False)

    # plt.figure(6845115566)
    # y2 = np.linspace(start=0.,stop=1.,num=100)
    # plt.plot(y2,planProp[planets[i]]['phaseFuncMeldedInverse'](y2))
    # plt.show(block=False)

    # plt.figure(345347766)
    # y3 = np.linspace(start=0.,stop=180.,num=100)
    # plt.plot(y3,planProp[planets[i]]['phaseFuncMelded'](y3))
    # plt.show(block=False)

    # plt.figure(345347766)
    # y3 = np.linspace(start=0.,stop=180.,num=100)
    # plt.plot(-planProp[planets[i]]['phaseFuncMelded'](y3),y3)
    # plt.show(block=False)

    # tmp = planProp[planets[i]]['phaseFuncMeldedInverse'](-planProp[planets[i]]['phaseFuncMelded'](y3))

    #### Generate Population of Planets
    tmp_num= 10**6
    M0 = np.random.uniform(2.*np.pi, size=int(tmp_num)) #Mean Anomaly
    #M = E - e sin(E) SO FOR CIRCULAR, M=E
    E0 = M0
    # tan(v/2) = sqrt((1+e)/(1-e)) tan(E/2), SO FOR CIRCULAR, nu=E
    nu0 = E0

    #For Planet i
    maxIncDeltai = np.max(np.abs(np.pi/2.-limitingIncsi))
    minInci = np.min(limitingIncsi)
    C = 0.5*(np.cos(minInci*u.rad) - np.cos(np.pi/2.*u.rad))# + ik2*u.deg))
    tmp_inci = ((np.arccos(np.cos(minInci*u.rad) - 2.*C*np.random.uniform(size=tmp_num))).to('rad')).value

    tmp_smai = np.ones(tmp_num)*planProp[planets[i]]['a']*u.m.to('AU')
    starMass = const.M_sun
    periods = (2.*np.pi*np.sqrt((tmp_smai*u.AU)**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value #calculate periods for all planets
    tmp_ei = np.zeros([tmp_num])
    tmp_wi = np.zeros([tmp_num])
    tmp_Wi = np.zeros([tmp_num])
    tmp_pi = np.ones(tmp_num)*planProp[planets[i]]['p']
    tmp_Rpi = np.ones(tmp_num)*planProp[planets[i]]['R']*u.m.to('AU')

    tmp_betasi = np.arccos(np.sin(tmp_inci)*np.sin(nu0))*180./np.pi #in deg
    
    s0i = separation_from_alpha_ap(tmp_betas*np.pi/180.,tmp_smai)
    dmag0i = deltaMag(tmp_pi, tmp_Rpi*u.AU, tmp_smai*u.AU, planProp[planets[i]]['phaseFuncMelded'](tmp_betas))

    #### Number of Planets i Within 1 sigma, Monte Carlo
    inds1Sigmai = np.where((np.abs(s0i-sk) < 1.*uncertainty_s)*(np.abs(dmag0i-dmagk) < 1.*uncertainty_dmag))[0]
    inds2Sigmai = np.where((np.abs(s0i-sk) < 2.*uncertainty_s)*(np.abs(dmag0i-dmagk) < 2.*uncertainty_dmag))[0]
    inds3Sigmai = np.where((np.abs(s0i-sk) < 3.*uncertainty_s)*(np.abs(dmag0i-dmagk) < 3.*uncertainty_dmag))[0]
    fracIn1Sigmai = len(inds1Sigmai)/tmp_num
    fracIn2Sigmai = len(inds2Sigmai)/tmp_num
    fracIn3Sigmai = len(inds3Sigmai)/tmp_num

    #For Planet j
    maxIncDeltaj = np.max(np.abs(np.pi/2.-limitingIncsj))
    minIncj = np.min(limitingIncsj)
    C = 0.5*(np.cos(minIncj*u.rad) - np.cos(np.pi/2.*u.rad))# + ik2*u.deg))
    tmp_incj = ((np.arccos(np.cos(minIncj*u.rad) - 2.*C*np.random.uniform(size=tmp_num))).to('rad')).value

    tmp_smaj = np.ones(tmp_num)*planProp[planets[j]]['a']*u.m.to('AU')
    periods = (2.*np.pi*np.sqrt((tmp_smaj*u.AU)**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value #calculate periods for all planets
    tmp_ej = np.zeros([tmp_num])
    tmp_wj = np.zeros([tmp_num])
    tmp_Wj = np.zeros([tmp_num])
    tmp_pj = np.ones(tmp_num)*planProp[planets[j]]['p']
    tmp_Rpj = np.ones(tmp_num)*planProp[planets[j]]['R']*u.m.to('AU')

    tmp_betasj = np.arccos(np.sin(tmp_incj)*np.sin(nu0))*180./np.pi #in deg
    
    s0j = separation_from_alpha_ap(tmp_betasj*np.pi/180.,tmp_smaj)
    dmag0j = deltaMag(tmp_pj, tmp_Rpj*u.AU, tmp_smaj*u.AU, planProp[planets[j]]['phaseFuncMelded'](tmp_betasj))

    #### Number of Planets i Within 1 sigma, Monte Carlo
    inds1Sigmaj = np.where((np.abs(s0j-sk) < 1.*uncertainty_s)*(np.abs(dmag0j-dmagk) < 1.*uncertainty_dmag))[0]
    inds2Sigmaj = np.where((np.abs(s0j-sk) < 2.*uncertainty_s)*(np.abs(dmag0j-dmagk) < 2.*uncertainty_dmag))[0]
    inds3Sigmaj = np.where((np.abs(s0j-sk) < 3.*uncertainty_s)*(np.abs(dmag0j-dmagk) < 3.*uncertainty_dmag))[0]
    fracIn1Sigmaj = len(inds1Sigmaj)/tmp_num
    fracIn2Sigmaj = len(inds2Sigmaj)/tmp_num
    fracIn3Sigmaj = len(inds3Sigmaj)/tmp_num

    print('(i,j): (' + str(i) + ',' + str(j) + ')' + ' & ' + str(np.round(fracIn1Sigmai*100.,1)) + ' & ' + str(np.round(fracIn1Sigmaj*100.,1)) + ' & ' + str(np.round(fracIn2Sigmai*100.,1)) + ' & ' + str(np.round(fracIn2Sigmaj*100.,1)) + ' & ' + str(np.round(fracIn3Sigmai*100.,1)) + ' & ' + str(np.round(fracIn3Sigmaj*100.,1)))


    #We need to rerun for the second intersection
    if (i,j) == (3,4):
        limitingIncsi = limitingIncFuncs(intTypes, sk_34, dmagk_34, uncertainty_dmag, uncertainty_s, planProp[planets[i]]['a']*u.m.to('AU'), planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m.to('AU'), pm, planProp[planets[i]]['phaseFuncMeldedInverse'])
        limitingIncsj = limitingIncFuncs(intTypes, sk_34, dmagk_34, uncertainty_dmag, uncertainty_s, planProp[planets[j]]['a']*u.m.to('AU'), planProp[planets[j]]['p'], planProp[planets[j]]['R']*u.m.to('AU'), pm, planProp[planets[j]]['phaseFuncMeldedInverse'])

        #For Planet i
        maxIncDeltai = np.max(np.abs(np.pi/2.-limitingIncsi))
        minInci = np.min(limitingIncsi)
        C = 0.5*(np.cos(minInci*u.rad) - np.cos(np.pi/2.*u.rad))# + ik2*u.deg))
        tmp_inci = ((np.arccos(np.cos(minInci*u.rad) - 2.*C*np.random.uniform(size=tmp_num))).to('rad')).value

        tmp_smai = np.ones(tmp_num)*planProp[planets[i]]['a']*u.m.to('AU')
        starMass = const.M_sun
        periods = (2.*np.pi*np.sqrt((tmp_smai*u.AU)**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value #calculate periods for all planets
        tmp_ei = np.zeros([tmp_num])
        tmp_wi = np.zeros([tmp_num])
        tmp_Wi = np.zeros([tmp_num])
        tmp_pi = np.ones(tmp_num)*planProp[planets[i]]['p']
        tmp_Rpi = np.ones(tmp_num)*planProp[planets[i]]['R']*u.m.to('AU')

        tmp_betasi = np.arccos(np.sin(tmp_inci)*np.sin(nu0))*180./np.pi #in deg
        
        s0i = separation_from_alpha_ap(tmp_betas*np.pi/180.,tmp_smai)
        dmag0i = deltaMag(tmp_pi, tmp_Rpi*u.AU, tmp_smai*u.AU, planProp[planets[i]]['phaseFuncMelded'](tmp_betas))

        #### Number of Planets i Within 1 sigma, Monte Carlo
        inds1Sigmai = np.where((np.abs(s0i-sk_34) < 1.*uncertainty_s)*(np.abs(dmag0i-dmagk_34) < 1.*uncertainty_dmag))[0]
        inds2Sigmai = np.where((np.abs(s0i-sk_34) < 2.*uncertainty_s)*(np.abs(dmag0i-dmagk_34) < 2.*uncertainty_dmag))[0]
        inds3Sigmai = np.where((np.abs(s0i-sk_34) < 3.*uncertainty_s)*(np.abs(dmag0i-dmagk_34) < 3.*uncertainty_dmag))[0]
        fracIn1Sigmai = len(inds1Sigmai)/tmp_num
        fracIn2Sigmai = len(inds2Sigmai)/tmp_num
        fracIn3Sigmai = len(inds3Sigmai)/tmp_num

        #For Planet j
        maxIncDeltaj = np.max(np.abs(np.pi/2.-limitingIncsj))
        minIncj = np.min(limitingIncsj)
        C = 0.5*(np.cos(minIncj*u.rad) - np.cos(np.pi/2.*u.rad))# + ik2*u.deg))
        tmp_incj = ((np.arccos(np.cos(minIncj*u.rad) - 2.*C*np.random.uniform(size=tmp_num))).to('rad')).value

        tmp_smaj = np.ones(tmp_num)*planProp[planets[j]]['a']*u.m.to('AU')
        periods = (2.*np.pi*np.sqrt((tmp_smaj*u.AU)**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value #calculate periods for all planets
        tmp_ej = np.zeros([tmp_num])
        tmp_wj = np.zeros([tmp_num])
        tmp_Wj = np.zeros([tmp_num])
        tmp_pj = np.ones(tmp_num)*planProp[planets[j]]['p']
        tmp_Rpj = np.ones(tmp_num)*planProp[planets[j]]['R']*u.m.to('AU')

        tmp_betasj = np.arccos(np.sin(tmp_incj)*np.sin(nu0))*180./np.pi #in deg
        
        s0j = separation_from_alpha_ap(tmp_betasj*np.pi/180.,tmp_smaj)
        dmag0j = deltaMag(tmp_pj, tmp_Rpj*u.AU, tmp_smaj*u.AU, planProp[planets[j]]['phaseFuncMelded'](tmp_betasj))

        #### Number of Planets i Within 1 sigma, Monte Carlo
        inds1Sigmaj = np.where((np.abs(s0j-sk_34) < 1.*uncertainty_s)*(np.abs(dmag0j-dmagk_34) < 1.*uncertainty_dmag))[0]
        inds2Sigmaj = np.where((np.abs(s0j-sk_34) < 2.*uncertainty_s)*(np.abs(dmag0j-dmagk_34) < 2.*uncertainty_dmag))[0]
        inds3Sigmaj = np.where((np.abs(s0j-sk_34) < 3.*uncertainty_s)*(np.abs(dmag0j-dmagk_34) < 3.*uncertainty_dmag))[0]
        fracIn1Sigmaj = len(inds1Sigmaj)/tmp_num
        fracIn2Sigmaj = len(inds2Sigmaj)/tmp_num
        fracIn3Sigmaj = len(inds3Sigmaj)/tmp_num

        print('(i,j): (' + str(i) + ',' + str(j) + ')' + ' & ' + str(np.round(fracIn1Sigmai*100.,1)) + ' & ' + str(np.round(fracIn1Sigmaj*100.,1)) + ' & ' + str(np.round(fracIn2Sigmai*100.,1)) + ' & ' + str(np.round(fracIn2Sigmaj*100.,1)) + ' & ' + str(np.round(fracIn3Sigmai*100.,1)) + ' & ' + str(np.round(fracIn3Sigmaj*100.,1)))
    ####







    # n=1. #number of sigma
    # ####Classify Intersection Type
    # ilower = np.arcsin(np.cos(np.arcsin((sk+n*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU')))))
    # iupper = np.arcsin(np.cos(np.arcsin((sk-n*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU')))))
    # #classify orbit type
    # indsWhereGTiupper = np.where(tmp_inc > iupper)[0] #bounds are ilower and iupper
    # indsWhereBetween = np.where((tmp_inc < iupper)*(tmp_inc > ilower))[0] #bounds is 0 and ilower
    # indsWhereLTilower = np.where(tmp_inc < ilower)[0] #no intersection
    # #calculate times
    # tlower = np.zeros(len(tmp_inc))
    # tlower[indsWhereGTiupper] = np.arcsin(np.cos(np.arcsin((sk+n*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU'))))/np.sin(tmp_inc[indsWhereGTiupper]))*periods[indsWhereGTiupper]/(2.*np.pi)
    # tlower[indsWhereBetween] = np.arcsin(np.cos(np.arcsin((sk+n*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU'))))/np.sin(tmp_inc[indsWhereBetween]))*periods[indsWhereBetween]/(2.*np.pi)
    # tupper = np.zeros(len(tmp_inc))
    # tlower[indsWhereGTiupper] = np.arcsin(np.cos(np.arcsin((sk-n*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU'))))/np.sin(tmp_inc[indsWhereGTiupper]))*periods[indsWhereGTiupper]/(2.*np.pi)
    # tlower[np.isnan(tlower)] = 0.
    # tupper[np.isnan(tupper)] = 0.
    # dt = 2.*np.abs(tupper - tlower) #multiply by 2 because of symmetry

    # #Running with a sweep
    # inc_sweep = np.linspace(start=0.,stop=np.pi/2.,num=1000)
    # ####Classify Intersection Type
    # #DUPLICATE ilower = np.arcsin(np.cos(np.arcsin((sk+n*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU')))))
    # #DUPLICATE iupper = np.arcsin(np.cos(np.arcsin((sk-n*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU')))))
    # #classify orbit type
    # #ORIGINAL indsWhereGTiupper_sweep = np.where(inc_sweep > iupper)[0] #bounds are ilower and iupper
    # # indsWhereBetween_sweep = np.where((inc_sweep < iupper)*(inc_sweep > ilower))[0] #bounds is 0 and ilower
    # # indsWhereLTilower_sweep = np.where(inc_sweep < ilower)[0] #no intersection
    # indsWhereGTiupper_sweep = np.where(inc_sweep > np.pi/2. - ilower)[0] #bounds are ilower and iupper
    # indsWhereBetween_sweep = np.where((inc_sweep > np.pi/2. - iupper)*(inc_sweep < np.pi/2. - ilower))[0] #bounds is 0 and ilower
    # indsWhereLTilower_sweep = np.where(inc_sweep < iupper)[0] #no intersection
    # #calculate times
    # tlower_sweep = np.zeros(len(inc_sweep))
    # tlower_sweep[indsWhereGTiupper_sweep] = np.arcsin(np.cos(np.arcsin((sk+n*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU'))))/np.sin(inc_sweep[indsWhereGTiupper_sweep]))*periods[indsWhereGTiupper_sweep]/(2.*np.pi)
    # tlower_sweep[indsWhereBetween_sweep] = np.arcsin(np.cos(np.arcsin((sk+n*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU'))))/np.sin(inc_sweep[indsWhereBetween_sweep]))*periods[indsWhereBetween_sweep]/(2.*np.pi)
    # tupper_sweep = np.zeros(len(inc_sweep))
    # tupper_sweep[indsWhereGTiupper_sweep] = np.arcsin(np.cos(np.arcsin((sk-n*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU'))))/np.sin(inc_sweep[indsWhereGTiupper_sweep]))*periods[indsWhereGTiupper_sweep]/(2.*np.pi)
    # tlower_sweep[np.isnan(tlower_sweep)] = 0.
    # tupper_sweep[np.isnan(tupper_sweep)] = 0.
    # dt_sweep = 2.*np.abs(tupper_sweep - tlower_sweep) #multiply by 2 because of symmetry

    # # plt.figure(num=516843138)
    # # plt.scatter(inc_sweep,dt_sweep,color='black')
    # # plt.show(block=False)

    # # #Calculate true anomalies of the limits of the planets
    # # tm1Sigma = np.arcsin(np.cos(np.arcsin((sk-1.*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU'))))/np.sin(tmp_inc))*periods/(2.*np.pi)
    # # tp1Sigma = np.arcsin(np.cos(np.arcsin((sk+1.*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU'))))/np.sin(tmp_inc))*periods/(2.*np.pi)

    # # tm2Sigma = np.arcsin(np.cos(np.arcsin((sk-2.*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU'))))/np.sin(tmp_inc))*periods/(2.*np.pi)
    # # tp2Sigma = np.arcsin(np.cos(np.arcsin((sk+2.*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU'))))/np.sin(tmp_inc))*periods/(2.*np.pi)

    # # tm3Sigma = np.arcsin(np.cos(np.arcsin((sk-3.*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU'))))/np.sin(tmp_inc))*periods/(2.*np.pi)
    # # tp3Sigma = np.arcsin(np.cos(np.arcsin((sk+3.*uncertainty_s)/(planProp[planets[i]]['a']*u.m.to('AU'))))/np.sin(tmp_inc))*periods/(2.*np.pi)

    
    # def fi(inc):
    #     """ The probability density function of inclinations
    #     inc in radians
    #     """
    #     return np.sin(inc)/2.

    # def func1(inc,sma,uncertainty_s,sk,periods):
    #     """For inclinations between the separation uncertainty upper bound and smin of the planet
    #     """
    #     tis = deltatFunc(inc,sma,p,R,periods,sigma_s,uncertainty_dmag,sk,dmagk,intTypes,i)
    #     return fi(inc)*2.*np.abs(tilower-0.)

    # def func2(inc,sma,uncertainty_s,sk,periods):
    #     """ For inclinations between the lower separation uncertainty bound and upper separation uncertainty bound
    #     """
    #     #tisps, tisms, tidmagpdmag, tidmagmdmag = t_ifunc(inc,sma,sigma_s,sk,periods)
    #     tis = deltatFunc(inc,sma,p,R,periods,sigma_s,uncertainty_dmag,sk,dmagk,intTypes,i)
    #     return fi(inc)*2.*np.abs(tis[1]-tis[0])

    # def t_ifunc(inc,sma,sigma_s,sk,periods,dmagk,uncertainty_dmag,i,p,R):
    #     tisps = np.arcsin(np.cos(np.arcsin((sk+sigma_s)/sma))/np.sin(inc))*periods/(2.*np.pi) #tilower
    #     tisms = np.arcsin(np.cos(np.arcsin((sk-sigma_s)/sma))/np.sin(inc))*periods/(2.*np.pi) #tiupper
    #     if i == 1: #its venus
    #         #do tidmagpdmag
    #         if planProp[planets[i]]['threedmagIntLimits'][1] < dmagk + uncertainty_dmag: #beta is super large, near 180 deg
    #             tidmagpdmag = periods/(2.*np.pi) * np.arcsin( 1./np.sin(inc) *np.cos( planProp[planets[i]]['phaseFuncMeldedInverse'][0](- sma**2.* 10**(-(dmagk+uncertainty_dmag)/2.5) /(p * R**2.)) ) )
    #         elif planProp[planets[i]]['threedmagIntLimits'][1] > dmagk + uncertainty_dmag and planProp[planets[i]]['threedmagIntLimits'][0] < dmagk + uncertainty_dmag: #dmag is between local min and local max, all 3 potential intersections must be calculated
    #             tidmagpdmag0 = periods/(2.*np.pi) * np.arcsin( 1./np.sin(inc) *np.cos( planProp[planets[i]]['phaseFuncMeldedInverse'][0](- sma**2.* 10**(-(dmagk+uncertainty_dmag)/2.5) /(p * R**2.)) ) )
    #             tidmagpdmag1 = periods/(2.*np.pi) * np.arcsin( 1./np.sin(inc) *np.cos( planProp[planets[i]]['phaseFuncMeldedInverse'][1](sma**2.* 10**(-(dmagk+uncertainty_dmag)/2.5) /(p * R**2.)) ) )
    #             tidmagpdmag2 = periods/(2.*np.pi) * np.arcsin( 1./np.sin(inc) *np.cos( planProp[planets[i]]['phaseFuncMeldedInverse'][2](- sma**2.* 10**(-(dmagk+uncertainty_dmag)/2.5) /(p * R**2.)) ) )
    #             #NEED TO DO SOME ERROR HANDLING HERE
    #         else:
    #             tidmagpdmag = periods/(2.*np.pi) * np.arcsin( 1./np.sin(inc) *np.cos( planProp[planets[i]]['phaseFuncMeldedInverse'][2](- sma**2.* 10**(-(dmagk+uncertainty_dmag)/2.5) /(p * R**2.)) ) )
            
    #         #do tidmagmdmag
    #         if planProp[planets[i]]['threedmagIntLimits'][1] < dmagk - uncertainty_dmag: #beta is super large, near 180 deg
    #             tidmagmdmag = periods/(2.*np.pi) * np.arcsin( 1./np.sin(inc) *np.cos( planProp[planets[i]]['phaseFuncMeldedInverse'][0](- sma**2.* 10**(-(dmagk-uncertainty_dmag)/2.5) /(p * R**2.)) ) )
    #         elif planProp[planets[i]]['threedmagIntLimits'][1] > dmagk - uncertainty_dmag and planProp[planets[i]]['threedmagIntLimits'][0] < dmagk - uncertainty_dmag: #dmag is between local min and local max, all 3 potential intersections must be calculated
    #             tidmagmdmag0 = periods/(2.*np.pi) * np.arcsin( 1./np.sin(inc) *np.cos( planProp[planets[i]]['phaseFuncMeldedInverse'][0](- sma**2.* 10**(-(dmagk-uncertainty_dmag)/2.5) /(p * R**2.)) ) )
    #             tidmagmdmag1 = periods/(2.*np.pi) * np.arcsin( 1./np.sin(inc) *np.cos( planProp[planets[i]]['phaseFuncMeldedInverse'][1](sma**2.* 10**(-(dmagk-uncertainty_dmag)/2.5) /(p * R**2.)) ) )
    #             tidmagmdmag2 = periods/(2.*np.pi) * np.arcsin( 1./np.sin(inc) *np.cos( planProp[planets[i]]['phaseFuncMeldedInverse'][2](- sma**2.* 10**(-(dmagk-uncertainty_dmag)/2.5) /(p * R**2.)) ) )
    #             #NEED TO DO SOME ERROR HANDLING HERE
    #         else:
    #             tidmagmdmag = periods/(2.*np.pi) * np.arcsin( 1./np.sin(inc) *np.cos( planProp[planets[i]]['phaseFuncMeldedInverse'][2](- sma**2.* 10**(-(dmagk-uncertainty_dmag)/2.5) /(p * R**2.)) ) )
    #     else:
    #         tidmagpdmag = periods/(2.*np.pi)* np.arcsin( 1./np.sin(inc) *np.cos( planProp[planets[i]]['phaseFuncMeldedInverse'](- sma**2.* 10**(-(dmagk+uncertainty_dmag)/2.5) /(p * R**2.) ,planetInd) ) )
    #         tidmagmdmag = periods/(2.*np.pi)* np.arcsin( 1./np.sin(inc) *np.cos( planProp[planets[i]]['phaseFuncMeldedInverse'](- sma**2.* 10**(-(dmagk-uncertainty_dmag)/2.5) /(p * R**2.) ,planetInd) ) )
    #     return tisps, tisms, tidmagpdmag, tidmagmdmag


    # def deltatFunc(inc2,sma,p,R,periods,uncertainty_s,uncertainty_dmag,sk,dmagk,intTypes,i,extraOutputBool):
    #     """
    #     Calculates the time between a planet intersecting two limits
    #     inc
    #     sma
    #     p
    #     R
    #     periods
    #     uncertainty_s
    #     uncertainty_dmag
    #     sk
    #     dmagk
    #     intTypes - the uncertainty intersection type, list of 2 strings fo form 's-a'
    #     i - the planet index
    #     betweenBool - indicates whether the delta is between the two values, or between the values and 0
    #     """
    #     tis = list()
    #     for intType in intTypes:
    #         if '+' in intType:
    #             pm = 1.
    #         else: #'-' in intType
    #             pm = -1.

    #         if 's' in intType: #Calculate the time of this orbit intersection using the separation form
    #             if 'a' in intType: #the intersection occurs where beta>90deg
    #                 #will need to substitute in np.pi - beta in the relevant bit of code
    #                 beta = np.pi-np.arcsin((sk + pm*uncertainty_s)/sma)
    #                 ti = (np.pi - np.arcsin(np.cos( beta )/np.sin(inc2)))*periods/(2.*np.pi)
    #             else: #'b' in intType
    #                 #Use the normal form, 0<beta<np.pi/2
    #                 beta = np.arcsin((sk + pm*uncertainty_s)/sma)
    #                 ti = np.arcsin(np.cos( beta )/np.sin(inc2))*periods/(2.*np.pi)

                
    #             #WILL NEED TO CHECK IF THIS NORMAL ARCSIN IS OK. IT MIGHT NOT NEED ADJUSTING BECAUSE OF THE COS

    #         # tisps = np.arcsin(np.cos(np.arcsin((sk+uncertainty_s)/sma))/np.sin(inc))*periods/(2.*np.pi) #tilower
    #         # tisms = np.arcsin(np.cos( np.pi-np.arcsin((sk-uncertainty_s)/sma) )/np.sin(inc))*periods/(2.*np.pi) #tiupper

    #         else: #'d' in intType
    #             #Use the dmag equation for time
    #             if i == 1: #its venus
    #                 #do tidmagpdmag
    #                 if planProp[planets[i]]['threedmagIntLimits'][1] < dmagk + uncertainty_dmag: #beta is super large, near 180 deg
    #                     ti = periods/(2.*np.pi) * np.arcsin( 1./np.sin(inc2) *np.cos( np.pi/180.*planProp[planets[i]]['phaseFuncMeldedInverse'][0](- sma**2.* 10**(-(dmagk + pm*uncertainty_dmag)/2.5) /(p * R**2.)) ) )
    #                 elif planProp[planets[i]]['threedmagIntLimits'][1] > dmagk - uncertainty_dmag and planProp[planets[i]]['threedmagIntLimits'][0] < dmagk + uncertainty_dmag: #dmag is between local min and local max, all 3 potential intersections must be calculated
    #                     ti0 = periods/(2.*np.pi) * np.arcsin( 1./np.sin(inc2) *np.cos( np.pi/180.*planProp[planets[i]]['phaseFuncMeldedInverse'][0](- sma**2.* 10**(-(dmagk + pm*uncertainty_dmag)/2.5) /(p * R**2.)) ) )
    #                     ti1 = periods/(2.*np.pi) * np.arcsin( 1./np.sin(inc2) *np.cos( np.pi/180.*planProp[planets[i]]['phaseFuncMeldedInverse'][1](sma**2.* 10**(-(dmagk + pm*uncertainty_dmag)/2.5) /(p * R**2.)) ) )
    #                     ti2 = periods/(2.*np.pi) * np.arcsin( 1./np.sin(inc2) *np.cos( np.pi/180.*planProp[planets[i]]['phaseFuncMeldedInverse'][2](- sma**2.* 10**(-(dmagk + pm*uncertainty_dmag)/2.5) /(p * R**2.)) ) )
    #                     #NEED TO DO SOME ERROR HANDLING HERE, not necessary for this problem it seems
    #                     #planProp[planets[i]]['threebetaLimits']
    #                 else:
    #                     beta = np.pi/180.*planProp[planets[i]]['phaseFuncMeldedInverse'][2](- sma**2.* 10**(-(dmagk + pm*uncertainty_dmag)/2.5) /(p * R**2.))
    #                     if 'a' in intType:
    #                         ti = periods/(2.*np.pi) * (np.pi - np.arcsin( np.cos( beta )/np.sin(inc2) ))
    #                     else: #'b' in intType
    #                         ti = periods/(2.*np.pi) * np.arcsin( np.cos( beta )/np.sin(inc2) )
    #             else: #other planets
    #                 beta = np.pi/180.*planProp[planets[i]]['phaseFuncMeldedInverse'](- sma**2.* 10**(-(dmagk + pm*uncertainty_dmag)/2.5) /(p * R**2.))
    #                 if 'a' in intType:
    #                     ti = periods/(2.*np.pi) * (np.pi - np.arcsin(np.cos(beta)/np.sin(inc2)))
    #                 else:
    #                     ti = periods/(2.*np.pi) * np.arcsin(np.cos(beta)/np.sin(inc2))

    #         #assert not np.isnan(ti), 'whoops nan'
    #         tis.append(ti)

    #     # #Do corrections for if the inclination 
    #     # if 'a' in intTypes[0] and 'a' in intTypes[1]:
    #     #     tis[1] = np.pi/2.*periods/(2.*np.pi)
    #     # elif 'b' in intTypes[0] and 'b' in intTypes[1]:
    #     #     tis[0] = np.pi*periods/(2.*np.pi)
    #     # else:
    #     #     print(saltyburrito)
    #     #     #need to handle this case


    #     #assert not np.isnan(tis[1]), 'whoops nan'
    #     #assert not np.isnan(tis[0]), 'whoops nan'
    #     out = np.abs(tis[1]-tis[0])
    #     if np.isnan(out):
    #         #print() #should I print out the information
    #         out = 0.
    #     assert isinstance(out,float), 'dtype of out is incorrect, needs to be a float'
    #     #assert out >= 0., 'the delta t is not positive'

    #     if extraOutputBool is True:
    #         return fi(inc2)*out, np.asarray(tis)
    #     else:
    #         return fi(inc2)*out
        
    # #DO CALCS FOR SMALLER PLANET FIRST
    # period_smaller = (2.*np.pi*np.sqrt((planProp[planets[i]]['a']*u.m.to('AU')*u.AU)**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('day').value #calculate periods for planet in days
    # intTypes = sigmaIntTypeDict[(i,j)][0][0] #[0][0] indicates smaller planet and 1sigma

    # # inc2 = limitingIncs[1]
    # # out,tis = deltatFunc(inc2,planProp[planets[i]]['a']*u.m.to('AU'),planProp[planets[i]]['p'],planProp[planets[i]]['R']*u.m.to('AU'),period_smaller,uncertainty_s,uncertainty_dmag,sk,dmagk,intTypes,i,True)


    # #### THIS PLOT VERIFIES LIMITINGINCS ARE CALCULATED CORRECTLY
    # nus = np.linspace(start=0.,stop=2.*np.pi,num=200)
    # limitingBeta0 = np.arccos(np.sin(limitingIncs[0])*np.sin(nus))
    # limitingBeta1 = np.arccos(np.sin(limitingIncs[1])*np.sin(nus))
    # #betas0 = np.linspace(start=np.min(limitingBeta0,limitingBeta1),stop=np.max(limitingBeta0,limitingBeta1),num=300)
    # # seps0 = planProp[planets[i]]['a']*u.m.to('AU')*np.sin(limitingBeta0)
    # # dmags0 = -2.5*np.log10(planProp[planets[i]]['p']*(planProp[planets[i]]['R']*u.m.to('AU'))**2./(planProp[planets[i]]['a']*u.m.to('AU'))**2.*planProp[planets[i]]['phaseFuncMelded'](limitingBeta0))
    # seps0 = separation_from_alpha_ap(limitingBeta0,planProp[planets[i]]['a']*u.m).to('AU').value#planProp[planets[i]]['a']*u.m.to('AU')*np.sin(limitingBeta1)
    # dmags0 = deltaMag(planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m, planProp[planets[i]]['a']*u.m, planProp[planets[i]]['phaseFuncMelded'](limitingBeta0*180./np.pi))#-2.5*np.log10(planProp[planets[i]]['p']*(planProp[planets[i]]['R']*u.m.to('AU'))**2./(planProp[planets[i]]['a']*u.m.to('AU'))**2.*planProp[planets[i]]['phaseFuncMelded'](limitingBeta1))
    # seps1 = separation_from_alpha_ap(limitingBeta1,planProp[planets[i]]['a']*u.m).to('AU').value#planProp[planets[i]]['a']*u.m.to('AU')*np.sin(limitingBeta1)
    # dmags1 = deltaMag(planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m, planProp[planets[i]]['a']*u.m, planProp[planets[i]]['phaseFuncMelded'](limitingBeta1*180./np.pi))#-2.5*np.log10(planProp[planets[i]]['p']*(planProp[planets[i]]['R']*u.m.to('AU'))**2./(planProp[planets[i]]['a']*u.m.to('AU'))**2.*planProp[planets[i]]['phaseFuncMelded'](limitingBeta1))
    # plt.figure(543548321311111111)
    # plt.plot(seps0,dmags0)
    # plt.plot(seps1,dmags1)
    # plt.scatter(sk,dmagk)
    # indOfMin = np.argmin(np.abs(np.pi/2.-nus))
    # plt.scatter(seps0[indOfMin],dmags0[indOfMin],color='red')

    # #out,tis = deltatFunc(limitingIncs[0],planProp[planets[i]]['a']*u.m.to('AU'),planProp[planets[i]]['p'],planProp[planets[i]]['R']*u.m.to('AU'),period_smaller,uncertainty_s,uncertainty_dmag,sk,dmagk,intTypes,i,True)
    # #find where along limitingBeta0 (since limitingInc0 produces the furthest left point, we need to find the nu where limitingInc0 produces an intersection)
    # nusToEvalAt = np.asarray(np.max(tis))*2.*np.pi/period_smaller
    # betasToEvalAt = np.arccos(np.sin(np.max(limitingIncs))*np.sin(nusToEvalAt))
    # sepsToEvalAt = separation_from_alpha_ap(betasToEvalAt,planProp[planets[i]]['a']*u.m).to('AU').value#planProp[planets[i]]['a']*u.m.to('AU')*np.sin(limitingBeta1)
    # dmagsToEvalAt = deltaMag(planProp[planets[i]]['p'], planProp[planets[i]]['R']*u.m, planProp[planets[i]]['a']*u.m, planProp[planets[i]]['phaseFuncMelded'](betasToEvalAt*180./np.pi))
    # plt.scatter(sepsToEvalAt,dmagsToEvalAt,color='green')

    # plt.plot([sk-uncertainty_s,sk-uncertainty_s,sk+uncertainty_s,sk+uncertainty_s,sk-uncertainty_s],[dmagk+uncertainty_dmag,dmagk-uncertainty_dmag,dmagk-uncertainty_dmag,dmagk+uncertainty_dmag,dmagk+uncertainty_dmag],color='darkblue')
    # plt.show(block=False)
    # ###

    # if 'a' in intTypes[0] and 'a' in intTypes[1]:
    #     integrand1 = integrate.quad(deltatFunc,np.min(limitingIncs),np.max(limitingIncs),args=(planProp[planets[i]]['a']*u.m.to('AU'),planProp[planets[i]]['p'],planProp[planets[i]]['R']*u.m.to('AU'),period_smaller,uncertainty_s,uncertainty_dmag,sk,dmagk,intTypes,i,False))[0]
    #     integrand2 = integrate.quad(deltatFunc,np.max(limitingIncs),np.pi/2.,args=(planProp[planets[i]]['a']*u.m.to('AU'),planProp[planets[i]]['p'],planProp[planets[i]]['R']*u.m.to('AU'),period_smaller,uncertainty_s,uncertainty_dmag,sk,dmagk,intTypes,i,False))[0]
    
    # elif 'b' in intTypes[0] and 'b' in intTypes[1]:
    #     integrand1 = integrate.quad(deltatFunc,np.min(limitingIncs),np.max(limitingIncs),args=(planProp[planets[i]]['a']*u.m.to('AU'),planProp[planets[i]]['p'],planProp[planets[i]]['R']*u.m.to('AU'),period_smaller,uncertainty_s,uncertainty_dmag,sk,dmagk,intTypes,i,False))[0]
    #     integrand2 = integrate.quad(deltatFunc,np.max(limitingIncs),np.pi/2.,args=(planProp[planets[i]]['a']*u.m.to('AU'),planProp[planets[i]]['p'],planProp[planets[i]]['R']*u.m.to('AU'),period_smaller,uncertainty_s,uncertainty_dmag,sk,dmagk,intTypes,i,False))[0]
    

    # tavg = 2.*integrand1 + 2.*integrand2
    # fracOrbit = tavg/period_smaller

    # print('(i,j): ' + '(' + str(i) + ',' + str(j) + ') tavg integration: ' + str(tavg) + ' fracOfOrbit integration: ' + str(fracOrbit) + ' fracIn1Sigma: ' + str(fracIn1Sigma))

    # assert np.abs(fracOrbit-fracIn1Sigma) < 0.01, 'the errors are large'





"""
Equations0
beta = np.arccos(np.sin(inc)*np.sin(nu))
s = a*np.sin(beta)
The inverse
beta = np.arcsin(s/a)
nu = np.arcsin(np.cos(beta)/np.sin(inc))
with  things substituted in
nu = np.arcsin(np.cos(np.arcsin(s/a))/np.sin(inc))


"""


