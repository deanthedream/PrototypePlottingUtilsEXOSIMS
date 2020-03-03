#Calculating Dynamic Completeness by Subtype
"""
This calculates dmag, s, theta, and D as functions of time
"""

import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import numpy as np
from EXOSIMS.util.eccanom import eccanom
from keplertools import fun #From dmitry's keplertools Git Repo
from astropy import units as u
from EXOSIMS.util.calc_dyn_comp_subtype_support import calc_nu_Ee
from EXOSIMS.util.calc_dyn_comp_subtype_support import meananom
#from EXOSIMS.util.calc_dyn_comp_subtype_support import calc_r
from EXOSIMS.util.calc_dyn_comp_subtype_support import calc_rsbetaphidmag

folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts/'))
filename = 'compSubtype3.json'

scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile,nopar=True)
print('Done Generating Sim Object')

#### Generate 10^8 planets
SU = sim.SimulatedUniverse
PP = SU.PlanetPopulation
TL = sim.TargetList
nPlan = 10**6
#DELETE I, dI, O, w = PP.gen_angles(nPlans)
#DELETE a, e, p, Rp = PP.gen_plan_params(nPlans)
#KEEP
# a, e, p, Rp, I, dI, O, w, E, nu, M, s, beta, Phi, dMag, bini, binj, earthLike = sim.Completeness.genplans_all(nPlan, TL)
# print('Done genplans_all')
dMagLim = sim.Completeness.dMagLim
IWA = sim.OpticalSystem.IWA
OWA = sim.OpticalSystem.OWA
a, e, p, Rp, I, dI, O, w, E, nu, M, s, beta, Phi, dMag, bini, binj, earthLike = sim.Completeness.genVisiblePlanets(nPlan, TL, IWA, OWA, dMagLim, dist_to_star=10.*u.pc)


#### Move Planets and calculate dynamic completeness using Corey's method - Look into Corey's method
dMagLim = sim.Completeness.dMagLim
IWA = sim.OpticalSystem.IWA
OWA = sim.OpticalSystem.OWA
t_range = np.linspace(start=0,stop=365*12,num=1200)
visibile_by_t = visibilityStates_overt(a, e, p, Rp, I, dI, O, w, E, nu, M, t_range, erange, IWA, OWA, dMagLim=23., mu=u.GM_sun.value)
print('Success!')

#Maybe we can intelligently calculate dynamic completeness based on the planet's known subtype
#and uncertainty in instrument parameters

#Things to Track
#Is the planet currently visible by the instrument (is the current index true)
#Has the planet been visible in the past by the instrument
#Total Number of Times the planet has been visible by the instrument
#Time since the start of the last visibility

def visibilityStates_overt(a, e, p, Rp, I, dI, O, w, E, nu, M, t_range, erange, IWA, OWA, dMagLim=23., mu=u.GM_sun.value):
    """
    Args:
        KOE STATES + M, E
        t_range
        erange
        dMagLim
        mu
    Returns:
        visible_by_t ():
            has shape [len(a),len(t_range)]
        visibleInPast ():
            has len = len(t)-1, integers counting number of times planet is visible again
    """
    visible_by_t = dict()#list()
    visiblet = list()
    rt = list()
    Et = list()
    nut = list()
    Mt = list()
    st = list()
    betat = list()
    phit = list()
    dmagt = list()
    for tind in np.arange(len(t_range)):
        visible1, r1, E1, nu1, M1, s1, beta1, phi1, dmag1 = visibilityStates(a, e, p, Rp, I, dI, O, w, E, nu, M, t_range[tind], erange, IWA, OWA, dMagLim=dMagLim, mu=u.GM_sun.value)
        visiblet.append(visible1)
        rt.append(r1)
        Et.append(E1)
        nut.append(nu1)
        Mt.append(M1)
        st.append(s1)
        betat.append(beta1)
        phit.append(phi1)
        dmagt.append(dmag1)
    visible = np.asarray(visiblet)
    r = np.asarray(rt) #dims [len(t_range),len(sInds)]
    E = np.asarray(Et)
    nu = np.asarray(nut)
    M = np.asarray(Mt)
    s = np.asarray(st)
    beta = np.asarray(betat)
    phi = np.asarray(phit)
    dmag = np.asarray(dmagt)

    #A temporary array calculating bit flips from 0 to 1
    visibleInPast = list()
    for sInd in np.arange(len(a)):
        tmp = visible[1:,sInd] - visible[:-1,sInd] #Checking for bit flips. produces 1,0,-1
        tmp2 = (tmp > 0.99) #Should be a true false array
        tmp3 = np.cumsum(tmp2)
        visibleInPast.append(tmp3)
    visibleInPast = visibleInPast.T

    visible_by_t = {'visible':visible, 'r':r, 'E':E, 'nu':nu, 'M':M, 's':s, 'beta':beta, 'phi':phi, 'dmag':dmag, \
            'visibleInPast':visibleInPast, 't':t_range}

    return visible_by_t

def visibilityStates_now(a, e, p, Rp, I, dI, O, w, E, nu, M, t, erange, IWA, OWA, dMagLim=23., mu=u.GM_sun.value):
    """ Calculates whether the planet is visible or not at the given time past t=0
    Args:
        t (ndarray):
            time since t=0 to calculate new visibility states for
    visible:
        indicates visibility right now
    visibleInPast:
        0-never visible yet
        1-visible once
        2-visible twice ...
    timeLastVisibilityStart:

    """
    nPlan = len(a) #Number of planets

    # Calculate the mean anomaly for the planet population after the time period
    M1 = meananom(mu, a, t, M0)
    
    # Calculate the anomalies for each planet after the time period passes
    #E1 = fun.eccanom(M1.value, e)
    #nu1 = fun.trueanom(E1, e)

    # theta = nu + w_p.value
    # r = calc_r(erange,e,M)
    # #r = a_p*(1.-e_p**2.)/(1.+e_p*np.cos(nu))
    # #Calculate Planet-Star Separation
    # s = r*np.sqrt(np.cos(w+nu)**2.+np.sin(w+nu)**2.*np.cos(I)**2.)
    # #Calculate Planet Phase Angle
    # beta = np.arcsin(s/r)

    # # phase function value
    # Phi = self.PlanetPhysicalModel.calc_Phi(beta)
    # # calculate dMag
    # dMag = deltaMag(p,Rp,r,Phi)
    #s = (r.value/4.)*np.sqrt(4.*np.cos(2.*I_p.value) + 4.*np.cos(2.*theta)-2.*np.cos(2.*I_p.value-2.*theta) \
    #     - 2.*np.cos(2.*I_p.value+2.*theta) + 12.) #From 
    #beta = np.arccos(-np.sin(I_p)*np.sin(theta))
    #phi = (np.sin(beta.value) + (np.pi - beta.value)*np.cos(beta.value))/np.pi
    #dMag = deltaMag(p_p, Rp_p.to(u.km), r.to(u.AU), phi)

    r1, e1, E1,  nu1, s1, beta1, phi1, dmag1 = calc_rsbetaphidmag(erange, e, M1, w, I, p, Rp, a)

    #COULD I CALCULATE A RANGE OF NU M OR E WHERE THE PLANET IS VISIBLE?
    #WOULD THIS BE A MORE EFFICIENT METHOD?

    #need p,Rp,r,beta,w,nu,I

    s1 = calc_s(r,w,nu,I)

    beta = np.arcsin(s/r)

    Phi = phi_lambert(beta) #for deltaMag
    del beta

    dmag1 = deltaMag(p,Rp,r,Phi)
    del Phi

    smin_instLimit = IWA.to(u.arcsec).value*dist_to_star.to(u.pc).value
    smax_instLimit = OWA.to(u.arcsec).value*dist_to_star.to(u.pc).value
    #min_separation = IWA.to(u.arcsec).value*dist_to_star.to(u.pc).value
    #max_separation = OWA.to(u.arcsec).value*dist_to_star.to(u.pc).value

    # Right now visible planets is an array with the size of the number of potential_planets
    # I need to convert it back to size of the total number of planets with each
    #visible = (s1 > min_separation) & (s1 < max_separation) & (dmag1 < dMagLim)
    visibleIWA = (s1 > min_separation)
    visibleOWA = (s1 < max_separation)
    visibledMag = (dmag1 < dMagLim)


    return visible, r1, E1, nu1, M1, s1, beta1, phi1, dmag1

    #visibleInPast


# def dynComp():
#     return dynComp

# def compSubtype_at_t():
#     return 


def cij(a, e, M0, I, w, O, Rp, p, t, mu, potential_planets, d, IWA, OWA, dMag0):
    '''
    Calculates the dynamic completeness value of the second visit to a star
    
    Args:
        a (ndarray of Astropy quantities):
            Semi-major axis
        e (ndarray):
            Eccentricity
        M0 (ndarray of Astropy quantities):
            Mean anomaly
        I (Astropy quantity):
            Inclination (rad)
        w (Astropy quantity):
            Argument of periastron (rad)
        O (Astropy quantity):
            Longitude of the ascending node (rad)
        Rp (Astropy quantity):
            Planetary radius
        p (float):
            Geometric albedo of planets
        t (float):
            Time progressed in seconds
        mu (Astropy quantity):
            Gravitational parameter
        potential_planets (ndarray of booleans):
            A true value indicates that the planet has not been eliminated from the search around this planet
        d (astropy quantity):
            Distance to star
        IWA (astropy quantity):
            Telescope's inner working angle
        OWA (astropy quantity):
            Telescope's outer working angle
        dMag0 (float):
            Telescope's limiting difference in brightness between the star and planet
    Returns:
        c2j (ndarray):
            Dynamic completeness value
    '''
    total_planets = len(a) #total number of planets
    
    # Get indices for which planets are to be propagated
    planet_indices = np.arange(total_planets) #np.linspace(0, total_planets-1, total_planets).astype(int)
    potential_planet_indices = planet_indices[potential_planets]
    
    # Get the values for the propagated planets
    a_p = a[potential_planets]
    e_p = e[potential_planets]
    M0_p = M0[potential_planets]
    I_p = I[potential_planets]
    w_p = w[potential_planets]
    Rp_p = Rp[potential_planets]
    p_p = p[potential_planets]
    
    # Calculate the mean anomaly for the planet population after the time period
    M1 = meananom(mu, a_p, t, M0_p)
    
    # Calculate the anomalies for each planet after the time period passes
    E = fun.eccanom(M1.value, e_p)
    nu = fun.trueanom(E, e_p)
    
    theta = nu + w_p.value
    r = a_p*(1.-e_p**2.)/(1.+e_p*np.cos(nu))
    s = (r.value/4.)*np.sqrt(4.*np.cos(2.*I_p.value) + 4.*np.cos(2.*theta)-2.*np.cos(2.*I_p.value-2.*theta) \
         - 2.*np.cos(2.*I_p.value+2.*theta) + 12.) #From 
    beta = np.arccos(-np.sin(I_p)*np.sin(theta))
    phi = (np.sin(beta.value) + (np.pi - beta.value)*np.cos(beta.value))/np.pi
    dMag = deltaMag(p_p, Rp_p.to(u.km), r.to(u.AU), phi)

    min_separation = IWA.to(u.arcsec).value*dist_to_star.to(u.pc).value
    max_separation = OWA.to(u.arcsec).value*dist_to_star.to(u.pc).value
    
    # Right now visible planets is an array with the size of the number of potential_planets
    # I need to convert it back to size of the total number of planets with each
    visible_planets = (s > min_separation) & (s < max_separation) & (dMag < dMag0)
    
    # Calculate the completeness
    cij = np.sum(visible_planets)/float(np.sum(potential_planets))
    
    # Create an array with all of the visible planets with their original indices
    visible_planet_indices = potential_planet_indices[visible_planets]
    full_visible_planets = np.zeros(total_planets, dtype=bool)
    full_visible_planets[visible_planet_indices] = True
    

    return [cij, full_visible_planets]








