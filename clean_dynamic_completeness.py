#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 17:12:06 2019

@author: corey
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keplertools import fun
import astropy
import pickle
from astropy import units as u
import EXOSIMS.PlanetPopulation as PlanetPopulation
from EXOSIMS.util.deltaMag import deltaMag
import EXOSIMS,EXOSIMS.MissionSim,os.path
from statistics import median
import multiprocessing as mp
import logging
import json
from scipy import optimize
from timeit import default_timer as timer


def meananom(mu, a, t, M0=0, to=0):
    '''
    Finds the mean anomaly after a period of time
    Uses astropy units
    
    Args:
        mu (float or ndarray):
            Gravitational parameter
        a (float or ndarray):
            Semi-major axis 
        t (float):
            Time progressed
        Mo (Astropy quantity):
            Initial mean anomaly
        to (float):
            Initial time
    Returns:
        M (float or ndarray):
            Mean anomaly (rad)
    '''
    # Get all the inputs into standard SI values
    t = t.to(u.s)
    a = a.to(u.m)
    mu = mu.to(u.m**3/(u.s**2))
    
    # Calculate mean anomaly
    #    M = np.add(np.sqrt(mu/a**3)*(t-to),Mo)
    M1= np.sqrt(mu/a**3)*(t-to)*u.rad
    M2= M0
    M = (M1+M2)%(2*np.pi*u.rad)
    
    return M



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


N0 = 100000 # Planets around the star
time_steps = 100 # How many observations are made
plot_type = 'linear'
start_exp = 1
end_exp = 8
distances = [5]*u.pc
observations = [False, True, False, True]
time_of_observation = [1e5, 1e7, 5e7, 1e8]*u.s

WFIRST_IWA = 0.15 * u.arcsec
WFIRST_OWA = 0.428996 * u.arcsec
WFIRST_dMag0 = 22.5
scriptfile = os.path.join(EXOSIMS.__path__[0],'Scripts','DC_test.json')
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
mu = sim.PlanetPopulation.mu
a, e, p, Rp = sim.PlanetPopulation.gen_plan_params(N0)
I, O, w = sim.PlanetPopulation.gen_angles(N0)
I = I.to(u.rad)
w = w.to(u.rad)
O = O.to(u.rad)
M0 = np.random.uniform(0, 2*np.pi, int(N0))*u.rad # mean anomaly

initial_potential_planets = np.ones(N0, dtype=bool)

if plot_type == 'log':
    ts = np.logspace(start_exp, end_exp, num=time_steps)
elif plot_type == 'linear':
    ts = np.linspace(10**start_exp, 10**end_exp, num=time_steps)


c1j_list = []


for dist_to_star in distances:
    
    # Calculate the initial completion value at t=0 and keep track of the plaents that are seen at that time
    [c1j, potential_planets] = cij(a, e, M0, I, w, O, Rp, p, 0*u.s, mu, initial_potential_planets, dist_to_star, WFIRST_IWA, WFIRST_OWA, WFIRST_dMag0)
    

    c1j_list.append(c1j)
    c2j_vals_list = []

    for i in range(time_steps):
        [c2j_val, visible_planets] = cij(a, e, M0, I, w, O, Rp, p, ts[i]*u.s, mu, potential_planets, dist_to_star, WFIRST_IWA, WFIRST_OWA, WFIRST_dMag0)

        c2j_vals_list.append(c2j_val)
        
        print('Observation: ' + str(i)  +  '/' + str(observations), end='\r')
    
    c2j_vals_list.append(c2j_val)

# Eliminating planets check
potential_planets = initial_potential_planets
for i in range(len(observations)):
    planet_observed = observations[i]
    t = time_of_observation[i]
    [cij_val, visible_planets] = cij(a, e, M0, I, w, O, Rp, p, 0*u.s, mu, potential_planets, dist_to_star, WFIRST_IWA, WFIRST_OWA, WFIRST_dMag0)
    if not planet_observed:
        potential_planets = np.logical_not(visible_planets)
    
# Log scale
fig, ax = plt.subplots()
#plt.scatter(ts,c2j_vals,s=2)
ts = ts / 3.154e7 # Convert to years
#t_opt_list = t_opt_list / 3.154e7
for i in range(len(c2j_vals_list)):
    ax.plot(ts, c2j_vals_list[i], label = str(distances[i]))
    # ax.plot(t_opt_list[i]/ 3.154e7, c2j_opt_list[i], marker='o')
    
leg = ax.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')

if plot_type == 'log':
    plt.xscale('log')
plt.xlabel('Delay time of second search (yr)')
plt.ylabel('Dynamic completeness, c2j')
title = 'Planets: ' + str(int(N0)) + ', Observations: ' + str(observations) #+ ', Marker Time: ' +  str(round(time_cost, 2)) + 's'
plt.title(title)
plt.savefig('C2j_WFIRST_' + str(dist_to_star) + '.png', dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight')
#filename = 'data/'+plot_type+str(start_exp)+'-'+str(end_exp)+'_Planets-'+str(int(N0))+'_Observations-'+str(observations)+'_Period-' + str(T) + '.p'
