"""

Instructions for to interface with ipac using API
https://exoplanetarchive.ipac.caltech.edu/applications/DocSet/index.html?doctree=/docs/docmenu.xml&startdoc=item_1_01

Written by: Dean Keithly
Written On:4/1/2019
"""

import os.path
import urllib2
import json
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.integrate import cumtrapz

baseURL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?"
tablebaseURL = "table="
tableInput = "exoplanets" # exoplanets to query exoplanet table
columnsbaseURL = "&select=" # Each table input must be separated by a comma
columnsInputList = ['pl_hostname','ra','dec','pl_discmethod','pl_pnum','pl_orbper','pl_orbsmax','pl_orbeccen',\
                    'pl_orbincl','pl_bmassj','pl_radj','st_dist','pl_tranflag','pl_rvflag','pl_imgflag',\
                    'pl_astflag','pl_omflag','pl_ttvflag', 'st_mass', 'pl_discmethod']
                    #https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html for explanations

"""
pl_hostname - Stellar name most commonly used in the literature.
ra - Right Ascension of the planetary system in decimal degrees.
dec - Declination of the planetary system in decimal degrees.
pl_discmethod - Method by which the planet was first identified.
pl_pnum - Number of planets in the planetary system.
pl_orbper - Time the planet takes to make a complete orbit around the host star or system.
pl_orbsmax - The longest radius of an elliptic orbit, or, for exoplanets detected via gravitational microlensing or direct imaging,\
            the projected separation in the plane of the sky. (AU)
pl_orbeccen - Amount by which the orbit of the planet deviates from a perfect circle.
pl_orbincl - Angular distance of the orbital plane from the line of sight.
pl_bmassj - Best planet mass estimate available, in order of preference: Mass, M*sin(i)/sin(i), or M*sin(i), depending on availability,\
            and measured in Jupiter masses. See Planet Mass M*sin(i) Provenance (pl_bmassprov) to determine which measure applies.
pl_radj - Length of a line segment from the center of the planet to its surface, measured in units of radius of Jupiter.
st_dist - Distance to the planetary system in units of parsecs. 
pl_tranflag - Flag indicating if the planet transits its host star (1=yes, 0=no)
pl_rvflag -     Flag indicating if the planet host star exhibits radial velocity variations due to the planet (1=yes, 0=no)
pl_imgflag - Flag indicating if the planet has been observed via imaging techniques (1=yes, 0=no)
pl_astflag - Flag indicating if the planet host star exhibits astrometrical variations due to the planet (1=yes, 0=no)
pl_omflag -     Flag indicating whether the planet exhibits orbital modulations on the phase curve (1=yes, 0=no)
pl_ttvflag -    Flag indicating if the planet orbit exhibits transit timing variations from another planet in the system (1=yes, 0=no).\
                Note: Non-transiting planets discovered via the transit timing variations of another planet in the system will not have\
                 their TTV flag set, since they do not themselves demonstrate TTVs.
st_mass - Amount of matter contained in the star, measured in units of masses of the Sun.
pl_discmethod - Method by which the planet was first identified.
"""

columnsInput = ','.join(columnsInputList)
formatbaseURL = '&format='
formatInput = 'json' #https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html#format

# Different acceptable "Inputs" listed at https://exoplanetarchive.ipac.caltech.edu/applications/DocSet/index.html?doctree=/docs/docmenu.xml&startdoc=item_1_01


myURL = baseURL + tablebaseURL + tableInput + columnsbaseURL + columnsInput + formatbaseURL + formatInput
response = urllib2.urlopen(myURL)
#html = response.read()
data = json.load(response)

fig = plt.figure(987654632)
ax = fig.add_subplot(1,1,1)
ylim = [1e9, -1.]
xlim = [1e9, -1.]
nanInds = list()
noDiscType = list()
nanpl_radjInds = list()
nanst_massInds = list()
nanpl_orbperInds = list()
pl_orbeccen = list()
plt_orbeccenISNONE = list()
for i in np.arange(len(data)):
    #### Determine Original Detection Device
    ec = None
    m = None
    fc = None
    if data[i]['pl_discmethod'] == 'Radial Velocity': # Is radial velocity
        ec = 'red'
        m = 'o'
        fc = 'none'
    if data[i]['pl_discmethod'] == 'Transit': # is transit
        ec = 'green'
        m = 's'
        fc = 'none'
    if data[i]['pl_discmethod'] == 'Microlensing': # is microlensing
        ec = 'purple'
        m = "^"
        fc = 'purple'
    if data[i]['pl_discmethod'] == 'Imaging': # is imaging
        ec = 'blue'
        m = '*'
        fc = 'blue'
    if data[i]['pl_discmethod'] == 'Transit Timing Variations' or data[i]['pl_discmethod'] == 'Eclipse Timing Variations': # is timing variations
        ec = 'yellow'
        m = 'p'
        fc = 'yellow'
    if data[i]['pl_discmethod'] == 'Orbital Brightness Modulation': # is oribtal brightness modulation
        ec = 'orange'
        m = 's'
        fc = 'orange'
    if data[i]['pl_discmethod'] == 'Astrometry': # is astrometry
        ec = 'grey'
        m = 'o'
        fc = 'grey'

    if ec == None:
        noDiscType.append(i)
    # Omitts pulsar timing

    #Calibrate Limits
    # if data[i]['pl_orbsmax'] == None or np.isnan(data[i]['pl_orbsmax']):
    #     nanInds.append(i)
    if data[i]['pl_radj'] == None or np.isnan(data[i]['pl_radj']):
        nanpl_radjInds.append(i)
        continue
    if data[i]['st_mass'] == None or np.isnan(data[i]['st_mass']):
        nanst_massInds.append(i)
        continue
    if data[i]['pl_orbper'] == None or np.isnan(data[i]['pl_orbper']):
        nanpl_orbperInds.append(i)
        continue

    sma = ((data[i]['st_mass']*const.M_sun*const.G)**(1./3.) *\
            (((data[i]['pl_orbper']*u.d).to('second')/(2.*np.pi))**2.)**(1./3.)).to('AU').value # from First Pages of Vallado

    # if data[i]['pl_orbsmax'] < xlim[0]:
    #     xlim[0] = data[i]['pl_orbsmax']
    # elif data[i]['pl_orbsmax'] > xlim[1]:
    #     xlim[1] = data[i]['pl_orbsmax']
    if sma < xlim[0]:
        xlim[0] = sma
    elif sma > xlim[1]:
        xlim[1] = sma
    if data[i]['pl_radj'] < ylim[0]:
        ylim[0] = data[i]['pl_radj']
    elif data[i]['pl_radj'] > ylim[1]:
        ylim[1] = data[i]['pl_radj']


    #data[i]['pl_orbsmax']
    ax.scatter(sma,data[i]['pl_radj'], edgecolors=ec, linewidths=1, marker=m, c=fc)#facecolors=fc)

    #### pl_orbeccen
    if data[i]['pl_orbeccen'] == None or np.isnan(data[i]['pl_orbeccen']):
        plt_orbeccenISNONE.append(data[i]['pl_orbeccen'])
        continue
    pl_orbeccen.append(data[i]['pl_orbeccen'])


def setOfStarsWithKnownPlanets(data):
    """
    Args:
        data (dict) - dict containing the pl_hostname of each star
    """
    starNames = list()
    for i in np.arange(len(data)):
        starNames.append(data[i]['pl_hostname'])
    return list(set(starNames))


print(nanInds)
print(len(nanInds))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(bottom=0.5*ylim[0],top=1.5*ylim[1])
ax.set_xlim(left=0.5*xlim[0],right=1.5*xlim[1])
ax.set_xlabel('Largest Observed Separation (AU)')
ax.set_ylabel('Planet Radius (R_j)')
plt.show(block=False)




n, bins, patches = plt.figure(665465461286584).add_subplot(1,1,1).hist(pl_orbeccen, bins=1000)
plt.show(block=False)
plt.close(665465461286584) # doing this just to descroy above plot Replace with numpy.histogram in future
cdf = np.cumsum(n)#cumtrapz(n, bins[:-1], initial=0.)
cdf = cdf/np.max(cdf)

#### Plot Histogram of Orbital Eccentricity with CDF Overlay
fig2 = plt.figure(321354684)
ax2 = fig2.add_subplot(1,1,1)
n2, bins2, patches2 = ax2.hist(np.asarray(pl_orbeccen),zorder=8,color='black')
ax2.set_xlabel('Oribtal Eccentricity')
ax3 = ax2.twinx()
ax3.plot(bins[:-1],cdf,zorder=10, color='red')
ax3.set_ylabel('Eccentricty CDF')
ax2.set_xlim(left=0.,right=1.)
ax2.set_ylim(bottom=0.,top=np.sum(n2))
ax3.set_ylim(bottom=0.,top=1.)
plt.show(block=False)
len(plt_orbeccenISNONE)
len(data)




starsWithPlanets = setOfStarsWithKnownPlanets(data)

