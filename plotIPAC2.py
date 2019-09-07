"""
same as plotIPAC but omits RV, astrometric,...
Written by: Dean Keithly
Written On:4/1/2019
"""

import os.path
#import urllib2 #for python3.5 and before
import urllib3 # for python3.6 and after
import json
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
#from scipy.integrate import cumtrapz
import datetime
import re
import glob
try:
    import cPickle as pickle
except:
    import pickle
import os
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim


def constructIPACurl(tableInput="exoplanets", columnsInputList=['pl_hostname','ra','dec','pl_discmethod','pl_pnum','pl_orbper','pl_orbsmax','pl_orbeccen',\
    'pl_orbincl','pl_bmassj','pl_radj','st_dist','pl_tranflag','pl_rvflag','pl_imgflag',\
    'pl_astflag','pl_omflag','pl_ttvflag', 'st_mass', 'pl_discmethod'],\
    formatInput='json'):
    """
    Extracts Data from IPAC
    Instructions for to interface with ipac using API
    https://exoplanetarchive.ipac.caltech.edu/applications/DocSet/index.html?doctree=/docs/docmenu.xml&startdoc=item_1_01
    Args:
        tableInput (string) - describes which table to query
        columnsInputList (list) - List of strings from https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html 
        formatInput (string) - string describing output type. Only support JSON at this time
    """
    baseURL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?"
    tablebaseURL = "table="
    # tableInput = "exoplanets" # exoplanets to query exoplanet table
    columnsbaseURL = "&select=" # Each table input must be separated by a comma
    # columnsInputList = ['pl_hostname','ra','dec','pl_discmethod','pl_pnum','pl_orbper','pl_orbsmax','pl_orbeccen',\
    #                     'pl_orbincl','pl_bmassj','pl_radj','st_dist','pl_tranflag','pl_rvflag','pl_imgflag',\
    #                     'pl_astflag','pl_omflag','pl_ttvflag', 'st_mass', 'pl_discmethod']
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
    # formatInput = 'json' #https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html#format

    # Different acceptable "Inputs" listed at https://exoplanetarchive.ipac.caltech.edu/applications/DocSet/index.html?doctree=/docs/docmenu.xml&startdoc=item_1_01

    myURL = baseURL + tablebaseURL + tableInput + columnsbaseURL + columnsInput + formatbaseURL + formatInput
    #response = urllib2.urlopen(myURL) #python3.5
    http = urllib3.PoolManager()
    r = http.request('GET', myURL)
    data = json.loads(r.data.decode('utf-8'))

    #data = json.load(response) # python3.5
    return data

def setOfStarsWithKnownPlanets(data):
    """ From the data dict created in this script, this method extracts the set of unique star names
    Args:
        data (dict) - dict containing the pl_hostname of each star
    """
    starNames = list()
    for i in np.arange(len(data)):
        starNames.append(data[i]['pl_hostname'])
    return list(set(starNames))


data = constructIPACurl()
PPoutpath = ''
folder = ''

Rj = 71492.
Re = 12756./2.

#### Plot "Penny Plot" as Rp vs SMA
fig = plt.figure(987654632)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
ax = fig.add_subplot(1,1,1)
ylim = [1e9, -1.]
xlim = [1e9, -1.]
nanpl_orbsmaxInds = list()
noDiscType = list()
nanpl_radjInds = list()
nanst_massInds = list()
nanpl_orbperInds = list()
pl_orbeccen = list()
plt_orbeccenISNONE = list()
st_dists = list()

# Define coloring for Each Detection Type
dataLabels = {}
dataLabels['Radial Velocity'] = {'ec':'red','m':'o','fc':'none','zorder':5,'alpha':0,'label':'Radial Velocity'}
dataLabels['Transit'] = {'ec':'green','m':'s','fc':'none','zorder':1,'alpha':0.4,'label':'Transit'}
dataLabels['Microlensing'] = {'ec':'purple','m':"^",'fc':'purple','zorder':5,'alpha':0,'label':'Microlensing'}
dataLabels['Imaging'] = {'ec':'blue','m':'*','fc':'blue','zorder':5,'alpha':1,'label':'Imaging'}
dataLabels['Transit Timing Variations'] = {'ec':'yellow','m':'p','fc':'yellow','zorder':5,'alpha':0,'label':'Timing Variations'}
dataLabels['Eclipse Timing Variations'] = {'ec':'yellow','m':'p','fc':'yellow','zorder':5,'alpha':0,'label':'Eclipse Timing Variations'}
dataLabels['Orbital Brightness Modulation'] = {'ec':'orange','m':'s','fc':'orange','zorder':5,'alpha':0,'label':'Orbital Brightness\nModulation'}
dataLabels['Astrometry'] = {'ec':'grey','m':'o','fc':'grey','zorder':5,'alpha':0,'label':'Astrometry'}
for i in np.arange(len(data)):
    #### Determine Original Detection Device and apply coloring
    ec = None
    m = None
    fc = None
    zorder = 10
    alpha=1
    if data[i]['pl_discmethod'] in dataLabels:
        ec=dataLabels[data[i]['pl_discmethod']]['ec']
        m=dataLabels[data[i]['pl_discmethod']]['m']
        fc=dataLabels[data[i]['pl_discmethod']]['fc']
        zorder=dataLabels[data[i]['pl_discmethod']]['zorder']
        alpha=dataLabels[data[i]['pl_discmethod']]['alpha']

    if ec == None:
        noDiscType.append(i)
    # Omitts pulsar timing

    #### Check if Any do not possess sma or ability to calculate sma
    #Append stuff to error lists for accounting purposes
    if (data[i]['pl_radj'] == None or np.isnan(data[i]['pl_radj']))\
        or ((data[i]['pl_orbsmax'] == None or np.isnan(data[i]['pl_orbsmax'])) \
            and (data[i]['st_mass'] == None or np.isnan(data[i]['st_mass'])\
                or data[i]['pl_orbper'] == None or np.isnan(data[i]['pl_orbper']))):

        if data[i]['pl_orbsmax'] == None or np.isnan(data[i]['pl_orbsmax']):
            nanpl_orbsmaxInds.append(i)
        if (data[i]['st_mass'] == None or np.isnan(data[i]['st_mass'])\
                or data[i]['pl_orbper'] == None or np.isnan(data[i]['pl_orbper'])):
            if data[i]['st_mass'] == None or np.isnan(data[i]['st_mass']):
                nanst_massInds.append(i)
            if data[i]['pl_orbper'] == None or np.isnan(data[i]['pl_orbper']):
                nanpl_orbperInds.append(i)
        if data[i]['pl_radj'] == None or np.isnan(data[i]['pl_radj']):
            nanpl_radjInds.append(i)
        continue
    #If pl_orbsmax, use for SMA
    if not data[i]['pl_orbsmax'] == None and not np.isnan(data[i]['pl_orbsmax']):
        sma = data[i]['pl_orbsmax']
    #If can calculate SMA, calculate SMA
    if not data[i]['st_mass'] == None and not np.isnan(data[i]['st_mass'])\
         and not data[i]['pl_orbper'] == None and not np.isnan(data[i]['pl_orbper']):
        sma = ((data[i]['st_mass']*const.M_sun*const.G)**(1./3.) *\
                (((data[i]['pl_orbper']*u.d).to('second')/(2.*np.pi))**2.)**(1./3.)).to('AU').value # from First Pages of Vallado
    assert not sma == None
    assert not data[i]['pl_radj'] == None

    #### Calibrate Limits
    if sma < xlim[0]:
        xlim[0] = sma
    elif sma > xlim[1]:
        xlim[1] = sma
    if data[i]['pl_radj']*Rj/Re < ylim[0]:
        ylim[0] = data[i]['pl_radj']*Rj/Re
    elif data[i]['pl_radj']*Rj/Re > ylim[1]:
        ylim[1] = data[i]['pl_radj']*Rj/Re

    ax.scatter(sma,data[i]['pl_radj']*Rj/Re, edgecolors=ec, linewidths=1, marker=m, c=fc, s=15, zorder=zorder, alpha=alpha)#facecolors=fc)

    #### pl_orbeccen
    if data[i]['pl_orbeccen'] == None or np.isnan(data[i]['pl_orbeccen']):
        plt_orbeccenISNONE.append(data[i]['pl_orbeccen'])
        continue
    pl_orbeccen.append(data[i]['pl_orbeccen'])

    st_dists.append(data[i]['st_dist'])

for key in dataLabels:
    if key == 'Eclipse Timing Variations':
        continue # skip this one
    if not key in ['Transit','Imaging']:#THIS SKIPS ALL OTHER DETECTION TECHNIQUES
        continue #THIS SKIPS ALL OTHER DETECTION TECHNIQUES
    ax.scatter(10**10,1,edgecolors=dataLabels[key]['ec'],\
        linewidths=1, marker=dataLabels[key]['m'],\
        c=dataLabels[key]['fc'],\
        s=15, zorder=dataLabels[key]['zorder'],\
        alpha=dataLabels[key]['alpha'],\
        label=dataLabels[key]['label'])


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(bottom=0.5*ylim[0],top=1.5*ylim[1])
ax.set_xlim(left=0.5*xlim[0],right=1.5*xlim[1])
ax.set_xlabel('Largest Observed Planet Star Separation s, in (AU)', weight='bold')
ax.set_ylabel(r'Planet Radius $R_p$, in ($R_\oplus$)', weight='bold')
ax.legend(loc='lower right')
plt.show(block=False)


date = datetime.datetime.now().strftime("%d_%b_%Y_%H_%M_%S_%f")
#date = str(datetime.datetime.now(),'utf-8')#Python3.5 unicode(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'pennyPlot_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
#plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))



#### Add solar system planets to pennyplot
kmtoAU =6.68459*10**-9.
solarSystemAU = np.asarray([778.6, 1433.5, 2872.5, 4495.1, 149.6, 108.2, 227.9, 57.9, 0.384, 5906.4])*10**6.*kmtoAU
solarSystemAU = [0.39, 0.72, 1.00, 1.52, 5.20, 9.54, 19.2, 30.1, 39.4 ]
# Mercury 0.39 
# Venus 0.72 
# Earth 1.00 
# Mars 1.52 
# Jupiter 5.20 
# Saturn 9.54 
# Uranus 19.2 
# Neptune 30.1
# Pluto 39.4 
#JUPITER SATURN URANUS NEPTUNE EARTH VENUS MARS MERCURY MOON PLUTO 
Rj = 71492.
Re = 12756./2.
#solarSystemR = np.asarray([71492., 60268., 25559., 24764., 6378.1, 6051.8, 3396.2, 2439.7, 1738.1, 1195.])/Rj
# MERCURY  4879   
# VENUS 12,104   
# EARTH 12,756  
# MARS 6792   
# JUPITER 142,984    
# SATURN  120,536    
# URANUS  51,118    
# NEPTUNE 49,528    
# PLUTO 2370
                   
solarSystemR = np.asarray([4879., 12104., 12756., 6792., 142984., 120536., 51118., 49528., 2370.])/2./Rj
#in km# 1   Jupiter 71492# 2   Saturn  60268# 3   Uranus  25559# 4   Neptune 24764# 5   Earth   6378.1# 6   Venus   6051.8# 7   Mars    3396.2# 8   Mercury 2439.7# 9   Moon    1738.1# 10  Pluto   1195
ax.scatter(solarSystemAU,solarSystemR*Rj/Re, edgecolors='k', linewidths=1, marker='o', c='k', s=15, zorder=10, alpha=1.0, label='Solar System Bodies')
plt.show(block=False)
fname = 'pennyPlotwSolarPlanets2_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
#plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))
#########################

#### Add exoplanets
folder = '/home/dean/Documents/SIOSlab/EXOSIMSres/HabExCompSpecPriors_HabEx_4m_TSDD_pop100DD_revisit_20180424/HabEx_CSAG13_PPSAG13' #HABEX
alpha=0.005 #HabEx
num = 50 #HabEx
# folder = '/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_40519/WFIRSTcycle6core_CKL2_PPKL2' #WFIRST
# alpha=0.1 #WFIRST
# num=None #WFIRST
PPoutpath = './'

if not os.path.exists(folder):#Folder must exist
    raise ValueError('%s not found'%folder)
if not os.path.exists(PPoutpath):#PPoutpath must exist
    raise ValueError('%s not found'%PPoutpath) 
outspecfile = os.path.join(folder,'outspec.json')
if not os.path.exists(outspecfile):#outspec file not found
    raise ValueError('%s not found'%outspecfile) 
try:
    with open(outspecfile, 'rb') as g:
        outspec = json.load(g)
except:
    vprint('Failed to open outspecfile %s'%outspecfile)
    pass

det_radii = list()
det_SMA = list()
starInd = list()

pklfiles = glob.glob(os.path.join(folder,'*.pkl'))
for counter,f in enumerate(pklfiles[0:num]):
    print("%d/%d"%(counter,len(pklfiles)))
    with open(f, 'rb') as g:
        res = pickle.load(g, encoding='latin1')

    for i in np.arange(len(res['DRM'])):
        dInds = np.where(res['DRM'][i]['det_status'] == 1)[0]
        if len(dInds) > 0:
            for ind in dInds:
                planInd = res['DRM'][i]['plan_inds'][dInds][0]
                det_radii.append(res['systems']['Rp'][planInd].value) 
                det_SMA.append(res['systems']['a'][planInd].value)
                starInd.append(res['DRM'][i]['star_ind'])

ax.scatter(np.asarray(det_SMA), np.asarray(det_radii), edgecolors='purple', linewidths=1, marker='o', c='purple', s=5, zorder=10, alpha=alpha)
#ax.scatter(.000001,.000001, edgecolors='purple', linewidths=1, marker='o', c='purple', s=5, zorder=10, alpha=1, label='WFIRST')
ax.scatter(.000001,.000001, edgecolors='purple', linewidths=1, marker='o', c='purple', s=5, zorder=10, alpha=1, label='HabEx')
plt.legend()
plt.show(block=False)
fname = 'pennyPlotwSolarPlanetsw2_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
#plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))
################################################################



#### Plot Histogram of Orbital Eccentricity with CDF Overlay
n, bins, patches = plt.figure(665465461286584).add_subplot(1,1,1).hist(pl_orbeccen, bins=1000)
plt.show(block=False)
plt.close(665465461286584) # doing this just to descroy above plot Replace with numpy.histogram in future
cdf = np.cumsum(n)#cumtrapz(n, bins[:-1], initial=0.)
cdf = cdf/np.max(cdf)

fig2 = plt.figure(321354684)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
ax2 = fig2.add_subplot(1,1,1)
n2, bins2, patches2 = ax2.hist(np.asarray(pl_orbeccen),zorder=8,color='black')
ax2.set_xlabel('Oribital Eccentricity', weight='bold')
ax3 = ax2.twinx()
ax3.plot(bins[:-1],cdf*100.,zorder=10, color='red')
ax2.set_ylabel('Eccentricty Frequency (count)', weight='bold')
ax3.set_ylabel('Eccentricty CDF (%)', weight='bold')
ax2.set_xlim(left=0.,right=1.)
ax2.set_ylim(bottom=0.,top=np.sum(n2))
ax3.set_ylim(bottom=0.,top=100.)
plt.show(block=False)

fname = 'IPACeccenHist2_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))



#### Planet Data Stats

len(plt_orbeccenISNONE)
len(data)

lines = list()
lines.append('From the confirmed exoplanet data table downloaded from IPAC on ' + str(date) + ' the following summaries apply')
lines.append('The table contains ' + str(len(data)) + ' lines')
lines.append('pl_orbsmax has ' + str(len(nanpl_orbsmaxInds)) + ' None or nan data fields')
lines.append('st_mass has ' + str(len(nanst_massInds)) + ' None or nan data fields')
lines.append('nanpl_orbperInds has ' + str(len(nanpl_orbperInds)) + ' None or nan data fields')
lines.append('pl_radj has ' + str(len(nanpl_radjInds)) + ' None or nan data fields')
lines.append('pl_orbeccen has ' + str(len(plt_orbeccenISNONE)) + ' None or nan data fileds')



starsWithPlanets = setOfStarsWithKnownPlanets(data)



#### Plot histogram of exoplanets around stars for each exoplanet
plt.figure(365468461)
plt.hist(st_dists, bins=np.logspace(np.log10(np.min(st_dists)),np.log10(np.max(st_dists))), color='red', alpha=0.5, label='Confirmed Exoplanets')#, 80)
plt.gca().set_xscale("log")
plt.xlabel('Host Star Distance (pc)', weight='bold')
plt.ylabel('Counts', weight='bold')
plt.legend()
plt.show(block=False)
fname = 'confirmedPlanStarDists2_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
#plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))

#### Plot histogram of stars
if not os.path.exists(folder):#Folder must exist
    raise ValueError('%s not found'%folder)
if not os.path.exists(PPoutpath):#PPoutpath must exist
    raise ValueError('%s not found'%PPoutpath) 
outspecfile = os.path.join(folder,'outspec.json')
if not os.path.exists(outspecfile):#outspec file not found
    raise ValueError('%s not found'%outspecfile) 
try:
    with open(outspecfile, 'rb') as g:
        outspec = json.load(g)
except:
    vprint('Failed to open outspecfile %s'%outspecfile)
    pass
outspec['cachedir'] = '/home/dean/.EXOSIMS/cache'
#Create Mission Object To Extract Some Plotting Limits
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=None, nopar=True, **outspec)


#Adding target star hist and 30pc mark
plt.hist(sim.SimulatedUniverse.TargetList.dist.value, bins=np.logspace(np.log10(0.95*np.min(sim.SimulatedUniverse.TargetList.dist.value)),np.log10(30.)), color='blue', alpha=0.5, label='Target List Stars')#, 80)
plt.plot([30.,30.],[0.,75.], color='k', label='30pc Line')
plt.legend()
fname = 'confandTLstarDistsAndTargetStars2_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
#plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))



#### Plot histogram
plt.figure(5617321)
plt.hist(sim.SimulatedUniverse.TargetList.dist.value, bins=np.logspace(np.log10(0.95*np.min(sim.SimulatedUniverse.TargetList.dist.value)),np.log10(30.)), color='blue', alpha=0.5, label='Target List Stars')#, 80)
plt.hist(st_dists, bins=np.logspace(np.log10(0.95*np.min(sim.SimulatedUniverse.TargetList.dist.value)),np.log10(30.)), color='red', alpha=0.5, label='Confirmed Exoplanets')#, 80)
plt.gca().set_xscale("log")
plt.xlabel('Host Star Distance (pc)', weight='bold')
plt.ylabel('Counts', weight='bold')
plt.legend()
plt.show(block=False)
fname = 'confandTLstarDists2_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
#plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))

