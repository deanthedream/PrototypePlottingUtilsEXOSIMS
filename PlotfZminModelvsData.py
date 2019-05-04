import os
from EXOSIMS.util.vprint import vprint
import random as myRand
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
try:
    import cPickle as pickle
except:
    import pickle
import os
import numpy as np
#from pylab import *
from numpy import nan
if not 'DISPLAY' in os.environ.keys(): #Check environment for keys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
else:
    import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import json
from EXOSIMS.util.read_ipcluster_ensemble import gen_summary
from EXOSIMS.util.read_ipcluster_ensemble import read_all
from numpy import linspace
from matplotlib.ticker import NullFormatter, MaxNLocator
from matplotlib import ticker
import astropy.units as u
import matplotlib.patheffects as PathEffects
import datetime
import re
from EXOSIMS.util.vprint import vprint
from astropy.coordinates import SkyCoord
from scipy.interpolate import griddata, interp1d
import astropy.constants as const
from matplotlib.colors import LogNorm
# folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/EXOSIMS/EXOSIMS/Scripts'))
# filename = 'sS_AYO6.json'#'sS_protoTimeKeeping.json'#'sS_AYO3.json'#sS_SLSQPstatic_parallel_ensembleJTWIN.json'#'sS_JTwin.json'#'sS_AYO4.json'#'sS_AYO3.json'
# #filename = 'sS_intTime6_KeplerLike2.json'
# scriptfile = os.path.join(folder,filename)
# sim = EXOSIMS.MissionSim.MissionSim(scriptfile)

# #sim.run_sim()

# sim.SurveySimulation.DRM

# from pylab import *
# import numpy as np
beta = np.arange(0,90)*1.0
lonMin = np.zeros(beta.shape[0])
# for i in np.arange(beta.shape[0]):
#     lonMin[i] = sim.ZodiacalLight.Lon_at_model11LambdaPartialZero(beta[i])

# fig = plt.figure(1)
# #plot(beta,lonMin)
# plt.plot(lonMin,beta)
# plt.xlabel('lonMin')
# plt.ylabel('Beta')
# plt.xlim(0,180)
# plt.ylim(0,90)
# plt.show(block=False)
# fig.savefig('/home/dean/Documents/SIOSlab/fZminModelBetavsLon'+'.png')

## Load Stark stuff for fZ calculation Look at starkAYO target List

PPoutpath = './'
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_405_19/WFIRSTcycle6core_CKL2_PPKL2'))
#filename = 'outspec.json'#'sS_protoTimeKeeping.json'#'sS_AYO3.json'#sS_SLSQPstatic_parallel_ensembleJTWIN.json'#'sS_JTwin.json'#'sS_AYO4.json'#'sS_AYO3.json'
#filename = 'sS_intTime6_KeplerLike2.json'
#scriptfile = os.path.join(folder,filename)
#sim2 = EXOSIMS.MissionSim.MissionSim(scriptfile=None, )
outspecPath = os.path.join(folder,'outspec.json')
try:
    with open(outspecPath, 'rb') as g:
        outspec = json.load(g)
except:
    vprint('Failed to open outspecfile %s'%outspecPath)
    pass

#Create Simulation Object
sim2 = EXOSIMS.MissionSim.MissionSim(scriptfile=None, nopar=True, **outspec)
SS = sim2.SurveySimulation
ZL = SS.ZodiacalLight
COMP = SS.Completeness
OS = SS.OpticalSystem
Obs = SS.Observatory
TL = SS.TargetList
TK = SS.TimeKeeping

sInds = np.arange(TL.nStars)
mode = sim2.SurveySimulation.mode

#fZmin, absTimefZmin = sim2.ZodiacalLight.calcfZmin(sInds, Obs, TL, TK, mode, SS.cachefname)

fZQuads = ZL.calcfZmin(sInds, Obs, TL, TK, SS.mode, SS.cachefname) # find fZmin to use in intTimeFilter
fZmin, absTimefZmin = ZL.extractfZmin_fZQuads(fZQuads)
        

#DELETE
# # #dec = np.zeros(len(imat2))
# # #for i in np.arange(len(imat2)):
# ra = TL.coords.ra.value*np.pi/180. -np.pi#The right ascension of the stars in the heliocentric ecliptic fixed frame
# #ra2 = ra[comp > 0.]
# dec = TL.coords.dec.value*np.pi/180.#The declinations of the stars in the heliocentric ecliptic fixed frame
# #dec2 = dec[comp > 0.]
# x = np.cos(dec)*np.cos(ra)#When dec2 =0, x/y=1
# y = np.cos(dec)*np.sin(ra)
# z = np.sin(dec)
# r_stars_equat = np.asarray([[x[i],y[i],z[i]] for i in np.arange(len(x))])
# r_stars_equat = np.divide(r_stars_equat,np.asarray([np.linalg.norm(r_stars_equat,axis=1).tolist()]).T)*u.AU#Target stars in the equatorial coordinate frame
# r_stars_eclip = Obs.equat2eclip(r_stars_equat,TK.currentTimeAbs,rotsign=1).value#target stars in the heliocentric ecliptic frame
# hEclipLat = np.arcsin(r_stars_eclip[:,2])
# hEclipLon = np.arctan2(r_stars_eclip[:,1],r_stars_eclip[:,0])
# for i in np.arange(absTimefZmin.shape[0]):
#     r_stars_eclip = Obs.equat2eclip(r_stars_equat,sim.TimeKeeping.currentTimeAbs,rotsign=1).value#target stars in the heliocentric ecliptic frame
#     hEclipLat = np.arcsin(r_stars_eclip[:,2])
#     hEclipLon = np.arctan2(r_stars_eclip[:,1],r_stars_eclip[:,0])



tmpdec = np.zeros(absTimefZmin.shape[0])
ra = np.zeros(absTimefZmin.shape[0])
r_obs = np.zeros((absTimefZmin.shape[0],3))
ra_obs = np.zeros(absTimefZmin.shape[0])
tmpdec_obs = np.zeros(absTimefZmin.shape[0])
ra_diff = np.zeros(absTimefZmin.shape[0])
dec_diff = np.zeros(absTimefZmin.shape[0])
for i in np.arange(absTimefZmin.shape[0]):
    r_targ = TL.starprop(sInds[i].astype(int),absTimefZmin[i],True) # spacecraft to 
    c = SkyCoord(r_targ[:,0],r_targ[:,1],r_targ[:,2],representation='cartesian')
    ra[i] = np.arctan2(c.y.value/np.sqrt(c.y.value**2. + c.x.value**2.),c.x.value/np.sqrt(c.y.value**2. + c.x.value**2.))
    tmpdec[i] = np.arcsin(np.abs(c.z.value[0])/np.sqrt(c.y.value**2. + c.x.value**2. + c.z.value**2.))
    # c.representation = 'spherical'
    # tmpdec[i] = c.dec.value
    # ra[i] = c.ra.value

    r_obs[i] = Obs.orbit(absTimefZmin[i], True)[0] # Sun to Spacecraft vector in HEE
    c2 = SkyCoord(r_obs[i][0],r_obs[i][1],r_obs[i][2],representation='cartesian')
    ra_obs[i] = np.arctan2(c2.y.value/np.sqrt(c2.y.value**2. + c2.x.value**2.),c2.x.value/np.sqrt(c2.y.value**2. + c2.x.value**2.))
    tmpdec_obs[i] = np.arcsin(np.abs(c2.z.value)/1.)
    # c2.representation = 'spherical'
    # tmpdec_obs[i] = c2.dec.value
    # ra_obs[i] = c2.ra.value
    # if ra_obs[i] < 0.:
    #     ra_obs[i] = -ra_obs[i]
    ra_diff[i] = 180./np.pi*np.max([ra[i]-ra_obs[i],ra_obs[i]-ra[i]])
    if ra_diff[i] > 180.:
        ra_diff[i] = (360.-ra_diff[i])
    ra_diff[i] = 180.-ra_diff[i]#convert from r_sun/SC=180. to r_solar/SC=0
    dec_diff[i] = 180./np.pi*np.max([tmpdec[i]-tmpdec_obs[i],tmpdec_obs[i]-tmpdec[i]])


#### Calculate fZ grid
# table 17 in Leinert et al. (1998)
# Zodiacal Light brightness function of solar LON (rows) and LAT (columns)
# values given in W m−2 sr−1 μm−1 for a wavelength of 500 nm
#path = os.path.split(inspect.getfile(self.__class__))[0]
lon = np.linspace(start=0.,stop=180.,num=361)
lat = np.linspace(start=0.,stop=90.,num=181)
path = '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/ZodiacalLight/'
Izod = np.loadtxt(os.path.join(path, 'Leinert98_table17.txt'))*1e-8 # W/m2/sr/um
# create data point coordinates
lon_pts = np.array([0., 5., 10., 15., 20., 25., 30., 35., 40., 45., 60., 75., 90.,
        105., 120., 135., 150., 165., 180.]) # deg
lat_pts = np.array([0., 5., 10., 15., 20., 25., 30., 45., 60., 75., 90.]) # deg
y_pts, x_pts = np.meshgrid(lat_pts, lon_pts)
points = np.array(zip(np.concatenate(x_pts), np.concatenate(y_pts)))
# create data values, normalized by (90,0) value
z = Izod/Izod[12,0]
values = z.reshape(z.size)

fZ = np.zeros([len(lon),len(lat)])
xi = list()
for lo in np.arange(len(lon)):
    for la in np.arange(len(lat)):
        xi.append([lon[lo],lat[la]])
xi = np.asarray(xi)
fZ = np.zeros([len(xi), 2])
# interpolates 2D
fbeta = griddata(points, values, xi)#zip([lon[lo]], [lat[la]]))#lon[lo], lat[la]))

# wavelength dependence, from Table 19 in Leinert et al 1998
# interpolated w/ a quadratic in log-log space
lam = SS.mode['lam']
zodi_lam = np.array([0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2, 2.2, 3.5,
        4.8, 12., 25., 60., 100., 140.]) # um
zodi_Blam = np.array([2.5e-8, 5.3e-7, 2.2e-6, 2.6e-6, 2.0e-6, 1.3e-6,
        1.2e-6, 8.1e-7, 1.7e-7, 5.2e-8, 1.2e-7, 7.5e-7, 3.2e-7, 1.8e-8,
        3.2e-9, 6.9e-10]) # W/m2/sr/um
x = np.log10(zodi_lam)
y = np.log10(zodi_Blam)
logf = interp1d(x, y, kind='quadratic')
f = 10.**(logf(np.log10(lam.to('um').value)))*u.W/u.m**2./u.sr/u.um
h = const.h                             # Planck constant
c = const.c                             # speed of light in vacuum
ephoton = h*c/lam/u.ph                  # energy of a photon
F0 = TL.OpticalSystem.F0(lam)           # zero-magnitude star (in ph/s/m2/nm)
f_corr = f/ephoton/F0                   # color correction factor

#fZ[lo,la]
fZ = fbeta*f_corr.to('1/arcsec2')
fZ2 = np.zeros([len(lon),len(lat)])
for lo in np.arange(len(lon)):
    for la in np.arange(len(lat)):
        fZ2[lo,la] = np.round(fZ[lo*len(lat) + la].value,14)
fZ2 = np.asarray(fZ2)
#### Calc fZmin ########################################
fZlaminInds = list()
fZlamin = list()
tmpfZ2 = fZ2
tmpfZ2 = np.nan_to_num(tmpfZ2)
tmpfZ2[tmpfZ2==0.] = 1000.
for la in np.arange(len(lat)):
    fZlaminInds.append(np.argmin(np.asarray(tmpfZ2)[:,la]))
    fZlamin.append(np.min(np.asarray(tmpfZ2)[:,la]))
#####################################################

fig2 = plt.figure(2,figsize=(6,4))
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold') 
fZ2[fZ2>=1e-8*1.000001]= np.nan
#CS = plt.contourf(lon, lat, np.transpose(fZ2), locator=ticker.LogLocator())#xi[:,0],xi[:,1]
#CS = plt.imshow(np.flip(fZ2,axis=1), extent=[lon.min(), lon.max(), lat.min(), lat.max()], norm=LogNorm(vmin=np.nanmin(fZ2), vmax=np.nanmax(fZ2)))
CS = plt.pcolormesh(lon, lat, np.transpose(fZ2), norm=LogNorm(vmin=np.nanmin(fZ2), vmax=1e-8),linewidth=0,rasterized=True)#vmin=np.nanmin(fZ2), vmax=np.nanmax(fZ2))#, locator=ticker.LogLocator())
CS.set_edgecolor('face')
plt.scatter(ra_diff, dec_diff, color='black', alpha=0.5, s=2)
plt.scatter(lon[fZlaminInds],lat, color='red',marker='s', alpha=0.5, s=2)
cbar = plt.colorbar(CS)
cbar.set_label('Zodiacal Light Intensity\n' + r'$f_Z(l,b)$ in $W m^{-2} sr^{-1} \mu m^{-1}$', weight='bold')
  

plt.xlabel('Geocentric Ecliptic Longitude, ' + r'$l$' + '\n' + r'$\ |l_{i/SC} - l_{SC/\odot}|$ ($^\circ$)', weight='bold')
plt.ylabel('Geocentric Ecliptic Latitude, ' + r'$b$' + '\n' + r'$\ |b_{i/SC} - b_{SC/\odot}|$ ($^\circ$)', weight='bold')
plt.xlim(0,180)
plt.ylim(0,90)
plt.subplots_adjust(bottom=0.2)
plt.show(block=False)

date = unicode(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'fZminCalculatedBetavsLon_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))


