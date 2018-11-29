#Plotting Fits Files


from pylab import *
from astropy.io import fits
import scipy.interpolate
import astropy.units as u
import numpy as np


#### occ_trans
pth = '/home/dean/Documents/exosims/fitFilesFolder/HabExMay3/G_VC6_occ_trans_asec.fits'
dat = fits.open(pth)[0].data
WA, D = (dat[0], dat[1]) if dat.shape[0] == 2 else (dat[:,0], dat[:,1])
Dinterp = scipy.interpolate.interp1d(WA.astype(float), D.astype(float),
                    kind='cubic', fill_value=fill, bounds_error=False)
WAint = np.asarray([0.045     , 0.045     , 0.19756445, 0.18430155, 0.13013536,
           0.045     , 0.045     , 1.41807054, 0.045     , 0.76020878,
           0.06810318, 0.0556777 , 0.045     , 0.045     , 0.045     ,
           0.045     , 0.045     , 0.045     , 0.045     , 0.045     ,
           0.045     , 0.045     , 0.045     , 0.045     , 0.045     ,
           0.045     , 0.63738941, 0.10301575, 0.09035079, 0.07160575,
           0.045     , 0.13248153, 0.045     , 0.045     , 0.045     ,
           0.06097561])*u.arcsec

lam = 500.*u.nm
syst_occ_trans = lambda l, s: np.array(Dinterp((s*lam/l).to('arcsec').value), ndmin=1)
#NO

#### core_thruput
pth = "/home/dean/Documents/exosims/fitFilesFolder/HabExMay3/G_VC6_core_thruput_asec.fits"
dat = fits.open(pth)[0].data
WA, D = (dat[0], dat[1]) if dat.shape[0] == 2 else (dat[:,0], dat[:,1])
Dinterp = scipy.interpolate.interp1d(WA.astype(float), D.astype(float),
                    kind='cubic', fill_value=fill, bounds_error=False)
syst_core_thruput = lambda l, s: np.array(Dinterp((s*lam/l).to('arcsec').value), ndmin=1)

#### mean_intensity
pth = "/home/dean/Documents/exosims/fitFilesFolder/HabExMay3/G_VC6_core_mean_intensity_asec.fits"
dat = fits.open(pth)[0].data
WA, D = (dat[0], dat[1]) if dat.shape[0] == 2 else (dat[:,0], dat[:,1])
Dinterp = scipy.interpolate.interp1d(WA.astype(float), D.astype(float),
                    kind='cubic', fill_value=fill, bounds_error=False)
syst_core_mean_intensity = lambda l, s: np.array(Dinterp((s*lam/l).to('arcsec').value), ndmin=1)


# #### core_area
# pth = 
# dat = fits.open(pth)[0].data
# WA, D = (dat[0], dat[1]) if dat.shape[0] == 2 else (dat[:,0], dat[:,1])
# Dinterp = scipy.interpolate.interp1d(WA.astype(float), D.astype(float),
#                     kind='cubic', fill_value=fill, bounds_error=False)
# syst_core_area = lambda l, s: np.array(Dinterp((s*lam/l).to('arcsec').value), ndmin=1)

