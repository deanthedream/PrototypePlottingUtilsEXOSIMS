# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.ZodiacalLight import ZodiacalLight
import numpy as np
import os, inspect
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d, griddata
from scipy.interpolate import CubicSpline

from scipy.optimize import minimize
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import time
from matplotlib.mlab import griddata as GD
from time import sleep
from matplotlib import cm
from pylab import rc
from pylab import rcParams
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

class KeithlyZL(ZodiacalLight):
    """Stark Zodiacal Light class
    
    This class contains all variables and methods necessary to perform
    Zodiacal Light Module calculations in exoplanet mission simulation using
    the model from Stark et al 2014 and Keithly et al 2018.
    
    """
    def __init__(self, magZ=23, magEZ=22, varEZ=0, **specs):
        """
        """
        ZodiacalLight.__init__(self, magZ=23, magEZ=22, varEZ=0, **specs)
        #Here we calculate the Zodiacal Light Model


    def fZ(self, Obs, TL, sInds, currentTime, mode):
        """Returns surface brightness of local zodiacal light using model11 approximation
        
        Args:
            Obs (Observatory module):
                Observatory class object
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            mode (dict):
                Selected observing mode
        
        Returns:
            fZ (astropy Quantity array):
                Surface brightness of zodiacal light in units of 1/arcsec2
        
        """
        # observatory positions vector in heliocentric ecliptic frame
        r_obs = Obs.orbit(currentTime, eclip=True)
        # observatory distances (projected in ecliptic plane)
        r_obs_norm = np.linalg.norm(r_obs[:,0:2], axis=1)*r_obs.unit
        # observatory ecliptic longitudes
        r_obs_lon = np.sign(r_obs[:,1])*np.arccos(r_obs[:,0]/r_obs_norm).to('deg').value
        # longitude of the sun
        lon0 = (r_obs_lon + 180) % 360
        
        # target star positions vector in heliocentric true ecliptic frame
        r_targ = TL.starprop(sInds, currentTime, eclip=True)
        # target star positions vector wrt observatory in ecliptic frame
        r_targ_obs = (r_targ - r_obs).to('pc').value
        # tranform to astropy SkyCoordinates
        coord = SkyCoord(r_targ_obs[:,0], r_targ_obs[:,1], r_targ_obs[:,2],
                representation='cartesian').represent_as('spherical')
        # longitude and latitude absolute values for Leinert tables
        lon = coord.lon.to('deg').value - lon0
        lat = coord.lat.to('deg').value
        lon = abs((lon + 180) % 360 - 180)
        lat = abs(lat)

        #Model 11 Params
        params = np.array([ 1.40425309e+00,  1.22778836e+00,  1.53803793e+00,  1.91316004e+00,
        2.91054227e+00,  2.89184866e+00, -1.06796214e-08, -2.51405713e-01,
        6.95613955e-01])

        def model11(lon,lat,params):
            #Defines the Model Calculating fbeta portion of fZ. This model fits Leinert98 to within 10%
            #This fit is to np.log(Izod/min(Izod))
            #To represent Izod from Stark.py, take np.exp(tmp)*Izodmin/Izod120
            tmp = 1/(params[0]*(np.cos(lon*np.pi/180 + np.pi)+params[1] + 1e-8))*(1/(np.cos(2*lat*np.pi/180 + np.pi)+params[2] + 1e-8) +  params[7]) + \
            1/(params[3]*(np.cos(lon*np.pi/180)+params[4] + 1e-8))*(np.cos(2*lat*np.pi/180) + params[8]) + \
            params[6]
            return tmp

        zodi_lam = np.array([0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2, 2.2, 3.5,
                4.8, 12, 25, 60, 100, 140]) # um #See Table 19 Leinert 1997
        zodi_Blam = np.array([2.5e-8, 5.3e-7, 2.2e-6, 2.6e-6, 2.0e-6, 1.3e-6,
                1.2e-6, 8.1e-7, 1.7e-7, 5.2e-8, 1.2e-7, 7.5e-7, 3.2e-7, 1.8e-8,
                3.2e-9, 6.9e-10]) # W/m2/sr/um #See Table 19 Leinert 1997
        self.logf2 = CubicSpline(np.log10(zodi_lam), zodi_Blam, bc_type='clamped')
        lam = mode['lam']

        def fZhat(lon,lat,lam,params,TL):#the new way we're doing it
            tmp = model11(lon,lat,params)
            #Izod[12,0] = 2.59e-06
            Izod120 = 2.59e-06
            #np.min(self.Izod) = 7.2e-07
            Izodmin = 7.2e-07
            fbeta = np.exp(tmp)*Izodmin/Izod120

            f = 10.**(self.logf2(np.log10(lam.to('um').value)))*u.W/u.m**2/u.sr/u.um
            h = const.h                             # Planck constant
            c = const.c                             # speed of light in vacuum
            ephoton = h*c/lam/u.ph                  # energy of a photon
            F0 = TL.OpticalSystem.F0(lam)           # zero-magnitude star (in ph/s/m2/nm)
            f_corr = f/ephoton/F0                   # color correction factor
            return fbeta*f_corr.to('1/arcsec2')

        return fZhat(lon,lat,lam,params,TL)

    def get_fZminValue(self, TL, sInds):
        """ Retrieves the minimum fZ value for the specified stars
        """
        #



    def get_fZminAbsTime(self):
        """
        Returns:
            AbsTime (Time Quantity Array) - The time at which the fzmin of the stars specified in sInds will occur
        """

    def Lon_at_model11LambdaPartialZero(self,beta):
        """Finds the Heliocentric Earth Ecliptic Longitude at a given beta that produces the minimum fZ
        Args:
            beta (deg):
                Heliocentric Earth Ecliptic latitude
        Returns:
            lon (deg):
                Heliocentric Earth Ecliptic longitude w.r.t solar longitude
        #This is the partial of model11 with respect to lambda (ecliptic longitude wrt solar longitude)
        #The equation was solved for where the partial is 0
        """
        #Model 11 Params
        params = np.array([ 1.40425309e+00,  1.22778836e+00,  1.53803793e+00,  1.91316004e+00,
        2.91054227e+00,  2.89184866e+00, -1.06796214e-08, -2.51405713e-01,
        6.95613955e-01])
        w0 = params[0]
        w1 = params[1]
        w2 = params[2]
        w3 = params[3]
        w4 = params[4]
        w5 = params[5]
        w6 = params[6]
        w7 = params[7]
        w8 = params[8]

        C1 = (w7 + (np.cos(2*beta*np.pi/180 + np.pi) + w2 + 1e-8)**(-1))*(np.pi/180)*(-1)
        C2 = (np.cos(2*beta*np.pi/180) + w8)*(np.pi/180)*(-1)
        if C1/C2 < 0:
            C1 = 200
            C2 = 1
        C3 = ((w0*C1)/(w3*C2))**(-0.5)*(w1 + 1e-8) - w4 - 1e-8
        C4 = ((w0*C1)/(w3*C2))**(-0.5)*w0 + w3


        #C3 = w1/w0 + 1e-8/w0 - (w4 + 1e-8)/(w0*(w0*C1)**(-0.5))
        #C4 = (w3*(w3*C2)**(-0.5))/(w0*(w0*C1)**(-0.5)) + 1
        lon = (180/np.pi)*(np.arccos(C3/C4))

        return lon