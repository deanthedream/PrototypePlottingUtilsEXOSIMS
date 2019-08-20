""" Telescope Limiting Magnitudes
    Based off 'Telescope Limiting Magnitudes' by: Bradley E. Schaefer

    Written By: Dean Keithly
"""

import numpy as np



def telescope_Limiting_Magnitude1(N,D):
    """ (1) from publication above. Correct to within 1-2 magnitudes
    Args:
        N (float) - some normalization constant. should range from 8.8 (Dmitroff and Baker 1945) to 10.7 (Sidgwick 1971) 
        D (float) - telescope aperture in inches
    Returns:
        m (float) - telescope limiting magnitude
    """
    m = N + 5.*np.log(D)
    return m

def telescope_Limiting_Magnitude2():
    """
    """
    I = 
    Fb = 1.41 #binocular to monocular correction
    Fe = Fe(B, kv, Z)
    Ft = Ft(n,t1,Ds,D)
    Fp = Fp(D,De,M)
    Fa = 1.0
    Fr = Fr(theta,M)
    Fsc = Fsc(De, M, D, B)
    Fc = Fc(B,V)
    Fs = 1.0
    Istar = I_star(I,Fb,Fe,Ft,Fp,Fa,Fr,Fsc,Fc,Fs)
    m = -16.57-2.5*np.log(Istar)


def Fe(B, kv, Z):
    """ Atmospheric Extinction correction
    Args:
        B (float) - observed surface brightness of the background
                in units of millimicroLamberts (m mu L
        kv (float) - extinction coefficient of the atmosphere
            (depends on effective eye wavelength: day-5550 Angstroms night-5100 Angstroms)
            (For good conditions kv is 0.2 mag per airmass, typical weather has kv is 0.30 mag per air mass)
        Z (float) - Zenith distance of the star
    Return:
        Fe (float) - tmospheric Extinction correction
    """
    if np.log(B) < 3.17:
        q = 1.2
    else: #np.log(B) >= 3.17
        q = 1.0
    Fe = np.exp(q*kv/np.cos(Z)/2.5)
    return Fe

def Ft(n,t1,Ds,D):
    """ Transmission Factor from newtonian telescope second mirror and
    optical transmission losses
    Args:
        n (int) - number of transmission surfaces
        t1 (float) - transmission surface loss factor
        Ds (float) - diameter of the secondary mirror
        D (float) - diameter of the primary mirror
    Return:
        Ft (float)
    """
    Ft = 1./(t1**np.float(n)*(1.- (Ds/D)**2.))
    return Ft

def Fp(D,De,M):
    """ Correction for light loss outside pupil
    Args:
        D (float) - telescope diameter
        De (float) - pupil diameter
        M (float) - magnification
    Returns:
        Fp (float) - light loss outside pupil correction factor
    """
    if De < D/M:
        Fp = (D/(M*De))**2.
    else: # De >= D/M
        Fp = 1.0
    return Fp

#Fm
#Fa

def Fr(theta,M):
    """ Approximate Correction Factor if source is extended
    Args:
        theta (float) - angular size of source in arc seconds
        M (float) - magnification factor of telescope
    Returns:
        Fr (floaT) - correction factor for extended source
    """
    if 2.*theta*M > 900.: # in arc minutes
        Fr = (2.*theta*M/900.)**0.5
    else: #2.*theta*M <= 900.
        Fr = 1.
    return Fr

def Fsc(De, M, D, B):
    """ Correction factor for ratio of efficiencies over
        utilized part of the eye
    Args:
        De (float) - 
        M (float) - 
        D (float) - 
        B (float) - 
    Returns:
        Fsc (float) - 
    """
    if De > D/M and np.log(B) > 3.17:
        Fsc = (De*M/D)*(1.-np.exp(-0.026*(D/M)**2.))/(1.-np.exp(-0.026*De**2.))
    elif De > D/M and np.log(B) <= 3.17:
        Fsc = (1.-(D/12.4/M)**4.)/(1.-(De/12.4)**4.)
    else: #not sure if this is right
        Fsc = 1.
    return Fsc

def F0(B,T=2360.):
    """ Normalization Factor for the sensitivity curves
    Args:
        B (float) - 
        T (float) - 
    Returns:
        F0 (float) - 
    """
    if np.log(B) < 3.17:
        F0 = Nn(T)/Nd(T)
    else: #np.log(B) >= 3.17
        F0 = 1.0
    return  F0

def Nn(T):

def Nd(T):

def Fv(B,T):
    """ Correction for night
        vision because the reported magnitudes of stars are in the
        V magnitude system which has a similar spectral response
        as the day vision sensitivity curve. The need for such a
        correction is apparent ifwe consider the case of two stars
        with equal V magnitude but different color.
    Args:
        B (float) - 
        T (float) - color temperature in kelvin
    Returns:
        Fv (float) - 
    """
    if np.log(B) < 3.17:
        Fv = Nd(T)/Nn(T)
    else: #np.log(B) >= 3.17
        Fv = 1.0
    return Fv

def Fc(B,V):
    """The two correction factors F0 and Fv can be combined into
        one color-correction factor which I will call Fc. This combination
        has the advantage that the normalization constants
        cancel out. I have evaluated the necessary integrals
        by numerical integration and have related the color temperature
        to the color index (that is, the (B —V)) of the star
        in question.
    Args:
        B (float) - 
        V (float) - 
    Returns:
        Fc (float) - 
    """
    if np.log(B) < 3.17:
        Fc = np.exp((1.-(B-V)/2.)/-2.5)
    else: #np.log(B) >= 3.17
        Fc = np.exp(0.)
    return Fc

def I_star(I,Fb,Fe,Ft,Fp,Fa,Fr,Fsc,Fc,Fs=1.):
    """
    Returns:
        Istar (float)
    """
    Istar = I*Fb*Fe*Ft*Fp*Fa*Fr*Fsc*Fc*Fs
    return Istar

def Bs(B,Fb,Ft,Fp,Fa,Fsc,Fm,Fc):
    """ Note: Best sites have brightness Bsprime 21.8 mag per arc sec squared
    Normal sites have brightness Bsprime 21.0 mag per arc sec squared
    Bs is related to Bsprime by Bs = 34.08*np.exp(20.7233-0.92104*Bsprime)
    """
    Bs = B*Fb*Ft*Fp*Fa*Fsc*Fm*Fc
    return Bs

def f_I():
    """
    Args:
        B (float) - observed surface brightness of background in milli micro Lambers m mu L
    Returns:
        I (float) - star brightness in footcandles
    """
    if np.log(B) < 3.17:
        C = np.exp(-9.8)
        K = np.exp(-1.9)
    else: #np.log(B) >= 3..17
        C = np.exp(-8.35)
        K = np.exp(-5.9)

    I = C*(1.+(K*B)**0.5)**2.
    return I
