import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

Re = 6371.
alt = 600.
a = Re+alt
e = 0.003 #as large as 0.01
i = np.pi/2.*80./90. #idk using 80 deg but it wont be that high
w = 0. #Argument of periapsis
W = 0. #longitude of the ascending node
nu = 0. #true anomaly
r_Earth_sun = np.asarray([1.,0.,0.])#Assumed Sun to Earth vector

def XYZ_from_KOE(a,e,i,w,W,nu):
    r = a*(1-e**2.)/(1.+e*np.cos(nu))
    return r*np.asarray([np.cos(W)*np.cos(w+nu) - np.sin(W)*np.sin(w+nu)*np.cos(i),\
        np.sin(W)*np.cos(w+nu) + np.cos(W)*np.sin(w+nu)*np.cos(i),\
        np.sin(w+nu)*np.sin(i)])


#### Compute the nu(s) where eclipse occurs
#Assuming the Earth is a sphere, its projection is a circle, and eclipse occurs exactly at this transition point (no unumbra or penumbra)
#I calculate when, in true anomaly, eclipses occur
#solve where Re^2 = Y^2 + Z^2

def quarticCoefficients(Re,a,e,i,omega,Omega):
    """ Calculates coefficients of the quartic expression solving for the intersection between a circle with radius r and ellipse with semi-major axis a
    semi-minor axis b, and the center of the circle at x and y.
    Coefficients for the quartic of form x**4 + A*x**3 + B*x**2 + C*x + D = 0
    Args:
        Re (numpy array):
            semi-major axis of the projected ellipse
        a (numpy array):
            semi-minor axis of the projected ellipse
        e (numpy array):
            x position of the center of the projected ellipse
        i (numpy array):
            y position of the center of the projected ellipse
        omega (numpy array):
            argument of periapsis in radians
        Omega (numpy array):
            longitude of the ascending node in radians
    Returns:
        A (numpy array):
            coefficients of x^3
        B (numpy array):
            coefficients of x^2
        C (numpy array):
            coefficients of x
        D (numpy array):
            constants
    """
    # a_0 = -Re**4.*e**4 + 2.*Re**2.*a**2.*e**6.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(i)**2 - 2.*Re**2.*a**2.*e**6.*np.sin(Omega)**2.*np.sin(omega)**2 - 2.*Re**2.*a**2.*e**6.*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*e**6.*np.sin(Omega)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*e**6.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2 - 2.*Re**2.*a**2.*e**6.*np.sin(omega)**2.*np.cos(Omega)**2 - 2.*Re**2.*a**2.*e**6.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*e**6.*np.cos(Omega)**2.*np.cos(omega)**2 - 4.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(i)**2 + 4.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2\
    #     + 4.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 4.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.cos(omega)**2 - 4.*Re**2.*a**2.*e**4.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2 + 4.*Re**2.*a**2.*e**4.*np.sin(omega)**2.*np.cos(Omega)**2 + 4.*Re**2.*a**2.*e**4.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 4.*Re**2.*a**2.*e**4.*np.cos(Omega)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(i)**2 - 2.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2 - 2.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.cos(omega)**2.\
    #     + 2.*Re**2.*a**2.*e**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2 - 2.*Re**2.*a**2.*e**2.*np.sin(omega)**2.*np.cos(Omega)**2 - 2.*Re**2.*a**2.*e**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*e**2.*np.cos(Omega)**2.*np.cos(omega)**2 - a**4.*e**8*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**4 + 2.*a**4.*e**8*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**2 - a**4.*e**8*np.sin(Omega)**4.*np.sin(omega)**4 - 2.*a**4.*e**8*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**4.*np.cos(omega)**2\
    #     + 4.*a**4.*e**8*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 2.*a**4.*e**8*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(omega)**2 - a**4.*e**8*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 2.*a**4.*e**8*np.sin(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 - a**4.*e**8*np.sin(Omega)**4.*np.cos(omega)**4 - 2.*a**4.*e**8*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**4 + 4.*a**4.*e**8*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**2 - 2.*a**4.*e**8*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2 - 4.*a**4.*e**8*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**2.\
    #     + 8*a**4.*e**8*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 4.*a**4.*e**8*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(omega)**2 - 2.*a**4.*e**8*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4 + 4.*a**4.*e**8*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4 - 2.*a**4.*e**8*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(omega)**4\
    #     - a**4.*e**8*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**4 + 2.*a**4.*e**8*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**2 - a**4.*e**8*np.sin(omega)**4.*np.cos(Omega)**4 - 2.*a**4.*e**8*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**2 + 4.*a**4.*e**8*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 - 2.*a**4.*e**8*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(omega)**2 - a**4.*e**8*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 2.*a**4.*e**8*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 - a**4.*e**8*np.cos(Omega)**4.*np.cos(omega)**4 + 4.*a**4.*e**6.*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**4.\
    #     - 8*a**4.*e**6.*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**2 + 4.*a**4.*e**6.*np.sin(Omega)**4.*np.sin(omega)**4 + 8*a**4.*e**6.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**4.*np.cos(omega)**2 - 16.*a**4.*e**6.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 8*a**4.*e**6.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(omega)**2 + 4.*a**4.*e**6.*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 - 8*a**4.*e**6.*np.sin(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 + 4.*a**4.*e**6.*np.sin(Omega)**4.*np.cos(omega)**4\
    #     + 8*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**4 - 16.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**2 + 8*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2 + 16.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**2 - 32.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 16.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(omega)**2 + 8*a**4.*e**6.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4 - 16.*a**4.*e**6.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4.\
    #     + 8*a**4.*e**6.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(omega)**4 + 4.*a**4.*e**6.*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**4 - 8*a**4.*e**6.*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**2 + 4.*a**4.*e**6.*np.sin(omega)**4.*np.cos(Omega)**4 + 8*a**4.*e**6.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**2 - 16.*a**4.*e**6.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 + 8*a**4.*e**6.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(omega)**2 + 4.*a**4.*e**6.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 - 8*a**4.*e**6.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4\
    #     + 4.*a**4.*e**6.*np.cos(Omega)**4.*np.cos(omega)**4 - 6.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**4 + 12.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**2 - 6.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(omega)**4 - 12.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**4.*np.cos(omega)**2 + 24.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 12.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(omega)**2 - 6.*a**4.*e**4.*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 12.*a**4.*e**4.*np.sin(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 - 6.*a**4.*e**4.*np.sin(Omega)**4.*np.cos(omega)**4.\
    #     - 12.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**4 + 24.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**2 - 12.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2 - 24.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**2 + 48*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 24.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(omega)**2\
    #     - 12.*a**4.*e**4.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4 + 24.*a**4.*e**4.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4 - 12.*a**4.*e**4.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(omega)**4 - 6.*a**4.*e**4.*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**4 + 12.*a**4.*e**4.*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**2 - 6.*a**4.*e**4.*np.sin(omega)**4.*np.cos(Omega)**4 - 12.*a**4.*e**4.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**2 + 24.*a**4.*e**4.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 - 12.*a**4.*e**4.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(omega)**2.\
    #     - 6.*a**4.*e**4.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 12.*a**4.*e**4.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 - 6.*a**4.*e**4.*np.cos(Omega)**4.*np.cos(omega)**4 + 4.*a**4.*e**2.*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**4 - 8*a**4.*e**2.*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**2 + 4.*a**4.*e**2.*np.sin(Omega)**4.*np.sin(omega)**4 + 8*a**4.*e**2.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**4.*np.cos(omega)**2 - 16.*a**4.*e**2.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 8*a**4.*e**2.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(omega)**2\
    #     + 4.*a**4.*e**2.*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 - 8*a**4.*e**2.*np.sin(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 + 4.*a**4.*e**2.*np.sin(Omega)**4.*np.cos(omega)**4 + 8*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**4 - 16.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**2 + 8*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2 + 16.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**2\
    #     - 32.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 16.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(omega)**2 + 8*a**4.*e**2.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4 - 16.*a**4.*e**2.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4 + 8*a**4.*e**2.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(omega)**4 + 4.*a**4.*e**2.*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**4 - 8*a**4.*e**2.*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**2 + 4.*a**4.*e**2.*np.sin(omega)**4.*np.cos(Omega)**4 + 8*a**4.*e**2.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**2.\
    #     - 16.*a**4.*e**2.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 + 8*a**4.*e**2.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(omega)**2 + 4.*a**4.*e**2.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 - 8*a**4.*e**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 + 4.*a**4.*e**2.*np.cos(Omega)**4.*np.cos(omega)**4 - a**4.*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**4 + 2.*a**4.*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**2 - a**4.*np.sin(Omega)**4.*np.sin(omega)**4\
    #     - 2.*a**4.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**4.*np.cos(omega)**2 + 4.*a**4.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 2.*a**4.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(omega)**2 - a**4.*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 2.*a**4.*np.sin(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 - a**4.*np.sin(Omega)**4.*np.cos(omega)**4 - 2.*a**4.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**4 + 4.*a**4.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**2 - 2.*a**4.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2 - 4.*a**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**2.\
    #     + 8*a**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 4.*a**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(omega)**2 - 2.*a**4.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4 + 4.*a**4.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4 - 2.*a**4.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(omega)**4 - a**4.*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**4\
    #     + 2.*a**4.*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**2 - a**4.*np.sin(omega)**4.*np.cos(Omega)**4\
    #     - 2.*a**4.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**2 + 4.*a**4.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 - 2.*a**4.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(omega)**2 - a**4.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 2.*a**4.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 - a**4.*np.cos(Omega)**4.*np.cos(omega)**4

    # a_1 = -4.*Re**4.*e**3 + 4.*Re**2.*a**2.*e**5.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(i)**2 - 4.*Re**2.*a**2.*e**5.*np.sin(Omega)**2.*np.sin(omega)**2 - 4.*Re**2.*a**2.*e**5.*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 4.*Re**2.*a**2.*e**5.*np.sin(Omega)**2.*np.cos(omega)**2 + 4.*Re**2.*a**2.*e**5.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2 - 4.*Re**2.*a**2.*e**5.*np.sin(omega)**2.*np.cos(Omega)**2 - 4.*Re**2.*a**2.*e**5.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 4.*Re**2.*a**2.*e**5.*np.cos(Omega)**2.*np.cos(omega)**2 - 8*Re**2.*a**2.*e**3*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(i)**2.\
    #     + 8*Re**2.*a**2.*e**3*np.sin(Omega)**2.*np.sin(omega)**2 + 8*Re**2.*a**2.*e**3*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 8*Re**2.*a**2.*e**3*np.sin(Omega)**2.*np.cos(omega)**2 - 8*Re**2.*a**2.*e**3*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2 + 8*Re**2.*a**2.*e**3*np.sin(omega)**2.*np.cos(Omega)**2 + 8*Re**2.*a**2.*e**3*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 8*Re**2.*a**2.*e**3*np.cos(Omega)**2.*np.cos(omega)**2 + 4.*Re**2.*a**2.*e*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(i)**2\
    #     - 4.*Re**2.*a**2.*e*np.sin(Omega)**2.*np.sin(omega)**2 - 4.*Re**2.*a**2.*e*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 4.*Re**2.*a**2.*e*np.sin(Omega)**2.*np.cos(omega)**2 + 4.*Re**2.*a**2.*e*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2 - 4.*Re**2.*a**2.*e*np.sin(omega)**2.*np.cos(Omega)**2 - 4.*Re**2.*a**2.*e*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 4.*Re**2.*a**2.*e*np.cos(Omega)**2.*np.cos(omega)**2

    # a_2 = -6.*Re**4.*e**2 + 2.*Re**2.*a**2.*e**6.*np.sin(Omega)**2.*np.sin(omega)**2 + 2.*Re**2.*a**2.*e**6.*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*e**6.*np.sin(omega)**2.*np.cos(Omega)**2 + 2.*Re**2.*a**2.*e**6.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(i)**2 - 6.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2 - 6.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*e**4.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2 - 6.*Re**2.*a**2.*e**4.*np.sin(omega)**2.*np.cos(Omega)**2.\
    #     - 6.*Re**2.*a**2.*e**4.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*e**4.*np.cos(Omega)**2.*np.cos(omega)**2 - 4.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(i)**2 + 6.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2 + 6.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 4.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.cos(omega)**2\
    #     - 4.*Re**2.*a**2.*e**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2\
    #     + 6.*Re**2.*a**2.*e**2.*np.sin(omega)**2.*np.cos(Omega)**2 + 6.*Re**2.*a**2.*e**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 4.*Re**2.*a**2.*e**2.*np.cos(Omega)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(i)**2 - 2.*Re**2.*a**2.*np.sin(Omega)**2.*np.sin(omega)**2 - 2.*Re**2.*a**2.*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*np.sin(Omega)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2 - 2.*Re**2.*a**2.*np.sin(omega)**2.*np.cos(Omega)**2 - 2.*Re**2.*a**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*np.cos(Omega)**2.*np.cos(omega)**2\
    #     - 2.*a**4.*e**8*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**2 + 2.*a**4.*e**8*np.sin(Omega)**4.*np.sin(omega)**4 + 2.*a**4.*e**8*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**4.*np.cos(omega)**2 - 4.*a**4.*e**8*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*a**4.*e**8*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(omega)**2 + 2.*a**4.*e**8*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4\
    #     - 2.*a**4.*e**8*np.sin(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 - 4.*a**4.*e**8*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**2\
    #     + 4.*a**4.*e**8*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2 + 4.*a**4.*e**8*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**2 - 8*a**4.*e**8*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 4.*a**4.*e**8*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(omega)**2 + 4.*a**4.*e**8*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4 - 4.*a**4.*e**8*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4\
    #     - 2.*a**4.*e**8*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**2 + 2.*a**4.*e**8*np.sin(omega)**4.*np.cos(Omega)**4 + 2.*a**4.*e**8*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**2 - 4.*a**4.*e**8*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 + 2.*a**4.*e**8*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(omega)**2 + 2.*a**4.*e**8*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 - 2.*a**4.*e**8*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 + 8*a**4.*e**6.*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**2 - 8*a**4.*e**6.*np.sin(Omega)**4.*np.sin(omega)**4 - 8*a**4.*e**6.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**4.*np.cos(omega)**2.\
    #     + 16.*a**4.*e**6.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 8*a**4.*e**6.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(omega)**2 - 8*a**4.*e**6.*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 8*a**4.*e**6.*np.sin(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 + 16.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**2 - 16.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2\
    #     - 16.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**2 + 32.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 16.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(omega)**2 - 16.*a**4.*e**6.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4 + 16.*a**4.*e**6.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4 + 8*a**4.*e**6.*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**2 - 8*a**4.*e**6.*np.sin(omega)**4.*np.cos(Omega)**4\
    #     - 8*a**4.*e**6.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**2 + 16.*a**4.*e**6.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 - 8*a**4.*e**6.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(omega)**2 - 8*a**4.*e**6.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 8*a**4.*e**6.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 - 12.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**2 + 12.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(omega)**4 + 12.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**4.*np.cos(omega)**2 - 24.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2.\
    #     + 12.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(omega)**2 + 12.*a**4.*e**4.*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 - 12.*a**4.*e**4.*np.sin(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 - 24.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**2 + 24.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2 + 24.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**2 - 48*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2\
    #     + 24.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(omega)**2 + 24.*a**4.*e**4.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4 - 24.*a**4.*e**4.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4 - 12.*a**4.*e**4.*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**2 + 12.*a**4.*e**4.*np.sin(omega)**4.*np.cos(Omega)**4 + 12.*a**4.*e**4.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**2 - 24.*a**4.*e**4.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 + 12.*a**4.*e**4.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(omega)**2 + 12.*a**4.*e**4.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4.\
    #     - 12.*a**4.*e**4.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 + 8*a**4.*e**2.*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**2 - 8*a**4.*e**2.*np.sin(Omega)**4.*np.sin(omega)**4 - 8*a**4.*e**2.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**4.*np.cos(omega)**2 + 16.*a**4.*e**2.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 8*a**4.*e**2.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(omega)**2\
    #     - 8*a**4.*e**2.*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 8*a**4.*e**2.*np.sin(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4\
    #     + 16.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**2 - 16.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2 - 16.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**2 + 32.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2\
    #     - 16.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(omega)**2 - 16.*a**4.*e**2.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4 + 16.*a**4.*e**2.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4 + 8*a**4.*e**2.*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**2 - 8*a**4.*e**2.*np.sin(omega)**4.*np.cos(Omega)**4 - 8*a**4.*e**2.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**2 + 16.*a**4.*e**2.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 - 8*a**4.*e**2.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(omega)**2.\
    #     - 8*a**4.*e**2.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 8*a**4.*e**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 - 2.*a**4.*np.sin(Omega)**4.*np.sin(omega)**4.*np.cos(i)**2 + 2.*a**4.*np.sin(Omega)**4.*np.sin(omega)**4 + 2.*a**4.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**4.*np.cos(omega)**2 - 4.*a**4.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*a**4.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(omega)**2\
    #     + 2.*a**4.*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 - 2.*a**4.*np.sin(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4 - 4.*a**4.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2.*np.cos(i)**2 + 4.*a**4.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2 + 4.*a**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**2 - 8*a**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 4.*a**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(omega)**2 + 4.*a**4.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4 - 4.*a**4.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4.\
    #      - 2.*a**4.*np.sin(omega)**4.*np.cos(Omega)**4.*np.cos(i)**2 + 2.*a**4.*np.sin(omega)**4.*np.cos(Omega)**4 + 2.*a**4.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**2 - 4.*a**4.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 + 2.*a**4.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(omega)**2 + 2.*a**4.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 - 2.*a**4.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**4

    # a_3 = -4.*Re**4.*e + 4.*Re**2.*a**2.*e**5.*np.sin(Omega)**2.*np.sin(omega)**2 + 4.*Re**2.*a**2.*e**5.*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 4.*Re**2.*a**2.*e**5.*np.sin(omega)**2.*np.cos(Omega)**2 + 4.*Re**2.*a**2.*e**5.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 8*Re**2.*a**2.*e**3*np.sin(Omega)**2.*np.sin(omega)**2 - 8*Re**2.*a**2.*e**3*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 8*Re**2.*a**2.*e**3*np.sin(omega)**2.*np.cos(Omega)**2 - 8*Re**2.*a**2.*e**3*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 4.*Re**2.*a**2.*e*np.sin(Omega)**2.*np.sin(omega)**2 + 4.*Re**2.*a**2.*e*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2\
    #     + 4.*Re**2.*a**2.*e*np.sin(omega)**2.*np.cos(Omega)**2 + 4.*Re**2.*a**2.*e*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2

    # a_4 = -Re**4 + 2.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2 + 2.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*e**4.*np.sin(omega)**2.*np.cos(Omega)**2 + 2.*Re**2.*a**2.*e**4.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 4.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2 - 4.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 4.*Re**2.*a**2.*e**2.*np.sin(omega)**2.*np.cos(Omega)**2 - 4.*Re**2.*a**2.*e**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*np.sin(Omega)**2.*np.sin(omega)**2\
    #     + 2.*Re**2.*a**2.*np.sin(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 2.*Re**2.*a**2.*np.sin(omega)**2.*np.cos(Omega)**2 + 2.*Re**2.*a**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - a**4.*e**8*np.sin(Omega)**4.*np.sin(omega)**4 - 2.*a**4.*e**8*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - a**4.*e**8*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 - 2.*a**4.*e**8*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2 - 4.*a**4.*e**8*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 2.*a**4.*e**8*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4\
    #     - a**4.*e**8*np.sin(omega)**4.*np.cos(Omega)**4 - 2.*a**4.*e**8*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 - a**4.*e**8*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 4.*a**4.*e**6.*np.sin(Omega)**4.*np.sin(omega)**4 + 8*a**4.*e**6.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 4.*a**4.*e**6.*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 8*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2 + 16.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 8*a**4.*e**6.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4\
    #     + 4.*a**4.*e**6.*np.sin(omega)**4.*np.cos(Omega)**4 + 8*a**4.*e**6.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 + 4.*a**4.*e**6.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 - 6.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(omega)**4 - 12.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 6.*a**4.*e**4.*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4\
    #     - 12.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2 - 24.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2\
    #     - 12.*a**4.*e**4.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4 - 6.*a**4.*e**4.*np.sin(omega)**4.*np.cos(Omega)**4 - 12.*a**4.*e**4.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 - 6.*a**4.*e**4.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 4.*a**4.*e**2.*np.sin(Omega)**4.*np.sin(omega)**4 + 8*a**4.*e**2.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2 + 4.*a**4.*e**2.*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 8*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2 + 16.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2\
    #     + 8*a**4.*e**2.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4 + 4.*a**4.*e**2.*np.sin(omega)**4.*np.cos(Omega)**4 + 8*a**4.*e**2.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 + 4.*a**4.*e**2.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 - a**4.*np.sin(Omega)**4.*np.sin(omega)**4 - 2.*a**4.*np.sin(Omega)**4.*np.sin(omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - a**4.*np.sin(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4\
    #     - 2.*a**4.*np.sin(Omega)**2.*np.sin(omega)**4.*np.cos(Omega)**2 - 4.*a**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2 - 2.*a**4.*np.sin(Omega)**2.*np.cos(Omega)**2.*np.cos(i)**4.*np.cos(omega)**4 - a**4.*np.sin(omega)**4.*np.cos(Omega)**4 - 2.*a**4.*np.sin(omega)**2.*np.cos(Omega)**4.*np.cos(i)**2.*np.cos(omega)**2 - a**4.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4

    a_0 = -Re**4.*e**4 + 4.*Re**2.*a**2.*e**6.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2. - 2.*Re**2.*a**2.*e**6.*np.sin(Omega)**2.*np.sin(i)**2. - 8.*Re**2.*a**2.*e**6.*np.sin(Omega)**2.*np.sin(omega)**2. + 4.*Re**2.*a**2.*e**6.*np.sin(Omega)**2. + 8.*Re**2.*a**2.*e**6.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega) + 4.*Re**2.*a**2.*e**6.*np.sin(omega)**2. - 2.*Re**2.*a**2.*e**6 - 8.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2.\
        + 4.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.sin(i)**2. + 16.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2. - 8.*Re**2.*a**2.*e**4.*np.sin(Omega)**2. - 16.*Re**2.*a**2.*e**4.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega) - 8.*Re**2.*a**2.*e**4.*np.sin(omega)**2. + 4.*Re**2.*a**2.*e**4 + 4.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2.\
        - 2.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(i)**2. - 8.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2. + 4.*Re**2.*a**2.*e**2.*np.sin(Omega)**2. + 8.*Re**2.*a**2.*e**2.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega) + 4.*Re**2.*a**2.*e**2.*np.sin(omega)**2. - 2.*Re**2.*a**2.*e**2. - a**4.*e**8.*np.sin(Omega)**4.*np.sin(i)**4 + 2.*a**4.*e**8.*np.sin(Omega)**2.*np.sin(i)**2. - a**4.*e**8 + 4.*a**4.*e**6.*np.sin(Omega)**4.*np.sin(i)**4.\
        - 8.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(i)**2. + 4.*a**4.*e**6 - 6.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(i)**4 + 12.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(i)**2.\
        - 6.*a**4.*e**4 + 4.*a**4.*e**2.*np.sin(Omega)**4.*np.sin(i)**4 - 8.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(i)**2. + 4.*a**4.*e**2. - a**4.*np.sin(Omega)**4.*np.sin(i)**4 + 2.*a**4.*np.sin(Omega)**2.*np.sin(i)**2. - a**4

    a_1 = 4.*Re**2.*e*(-Re**2.*e**2. + 2.*a**2.*e**4.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2. - a**2.*e**4.*np.sin(Omega)**2.*np.sin(i)**2. - 4.*a**2.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2. + 2.*a**2.*e**4.*np.sin(Omega)**2. + 4.*a**2.*e**4.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega) + 2.*a**2.*e**4.*np.sin(omega)**2. - a**2.*e**4 - 4.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2. + 2.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(i)**2.\
        + 8.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2. - 4.*a**2.*e**2.*np.sin(Omega)**2. - 8.*a**2.*e**2.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega) - 4.*a**2.*e**2.*np.sin(omega)**2. + 2.*a**2.*e**2. + 2.*a**2.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2. - a**2.*np.sin(Omega)**2.*np.sin(i)**2. - 4.*a**2.*np.sin(Omega)**2.*np.sin(omega)**2. + 2.*a**2.*np.sin(Omega)**2.\
        + 4.*a**2.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega) + 2.*a**2.*np.sin(omega)**2. - a**2)

    a_2 = -6.*Re**4.*e**2. - 2.*Re**2.*a**2.*e**6.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2. + 2.*Re**2.*a**2.*e**6.*np.sin(Omega)**2.*np.sin(i)**2. + 4.*Re**2.*a**2.*e**6.*np.sin(Omega)**2.*np.sin(omega)**2. - 2.*Re**2.*a**2.*e**6.*np.sin(Omega)**2. - 4.*Re**2.*a**2.*e**6.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega) - 2.*Re**2.*a**2.*e**6.*np.sin(omega)**2. + 2.*Re**2.*a**2.*e**6 + 8.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2.\
        - 6.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.sin(i)**2. - 16.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2. + 8.*Re**2.*a**2.*e**4.*np.sin(Omega)**2. + 16.*Re**2.*a**2.*e**4.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega) + 8.*Re**2.*a**2.*e**4.*np.sin(omega)**2. - 6.*Re**2.*a**2.*e**4 - 10*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2.\
        + 6.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(i)**2. + 20*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2. - 10*Re**2.*a**2.*e**2.*np.sin(Omega)**2. - 20*Re**2.*a**2.*e**2.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega) - 10*Re**2.*a**2.*e**2.*np.sin(omega)**2. + 6.*Re**2.*a**2.*e**2. + 4.*Re**2.*a**2.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2. - 2.*Re**2.*a**2.*np.sin(Omega)**2.*np.sin(i)**2. - 8.*Re**2.*a**2.*np.sin(Omega)**2.*np.sin(omega)**2.\
        + 4.*Re**2.*a**2.*np.sin(Omega)**2. + 8.*Re**2.*a**2.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega) + 4.*Re**2.*a**2.*np.sin(omega)**2. - 2.*Re**2.*a**2. - 2.*a**4.*e**8.*np.sin(Omega)**4.*np.sin(i)**4.*np.sin(omega)**2.\
        + 2.*a**4.*e**8.*np.sin(Omega)**4.*np.sin(i)**4 + 4.*a**4.*e**8.*np.sin(Omega)**4.*np.sin(i)**2.*np.sin(omega)**2. - 2.*a**4.*e**8.*np.sin(Omega)**4.*np.sin(i)**2. - 4.*a**4.*e**8.*np.sin(Omega)**3*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega) - 4.*a**4.*e**8.*np.sin(Omega)**3*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3 - 4.*a**4.*e**8.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2. + 4.*a**4.*e**8.*np.sin(Omega)**2.*np.sin(omega)**2.\
        - 2.*a**4.*e**8.*np.sin(Omega)**2. - 4.*a**4.*e**8.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega) - 4.*a**4.*e**8.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3 - 4.*a**4.*e**8.*np.sin(Omega)*np.sin(omega)**3*np.cos(Omega)**3*np.cos(i)**3*np.cos(omega) - 4.*a**4.*e**8.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)**3*np.cos(i)**3.*np.cos(omega)**3.\
        - 2.*a**4.*e**8.*np.sin(omega)**2. + 2.*a**4.*e**8 + 8.*a**4.*e**6.*np.sin(Omega)**4.*np.sin(i)**4.*np.sin(omega)**2. - 8.*a**4.*e**6.*np.sin(Omega)**4.*np.sin(i)**4 - 16.*a**4.*e**6.*np.sin(Omega)**4.*np.sin(i)**2.*np.sin(omega)**2. + 8.*a**4.*e**6.*np.sin(Omega)**4.*np.sin(i)**2. + 16.*a**4.*e**6.*np.sin(Omega)**3*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega) + 16.*a**4.*e**6.*np.sin(Omega)**3*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3.\
        + 16.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2. - 16.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**2. + 8.*a**4.*e**6.*np.sin(Omega)**2. + 16.*a**4.*e**6.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega) + 16.*a**4.*e**6.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3.\
        + 16.*a**4.*e**6.*np.sin(Omega)*np.sin(omega)**3*np.cos(Omega)**3*np.cos(i)**3*np.cos(omega) + 16.*a**4.*e**6.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)**3*np.cos(i)**3*np.cos(omega)**3 + 8.*a**4.*e**6.*np.sin(omega)**2. - 8.*a**4.*e**6 - 12.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(i)**4.*np.sin(omega)**2. + 12.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(i)**4 + 24.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(i)**2.*np.sin(omega)**2.\
        - 12.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(i)**2.\
        - 24.*a**4.*e**4.*np.sin(Omega)**3*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega) - 24.*a**4.*e**4.*np.sin(Omega)**3*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3 - 24.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2. + 24.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2. - 12.*a**4.*e**4.*np.sin(Omega)**2. - 24.*a**4.*e**4.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega)\
        - 24.*a**4.*e**4.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3 - 24.*a**4.*e**4.*np.sin(Omega)*np.sin(omega)**3*np.cos(Omega)**3*np.cos(i)**3*np.cos(omega) - 24.*a**4.*e**4.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)**3*np.cos(i)**3*np.cos(omega)**3 - 12.*a**4.*e**4.*np.sin(omega)**2. + 12.*a**4.*e**4 + 8.*a**4.*e**2.*np.sin(Omega)**4.*np.sin(i)**4.*np.sin(omega)**2. - 8.*a**4.*e**2.*np.sin(Omega)**4.*np.sin(i)**4\
        - 16.*a**4.*e**2.*np.sin(Omega)**4.*np.sin(i)**2.*np.sin(omega)**2. + 8.*a**4.*e**2.*np.sin(Omega)**4.*np.sin(i)**2. + 16.*a**4.*e**2.*np.sin(Omega)**3*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega) + 16.*a**4.*e**2.*np.sin(Omega)**3*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3 + 16.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2.\
        - 16.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2. + 8.*a**4.*e**2.*np.sin(Omega)**2. + 16.*a**4.*e**2.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega) + 16.*a**4.*e**2.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3 + 16.*a**4.*e**2.*np.sin(Omega)*np.sin(omega)**3*np.cos(Omega)**3*np.cos(i)**3*np.cos(omega) + 16.*a**4.*e**2.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)**3*np.cos(i)**3*np.cos(omega)**3\
        + 8.*a**4.*e**2.*np.sin(omega)**2. - 8.*a**4.*e**2. - 2.*a**4.*np.sin(Omega)**4.*np.sin(i)**4.*np.sin(omega)**2. + 2.*a**4.*np.sin(Omega)**4.*np.sin(i)**4 + 4.*a**4.*np.sin(Omega)**4.*np.sin(i)**2.*np.sin(omega)**2. - 2.*a**4.*np.sin(Omega)**4.*np.sin(i)**2. - 4.*a**4.*np.sin(Omega)**3*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega)\
        - 4.*a**4.*np.sin(Omega)**3*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3.\
        - 4.*a**4.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2. + 4.*a**4.*np.sin(Omega)**2.*np.sin(omega)**2. - 2.*a**4.*np.sin(Omega)**2. - 4.*a**4.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega)\
        - 4.*a**4.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3 - 4.*a**4.*np.sin(Omega)*np.sin(omega)**3*np.cos(Omega)**3*np.cos(i)**3*np.cos(omega) - 4.*a**4.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)**3*np.cos(i)**3*np.cos(omega)**3 - 2.*a**4.*np.sin(omega)**2. + 2.*a**4

    a_3 = 4.*Re**2.*e*(-Re**2. - a**2.*e**4.*(np.cos(-2.*Omega + i + 2.*omega) - np.cos(2.*Omega - i + 2.*omega) + np.cos(2.*Omega + i - 2.*omega) - np.cos(2.*Omega + i + 2.*omega))/8 + a**2.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2. + a**2.*e**4.*np.sin(i)**2.*np.cos(omega)**2. + a**2.*e**4.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2. + a**2.*e**2.*(np.cos(-2.*Omega + i + 2.*omega) - np.cos(2.*Omega - i + 2.*omega) + np.cos(2.*Omega + i - 2.*omega) - np.cos(2.*Omega + i + 2.*omega))/4.\
        - 2.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2. - 2.*a**2.*e**2.*np.sin(i)**2.*np.cos(omega)**2. - 2.*a**2.*e**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2. - a**2.*(np.cos(-2.*Omega + i + 2.*omega) - np.cos(2.*Omega - i + 2.*omega) + np.cos(2.*Omega + i - 2.*omega) - np.cos(2.*Omega + i + 2.*omega))/8 + a**2.*np.sin(Omega)**2.*np.sin(omega)**2. + a**2.*np.sin(i)**2.*np.cos(omega)**2.\
        + a**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2)

    a_4 = -Re**4 - Re**2.*a**2.*e**4.*(np.cos(-2.*Omega + i + 2.*omega) - np.cos(2.*Omega - i + 2.*omega) + np.cos(2.*Omega + i - 2.*omega) - np.cos(2.*Omega + i + 2.*omega))/4 + 2.*Re**2.*a**2.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2. + 2.*Re**2.*a**2.*e**4.*np.sin(i)**2.*np.cos(omega)**2. + 2.*Re**2.*a**2.*e**4.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2. + Re**2.*a**2.*e**2.*(np.cos(-2.*Omega + i + 2.*omega) - np.cos(2.*Omega - i + 2.*omega)\
        + np.cos(2.*Omega + i - 2.*omega) - np.cos(2.*Omega + i + 2.*omega))/2. - 4.*Re**2.*a**2.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2. - 4.*Re**2.*a**2.*e**2.*np.sin(i)**2.*np.cos(omega)**2. - 4.*Re**2.*a**2.*e**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2. - Re**2.*a**2.*(np.cos(-2.*Omega + i + 2.*omega) - np.cos(2.*Omega - i + 2.*omega) + np.cos(2.*Omega + i - 2.*omega) - np.cos(2.*Omega + i + 2.*omega))/4.\
        + 2.*Re**2.*a**2.*np.sin(Omega)**2.*np.sin(omega)**2. + 2.*Re**2.*a**2.*np.sin(i)**2.*np.cos(omega)**2.\
        + 2.*Re**2.*a**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2. - a**4.*e**8.*np.sin(Omega)**4.*np.sin(omega)**4 + 4.*a**4.*e**8.*np.sin(Omega)**3*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega) - 2.*a**4.*e**8.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2.*np.cos(omega)**2. - 6.*a**4.*e**8.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2.\
        + 4.*a**4.*e**8.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3 + 4.*a**4.*e**8.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)**3*np.cos(i)**3*np.cos(omega)**3 - a**4.*e**8.*np.sin(i)**4.*np.cos(omega)**4.\
        - 2.*a**4.*e**8.*np.sin(i)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4 - a**4.*e**8.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 4.*a**4.*e**6.*np.sin(Omega)**4.*np.sin(omega)**4 - 16.*a**4.*e**6.*np.sin(Omega)**3*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega) + 8.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2.*np.cos(omega)**2. + 24.*a**4.*e**6.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2.\
        - 16.*a**4.*e**6.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3 - 16.*a**4.*e**6.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)**3*np.cos(i)**3*np.cos(omega)**3 + 4.*a**4.*e**6.*np.sin(i)**4.*np.cos(omega)**4 + 8.*a**4.*e**6.*np.sin(i)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4 + 4.*a**4.*e**6.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4.\
        - 6.*a**4.*e**4.*np.sin(Omega)**4.*np.sin(omega)**4 + 24.*a**4.*e**4.*np.sin(Omega)**3*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega) - 12.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2.*np.cos(omega)**2. - 36.*a**4.*e**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2. + 24.*a**4.*e**4.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3.\
        + 24.*a**4.*e**4.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)**3*np.cos(i)**3*np.cos(omega)**3 - 6.*a**4.*e**4.*np.sin(i)**4.*np.cos(omega)**4 - 12.*a**4.*e**4.*np.sin(i)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4 - 6.*a**4.*e**4.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 + 4.*a**4.*e**2.*np.sin(Omega)**4.*np.sin(omega)**4.\
        - 16.*a**4.*e**2.*np.sin(Omega)**3*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega) + 8.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2.*np.cos(omega)**2.\
        + 24.*a**4.*e**2.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2. - 16.*a**4.*e**2.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3 - 16.*a**4.*e**2.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)**3*np.cos(i)**3*np.cos(omega)**3 + 4.*a**4.*e**2.*np.sin(i)**4.*np.cos(omega)**4 + 8.*a**4.*e**2.*np.sin(i)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4.\
        + 4.*a**4.*e**2.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4 - a**4.*np.sin(Omega)**4.*np.sin(omega)**4 + 4.*a**4.*np.sin(Omega)**3*np.sin(omega)**3*np.cos(Omega)*np.cos(i)*np.cos(omega) - 2.*a**4.*np.sin(Omega)**2.*np.sin(i)**2.*np.sin(omega)**2.*np.cos(omega)**2. - 6.*a**4.*np.sin(Omega)**2.*np.sin(omega)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**2.\
        + 4.*a**4.*np.sin(Omega)*np.sin(i)**2.*np.sin(omega)*np.cos(Omega)*np.cos(i)*np.cos(omega)**3 + 4.*a**4.*np.sin(Omega)*np.sin(omega)*np.cos(Omega)**3*np.cos(i)**3*np.cos(omega)**3 - a**4.*np.sin(i)**4.*np.cos(omega)**4 - 2.*a**4.*np.sin(i)**2.*np.cos(Omega)**2.*np.cos(i)**2.*np.cos(omega)**4 - a**4.*np.cos(Omega)**4.*np.cos(i)**4.*np.cos(omega)**4

    A = a_1/a_0
    B = a_2/a_0
    C = a_3/a_0
    D = a_4/a_0

    return A, B, C, D

def quarticSolutions_ellipse_to_Quarticipynb(A, B, C, D):
    """ Equations from ellipse_to_Quartic.ipynb solves the quartic 
    Uses the coefficients of the quartic to find
    Args:
        A (numpy array):
            coefficients of x^3
        B (numpy array):
            coefficients of x^2
        C (numpy array):
            coefficients of x
        D (numpy array):
            constants
    Returns:
        xreal (numpy array):
            an nx4 array contianing the solutions to the quartic expression
        delta (numpy array):
            indicator parameter for quartic solution types
        P (numpy array):
            indicator parameter for quartic solution types
        D2 (numpy array):
            indicator parameter for quartic solution types
        R (numpy array):
            indicator parameter for quartic solution types
        delta_0 (numpy array):
            indicator parameter for quartic solution types
    """
    #A bunch of simplifications
    p0 = (-3.*A**2./8.+B)**3.
    p1 = (A*(A**2./8.-B/2.)+C)**2.
    p2 = -A*(A*(3.*A**2./256.-B/16.)+C/4.)+D
    p3 = -3.*A**2./8.+B
    p4 = 2.*A*(A**2./8.-B/2.)
    p5 = -p0/108.-p1/8.+p2*p3/3.
    p6 = (p0/216.+p1/16.-p2*p3/6.+np.sqrt(p5**2./4.+(-p2-p3**2./12.)**3./27.))**(1./3.)
    p6Inds = np.where(p6==0.0)[0] #p6 being 0 causes divide by 0
    p6[p6Inds] = np.ones(len(p6Inds))*10**-10
    p7 = A**2./4.-2.*B/3.
    p8 = (2.*p2+p3**2./6.)/(3.*p6)
    #, (-2*p2-p3**2/6)/(3*p6)
    p9 = np.sqrt(-2.*p5**(1./3.)+p7)
    p10 = np.sqrt(2.*p6+p7+p8)
    p11 = A**2./2.-4.*B/3.

    #otherwise case
    x0 = -A/4. - p10/2. - np.sqrt(p11 - 2.*p6 - p8 + (2.*C + p4)/p10)/2.
    x1 = -A/4. - p10/2. + np.sqrt(p11 - 2.*p6 - p8 + (2.*C + p4)/p10)/2.
    x2 = -A/4. + p10/2. - np.sqrt(p11 - 2.*p6 - p8 + (-2.*C - p4)/p10)/2.
    x3 = -A/4. + p10/2. + np.sqrt(p11 - 2.*p6 - p8 + (-2.*C - p4)/p10)/2.
    zeroInds = np.where(p2 + p3**2./12. == 0)[0] #piecewise condition
    if len(zeroInds) != 0.:
        x0[zeroInds] = -A[zeroInds]/4. - p9[zeroInds]/2. - np.sqrt(p11[zeroInds] + 2.*np.cbrt(np.real(p5[zeroInds])) + (2.*C[zeroInds] + p4[zeroInds])/p9[zeroInds])/2.
        x1[zeroInds] = -A[zeroInds]/4. - p9[zeroInds]/2. + np.sqrt(p11[zeroInds] + 2.*np.cbrt(np.real(p5[zeroInds])) + (2.*C[zeroInds] + p4[zeroInds])/p9[zeroInds])/2.
        x2[zeroInds] = -A[zeroInds]/4. + p9[zeroInds]/2. - np.sqrt(p11[zeroInds] + 2.*np.cbrt(np.real(p5[zeroInds])) + (-2.*C[zeroInds] - p4[zeroInds])/p9[zeroInds])/2.
        x3[zeroInds] = -A[zeroInds]/4. + p9[zeroInds]/2. + np.sqrt(p11[zeroInds] + 2.*np.cbrt(np.real(p5[zeroInds])) + (-2.*C[zeroInds] - p4[zeroInds])/p9[zeroInds])/2.

    delta = 256.*D**3. - 192.*A*C*D**2. - 128.*B**2.*D**2. + 144.*B*C**2.*D - 27.*C**4.\
        + 144.*A**2.*B*D**2. - 6.*A**2.*C**2.*D - 80.*A*B**2.*C*D + 18.*A*B*C**3. + 16.*B**4.*D\
        - 4.*B**3.*C**2. - 27.*A**4.*D**2. + 18.*A**3.*B*C*D - 4.*A**3.*C**3. - 4.*A**2.*B**3.*D + A**2.*B**2.*C**2. #verified against wikipedia multiple times
    assert 0 == np.count_nonzero(np.imag(delta)), 'Not all delta are real'
    delta = np.real(delta)
    P = 8.*B - 3.*A**2.
    assert 0 == np.count_nonzero(np.imag(P)), 'Not all P are real'
    P = np.real(P)
    D2 = 64.*D - 16.*B**2. + 16.*A**2.*B - 16.*A*C - 3.*A**4. #is 0 if the quartic has 2 double roots 
    assert 0 == np.count_nonzero(np.imag(D2)), 'Not all D2 are real'
    D2 = np.real(D2)
    R = A**3. + 8.*C - 4.*A*B
    assert 0 == np.count_nonzero(np.imag(R)), 'Not all R are real'
    R = np.real(R)
    delta_0 = B**2. - 3.*A*C + 12.*D
    assert 0 == np.count_nonzero(np.imag(delta_0)), 'Not all delta_0 are real'
    delta_0 = np.real(delta_0)

    return np.asarray([x0, x1, x2, x3]).T, delta, P, D2, R, delta_0

def ellipseZFromX(xreal, Re):
    Z = np.sqrt(Re**2. - xreal**2.)
    return Z, -Z


#Note: we don't need to do any calculations checking whether intersections will occur or not because we know they will in this instance
A, B, C, D = quarticCoefficients(np.asarray([Re]),np.asarray([a]).astype('complex128'),np.asarray([e]),np.asarray([i]),np.asarray([w]),np.asarray([W]))
xreal, delta, P, D2, R, delta_0 = quarticSolutions_ellipse_to_Quarticipynb(A.astype('complex128'), B, C, D)
assert np.max(xreal.imag) < 1e-5, "The largest imaginary component is large" #if this is the case, I will need to add some filtering
zreal = ellipseZFromX(xreal,Re)
#TODO: FIX THIS +/- FIXER. EACH VALUE NEEDS TO BE EVALUATED WHETHER IT IS +/-
nus_tmp = np.arccos(xreal.real) #compute the nustogether

#Doing this because of the way the orbit is configured
#CONTAINS THE NUS SEPARATING BETWEEN ECLIPSE AND NON-ECLIPSE
nus_eclipse = np.asarray([nus_tmp[0][2],nus_tmp[0][0],2.*np.pi-nus_tmp[0][0],2.*np.pi-nus_tmp[0][2]]) #I am just taking the first and third because the firsst-second are nearly identical and the third-fourth are nearly identical

#compute nus in between to determine whether it is in eclipse
nus_eclipse_inbetween = np.asarray([nus_eclipse[0]+2.*np.pi-nus_eclipse[-1],nus_eclipse[1]-nus_eclipse[0],nus_eclipse[2]-nus_eclipse[1],nus_eclipse[3]-nus_eclipse[2]])/2.

#Check which nus range the 
pos_vect_eclipse = XYZ_from_KOE(a,e,i,w,W,nus_eclipse)

nu_fullrange = np.linspace(start=0.,stop=2.*np.pi,num=200)
pos_vect_fullOrbit = XYZ_from_KOE(a,e,i,w,W,nu_fullrange)


#### Plot planet and orbit
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_aspect("equal")

# draw cube
# r = [-1, 1]
# for s, e in combinations(np.array(list(product(r, r, r))), 2):
#     if np.sum(np.abs(s-e)) == r[1]-r[0]:
#         ax.plot3D(*zip(s, e), color="b")

# draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = Re*np.cos(u)*np.sin(v)
y = Re*np.sin(u)*np.sin(v)
z = Re*np.cos(v)
ax.plot_wireframe(x, y, z, color="r")

#Plot the whole orbit
ax.plot(pos_vect_fullOrbit[0],pos_vect_fullOrbit[1],pos_vect_fullOrbit[2],color='blue')
#Plot the sun vector
ax.plot(np.asarray([-1.5*Re,-Re]),np.asarray([0,0]),np.asarray([0,0]),color='orange')

#Plot the eclipse points
ax.scatter(pos_vect_eclipse[0],pos_vect_eclipse[1],pos_vect_eclipse[2],color='black',s=100)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# draw a point
#ax.scatter([0], [0], [0], color="g", s=100)

# # draw a vector
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import proj3d


# class Arrow3D(FancyArrowPatch):

#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
#         self._verts3d = xs, ys, zs

#     def draw(self, renderer):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         FancyArrowPatch.draw(self, renderer)

# a = Arrow3D([0, 1], [0, 1], [0, 1], mutation_scale=20,
#             lw=1, arrowstyle="-|>", color="k")
#ax.add_artist(a)
plt.show(block=False)
####


print(saltyburrito)

Amin = 0.24
Amax = 3.4
A = 3.4
C_d = 2.2
rho = 5*10**-13
m = 6
a = (6378+400)*1000
mu = 398600*10**9
T0 = 2.*np.pi*np.sqrt(a**3./mu)

#dndt = 1.5*A*C_d*a**(-2.0)*mu*rho/m


#The time we have until the two spacecraft need to be at the same separation
timeToGetToSeparations = 3*30*24*60*60 #3 months in seconds
numOrbits = int(np.floor(timeToGetToSeparations/T0))

th0 = 0.
th1 = 20./90*2.*np.pi #initial 20 minutes of separation between lead and following spacecraft
thd0 = np.sqrt(mu/a**3.)
thd1 = np.sqrt(mu/a**3.)

desiredSeparations = 2./90*2.*np.pi

def simpleInputs_to_At(firstStart, firstDur, timeBetween, secondDur, numOrbits):
    """ Converts simple inputs into spacecraft SA vs time inputs
    """
    #Error checking so we cannot start the maneuver too close to the end
    # print(firstStart)
    # print(firstDur)
    # print(timeBetween)
    # print(secondDur)
    # print(numOrbits)
    if firstStart + firstDur + timeBetween + secondDur > numOrbits:
        firstStart = numOrbits - firstDur - timeBetween - secondDur
        print("constraint violated")
    A0 = Amin*np.ones(numOrbits)
    A0[int(np.floor(firstStart)):int(np.floor(firstStart+firstDur))] = Amax*np.ones(int(np.floor(firstStart+firstDur))-int(np.floor(firstStart)))
    
    A1 = Amin*np.ones(numOrbits)
    secondStart = int(np.floor(firstStart + firstDur + timeBetween))
    secondEnd = int(np.floor(firstStart + firstDur + timeBetween + secondDur))
    A1[secondStart:secondEnd] = Amax*np.ones(secondEnd-secondStart)
    # plt.figure()
    # plt.plot(np.arange(len(A0)),A0,color='blue')
    # plt.plot(np.arange(len(A1)),A1,color='black')
    # plt.show(block=False)
    return A0, A1

def posVelDiff(x,th0_0,th1_0,C_d,rho,m,mu):
    # print(x)
    firstStart = x[0]
    firstDur = x[1]
    timeBetween = x[2]
    secondDur = x[3]
    A0, A1 = simpleInputs_to_At(firstStart, firstDur, timeBetween, secondDur, numOrbits)
    #Extract A(t)
    # A0 = x[:numOrbits]
    # A1 = x[numOrbits+1:] #double check these lengths are correct

    #Need to run through orbits iteratively
    a0 = np.zeros(numOrbits)
    a0[0] = a
    a1 = np.zeros(numOrbits)
    a1[0] = a
    T0 = 2.*np.pi*np.sqrt(a0**3./mu)
    T1 = 2.*np.pi*np.sqrt(a1**3./mu)
    th0 = np.zeros(numOrbits)
    th0[0] = th0_0
    th1 = np.zeros(numOrbits)
    th1[0] = th1_0
    n0 = np.zeros(numOrbits)
    n0[0] = np.sqrt(mu/a0[0]**3.)
    n1 = np.zeros(numOrbits)
    n1[0] = np.sqrt(mu/a1[0]**3.)
    dndt0 = np.zeros(numOrbits)
    dndt1 = np.zeros(numOrbits)
    for i in np.arange(numOrbits-1):
        dadt0 = -1.0*A0[i]*C_d*np.sqrt(a0[i])*np.sqrt(mu)*rho/m
        a0[i+1] = a0[i] + dadt0
        dadt1 = -1.0*A1[i]*C_d*np.sqrt(a1[i])*np.sqrt(mu)*rho/m
        a1[i+1] = a1[i] + dadt1
        T0[i+1] = 2.*np.pi*np.sqrt(a0[i+1]**3./mu)
        T1[i+1] = 2.*np.pi*np.sqrt(a1[i+1]**3./mu)
        dndt0[i+1] = 1.5*A0[i]*C_d*a0[i+1]**(-2.0)*mu*rho/m
        n0[i+1] = n0[i] + dndt0[i+1]*T0[i+1]
        dndt1[i+1] = 1.5*A1[i]*C_d*a1[i+1]**(-2.0)*mu*rho/m
        n1[i+1] = n1[i] + dndt1[i+1]*T1[i+1]
        th0[i+1] = th0[i] + n0[i+1]*T0[i+1]
        th1[i+1] = th1[i] + n1[i+1]*T1[i+1]
    #Evaluation function
    erf = np.abs(th0[-1] - th1[-1]) + 1000*np.abs(n0[-1] - n1[-1]) + 100000*n0[-1]
    print("th0: " + str(th0[-1]) + " th1: " + str(th1[-1]) + " n0: " + str(n0[-1]) + " n1: " + str(n1[-1]) + " erf: " + str(erf))
    #todo update to have constant separation
    #print(erf)
    return erf

#initial starting array of times
x0 = [int(np.floor(numOrbits/10)),int(np.floor(numOrbits/10)),int(np.floor(numOrbits/10)),int(np.floor(numOrbits/10))]
bounds = [(0,numOrbits),(0,numOrbits/3),(0,numOrbits),(0,numOrbits/3)]
#bounds = [(Amin,Amax) for i in np.arange(len(x0))]

# erf0 = posVelDiff(x0,th0,th1,C_d,rho,m,mu)
# x0[1] = 100
# erf1 = posVelDiff(x0,th0,th1,C_d,rho,m,mu)
# print(saltyburrito)

out = minimize(posVelDiff, x0, bounds=bounds, method='SLSQP', options={'eps':20}, args=(th0, th1,C_d,rho,m,mu))#constraints=[]) #constraints=const)
A_SV0, A_SV1 = simpleInputs_to_At(out['x'][0], out['x'][1], out['x'][2], out['x'][3], numOrbits)

plt.figure(1)
plt.plot(np.arange(len(A_SV0)),A_SV0,color='blue')
plt.plot(np.arange(len(A_SV1)),A_SV1,color='black')
plt.show(block=False)



#### NEED TO REFORMULATE FREE VARIABLES INTO: FIRST MANEUVER START, FIRST MANEUVER DURATION, TIME BETWEEN MANEUVERS, SECOND MANEUVER DURATION


