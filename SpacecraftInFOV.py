# Ground Station and Spacecraft FOV calculations
import numpy as np
import math as m

a = np.asarray([[1,0,0],[0,1,0],[0,0,1],[2,0,2]])
b = np.asarray([[1,0,0],[1,0,0],[1,0,0],[1,0,0]])


c = np.einsum('ij,ij->i',a,b)


def angleBetweenNandR(ns,rs):
    """ Computes the angle between the orbital plane normal vector n and the ground station look vector r
    Args:
        ndarray:
            ns - array of orbital plane normal vectors, size nx3
        ndarray:
            rs - array of ground station look vectors, size nx3
    Returns:
        ndarray:
            phis - array of angle between the orbital plane normal vectors and ground station look vectors, size n
    """
    c = np.einsum('ij,ij->i',ns,rs) #Dot product between vectors
    phis = np.arccos(c/(np.linalg.norm(ns,axis=1)*np.linalg.norm(rs,axis=1)))
    return phis

def classifyIntersectionTypesLevel1(phis,FOV):
    """ Determines inds of the separate KOE that have the given intersection type as none, parabola, or ellipse (line not included)
    Args:
        ndarray:
            phis
        ndarray:
            FOV
    Returns:
        ndarray:
            indsNoPlaneIntersection
        ndarray:
            indsPlaneParabola
        ndarray:
            indsPlaneEllipse
    """
    indsNoPlaneIntersection = np.where(phis < FOV + np.pi/2.)[0]
    indsPlaneParabola = np.where((FOV + np.pi/2. < phis)*(phis < np.pi/2.))[0]
    indsPlaneEllipse = np.where(np.pi/2. < phis)[0]
    return indsNoPlaneIntersection, indsPlaneParabola, indsPlaneEllipse

def latlonalt_to_r(lat,lon,alt): #Taken from LLIPPEEE latlonfuncs
    """ Verified agains https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
    Args:
        lat (float) - latitude in radians
        lon (float) - longitude in radians
        alt (float) - distance from center of Body
    Return:
        r (numpy array) - x,y,z distance from Mars Centered Inertial Frame in m
    """
    x = np.cos(lon)*np.cos(lat)
    y = np.sin(lon)*np.cos(lat)
    z = np.sin(lat)
    return np.asarray([x,y,z])*alt

def Rx(theta):
    """theta in radians
    """
    return np.matrix([[ 1, 0           , 0           ],
        [ 0, m.cos(theta),-m.sin(theta)],
        [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
    """theta in radians
    """
    return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
        [ 0           , 1, 0           ],
        [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
    """theta in radians
    """
    return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
        [ m.sin(theta), m.cos(theta) , 0 ],
        [ 0           , 0            , 1 ]])

def cross(a,b):
    c = np.asarray([a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]])
    return c

#Generate gs location
#Generate gs look vector
#Generate gs angular FOV
#### Generate Orbit #######################
import os
from exodetbox.projectedEllipse import *
from exodetbox.stats import *
#from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as random
import time
from astropy import constants as const
import astropy.units as u
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.planet_star_separation import planet_star_separation
import itertools
import datetime
import re
import matplotlib.gridspec as gridspec
from pandas.plotting import scatter_matrix
import pandas as pd 
import corner
from EXOSIMS.util.eccanom import *
from scipy.stats import multivariate_normal
from scipy.stats import norm
from exodetbox.trueAnomalyFromEccentricAnomaly import trueAnomalyFromEccentricAnomaly
from statsmodels.stats.weightstats import DescrStatsW
import csv
import itertools
import random as random2
import pickle
import imageio
import matplotlib.colors as colors

folder_load = '/home/dean/Documents/exosims/twoDetMC'
filename = 'HabEx_PPEarthlike.json'
#filename = 'HabEx_PPSAG13.json'

#filename = 'HabEx_PPJupiterlike.json'
scriptfile = os.path.join(folder_load,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
PPop = sim.PlanetPopulation
comp = sim.Completeness
OS = sim.OpticalSystem
ZL = sim.ZodiacalLight
TL = sim.TargetList
TL.BV[0] = 0.65 #http://spiff.rit.edu/classes/phys440/lectures/color/color.html
TL.Vmag[0] = 1. #reference star

cachefname = sim.Completeness.filename #used to append to files

#Generate Planets
n = 10**4
# rndInt = np.random.randint(low=0,high=1,size=1)
# rnduniform = np.random.uniform()
# if rnduniform > 0.95:
#     n = 10**5
# else:
#     n = 6*10**6
#n = int(np.abs(1-rndInt)*6*10**6 + rndInt*10**5)
inc, W, w = PPop.gen_angles(n,None)
W = W.to('rad').value
w = w.to('rad').value
#w correction caused in smin smax calcs
wReplacementInds = np.where(np.abs(w-1.5*np.pi)<1e-3)[0]
w[wReplacementInds] = w[wReplacementInds] - 0.001
wReplacementInds = np.where(np.abs(w-0.5*np.pi)<1e-3)[0]
w[wReplacementInds] = w[wReplacementInds] - 0.001
del wReplacementInds
inc = inc.to('rad').value
#inc[np.where(inc>np.pi/2.)[0]] = np.pi - inc[np.where(inc>np.pi/2.)[0]]
sma, e, p, Rp = PPop.gen_plan_params(n)
############################################################
#TODO: CHANGE ORBIT SMA UNITS AND START USING SPACECRAFT KOE AT SOME POINT

#Calculate Orbit Semi-minor Axis
smna = sma*np.sqrt(1.-e**2.) #The semi-minor axis of the orbital ellipse

#Calculate Orbit Normal
from exodetbox.projectedEllipse import *
v0 = np.zeros(sma.shape) #nu=0
r0 = xyz_3Dellipse(sma,e,W,w,inc,v0)
r0 = r0.reshape(3,10000)
v1 = np.zeros(sma.shape)+np.pi #nu=pi
r1 = xyz_3Dellipse(sma,e,W,w,inc,v1)
r1 = r1.reshape(3,10000)
v2 = np.zeros(sma.shape) + np.pi/2. #nu=pi/2
r2 = xyz_3Dellipse(sma,e,W,w,inc,v2)
r2 = r2.reshape(3,10000)
tmp = cross(r0,r2)
nhats = tmp/np.linalg.norm(tmp,axis=0) #normal vectors to the orbital ellipse
#Calculate Orbit Geometric Center
Oorbit = (r0+r1)/2. #The 3D center of the orbital ellipse
#Calculate Orbit Coordinate Reference Frame
xhats = (r1-r0)/np.linalg.norm(r1-r0,axis=0)
zhats = nhats
yhats = -cross(xhats,zhats) #get third component from first 2
#d of plane #the constant in the plane equation
#d_plane = nhats[0]*Os[0] + nhats[1]*Os[1] + nhats[2]*Os[2] #these are all 0

#Calculate the 3D position vectors of the 
r_sma_orbit = Oorbit + xhats*sma.value
r_smna_orbit = Oorbit + yhats*smna.value

#Ground Station Location
lat = 45.*np.pi/180.#lat location of ground station
lon = 85.*np.pi/180.#lon location of ground station
rs_gs = latlonalt_to_r(lat,lon,1.)#*u.earthRad) #in EarthRad

#Ground Station Look Vector
#temporary, improve later
# r_gslook = rs_gs.valuenp.dot(np.tile(rs_gs,(nhats.shape[1],1)).T,nhats)
# r_gslook = r_gslook * Rz(40.*np.pi/180.) * Rx(20.*np.pi/180.)
r_gslook = rs_gs * Rz(40.*np.pi/180.) * Rx(20.*np.pi/180.)
r_gslook = np.asarray(r_gslook)[0]

#Ground Station FOV
FOV = 2.*np.pi/180. #2 deg is approximately the size of the field of view of a small telescope
#FOV = 160.*np.pi/180. #the whole sky field of view

#Compute angle between GS look vector and orbit normal vector
r_gslooks = np.tile(r_gslook,(nhats.shape[1],1)).reshape(3,nhats.shape[1])
phis = angleBetweenNandR(nhats.T,r_gslooks.T)

#Compute Intersection Types for Each Spacecraft
indsNoPlaneIntersection, indsPlaneParabola, indsPlaneEllipse = classifyIntersectionTypesLevel1(phis,FOV)

#Calculate intersection between look vector and plane (the center of the ellipse)
#DELETE0 = nhats[0]*(rs_gs[0] + t*r_gslook[0]) + nhats[1]*(rs_gs[1] + t*r_gslook[1]) + nhats[2]*(rs_gs[2] + t*r_gslook[2]) #these are all 0
ts = -(nhats[0,indsPlaneEllipse]*rs_gs[0] + nhats[1,indsPlaneEllipse]*rs_gs[1] + nhats[2,indsPlaneEllipse]*rs_gs[2])/(nhats[0,indsPlaneEllipse]*r_gslook[0] + nhats[1,indsPlaneEllipse]*r_gslook[1] + nhats[2,indsPlaneEllipse]*r_gslook[2])
Olook = np.asarray([rs_gs[0] + ts*r_gslook[0],rs_gs[1] + ts*r_gslook[1],rs_gs[2] + ts*r_gslook[2]])

#nhats[0]*(rs_gs[0] + t*r_gslook[0]) + nhats[1]*(rs_gs[1] + t*r_gslook[1]) + nhats[2]*(rs_gs[2] + t*r_gslook[2]) #these are all 0

#### For the intersections that are an ellipse ##############################
#The projection of the location of the ground station into the orbital plane
r_gs_proj = np.tile(rs_gs,(nhats.shape[1],1)).T[:,indsPlaneEllipse] - nhats[:,indsPlaneEllipse]*np.dot(rs_gs,nhats[:,indsPlaneEllipse]) #Dot product between vectors#rs_gs
# From r_gs_proj to Olook
# = Olook - r_gs_proj
#Get component of r_gs in nhat (distance from r_gs location in orbital plane to r_gs)
dist_r_gs_in_nhat = np.einsum('ij,ij->j',np.tile(rs_gs,(nhats[:,indsPlaneEllipse].shape[1],1)).T,nhats[:,indsPlaneEllipse]) #distance from r_gs to orbital plane

#Get unit vector from r_gs_proj to Olook
OlookNorm = (Olook-r_gs_proj)/np.linalg.norm(Olook-r_gs_proj,axis=0)
#Get unit vector from origin to Olook
#OlookNorm = Olook/np.linalg.norm(Olook,axis=0)

#Calculate distance from r_gs_proj to Olook
dist_Olook = np.linalg.norm(Olook-r_gs_proj,axis=0)

#Calculate distance from r_gs_proj to FOV extrema
dist_fov_lower = np.tan(phis[indsPlaneEllipse]+FOV/2.)*dist_r_gs_in_nhat
dist_fov_upper = np.tan(phis[indsPlaneEllipse]-FOV/2.)*dist_r_gs_in_nhat

#3D position vector of lower and upper portion of FOV
r_FOV_lower = r_gs_proj+OlookNorm*dist_fov_lower
r_FOV_upper = r_gs_proj+OlookNorm*dist_fov_upper

#3D unit vector from Olook in the direction of semi-minor axis of the FOV ellipse in the orbital plane
r_FOV_leftNorm = -cross(OlookNorm, nhats[:,indsPlaneEllipse])
#TODO: GO BACK AND PLOT THIS TO ENSURE I GOT THIS RIGHT. I COULD HAVE DONE THIS SPECIFIC STEP

#Distance from the Ground Station to the center of the look vector
dist_rgs_Olook = np.linalg.norm(Olook - np.tile(rs_gs,(nhats.shape[1],1)).T[:,indsPlaneEllipse],axis=0)

#Calculate the Semi-minor axis of the look vector ellipse in the orbital plane
smna_look = dist_rgs_Olook*np.tan(FOV/2.)
#The Location in 3D space of the semi-minor axis FOV extent
r_smna_look = Olook + smna_look*r_FOV_leftNorm


#Extract components from 3D problem to formulate 2D problem, the orbital ellipse and FOV cone ellipse
# x_Olook = np.einsum('ij,ij->j',Olook,xhats[:,indsPlaneEllipse]) #combined into r_xy_look
# y_Olook = np.einsum('ij,ij->j',Olook,yhats[:,indsPlaneEllipse])
r_xy_Olook = np.asarray([np.einsum('ij,ij->j',Olook,xhats[:,indsPlaneEllipse]),np.einsum('ij,ij->j',Olook,yhats[:,indsPlaneEllipse])])
r_xy_FOVupper = np.asarray([np.einsum('ij,ij->j',r_FOV_upper,xhats[:,indsPlaneEllipse]),np.einsum('ij,ij->j',r_FOV_upper,yhats[:,indsPlaneEllipse])])
r_xy_smnalook = np.asarray([np.einsum('ij,ij->j',r_smna_look,xhats[:,indsPlaneEllipse]),np.einsum('ij,ij->j',r_smna_look,yhats[:,indsPlaneEllipse])])
r_xy_sma_orbit = np.asarray([np.einsum('ij,ij->j',r_sma_orbit[:,indsPlaneEllipse],xhats[:,indsPlaneEllipse]),np.einsum('ij,ij->j',r_sma_orbit[:,indsPlaneEllipse],yhats[:,indsPlaneEllipse])])
r_xy_smna_orbit = np.asarray([np.einsum('ij,ij->j',r_smna_orbit[:,indsPlaneEllipse],xhats[:,indsPlaneEllipse]),np.einsum('ij,ij->j',r_smna_orbit[:,indsPlaneEllipse],yhats[:,indsPlaneEllipse])])
r_xy_Oorbit = np.asarray([np.einsum('ij,ij->j',Oorbit[:,indsPlaneEllipse],xhats[:,indsPlaneEllipse]),np.einsum('ij,ij->j',Oorbit[:,indsPlaneEllipse],yhats[:,indsPlaneEllipse])])
#np.histogram(np.einsum('ij,ij->j',r_xy_FOVupper-r_xy_Olook,r_xy_smnalook-r_xy_Olook),bins=[-400,-1,-0.01,-0.001,0,0.001,0.01,1,10,100]) #confirms orthogonality of r_xy_FOVupper and r_xy_smnalook
#np.histogram(np.einsum('ij,ij->j',r_xy_sma_orbit-r_xy_Oorbit,r_xy_smna_orbit-r_xy_Oorbit),bins=[-400,-1,-0.01,-0.001,0,0.001,0.01,1,10,100]) #confirms orthogonality of r_xy_sma_orbit and r_xy_smna_orbit

#Center these 3D points at with respect to the geometric center of the 3D orbital ellipse
r_xy_Olook_cr = r_xy_Olook - r_xy_Oorbit
r_xy_FOVupper_cr = r_xy_FOVupper - r_xy_Oorbit
r_xy_smnalook_cr = r_xy_smnalook - r_xy_Oorbit
r_xy_sma_orbit_cr = r_xy_sma_orbit - r_xy_Oorbit
r_xy_smna_orbit_cr = r_xy_smna_orbit - r_xy_Oorbit
r_xy_Oorbit_cr = r_xy_Oorbit - r_xy_Oorbit

#Calculate the 2D rotation angle of r_xy_orbit from xhats
rotAng2D = np.arctan2(r_xy_sma_orbit[1]-r_xy_Oorbit[1],r_xy_sma_orbit[0]-r_xy_Oorbit[0]) #the angle between the x-axis and the sma of the orbital ellipse (the angle with which to derotate all points)

#Derotate these 2D points
r_xy_Olook_dr = np.asarray([np.cos(-rotAng2D)*r_xy_Olook_cr[0]  + np.sin(-rotAng2D)*r_xy_Olook_cr[0], -np.sin(-rotAng2D)*r_xy_Olook_cr[1]  + np.cos(-rotAng2D)*r_xy_Olook_cr[1]])
r_xy_FOVupper_dr = np.asarray([np.cos(-rotAng2D)*r_xy_FOVupper_cr[0]  + np.sin(-rotAng2D)*r_xy_FOVupper_cr[0], -np.sin(-rotAng2D)*r_xy_FOVupper_cr[1]  + np.cos(-rotAng2D)*r_xy_FOVupper_cr[1]])
r_xy_smnalook_dr = np.asarray([np.cos(-rotAng2D)*r_xy_smnalook_cr[0]  + np.sin(-rotAng2D)*r_xy_smnalook_cr[0], -np.sin(-rotAng2D)*r_xy_smnalook_cr[1]  + np.cos(-rotAng2D)*r_xy_smnalook_cr[1]])
r_xy_sma_orbit_dr = np.asarray([np.cos(-rotAng2D)*r_xy_sma_orbit_cr[0]  + np.sin(-rotAng2D)*r_xy_sma_orbit_cr[0], -np.sin(-rotAng2D)*r_xy_sma_orbit_cr[1]  + np.cos(-rotAng2D)*r_xy_sma_orbit_cr[1]])
r_xy_smna_orbit_dr = np.asarray([np.cos(-rotAng2D)*r_xy_smna_orbit_cr[0]  + np.sin(-rotAng2D)*r_xy_smna_orbit_cr[0], -np.sin(-rotAng2D)*r_xy_smna_orbit_cr[1]  + np.cos(-rotAng2D)*r_xy_smna_orbit_cr[1]])
r_xy_Oorbit_dr = np.asarray([np.cos(-rotAng2D)*r_xy_Oorbit_cr[0]  + np.sin(-rotAng2D)*r_xy_Oorbit_cr[0], -np.sin(-rotAng2D)*r_xy_Oorbit_cr[1]  + np.cos(-rotAng2D)*r_xy_Oorbit_cr[1]])
#now the orbital ellipse should be derotated

#Scale (divide) all points along x-axis by sma_orbit/smna_orbit
r_xy_Olook_dr_sc = np.asarray([r_xy_Olook_dr[0]*smna[indsPlaneEllipse]/sma[indsPlaneEllipse],r_xy_Olook_dr[1]])
r_xy_FOVupper_dr_sc = np.asarray([r_xy_FOVupper_dr[0]*smna[indsPlaneEllipse]/sma[indsPlaneEllipse],r_xy_FOVupper_dr[1]])
r_xy_smnalook_dr_sc = np.asarray([r_xy_smnalook_dr[0]*smna[indsPlaneEllipse]/sma[indsPlaneEllipse],r_xy_smnalook_dr[1]])
r_xy_sma_orbit_dr_sc = np.asarray([r_xy_sma_orbit_dr[0]*smna[indsPlaneEllipse]/sma[indsPlaneEllipse],r_xy_sma_orbit_dr[1]])
r_xy_smna_orbit_dr_sc = np.asarray([r_xy_smna_orbit_dr[0]*smna[indsPlaneEllipse]/sma[indsPlaneEllipse],r_xy_smna_orbit_dr[1]])
r_xy_Oorbit_dr_sc = np.asarray([r_xy_Oorbit_dr[0]*smna[indsPlaneEllipse]/sma[indsPlaneEllipse],r_xy_Oorbit_dr[1]])

#The orbital ellipse circular, but the FOV ellipse is now distorted
#The new sma and smna vectors must be calculated for the FOV ellipse
#Question: Are the r_xy_smnalook_dr_sc and r_xy_FOVupper_dr_sc vectors still orthogonal to one another? that would disprove the above statement, most are close, but they are not all orthogonal
#np.einsum('ij,ij->j',r_xy_FOVupper_dr_sc-r_xy_Olook_dr_sc,r_xy_smnalook_dr_sc-r_xy_Olook_dr_sc) #gives dot product between the two to demonstrate orthogonality, no but many are close to orthogonal
#Calculate sma_dr_sc, smna_dr_sc, and phi_dr_sc of look vector ellipse
OpCp = r_xy_smnalook_dr_sc - r_xy_Olook_dr_sc #Used to define slope of QQp line
QQphat = np.asarray([OpCp[1],-OpCp[0]])/np.linalg.norm(OpCp,axis=0) #QQp line unit vector
OpBp = r_xy_FOVupper_dr_sc - r_xy_Olook_dr_sc #Define center of QQp line
OpQ = OpBp - QQphat*np.linalg.norm(OpCp,axis=0)
OpQp = OpBp + QQphat*np.linalg.norm(OpCp,axis=0)
sma_dr_sc = (np.linalg.norm(OpQp,axis=0) + np.linalg.norm(OpQ,axis=0))/2. #Calculates the semi-major axis of the derotated and scaled FOV ellipse
smna_dr_sc = (np.linalg.norm(OpQp,axis=0) - np.linalg.norm(OpQ,axis=0))/2. #Calculates the semi-minor axis of the derotated and scaled FOV ellipse
psi = 0.5*(np.arctan2(OpQ[1],OpQ[0]) + np.arctan2(OpQp[1],OpQp[0])) #Calculate the angle in radians the semi-major axis of the derotated and scaled FOV ellipse makes with the x-axis



# def theta_limits(x, y, th, h, k, psi, a_lscdr, b_lscdr):
#     """Takes in some ellipse in 3D space and 1) determines whether the origin is inside the ellipse, returns this boolean array and
#     2) calculates the angular limits if it is not inside the ellipse and
#     3) returns the inds where
#     Args:
#         ndarray:
#             x
#         ndarray:
#             y
#         ndarray:
#             th
#         ndarray:
#             h
#         ndarray:
#             k
#         ndarray:
#             psi
#         ndarray:
#             a_lscdr
#         ndarray:
#             b_lscdr
#     Returns:
#         ndarray:
#             insideEllipseBool
#         ndarray:
#             thetas
#     """
#     insideEllipseBool = ((x-h)/a_lscdr)**2. + ((y-k)/b_lscdr)**2. <= 1. #A boolean to determine whether the origin is inside the ellipse or not

#     theta0 = -np.arccos(-a_lscdr*b_lscdr*np.sqrt(a_lscdr**2.*np.cos(psi)**2. + b_lscdr**2.*np.sin(psi)**2. - h**2.*np.sin(psi)**4. - 2.*h**2.*np.sin(psi)**2.*np.cos(psi)**2.\
#         - h**2.*np.cos(psi)**4. + 2.*h*x*np.sin(psi)**4. + 4*h*x*np.sin(psi)**2.*np.cos(psi)**2. + 2.*h*x*np.cos(psi)**4. - x**2.*np.sin(psi)**4. - 2.*x**2.*np.sin(psi)**2.*np.cos(psi)**2.\
#         - x**2.*np.cos(psi)**4.)/(x*(a_lscdr**2.*np.cos(psi)**2. + b_lscdr**2.*np.sin(psi)**2.)) + (-a_lscdr**2.*h*np.sin(psi)*np.cos(psi) + a_lscdr**2.*k*np.cos(psi)**2. + a_lscdr**2.*x*np.sin(psi)*np.cos(psi)\
#         + b_lscdr**2.*h*np.sin(psi)*np.cos(psi) + b_lscdr**2.*k*np.sin(psi)**2. - b_lscdr**2.*x*np.sin(psi)*np.cos(psi))/(x*(a_lscdr**2.*np.cos(psi)**2. + b_lscdr**2.*np.sin(psi)**2.))) + 2.*np.pi
#     theta1 = -np.arccos(a_lscdr*b_lscdr*np.sqrt(a_lscdr**2.*np.cos(psi)**2. + b_lscdr**2.*np.sin(psi)**2. - h**2.*np.sin(psi)**4. - 2.*h**2.*np.sin(psi)**2.*np.cos(psi)**2.\
#         - h**2.*np.cos(psi)**4. + 2.*h*x*np.sin(psi)**4. + 4*h*x*np.sin(psi)**2.*np.cos(psi)**2. + 2.*h*x*np.cos(psi)**4. - x**2.*np.sin(psi)**4. - 2.*x**2.*np.sin(psi)**2.*np.cos(psi)**2.\
#         - x**2.*np.cos(psi)**4.)/(x*(a_lscdr**2.*np.cos(psi)**2. + b_lscdr**2.*np.sin(psi)**2.)) + (-a_lscdr**2.*h*np.sin(psi)*np.cos(psi) + a_lscdr**2.*k*np.cos(psi)**2. + a_lscdr**2.*x*np.sin(psi)*np.cos(psi)\
#         + b_lscdr**2.*h*np.sin(psi)*np.cos(psi) + b_lscdr**2.*k*np.sin(psi)**2. - b_lscdr**2.*x*np.sin(psi)*np.cos(psi))/(x*(a_lscdr**2.*np.cos(psi)**2. + b_lscdr**2.*np.sin(psi)**2.))) + 2.*np.pi
#     theta2 = np.arccos(-a_lscdr*b_lscdr*np.sqrt(a_lscdr**2.*np.cos(psi)**2. + b_lscdr**2.*np.sin(psi)**2. - h**2.*np.sin(psi)**4. - 2.*h**2.*np.sin(psi)**2.*np.cos(psi)**2.\
#         - h**2.*np.cos(psi)**4. + 2.*h*x*np.sin(psi)**4. + 4*h*x*np.sin(psi)**2.*np.cos(psi)**2. + 2.*h*x*np.cos(psi)**4. - x**2.*np.sin(psi)**4. - 2.*x**2.*np.sin(psi)**2.*np.cos(psi)**2.\
#         - x**2.*np.cos(psi)**4.)/(x*(a_lscdr**2.*np.cos(psi)**2. + b_lscdr**2.*np.sin(psi)**2.)) + (-a_lscdr**2.*h*np.sin(psi)*np.cos(psi) + a_lscdr**2.*k*np.cos(psi)**2. + a_lscdr**2.*x*np.sin(psi)*np.cos(psi)\
#         + b_lscdr**2.*h*np.sin(psi)*np.cos(psi) + b_lscdr**2.*k*np.sin(psi)**2. - b_lscdr**2.*x*np.sin(psi)*np.cos(psi))/(x*(a_lscdr**2.*np.cos(psi)**2. + b_lscdr**2.*np.sin(psi)**2.)))
#     theta3 = np.arccos(a_lscdr*b_lscdr*np.sqrt(a_lscdr**2.*np.cos(psi)**2. + b_lscdr**2.*np.sin(psi)**2. - h**2.*np.sin(psi)**4. - 2.*h**2.*np.sin(psi)**2.*np.cos(psi)**2.\
#         - h**2.*np.cos(psi)**4. + 2.*h*x*np.sin(psi)**4. + 4*h*x*np.sin(psi)**2.*np.cos(psi)**2. + 2.*h*x*np.cos(psi)**4. - x**2.*np.sin(psi)**4. - 2.*x**2.*np.sin(psi)**2.*np.cos(psi)**2.\
#         - x**2.*np.cos(psi)**4.)/(x*(a_lscdr**2.*np.cos(psi)**2. + b_lscdr**2.*np.sin(psi)**2.)) + (-a_lscdr**2.*h*np.sin(psi)*np.cos(psi) + a_lscdr**2.*k*np.cos(psi)**2. + a_lscdr**2.*x*np.sin(psi)*np.cos(psi)\
#         + b_lscdr**2.*h*np.sin(psi)*np.cos(psi) + b_lscdr**2.*k*np.sin(psi)**2. - b_lscdr**2.*x*np.sin(psi)*np.cos(psi))/(x*(a_lscdr**2.*np.cos(psi)**2. + b_lscdr**2.*np.sin(psi)**2.)))
#     return insideEllipseBool, np.asarray([theta0,theta1,theta2,theta3])

#Calculate derotated, centered FOV ellipse sma_dr



#############################################################################

#Circularize the orbital ellipse
#i.e. 



