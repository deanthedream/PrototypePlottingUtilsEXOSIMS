# Ground Station and Spacecraft FOV calculations
import numpy as np
import math as m

#DELETE
#a = np.asarray([[1,0,0],[0,1,0],[0,0,1],[2,0,2]])
#b = np.asarray([[1,0,0],[1,0,0],[1,0,0],[1,0,0]])
#c = np.einsum('ij,ij->i',a,b)


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
FOV = 20.*np.pi/180. #2 deg is approximately the size of the field of view of a small telescope
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


# Use ellipse equation to determine whether the circle center is within the bounds of the ellipse.
#If it is, we can reformulate the problem and compute whether the smin, smax, slmin, slmax which can be used to find
#borbit relative to these and classify how many intersections will occur



def calcMasterIntersections_scFOV(x,y,b_orbit,dmajorp,dminorp,psi):
    """ A method for calculating the nu and times of orbit and circle intersections as well as extrema
    In the memory efficient method, there are approximately 349 Bytes per planet
    When plotting, extra variables are saved resulting in approximately 373 Bytes per planet
    Args:
    ndarray:
        x,y,b_orbit,dmajorp,dminorp,psi
    Returns:
        dmajorp (numpy array): 
            the semi-major axis of the projected ellipse
        dminorp (numpy array):
            the semi-minor axis of the projected ellipse
        theta_OpQ_X (numpy array):
            the angle formed between point Q, the geometric center of
            the projected ellipse, and the X-axis
        theta_OpQp_X (numpy array):
            the angle formed between point Q, the geometric center of
            the projected ellipse, and the X-axis
        Op (numpy array):
            the geometric center of the projected ellipse
        x (numpy array):
            the x component of the projected star location
        y (numpy array):
            the y component of the projected star location
        phi (numpy array):
            angle from X-axis to semi-minor axis of projected ellipse 

        xreal (numpy array):
        only2RealInds (numpy array):
        yrealAllRealInds (numpy array):
        fourIntInds (numpy array):
        twoIntOppositeXInds (numpy array):
        twoIntSameYInds (numpy array):
        yrealImagInds (numpy array):
        minSepPoints_x (numpy array):
        minSepPoints_y (numpy array):
        maxSepPoints_x (numpy array):
        maxSepPoints_y (numpy array):
        lminSepPoints_x (numpy array):
        lminSepPoints_y (numpy array):
        lmaxSepPoints_x (numpy array):
        lmaxSepPoints_y (numpy array):
        minSep (numpy array):
        maxSep (numpy array):
        lminSep (numpy array):
        lmaxSep (numpy array):
        errors_fourInt0 (numpy array):
        errors_fourInt1 (numpy array):
        errors_fourInt2 (numpy array):
        errors_fourInt3 (numpy array):
        errors_twoIntSameY0 (numpy array):
        errors_twoIntSameY1 (numpy array):
        errors_twoIntOppositeX0 (numpy array):
        errors_twoIntOppositeX1 (numpy array):
        errors_IntersectionsOnly2X0 (numpy array):
        errors_IntersectionsOnly2X1 (numpy array):
        type0_0Inds (numpy array):\
        type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,type2_0Inds,type2_1Inds,type2_2Inds,\
        type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds
        fourInt_x (numpy array):
        fourInt_y (numpy array):
        twoIntSameY_x (numpy array):
        twoIntSameY_y (numpy array):
        twoIntOppositeX_x (numpy array):
        twoIntOppositeX_y (numpy array):
        xIntersectionsOnly2 (numpy array):
        yIntersectionsOnly2 (numpy array):
        typeInds0 (numpy array):
        typeInds1 (numpy array):
        typeInds2 (numpy array):
        typeInds3 (numpy array):
    """
    #### Calculate X,Y Position of Minimum and Maximums with Quartic
    A, B, C, D = quarticCoefficients_smin_smax_lmin_lmax(dmajorp.astype('complex128'), dminorp, np.abs(x), np.abs(y)) #calculate the quartic solutions to the min-max separation problem
    xreal, _, _, _, _, _ = quarticSolutions_ellipse_to_Quarticipynb(A.astype('complex128'), B, C, D)
    del A, B, C, D #delting for memory efficiency

    #AT LEAST ONE INSTANCE OF THIS OCCURED WHERE THE ELLIPSE IS EFFECTIVELY A CIRCLE AND THE THE SMA IS >> b_orbit
    xreal.real = np.abs(xreal) #all solutions should be positive

    #### Technically, each row must have at least 2 solutions, but whatever
    yreal = ellipseYFromX(xreal.astype('complex128'), dmajorp, dminorp) #Calculates the y values corresponding to the x values in the first quadrant of an ellipse
    ####

    #### Calculate Minimum, Maximum, Local Minimum, Local Maximum Separations
    minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y,\
        minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds = smin_smax_slmin_slmax(len(x), xreal, yreal, np.abs(x), np.abs(y), x, y)
    lminSepPoints_x = np.real(lminSepPoints_x)
    lminSepPoints_y = np.real(lminSepPoints_y)
    lmaxSepPoints_x = np.real(lmaxSepPoints_x)
    lmaxSepPoints_y = np.real(lmaxSepPoints_y)
    ####

    #### Ellipse Circle Intersection #######################################################################
    only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3,\
        fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
        twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2, twoIntSameYInds,\
        type0_0Inds,type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,\
        type2_0Inds,type2_1Inds,type2_2Inds,type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,\
        _ = ellipseCircleIntersections(b_orbit, dmajorp, dminorp, np.abs(x), np.abs(y), x, y, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds)
    # if plotBool == False:
    #     del typeInds0, typeInds1, typeInds2, typeInds3
    #     del type0_0Inds,type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds
    #     del type2_0Inds,type2_1Inds,type2_2Inds,type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds
    ####

    #### Correct Ellipse Circle Intersections fourInt1 ####################################
    fourInt_x[:,0], fourInt_y[:,0] = intersectionFixer_pm(x, y, fourInt_x[:,0], fourInt_y[:,0], yrealAllRealInds[fourIntInds], b_orbit[yrealAllRealInds[fourIntInds]]) #necessary because a minority of cases occur in quadrant 3
    fourInt_x[:,1], fourInt_y[:,1] = intersectionFixer_pm(x, y, fourInt_x[:,1], fourInt_y[:,1], yrealAllRealInds[fourIntInds], b_orbit[yrealAllRealInds[fourIntInds]]) #necessary because a minority of cases occur in quadrant 4
    fourInt_x[:,2], fourInt_y[:,2] = intersectionFixer_pm(x, y, fourInt_x[:,2], fourInt_y[:,2], yrealAllRealInds[fourIntInds], b_orbit[yrealAllRealInds[fourIntInds]])
    fourInt_x[:,3], fourInt_y[:,3] = intersectionFixer_pm(x, y, fourInt_x[:,3], fourInt_y[:,3], yrealAllRealInds[fourIntInds], b_orbit[yrealAllRealInds[fourIntInds]])
    #### Correct Ellipse Circle Intersections twoIntSameY0
    twoIntSameY_x[:,0], twoIntSameY_y[:,0] = intersectionFixer_pm(x, y, twoIntSameY_x[:,0], twoIntSameY_y[:,0], yrealAllRealInds[twoIntSameYInds], b_orbit[yrealAllRealInds[twoIntSameYInds]])
    #### Correct Ellipse Circle Intersections twoIntSameY1 
    twoIntSameY_x[:,1], twoIntSameY_y[:,1] = intersectionFixer_pm(x, y, twoIntSameY_x[:,1], twoIntSameY_y[:,1], yrealAllRealInds[twoIntSameYInds], b_orbit[yrealAllRealInds[twoIntSameYInds]])
    #### Correct Ellipse Circle Intersections twoIntOppositeX0
    twoIntOppositeX_x[:,0], twoIntOppositeX_y[:,0] = intersectionFixer_pm(x, y, twoIntOppositeX_x[:,0], twoIntOppositeX_y[:,0], yrealAllRealInds[twoIntOppositeXInds], b_orbit[yrealAllRealInds[twoIntOppositeXInds]])
    #### Correct Ellipse Circle Intersections twoIntOppositeX1 
    twoIntOppositeX_x[:,1], twoIntOppositeX_y[:,1] = intersectionFixer_pm(x, y, twoIntOppositeX_x[:,1], twoIntOppositeX_y[:,1], yrealAllRealInds[twoIntOppositeXInds], b_orbit[yrealAllRealInds[twoIntOppositeXInds]])
    #### COULD RUN ON OTHER CASES #########################################################

    #### Rerotate Extrema and Intersection Points
    minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,\
        fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr = \
        rerotateExtremaAndIntersectionPoints(minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y,\
        fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
        psi, np.asarray([x,y]), yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds)
    # if plotBool == False:
    #     del minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y
    #     del fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2
    del minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr
    del fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr
    ####

    # #### Memory Calculations
    # #Necessary Variables
    # if plotBool == True:
    #     memory_necessary = [inc.nbytes,w.nbytes,W.nbytes,sma.nbytes,e.nbytes,dmajorp.nbytes,dminorp.nbytes,\
    #         Op.nbytes,x.nbytes,y.nbytes,Phi.nbytes,xreal.nbytes,only2RealInds.nbytes,yrealAllRealInds.nbytes,fourIntInds.nbytes,twoIntOppositeXInds.nbytes,twoIntSameYInds.nbytes,\
    #         nu_minSepPoints.nbytes,nu_maxSepPoints.nbytes,nu_lminSepPoints.nbytes,nu_lmaxSepPoints.nbytes,nu_fourInt.nbytes,nu_twoIntSameY.nbytes,nu_twoIntOppositeX.nbytes,nu_IntersectionsOnly2.nbytes,\
    #         minSepPoints_x.nbytes, minSepPoints_y.nbytes, maxSepPoints_x.nbytes, maxSepPoints_y.nbytes, lminSepPoints_x.nbytes, lminSepPoints_y.nbytes, lmaxSepPoints_x.nbytes,\
    #         lmaxSepPoints_y.nbytes, minSep.nbytes, maxSep.nbytes, lminSep.nbytes, lmaxSep.nbytes, yrealImagInds.nbytes,\
    #         t_minSep.nbytes,t_maxSep.nbytes,t_lminSep.nbytes,t_lmaxSep.nbytes,t_fourInt0.nbytes,t_fourInt1.nbytes,t_fourInt2.nbytes,t_fourInt3.nbytes,\
    #         t_twoIntSameY0.nbytes,t_twoIntSameY1.nbytes,t_twoIntOppositeX0.nbytes,t_twoIntOppositeX1.nbytes,t_IntersectionOnly20.nbytes,t_IntersectionOnly21.nbytes]
    #     print('memory_necessary Used: ' + str(np.sum(memory_necessary)/10**9) + ' GB')

    #if plotBool == False:
        #return dmajorp,dminorp,_,_,x,y,xreal,only2RealInds,\
        #     yrealAllRealInds,fourIntInds,twoIntOppositeXInds,twoIntSameYInds,yrealImagInds,\
        #     _,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_
    # return dmajorp,dminorp,_,_,x,y,xreal,only2RealInds,yrealAllRealInds,\
    #     fourIntInds,twoIntOppositeXInds,twoIntSameYInds,yrealImagInds,\
    #     minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep,\
    #     errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,errors_twoIntSameY0,\
    #     errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,_,\
    #     _,_,_,_,_,_,_,_,_,_,_,_,\
    #     _,_,_,_,_,_,_,fourInt_x,fourInt_y,twoIntSameY_x,twoIntSameY_y,twoIntOppositeX_x,\
    #     twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,_,_,_,_

    # else:
    return dmajorp,dminorp,x,y,xreal,only2RealInds,yrealAllRealInds,\
        fourIntInds,twoIntOppositeXInds,twoIntSameYInds,yrealImagInds,\
        minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep,\
        type0_0Inds,\
        type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,type2_0Inds,type2_1Inds,type2_2Inds,\
        type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,fourInt_x,fourInt_y,twoIntSameY_x,twoIntSameY_y,twoIntOppositeX_x,\
        twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3


def plotDerotatedIntersectionsMinMaxStarLocBounds_scFOV(ind, sma, e, W, w, inc, x, y, dmajorp, dminorp, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
    minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
    lmaxSepPoints_x, lmaxSepPoints_y, twoIntSameYInds,\
    maxSepPoints_x, maxSepPoints_y, twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
    type0_0Inds, type0_1Inds, type0_2Inds, type0_3Inds, type0_4Inds, type1_0Inds, type1_1Inds, type1_2Inds, type1_3Inds, type1_4Inds,\
    type2_0Inds, type2_1Inds, type2_2Inds, type2_3Inds, type2_4Inds, type3_0Inds, type3_1Inds, type3_2Inds, type3_3Inds, type3_4Inds, num, s_circ=None):
    """
    """
    if s_circ is None: #defines the circle radius
        s_circ = np.ones(len(sma),dtype=float)

    plt.close(num)
    fig = plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    ca = plt.gca()
    ca.axis('equal')
    #DELETEplt.scatter([xreal[ind,0],xreal[ind,1],xreal[ind,2],xreal[ind,3]], [yreal[ind,0],yreal[ind,1],yreal[ind,2],yreal[ind,3]], color='purple')
    plt.scatter([0],[0],color='purple')
    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    #new plot stuff
    Erange = np.linspace(start=0.,stop=2*np.pi,num=400)
    plt.plot([-dmajorp[ind],dmajorp[ind]],[0,0],color='purple',linestyle='--',zorder=2) #major
    plt.plot([0,0],[-dminorp[ind],dminorp[ind]],color='purple',linestyle='--',zorder=2) #minor
    xellipsetmp = dmajorp[ind]*np.cos(Erange)
    yellipsetmp = dminorp[ind]*np.sin(Erange)
    plt.plot(xellipsetmp,yellipsetmp,color='purple')
    plt.scatter(x[ind],y[ind],color='black',marker='o')
    if ind in only2RealInds[typeInds0]:
        plt.scatter(x[ind],y[ind],edgecolors='teal',marker='o',s=64,facecolors='none')
    if ind in only2RealInds[typeInds1]:
        plt.scatter(x[ind],y[ind],edgecolors='red',marker='o',s=64,facecolors='none')
    if ind in only2RealInds[typeInds2]:
        plt.scatter(x[ind],y[ind],edgecolors='blue',marker='o',s=64,facecolors='none')
    if ind in only2RealInds[typeInds3]:
        plt.scatter(x[ind],y[ind],edgecolors='magenta',marker='o',s=64,facecolors='none')

    c_ae = dmajorp[ind]*np.sqrt(1-dminorp[ind]**2/dmajorp[ind]**2)
    plt.scatter([-c_ae,c_ae],[0,0],color='blue')

    # #Plot Min Sep Circle
    # x_circ = minSep[ind]*np.cos(vs)
    # y_circ = minSep[ind]*np.sin(vs)
    # plt.plot(x[ind]+x_circ,y[ind]+y_circ,color='teal')
    # #Plot Max Sep Circle
    # x_circ2 = maxSep[ind]*np.cos(vs)
    # y_circ2 = maxSep[ind]*np.sin(vs)
    # plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='red')
    #Plot Min Sep Ellipse Intersection
    plt.scatter(minSepPoints_x[ind],minSepPoints_y[ind],color='teal',marker='D',zorder=3)
    #Plot Max Sep Ellipse Intersection
    plt.scatter(maxSepPoints_x[ind],maxSepPoints_y[ind],color='red',marker='D',zorder=3)

    if ind in yrealAllRealInds:
        tind = np.where(yrealAllRealInds == ind)[0]
        # #Plot lminSep Circle
        # x_circ2 = lminSep[tind]*np.cos(vs)
        # y_circ2 = lminSep[tind]*np.sin(vs)
        # plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='magenta')
        # #Plot lmaxSep Circle
        # x_circ2 = lmaxSep[tind]*np.cos(vs)
        # y_circ2 = lmaxSep[tind]*np.sin(vs)
        # plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='gold')
        #### Plot Local Min
        plt.scatter(lminSepPoints_x[tind], lminSepPoints_y[tind],color='magenta',marker='D',zorder=3)
        #### Plot Local Max Points
        plt.scatter(lmaxSepPoints_x[tind], lmaxSepPoints_y[tind],color='gold',marker='D',zorder=3)

    #### r Intersection test
    x_circ2 = s_circ[ind]*np.cos(vs)
    y_circ2 = s_circ[ind]*np.sin(vs)
    plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='black')
    #### Intersection Points
    if ind in yrealAllRealInds[fourIntInds]:
        yind = np.where(yrealAllRealInds[fourIntInds] == ind)[0]
        plt.scatter(fourInt_x[yind],fourInt_y[yind], color='black',marker='o')
    elif ind in yrealAllRealInds[twoIntSameYInds]: #Same Y
        yind = np.where(yrealAllRealInds[twoIntSameYInds] == ind)[0]
        plt.scatter(twoIntSameY_x[yind],twoIntSameY_y[yind], color='black',marker='o')
    elif ind in yrealAllRealInds[twoIntOppositeXInds]: #Same X
        yind = np.where(yrealAllRealInds[twoIntOppositeXInds] == ind)[0]
        plt.scatter(twoIntOppositeX_x[yind],twoIntOppositeX_y[yind], color='black',marker='o')
        #### Type0
    elif ind in only2RealInds[type0_0Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
        print('plotted')
    elif ind in only2RealInds[type0_1Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type0_2Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type0_3Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type0_4Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
        #### Type1
    elif ind in only2RealInds[type1_0Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type1_1Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type1_2Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type1_3Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type1_4Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
        #### Type2
    elif ind in only2RealInds[type2_0Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type2_1Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type2_2Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type2_3Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type2_4Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
        #### Type3
    elif ind in only2RealInds[type3_0Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type3_1Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type3_2Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type3_3Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')
    elif ind in only2RealInds[type3_4Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='black',marker='o')

    # Plot Star Location Type Dividers
    xran = np.linspace(start=(dmajorp[ind]*(dmajorp[ind]**2*(dmajorp[ind] - dminorp[ind])*(dmajorp[ind] + dminorp[ind]) - dminorp[ind]**2*np.sqrt(3*dmajorp[ind]**4 + 2*dmajorp[ind]**2*dminorp[ind]**2 + 3*dminorp[ind]**4))/(2*(dmajorp[ind]**4 + dminorp[ind]**4))),\
        stop=(dmajorp[ind]*(dmajorp[ind]**2*(dmajorp[ind] - dminorp[ind])*(dmajorp[ind] + dminorp[ind]) + dminorp[ind]**2*np.sqrt(3*dmajorp[ind]**4 + 2*dmajorp[ind]**2*dminorp[ind]**2 + 3*dminorp[ind]**4))/(2*(dmajorp[ind]**4 + dminorp[ind]**4))), num=3, endpoint=True)
    ylineQ1 = xran*dmajorp[ind]/dminorp[ind] - dmajorp[ind]**2/(2*dminorp[ind]) + dminorp[ind]/2 #between first quadrant a,b
    ylineQ4 = -xran*dmajorp[ind]/dminorp[ind] + dmajorp[ind]**2/(2*dminorp[ind]) - dminorp[ind]/2 #between 4th quadrant a,b
    plt.plot(xran, ylineQ1, color='brown', linestyle='-.', )
    plt.plot(-xran, ylineQ4, color='grey', linestyle='-.')
    plt.plot(-xran, ylineQ1, color='orange', linestyle='-.')
    plt.plot(xran, ylineQ4, color='red', linestyle='-.')
    plt.xlim([-1.2*dmajorp[ind],1.2*dmajorp[ind]])
    plt.ylim([-1.2*dminorp[ind],1.2*dminorp[ind]])
    #plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + '\nw: ' + str(np.round(w[ind],4)) + ' inc: ' + str(np.round(inc[ind],4)))
    plt.show(block=False)


#Create Intersection Arrays, intersections between 
nu_intersections = np.zeros((len(psi),4))*np.nan
x_intersections = np.zeros((len(psi),4))*np.nan
y_intersections = np.zeros((len(psi),4))*np.nan


# Under these conditions, the orbit circle center is within the elliptical FOV and we can use methods from exodetbox-projectedEllipse to determine intersection locations/quantity
indsOrbitCenterInsideFOVEllipse = np.where(((0.-r_xy_Olook_dr_sc[0])*np.cos(psi) + (0.-r_xy_Olook_dr_sc[1])*np.sin(psi))**2./sma_dr_sc**2. + ((0.-r_xy_Olook_dr_sc[0])*np.sin(psi) - (0.-r_xy_Olook_dr_sc[1])*np.cos(psi))**2./smna_dr_sc**2.<=1.)[0]
indsOrbitCenterOutsideFOVEllipse = np.where(((0.-r_xy_Olook_dr_sc[0])*np.cos(psi) + (0.-r_xy_Olook_dr_sc[1])*np.sin(psi))**2./sma_dr_sc**2. + ((0.-r_xy_Olook_dr_sc[0])*np.sin(psi) - (0.-r_xy_Olook_dr_sc[1])*np.cos(psi))**2./smna_dr_sc**2.>1.)[0]


#### Inds Inside FOV Ellipse
if len(indsOrbitCenterInsideFOVEllipse) > 0: #There is at least one orbit with orbital ellipse geometric center inside FOV ellipse
    #Generally, this method will only be triggered when it is a spacecraft looking back at the Earth, but could also be a 
    
    gamma = np.arctan2(-r_xy_Olook_dr_sc[1,indsOrbitCenterInsideFOVEllipse],-r_xy_Olook_dr_sc[0,indsOrbitCenterInsideFOVEllipse])#-psi[indsOrbitCenterInsideFOVEllipse] #rotational angle from X-axis to Olook->Oorbit vector
    #r_Olook_Oorbit = np.asarray([dist_rgs_Olook[indsOrbitCenterInsideFOVEllipse]*np.cos(tmp_theta - psi[indsOrbitCenterInsideFOVEllipse]), dist_rgs_Olook[indsOrbitCenterInsideFOVEllipse]*np.sin(tmp_theta - psi[indsOrbitCenterInsideFOVEllipse])])

    #We need to reformulate the FOV ellipse - orbit circle problem so the ellipse is centered at the origin and sma of FOV ellipse is aligned along x-axis
    x_orbitCenter_FOVCenter = -r_xy_Olook_dr_sc[0,indsOrbitCenterInsideFOVEllipse]*np.cos(-psi[indsOrbitCenterInsideFOVEllipse]) + r_xy_Olook_dr_sc[1,indsOrbitCenterInsideFOVEllipse]*np.sin(-psi[indsOrbitCenterInsideFOVEllipse])
    y_orbitCenter_FOVCenter = -r_xy_Olook_dr_sc[0,indsOrbitCenterInsideFOVEllipse]*np.sin(-psi[indsOrbitCenterInsideFOVEllipse]) - r_xy_Olook_dr_sc[1,indsOrbitCenterInsideFOVEllipse]*np.cos(-psi[indsOrbitCenterInsideFOVEllipse])

    dmajorp,dminorp,x,y,xreal,only2RealInds,yrealAllRealInds,\
        fourIntInds,twoIntOppositeXInds,twoIntSameYInds,yrealImagInds,\
        minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep,\
        type0_0Inds,\
        type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,type2_0Inds,type2_1Inds,type2_2Inds,\
        type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,fourInt_x,fourInt_y,twoIntSameY_x,twoIntSameY_y,twoIntOppositeX_x,\
        twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3 = calcMasterIntersections_scFOV(x_orbitCenter_FOVCenter,y_orbitCenter_FOVCenter,\
        smna[indsPlaneEllipse[indsOrbitCenterInsideFOVEllipse]].value,sma_dr_sc[indsOrbitCenterInsideFOVEllipse],smna_dr_sc[indsOrbitCenterInsideFOVEllipse],psi[indsOrbitCenterInsideFOVEllipse])

    #### Plot fourIntInds, looks good
    #for i in np.arange(20):
    ind = yrealAllRealInds[np.random.choice(fourIntInds)]
    ind99 = np.where(yrealAllRealInds == ind)[0]
    ind98 = np.where(fourIntInds == ind99)[0]
    blank = np.arange(len(yrealAllRealInds))
    num=int(68743132138431321)#+i)
    plotDerotatedIntersectionsMinMaxStarLocBounds_scFOV(ind, blank, blank, blank, blank, blank, x_orbitCenter_FOVCenter, y_orbitCenter_FOVCenter, sma_dr_sc[indsOrbitCenterInsideFOVEllipse], smna_dr_sc[indsOrbitCenterInsideFOVEllipse], only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
    minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
    lmaxSepPoints_x, lmaxSepPoints_y, twoIntSameYInds,\
    maxSepPoints_x, maxSepPoints_y, twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
    type0_0Inds, type0_1Inds, type0_2Inds, type0_3Inds, type0_4Inds, type1_0Inds, type1_1Inds, type1_2Inds, type1_3Inds, type1_4Inds,\
    type2_0Inds, type2_1Inds, type2_2Inds, type2_3Inds, type2_4Inds, type3_0Inds, type3_1Inds, type3_2Inds, type3_3Inds, type3_4Inds, num, smna[indsPlaneEllipse[indsOrbitCenterInsideFOVEllipse]].value)
    ####

    #### Plot twoIndInds, looks good
    #for i in np.arange(20):
    ind = yrealAllRealInds[np.random.choice(twoIntSameYInds)]
    #ind99 = np.where(yrealAllRealInds == ind)[0]
    #ind98 = np.where(fourIntInds == ind99)[0]
    blank = np.arange(len(yrealAllRealInds))
    num=int(68743132138431321)#+i)
    plotDerotatedIntersectionsMinMaxStarLocBounds_scFOV(ind, blank, blank, blank, blank, blank, x_orbitCenter_FOVCenter, y_orbitCenter_FOVCenter, sma_dr_sc[indsOrbitCenterInsideFOVEllipse], smna_dr_sc[indsOrbitCenterInsideFOVEllipse], only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
    minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
    lmaxSepPoints_x, lmaxSepPoints_y, twoIntSameYInds,\
    maxSepPoints_x, maxSepPoints_y, twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
    type0_0Inds, type0_1Inds, type0_2Inds, type0_3Inds, type0_4Inds, type1_0Inds, type1_1Inds, type1_2Inds, type1_3Inds, type1_4Inds,\
    type2_0Inds, type2_1Inds, type2_2Inds, type2_3Inds, type2_4Inds, type3_0Inds, type3_1Inds, type3_2Inds, type3_3Inds, type3_4Inds, num, smna[indsPlaneEllipse[indsOrbitCenterInsideFOVEllipse]].value)
    ####

    #### x and y of FOV,orbit 4 intersections
    x_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]],0] = r_xy_Olook_dr_sc[0,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]] + fourInt_x[:,0]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]]) - fourInt_y[:,0]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]])
    x_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]],1] = r_xy_Olook_dr_sc[0,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]] + fourInt_x[:,1]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]]) - fourInt_y[:,1]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]])
    x_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]],2] = r_xy_Olook_dr_sc[0,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]] + fourInt_x[:,2]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]]) - fourInt_y[:,2]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]])
    x_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]],3] = r_xy_Olook_dr_sc[0,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]] + fourInt_x[:,3]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]]) - fourInt_y[:,3]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]])
    y_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]],0] = r_xy_Olook_dr_sc[1,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]] + fourInt_x[:,0]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]]) + fourInt_y[:,0]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]])
    y_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]],1] = r_xy_Olook_dr_sc[1,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]] + fourInt_x[:,1]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]]) + fourInt_y[:,1]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]])
    y_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]],2] = r_xy_Olook_dr_sc[1,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]] + fourInt_x[:,2]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]]) + fourInt_y[:,2]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]])
    y_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]],3] = r_xy_Olook_dr_sc[1,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]] + fourInt_x[:,3]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]]) + fourInt_y[:,3]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[fourIntInds]]])
    #2 Int Same Y Side
    x_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]],0] = r_xy_Olook_dr_sc[0,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]]] + twoIntSameY_x[:,0]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]]]) - twoIntSameY_y[:,0]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]]])
    x_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]],1] = r_xy_Olook_dr_sc[0,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]]] + twoIntSameY_x[:,1]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]]]) - twoIntSameY_y[:,1]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]]])
    y_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]],0] = r_xy_Olook_dr_sc[1,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]]] + twoIntSameY_x[:,0]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]]]) + twoIntSameY_y[:,0]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]]])
    y_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]],1] = r_xy_Olook_dr_sc[1,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]]] + twoIntSameY_x[:,1]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]]]) + twoIntSameY_y[:,1]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntSameYInds]]])

    #2 Int Opposite X Side
    x_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]],0] = r_xy_Olook_dr_sc[0,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]]] + twoIntOppositeX_x[:,0]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]]]) - twoIntOppositeX_y[:,0]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]]])
    x_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]],1] = r_xy_Olook_dr_sc[0,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]]] + twoIntOppositeX_x[:,1]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]]]) - twoIntOppositeX_y[:,1]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]]])
    y_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]],0] = r_xy_Olook_dr_sc[1,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]]] + twoIntOppositeX_x[:,0]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]]]) + twoIntOppositeX_y[:,0]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]]])
    y_intersections[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]],1] = r_xy_Olook_dr_sc[1,indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]]] + twoIntOppositeX_x[:,1]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]]]) + twoIntOppositeX_y[:,1]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[yrealAllRealInds[twoIntOppositeXInds]]])


    #### Plot Scaled Orbit at Origin with Scaled FOV ellipse
    num=54867312354343
    plt.close(num)
    plt.figure(num)
    nu = np.linspace(start=0.,stop=2.*np.pi,num=100)
    plt.scatter(0.,0.,color='black') #OK, scaled, derotated orbit center
    #OK
    plt.plot(smna[indsPlaneEllipse[indsOrbitCenterInsideFOVEllipse[ind]]]*np.cos(nu),smna[indsPlaneEllipse[indsOrbitCenterInsideFOVEllipse[ind]]]*np.sin(nu),color='black') #orbit should be at origin
    plt.scatter(x_intersections[indsOrbitCenterInsideFOVEllipse[ind]],y_intersections[indsOrbitCenterInsideFOVEllipse[ind]],color='black')

    x_ellipse = sma_dr_sc[indsOrbitCenterInsideFOVEllipse[ind]]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[ind]])*np.cos(nu) - smna_dr_sc[indsOrbitCenterInsideFOVEllipse[ind]]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[ind]])*np.sin(nu) + r_xy_Olook_dr_sc[0,indsOrbitCenterInsideFOVEllipse[ind]]
    y_ellipse = sma_dr_sc[indsOrbitCenterInsideFOVEllipse[ind]]*np.sin(psi[indsOrbitCenterInsideFOVEllipse[ind]])*np.cos(nu) + smna_dr_sc[indsOrbitCenterInsideFOVEllipse[ind]]*np.cos(psi[indsOrbitCenterInsideFOVEllipse[ind]])*np.sin(nu) + r_xy_Olook_dr_sc[1,indsOrbitCenterInsideFOVEllipse[ind]]
    plt.plot(x_ellipse,y_ellipse,color='purple')
    plt.scatter(r_xy_Olook_dr_sc[0,indsOrbitCenterInsideFOVEllipse[ind]],r_xy_Olook_dr_sc[1,indsOrbitCenterInsideFOVEllipse[ind]], color='purple')

    plt.gca().axis('equal')
    plt.show(block=False)
####

### Inds Outside FOV Ellipse
#NOTE REFORMULATE TO EXCLUDE indsOrbitCenterInsideFOVEllipse (ONLY CALCULATE WITH THE ONES WE NEED) 
#Get FOV ellipse Extremas
from scFOVEllipseSepExtrema import calc_FOVEllipseExtrema_QuarticCoefficients, calc_FOVEllipse_yscdrFromxscdr, calc_FOVEllipse_scdr_dydxEqn
A2, B2, C2, D2, A3, B3, C3, D3 = calc_FOVEllipseExtrema_QuarticCoefficients(sma_dr_sc[indsOrbitCenterOutsideFOVEllipse],smna_dr_sc[indsOrbitCenterOutsideFOVEllipse],r_xy_Olook_dr_sc[0,indsOrbitCenterOutsideFOVEllipse],r_xy_Olook_dr_sc[1,indsOrbitCenterOutsideFOVEllipse],psi[indsOrbitCenterOutsideFOVEllipse])
from exodetbox.projectedEllipse import *
xout2, delta2, P2, D22, R2, delta_02 = quarticSolutions_ellipse_to_Quarticipynb(A2.astype('complex128'), B2, C2, D2)
xout3, delta3, P3, D23, R3, delta_03 = quarticSolutions_ellipse_to_Quarticipynb(A3.astype('complex128'), B3, C3, D3)
xout_combined = np.concatenate((xout2,xout3),axis=1)


#where there are only 2 solutions, use them and use the equation to solve for y
argsortabsimag_xout_combined = np.argsort(np.abs(np.imag(xout_combined)),axis=1)
minImagInds = np.asarray([argsortabsimag_xout_combined[:,0],argsortabsimag_xout_combined[:,1]])
del argsortabsimag_xout_combined
#minImagInds = np.asarray([np.argsort(np.abs(np.imag(xout_combined)),axis=1)[:,0],np.argsort(np.abs(np.imag(xout_combined)),axis=1)[:,1]])
sortabsimag_xout_combined = np.argsort(np.abs(np.imag(xout_combined)),axis=1)
minImagVals = np.asarray([sortabsimag_xout_combined[:,0],sortabsimag_xout_combined[:,1]])
del sortabsimag_xout_combined

x0 = np.real(xout_combined[np.arange(minImagInds.shape[1]),minImagInds[0]])
x1 = np.real(xout_combined[np.arange(minImagInds.shape[1]),minImagInds[1]])
y0m, y0p = calc_FOVEllipse_yscdrFromxscdr(sma_dr_sc[indsOrbitCenterOutsideFOVEllipse],smna_dr_sc[indsOrbitCenterOutsideFOVEllipse],r_xy_Olook_dr_sc[0,indsOrbitCenterOutsideFOVEllipse],r_xy_Olook_dr_sc[1,indsOrbitCenterOutsideFOVEllipse],psi[indsOrbitCenterOutsideFOVEllipse],x0)
y1m, y1p = calc_FOVEllipse_yscdrFromxscdr(sma_dr_sc[indsOrbitCenterOutsideFOVEllipse],smna_dr_sc[indsOrbitCenterOutsideFOVEllipse],r_xy_Olook_dr_sc[0,indsOrbitCenterOutsideFOVEllipse],r_xy_Olook_dr_sc[1,indsOrbitCenterOutsideFOVEllipse],psi[indsOrbitCenterOutsideFOVEllipse],x1)
m0m = np.divide(y0m,x0)
m0p = np.divide(y0p,x0)
dydx0m = calc_FOCEllipse_scdr_dydxEqn(sma_dr_sc[indsOrbitCenterOutsideFOVEllipse],smna_dr_sc[indsOrbitCenterOutsideFOVEllipse],r_xy_Olook_dr_sc[0,indsOrbitCenterOutsideFOVEllipse],r_xy_Olook_dr_sc[1,indsOrbitCenterOutsideFOVEllipse],psi[indsOrbitCenterOutsideFOVEllipse],x0,y0m)
dydx0p = calc_FOCEllipse_scdr_dydxEqn(sma_dr_sc[indsOrbitCenterOutsideFOVEllipse],smna_dr_sc[indsOrbitCenterOutsideFOVEllipse],r_xy_Olook_dr_sc[0,indsOrbitCenterOutsideFOVEllipse],r_xy_Olook_dr_sc[1,indsOrbitCenterOutsideFOVEllipse],psi[indsOrbitCenterOutsideFOVEllipse],x0,y0p)
error0m = np.abs(dxdy0m + 1./m0m)
error0p = np.abs(dxdy0p + 1./m0p)

m1m = np.divide(y1m,x1)
m1p = np.divide(y1p,x1)
dydx1m = calc_FOCEllipse_scdr_dydxEqn(sma_dr_sc[indsOrbitCenterOutsideFOVEllipse],smna_dr_sc[indsOrbitCenterOutsideFOVEllipse],r_xy_Olook_dr_sc[0,indsOrbitCenterOutsideFOVEllipse],r_xy_Olook_dr_sc[1,indsOrbitCenterOutsideFOVEllipse],psi[indsOrbitCenterOutsideFOVEllipse],x1,y1m)
dydx1p = calc_FOCEllipse_scdr_dydxEqn(sma_dr_sc[indsOrbitCenterOutsideFOVEllipse],smna_dr_sc[indsOrbitCenterOutsideFOVEllipse],r_xy_Olook_dr_sc[0,indsOrbitCenterOutsideFOVEllipse],r_xy_Olook_dr_sc[1,indsOrbitCenterOutsideFOVEllipse],psi[indsOrbitCenterOutsideFOVEllipse],x1,y1p)
error1m = np.abs(dxdy1m + 1./m1m)
error1p = np.abs(dxdy1p + 1./m1p)

#numSols = np.sum(np.isnan(xout_combined),axis=1) #number of solutions


print(saltyburrito)

indsWhere0 = np.where(numSols == 0)[0]
indsWhere1 = np.where(numSols == 1)[0]
indsWhere2 = np.where(numSols == 2)[0]
indsWhere3 = np.where(numSols == 3)[0]
indsWhere4 = np.where(numSols == 4)[0]
indsWhere5 = np.where(numSols == 5)[0]
indsWhere6 = np.where(numSols == 6)[0]
indsWhere7 = np.where(numSols == 7)[0]


print(saltyburrito)

#Calculate FOV ellipse Intersections
from scFOVEllipseSepExtrema import calc_FOVEllipseIntersection_QuarticCoefficients
A4, B4, C4, D4 = calc_FOVEllipseIntersection_QuarticCoefficients(sma_dr_sc,smna_dr_sc,r_xy_Olook_dr_sc[0],r_xy_Olook_dr_sc[1],psi,smna[indsPlaneEllipse[indsOrbitCenterInsideFOVEllipse]].value)
xout4, delta4, P4, D24, R4, delta_04 = quarticSolutions_ellipse_to_Quarticipynb(A4.astype('complex128'), B4, C4, D4)








#Rescale Intersections along x
#x_intersections[indsOrbitCenterInsideFOVEllipse] * sma[indsPlaneEllipse[indsOrbitCenterInsideFOVEllipse]]/smna[indsPlaneEllipse[indsOrbitCenterInsideFOVEllipse]]

#Rerotate

#Re-translate Problem






from matplotlib.patches import Arc
def circarrow(self,diameter,centX,centY,startangle,angle,**kwargs):
    startarrow=kwargs.pop("startarrow",False)
    endarrow=kwargs.pop("endarrow",False)

    arc = Arc([centX,centY],diameter,diameter,angle=startangle,
          theta1=np.rad2deg(kwargs.get("head_length",1.5*3*.001)) if startarrow else 0,theta2=angle-(np.rad2deg(kwargs.get("head_length",1.5*3*.001)) if endarrow else 0),linestyle="-",color=kwargs.get("color","black"))
    self.axes().add_patch(arc)

    if startarrow:
        startX=diameter/2*np.cos(np.radians(startangle))
        startY=diameter/2*np.sin(np.radians(startangle))
        startDX=+.000001*diameter/2*np.sin(np.radians(startangle)+kwargs.get("head_length",1.5*3*.001))
        startDY=-.000001*diameter/2*np.cos(np.radians(startangle)+kwargs.get("head_length",1.5*3*.001))
        self.arrow(startX-startDX + centX,startY-startDY + centY,startDX,startDY,**kwargs)

    if endarrow:
        endX=diameter/2*np.cos(np.radians(startangle+angle))
        endY=diameter/2*np.sin(np.radians(startangle+angle))
        endDX=-.000001*diameter/2*np.sin(np.radians(startangle+angle)-kwargs.get("head_length",1.5*3*.001))
        endDY=+.000001*diameter/2*np.cos(np.radians(startangle+angle)-kwargs.get("head_length",1.5*3*.001))
        self.arrow(endX-endDX + centX,endY-endDY + centY,endDX,endDY,**kwargs)

import types
plt.circarrow = types.MethodType(circarrow,plt)




import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.patches as patches

#### Plot the centered, derotated, scaled ellipse problem
tmp_FOV = 30.*np.pi/180.
tmp_r_gs = np.asarray([0.55,0.651])
tmp_r_look = np.asarray([0.25,-0.65])
tmp_r_look_norm = tmp_r_look/np.linalg.norm(tmp_r_look)
t_mag = np.abs(tmp_r_gs[1]/tmp_r_look_norm[1])
num = 5618473214
plt.close(num)
fig3 = plt.figure(num=num)
ax3 = plt.gca()
ax3.scatter(0,0,color='black') #origin

# plt.arrow(0,0,tmp_r_gs[0],tmp_r_gs[1],color='black',width=0.005) #r_gs line
# plt.arrow(0,0,tmp_r_gs[0],0,color='black',linestyle='--',width=0.005) #r_gs_proj line
ax3.quiver(0,0,tmp_r_gs[0],tmp_r_gs[1], angles='xy', scale_units='xy', scale=1,color='black',width=0.005) #r_gs line
#ax3.quiver(0,0,tmp_r_gs[0],0, angles='xy', scale_units='xy', scale=1,color='black',linestyle='--',width=0.005) #r_gs_proj line
ax3.plot([0,tmp_r_gs[0]],[0,0], color='black',linestyle='--',linewidth=2.0) #r_gs_proj line
ax3.plot([tmp_r_gs[0],tmp_r_gs[0]],[0,tmp_r_gs[1]], color='black',linestyle='--',linewidth=2.0) #r_gs_proj line
#ax3.quiver(0,0,tmp_r_gs[0],0,angles='xy', scale_units='xy', scale=1, linestyle='dashed', facecolor='none', linewidth=2, width=0.0001, headwidth=300, headlength=500)

ax3.scatter(tmp_r_gs[0],0,color='black') #r_gs_proj location
ax3.scatter(tmp_r_gs[0],tmp_r_gs[1],color='black') #r_gs location

#plt.arrow(tmp_r_gs[0],tmp_r_gs[1],t_mag*tmp_r_look_norm[0],t_mag*tmp_r_look_norm[1],color='red',width=0.005) #look vector
ax3.quiver(tmp_r_gs[0],tmp_r_gs[1],t_mag*tmp_r_look_norm[0],t_mag*tmp_r_look_norm[1], angles='xy', scale_units='xy', scale=1,color='red',width=0.005) #look vector
#plt.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=21)

#Plot upper and lower FOV limits
tmp_r_look_norm_lower = np.asarray([tmp_r_look_norm[0]*np.cos(-FOV/2.) + tmp_r_look_norm[1]*-np.sin(-FOV/2.),tmp_r_look_norm[0]*np.sin(-FOV/2.) + tmp_r_look_norm[1]*np.cos(-FOV/2.)])
tmp_r_look_norm_lower = tmp_r_look_norm_lower/np.linalg.norm(tmp_r_look_norm_lower)
t_mag2 = np.abs(tmp_r_gs[1]/tmp_r_look_norm_lower[1])
tmp_r_look_norm_upper = np.asarray([tmp_r_look_norm[0]*np.cos(FOV/2.) + tmp_r_look_norm[1]*-np.sin(FOV/2.),tmp_r_look_norm[0]*np.sin(FOV/2.) + tmp_r_look_norm[1]*np.cos(FOV/2.)])
tmp_r_look_norm_upper = tmp_r_look_norm_upper/np.linalg.norm(tmp_r_look_norm_upper)
t_mag3 = np.abs(tmp_r_gs[1]/tmp_r_look_norm_upper[1])
ax3.quiver(tmp_r_gs[0],tmp_r_gs[1],t_mag2*tmp_r_look_norm_lower[0],t_mag2*tmp_r_look_norm_lower[1], angles='xy', scale_units='xy', scale=1,color='red',width=0.005)
ax3.quiver(tmp_r_gs[0],tmp_r_gs[1],t_mag3*tmp_r_look_norm_upper[0],t_mag3*tmp_r_look_norm_upper[1], angles='xy', scale_units='xy', scale=1,color='red',width=0.005)

#Plot lower dashed line
ax3.plot([tmp_r_gs[0],tmp_r_gs[0]+t_mag3*tmp_r_look_norm_upper[0]],[0,0], color='red',linewidth=2.0,linestyle='--')

#FOV intersection points
ax3.scatter(tmp_r_gs[0]+t_mag*tmp_r_look_norm[0],0,color='red') #plot FOV limit points upper
ax3.scatter(tmp_r_gs[0]+t_mag3*tmp_r_look_norm_upper[0],0,color='red') #plot FOV limit points upper
ax3.scatter(tmp_r_gs[0]+t_mag2*tmp_r_look_norm_lower[0],0,color='red') #plot FOV limit points lower

#nhat vector and label
ax3.quiver(tmp_r_gs[0],tmp_r_gs[1] + 0.03,0,0.1, angles='xy', scale_units='xy', scale=1,color='black',width=0.005)
ax3.text(0.47,tmp_r_gs[1]*1.05,r'$\hat{\underline{n}}$',weight='bold',fontdict={'fontsize':'x-large'})

#r_gs vector
ax3.text(0.13,0.25,r'$\underline{r}_{gs}$',weight='bold',fontdict={'fontsize':'x-large'})

#r_gs_proj

#r_look
ax3.text(0.8,0.075,r'$\underline{r}_{look}$',weight='bold',fontdict={'fontsize':'x-large'})

#O_look
ax3.text(0.76,-0.055,r'$O_{look}$',weight='bold',fontdict={'fontsize':'x-large'})

#phi
ax3.text(0.63,0.73,r'$\phi$',weight='bold',fontdict={'fontsize':'x-large'})
plt.circarrow(0.15,tmp_r_gs[0],tmp_r_gs[1],310,132,startarrow=False,endarrow=True,width=0.,head_width=.017,head_length=.015,length_includes_head=True,color="black")

#FOV/2
ax3.text(0.64,0.2,r'$\frac{FOV}{2}$',weight='bold',fontdict={'fontsize':'large'})
ax3.text(0.73,0.2,r'$\frac{FOV}{2}$',weight='bold',fontdict={'fontsize':'large'})


style = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowstyle=style, color="k")
#a3 = patches.FancyArrowPatch((0.6, 0.56), (0.55, 0.735), connectionstyle="arc3,rad=0.83", **kw)
#ax3.add_patch(a3)

plt.circarrow(0.7,tmp_r_gs[0],tmp_r_gs[1],281,10,startarrow=True,endarrow=True,width=0.,head_width=.017,head_length=.015,length_includes_head=True,color="black")
plt.circarrow(0.7,tmp_r_gs[0],tmp_r_gs[1],291,10,startarrow=True,endarrow=True,width=0.,head_width=.017,head_length=.015,length_includes_head=True,color="black")

ax3.set_xlim([0.-0.1*(tmp_r_gs[0] + t_mag3*tmp_r_look_norm_upper[0]),(tmp_r_gs[0] + t_mag3*tmp_r_look_norm_upper[0])*1.1])
ax3.set_ylim([0.-0.1*tmp_r_gs[1],tmp_r_gs[1]*1.3])
plt.axis('off')
plt.show(block=False)
plt.savefig('twoDFOVclassification' + '.svg')
plt.savefig('twoDFOVclassification' + '.png')
####










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



