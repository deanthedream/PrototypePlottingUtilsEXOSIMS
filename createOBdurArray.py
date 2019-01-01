#Written


import numpy as np
import math
try:
    import cPickle as pickle
except:
    import pickle
import os
from pylab import *
from numpy import nan
if not 'DISPLAY' in os.environ.keys(): #Check environment for keys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
import argparse
import json
from EXOSIMS.util.vprint import vprint
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
#from physt import *
#import physt
from mpl_toolkits.mplot3d import Axes3D #required for 3d plot
#import seaborn as sns
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from copy import deepcopy
import random
import itertools
import matplotlib as mpl
import datetime
import re


def generateEquadistantPointsOnSphere(N=100,PPoutpath='./'):
    """Generate uniform points on a sphere
    #https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    Args
    Return:
        xyzpoint (numpy array) - of points with dimensions [~N,3], xyzpoint[~N] = [x,y,z]
        ra_dec (numpy array) - of ra and dec of points with dimensions [~N,2] 
            where [0,0] = firstind, declination
            where [100,1] = 100th ind, right ascension
            ra is from 0 to 2pi
            dec is from 0 to pi
    #designed to be run from EXOSIMS/EXOSIMS/util
    """
    #close('all')
    point = list()
    ra_dec = list()
    r=1.
    Ncount = 0
    a = 4*np.pi*r**2./N
    d = np.sqrt(a)
    Mtheta = np.round(np.pi/d)
    dtheta = np.pi/Mtheta
    dphi = a/dtheta
    for m in np.arange(0,Mtheta):
        theta = np.pi*(m+0.5)/Mtheta
        Mphi = np.round(2*np.pi*np.sin(theta)/dphi)
        for n in np.arange(0,Mphi):
            phi = 2*np.pi*n/Mphi
            x = np.sin(theta)*np.cos(phi)
            y = np.sin(theta)*np.sin(phi)
            z = np.cos(theta)
            point.append([x,y,z])
            ra_dec.append([theta,phi])
            Ncount += 1

    fig = figure(num=5000)
    ax = fig.add_subplot(111, projection='3d')
    xyzpoint = np.asarray(point)
    ra_dec = np.asarray(ra_dec)
    prettifyPlot()
    ax.scatter(xyzpoint[:,0], xyzpoint[:,1], xyzpoint[:,2], color='k', marker='o')
    plt.title('Points Evenly Distributed on a Unit Sphere',weight='bold')
    ax.set_xlabel('x',weight='bold')
    ax.set_ylabel('y',weight='bold')
    ax.set_zlabel('z',weight='bold')
    plt.show(block=False)

    fname = 'PointsEvenlyDistributedOnaUnitSphere'
    plt.savefig(PPoutpath + fname + '.png')
    plt.savefig(PPoutpath + fname + '.svg')
    plt.savefig(PPoutpath + fname + '.eps')

    #output of form ra_dec[ind,ra/dec]
    return xyzpoint, ra_dec

def generateHistHEL(hEclipLon,PPoutpath='./'):
    """ Generates a Heliocentric Ecliptic Longitude Histogram
    Returns:
        numVsLonInterp2 - (interpolant) - this is the interpolant of the histogram of stars 
            located along the heliocentric ecliptic longitude
    """
    plt.figure(num=2000)
    prettifyPlot()
    h, edges = np.histogram(hEclipLon)
    xdiff = np.diff(edges)
    xcents = xdiff/2.+edges[:-1]
    numVsLonInterp = interp1d(xcents, h, kind='cubic',fill_value='extrapolate')
    ystart = numVsLonInterp(np.pi)
    yend = numVsLonInterp(-np.pi)
    yse = (ystart+yend)/2.
    h = np.append(h,yse)
    xcents = np.append(xcents,np.pi)
    h = np.insert(h, 0, yse, axis=0)
    xcents = np.insert(xcents, 0, -np.pi, axis=0)
    numVsLonInterp2 = CubicSpline(xcents,h,bc_type='periodic')
    tp = np.linspace(-np.pi,np.pi,num=100)
    plt.hist(hEclipLon)
    plt.plot(tp,numVsLonInterp2(tp))
    plt.xlim(-np.pi,np.pi)
    plt.xlabel('Heliocentric Ecliptic Longitude of Targets (rad)',weight='bold')
    plt.title('Histogram of Planned to Observe Targets',weight='bold')

    #Save Plots
    # Save to a File
    date = unicode(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date

    fname = 'HistogramPlannedTargetsToObserve_' + folder.split('/')[-1] + '_' + date
    plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
    plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='png', dpi=500)
    #DELETEplt.savefig(PPoutpath + fname + '.png')
    plt.show(block=False)

    targUnderSpline = numVsLonInterp2.integrate(-np.pi,np.pi)/xdiff#Integral of spline, tells how many targets are under spline
    sumh = sum(h[1:-1])#*xdiff[0]
    return numVsLonInterp2, targUnderSpline, sumh, xdiff, edges

def generatePlannedObsTimeHistHEL(edges,t_dets,comp,hEclipLon,PPoutpath='./'):
    edges[0] = -np.pi#Force edges to -pi and pi
    edges[-1] = np.pi
    t_bins = list()
    t_dets_tmp = t_dets[comp > 0.]
    for i in np.arange(len(edges)-1):
        t_bins.append(sum(t_dets_tmp[np.where((edges[i] <= hEclipLon)*(hEclipLon <= edges[i+1]))[0]]).value)
    #consistency check
    sum(t_dets)
    sum(np.asarray(t_bins))

    #Plot t_bins in Histogram
    left_edges = edges[:-1]
    right_edges = edges[1:]
    centers = (left_edges+right_edges)/2.
    t_bins = np.asarray(t_bins)
    widths = np.diff(edges)
    fig = plt.figure(num=2002,figsize=(10,3))
    prettifyPlot()
    plt.bar(centers,t_bins,width=widths,color='black')
    plt.xlabel('Heliocentric Ecliptic Longitude of Targets (rad)',weight='bold')
    plt.ylabel('Sum Integration Time (days)',weight='bold')
    plt.xlim([-np.pi,np.pi])
    plt.title('Histogram of Planned Time to Observe Targets',weight='bold')
    fig.tight_layout()

    #Save Plots
    # Save to a File
    date = unicode(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date

    fname = 'HistogramPlannedTargetTimeToObserve_' + folder.split('/')[-1] + '_' + date
    plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
    plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='png', dpi=500)
    plt.show(block=False)

def line2linev2(p0,v0,p1,v1):
    """ Find the closest points between two arbitrary lines, and the distance between them
    Args:
        p0 (numpy array) - a 3x1 numpy array of a point on line 0 
        v0 (numpy array) - a 3x1 numpy array of the line 0 vector
        p1 (numpy array) - a 3x1 numpy array of a point on line 1 
        v1 (numpy array) - a 3x1 numpy array of the line 1 vector
    Return:
        normdP (float) - the linear distance from q0 to q1
        dP (numpy array) - the vector from q0 to q1
        q0 (numpy array) - the point in 3D space of q0, the closest point on line 0
        q1 (numpy array) - the point in 3D space of q1, the closest point on line 1
        t0 (float) - from p0-t0*v0=q0 the amount of v0 to get from p0 to q0 
        t1 (float) - from p1-t1*v1=q1 the amount of v1 to get from p1 to q1
    """
    epsilon = 0.00000001
    a = np.dot(v0,v0)
    b = np.dot(v0,v1)
    c = np.dot(v1,v1)
    d = np.dot(v0,p1-p0)
    e = np.dot(v1,p1-p0)
    D = a*c - b*b#had 
    if D < epsilon:
        t0 = 0.0
        t1 = d/b if b>c else e/c
    else:
        t0 = (b*e - c*d) / D
        t1 = (a*e - b*d) / D
    dP = p1-p0 + t0*v0 - t1*v1#p1-p0 + (sc * v0) - (tc * v1)
    return np.linalg.norm(dP), dP, p0-t0*v0, p1-t1*v1, t0, t1

def sphericalAngles(ah,bh,ch,R=1.):
    """ Calculates Spherical Angles Between 3 points on the surface of a sphere
    http://mathworld.wolfram.com/SphericalTrigonometry.html
    Args:
        R (float) - Radius of the sphere
        ah (numpy array) - unit vector from O to A 
        bh (numpy array) - unit vector from O to B
        ch (numpy array) - unit vector from O to C
    Returns:
        A (float) - angle of arc A in radians
        B (float) - angle of arc B in radians
        C (float) - angle of arc C in radians
    """
    ah = ah/np.linalg.norm(ah)
    bh = bh/np.linalg.norm(bh)
    ch = ch/np.linalg.norm(ch)

    ap = np.arccos(np.dot(bh,ch)/R**2.)# angular lengths of sides (angle formed by BOC)
    bp = np.arccos(np.dot(ah,ch)/R**2.)
    cp = np.arccos(np.dot(ah,bh)/R**2.)
    assert ap >=0, 'angle must be positive'
    assert bp >=0, 'angle must be positive'
    assert cp >=0, 'angle must be positive'

    a = R*ap# arc length of sides
    b = R*bp
    c = R*cp
    A = np.arccos((np.cos(a)-np.cos(b)*np.cos(c))/(np.sin(b)*np.sin(c)))
    #A = np.arcsin(np.abs(np.linalg.norm(np.cross(np.cross(ah,bh),np.cross(ah,ch))))/\
    #                (np.abs(np.linalg.norm(np.cross(ah,bh)))*np.abs(np.linalg.norm(np.cross(ah,ch))))) # cosines rule for sides Smart 1960
    B = np.arcsin(np.sin(A)/np.sin(a)*np.sin(b)) # spherical triangle analagous law of sines
    C = np.arcsin(np.sin(A)/np.sin(a)*np.sin(c))
    # if A+B+C < np.pi:
    #     print saltyburrito
    return A, B, C

def sphericalArea(A,B,C,a,b,c,r=1.):
    """ Calculates the area on the sphere subtended by 3 arcs on its surface
    http://mathworld.wolfram.com/SphericalTriangle.html
    Args:
        r (float) - radius of the sphere
        A (float) - angle of arc A in radians
        B (float) - angle of arc B in radians
        C (float) - angle of arc C in radians
        a (numpy array) - vector from O to A 
        b (numpy array) - vector from O to B
        c (numpy array) - vector from O to C
    Returns:
        area (float) - area subtended in units^2
    """
    #assert A+B+C >= np.pi, 'The angles input are less than that required for a spherical triangle'
    areaDelta = (r**2.)*((A+B+C)-np.pi) # the delta in surface area caused by the angles of the edges on the surface
    v0 = b-a
    v1 = c-a
    areaNominal = 0.5*np.abs(np.linalg.norm(np.cross(v0,v1)))
    return areaNominal# + areaDelta

def latlonToxyz(lat,lon):
    """
    Args:
        lat (radians) - should be from -np.pi/2 to np.pi/2
        lon (radians) - should be from 0 to 2*np.pi
    Return:
        xyz (numpy array) - x,y,z points on unit sphere
    """
    x = 1.*np.cos(lat)*np.cos(lon)
    y = 1.*np.cos(lat)*np.sin(lon)
    z = 1.*np.sin(lat)
    return np.asarray([x,y,z])

def xyzTolonlat(pt):
    """
    Args:
        pt (numpy array) - numpy array of xyz point
    Returns:
        lat (float) - latitude in radians from -np.pi/2. to np.pi/2.
        lon (float) - longitude in radians from -np.pi to np.pi
    """
    pt = pt/np.linalg.norm(pt) #normalize
    lat = np.arcsin(pt[2])
    lon = np.arctan2(pt[1],pt[0])
    return lon,lat

def calculateClosestPoints(out1kv):
    """ Calculates the distance between each point and each other point, 
    finds inds of closest 7 points (including itself), 
    finds separation distance between point and closest point
    Args:
        out1kv (2d numpy array nx3) - 2d numpy array of floats where dim 1 is the number of points
            and dim 2 are the xyz components
    Returns:
        d_diff_pts_array (numpy array) - 2d numpy array of floats where dim 1 is the number of points
            and dim 2 are the distances between that point and all other points
        inds_of_closest (list of numpy arrays) - list of numpy arrays with length number of points
            where each array are integer indices of 7 closest points in out1kv
        diff_closest (list of numpy arrays) - list of numpy arrays with length number of points
            where each array are linear distances between point and 7 closest points
    """
    d_diff_pts_array = list() # for each point, distances from point to all other points
    inds_of_closest = list() # for each point, indicie of closest point
    diff_closest = list()
    for i in np.arange(len(out1kv)):
        xyzpoint = out1kv[i] # extract a single xyz point on sphere
        diff_pts = out1kv - xyzpoint # calculate angular difference between point spacing
        d_diff_pts = np.abs(np.linalg.norm(diff_pts,axis=1)) # calculate linear distance between points
        d_diff_pts_array.append(d_diff_pts)
        inds_of_closest.append(d_diff_pts_array[i].argsort()[:7])
        diff_closest.append(d_diff_pts_array[i][inds_of_closest[i]])
    d_diff_pts_array = np.asarray(d_diff_pts_array)
    return d_diff_pts_array, inds_of_closest, diff_closest

def plotClosestPoints(inds_of_closest, out1kv):
    """ Plots a unit sphere with all lines connecting points
    """
    plt.close(50067)
    fig = plt.figure(num=50067)
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Plot of all point-to-point connections on sphere')
    for i in np.arange(len(out1kv)):
        xyzpoint = out1kv[i] # extract a single xyz point on sphere
        plotted = list() #keeps track of index-to-index lines plotted
        for j in np.delete(inds_of_closest[i],0):
            if [i,j] in plotted or [j,i] in plotted:
                continue
            ax.plot([xyzpoint[0],out1kv[j,0]],[xyzpoint[1],out1kv[j,1]],[xyzpoint[2],out1kv[j,2]],color='red',zorder=0)
            plotted.append([i,j])
    ax.set_xlabel('X',weight='bold')
    ax.set_ylabel('Y',weight='bold')
    ax.set_zlabel('Z',weight='bold')
    plt.show(block=False)
    return fig, ax

def removeCoplanarConnectingLines(inds_of_closest, out1kv):
    """
    Problem: At least 4 points on the surface appear to be coplanar and ~equidistant causing the 6 closest connecting policy to fail
    #Solution For all points that share two connecting points and are NOT directly connected, IF the two points they share are connected, 
    delete the connection between the two shared points
    Args:
        inds_of_closest (list of numpy arrays) - list of numpy arrays with length #points on sphere where each numpy array
            are the 7 closest connecting points that HAVE intersecting lines
        out1kv (2d numpy array nx3) - 2d numpy array of floats where dim 1 is the number of points
            and dim 2 are the xyz components
    returns:
        inds_of_closest (list of numpy arrays) - list of numpy arrays with length #points on sphere where each numpy array
            are the 7 closest connecting points that DO NOT HAVE intersecting lines
    """
    fig, ax = plotClosestPoints(inds_of_closest, out1kv)
    ind_ind_observed = list()
    connectionsToDelete = list() # list of sets containing links to delete
    for ind0_1 in np.arange(len(inds_of_closest)):#iterate over all points
        for ind0_2 in inds_of_closest[ind0_1]:#np.arange(len(inds_of_closest[ind0_1])):#also iterate over all points
            wasBreak = False
            if ind0_1 == ind0_2: # Skip if inds are the same
                continue

            ms = set([ind0_1,ind0_2]) # a small set containing the current two inds
            if any([True if ms == iio else False for iio in ind_ind_observed]): #Determine if we have observed the complement i.e. if 1-3 observed, don't observe 3-1
                continue
            else:
                ind_ind_observed.append(ms) # we are looking at a new ind-ind pairing
            del ms

            # Skip if ind0_1 and ind0_2 are not connected
            if  not ind0_1 in set(inds_of_closest[ind0_2]) or \
                not ind0_2 in set(inds_of_closest[ind0_1]):
                continue

            if len(set(inds_of_closest[ind0_1]).intersection(set(inds_of_closest[ind0_2]))) >= 3: # if ind0_1 in inds_of_closest[ind0_2]:
                #Requirement where the inds we care about are directly connected
                #and least two of the other indices are in common
                intSet = set(inds_of_closest[ind0_1]).intersection(set(inds_of_closest[ind0_2]))
                intSet.remove(ind0_1)#remove inds that are directly connected
                intSet.remove(ind0_2)#remove inds that are directly connected

                # Only Care if ind3 in intSet 
                for ind3 in intSet:
                    #Already guaranteed ind0_1 and ind0_2
                    if len(set(inds_of_closest[ind3]).intersection(intSet)) > 1:
                        #all sets contain at least 1 because the set contains itself
                        #IF the len >1, this point is connected to ind1, ind2, and a 3rd point connected to ind1 and ind2
                        # Note I removed ind0_1 and ind0_2 from intSet causing us to use 1, would be > 3 otherwise

                        #We must take ind3, ind0_1, and ind0_2 and find 4th point
                        all4 = set(inds_of_closest[ind3]).intersection(set(inds_of_closest[ind0_1])).intersection(set(inds_of_closest[ind0_2])) #set of all 4 connected points
                        lineList = list(itertools.permutations(itertools.combinations(all4,r=2),r=2)) # generates a list of all vector comparisons to do

                        #Iterate over point0-point1 point2-point3 lines
                        wasBreak = False
                        skipList = list() # keeps track of line comparisons to skip in lineList
                        for tmpInd in np.arange(len(lineList)):
                            if tmpInd in skipList: # if line comparison is in skipList
                                continue
                            tmpInd0 = lineList[tmpInd][0][0] # Extract line inds
                            tmpInd1 = lineList[tmpInd][0][1]
                            tmpInd2 = lineList[tmpInd][1][0]
                            tmpInd3 = lineList[tmpInd][1][1]
                            p0 = out1kv[tmpInd0]
                            v0 = out1kv[tmpInd1] - p0
                            p1 = out1kv[tmpInd2]
                            v1 = out1kv[tmpInd3] - p1
                            out = line2linev2(p0,v0,p1,v1)
                            t0 = out[4]
                            t1 = out[5]
                            q0 = out[2]
                            q1 = out[3]
                            dP = out[1]
                            v0Hat = v0/np.linalg.norm(v0) # Var for Parallel Check
                            v1Hat = v1/np.linalg.norm(v1) # Var for Parallel Check

                            #All of these should intersect
                            pts = out1kv[list(all4)] #Each of the points
                            thresh = 1e-3 # intersection criteria choosen for points
                            if min(np.abs(np.linalg.norm(out1kv[list(all4)] - q0,axis=1))) < thresh: #If the lines intersect at a point
                                continue
                            elif np.abs(np.dot(v0Hat,v1Hat)) > 0.8: # The lines are approximately parallel
                                continue
                            elif not tmpInd1 in inds_of_closest[tmpInd0] or not tmpInd0 in inds_of_closest[tmpInd1]: # When tmpInd0 not in inds_of_closest[tmpInd0]... This patches that...
                                continue
                            elif np.abs(np.linalg.norm(dP)) < 0.05: #Threshold I define, the intersection is close
                                #print 'intersection Is close!' # Was purely for debugging purposes
                                #we just found the two lines that intersect, but not at one of the points
                                ms2 = set([tmpInd0,tmpInd1])
                                if not ms2 in connectionsToDelete:
                                    connectionsToDelete.append(ms2)
                                lltiTF = [True if (tmpInd0, tmpInd1) in lineList[llti] or \
                                         (tmpInd1, tmpInd0) in lineList[llti] else False for llti in np.arange(len(lineList))]
                                         #Create array indicating where invalid lines exist
                                for item in np.arange(len(lltiTF))[lltiTF]:
                                    skipList.append(item)

                                wasBreak = True
                                # ax.scatter(q0[0],q0[1],q0[2],color='purple',marker='+') # Used to show where intersections where
                                # ax.scatter(q1[0],q1[1],q1[2],color='green',marker='+')
                                ax.plot([p0[0],p0[0]-t0*v0[0]],[p0[1],p0[1]-t0*v0[1]],[p0[2],p0[2]-t0*v0[2]],color='red')
                                ax.plot([p1[0],p1[0]-t1*v1[0]],[p1[1],p1[1]-t1*v1[1]],[p1[2],p1[2]-t1*v1[2]],color='blue')
                                ax.scatter(p0[0],p0[1],p0[2],color='black')#starting points
                                ax.scatter(p1[0],p1[1],p1[2],color='black')#starting points
                                ax.plot([q0[0],q1[0]],[q0[1],q1[1]],[q0[2],q1[2]],color='purple')
                                ax.scatter(q0[0],q0[1],q0[2],color='purple')#ending points
                                ax.scatter(q1[0],q1[1],q1[2],color='purple')#ending points
                                break
                            else: #they do not intersect! duh!
                                continue

                        # ax.scatter(out1kv[ind0_1][0],out1kv[ind0_1][1],out1kv[ind0_1][2],color='blue',zorder=2)
                        # ax.scatter(out1kv[ind0_2][0],out1kv[ind0_2][1],out1kv[ind0_2][2],color='blue',zorder=2)
                        # ax.scatter(out1kv[ind3][0],out1kv[ind3][1],out1kv[ind3][2],color='blue',zorder=2)
                        show(block=False)
                    
                    if wasBreak == True: # we do not want to continue in this loop
                        break
            else: #There are insufficient intersections between points to merit line intersection analysis
                continue
            if  wasBreak == True: # we do not want to continue in this loop
                break
    for ind in np.arange(len(connectionsToDelete)):
        tmpInd0 = list(connectionsToDelete[ind])[0]
        tmpInd1 = list(connectionsToDelete[ind])[1]
        index = np.argwhere(inds_of_closest[tmpInd0]==tmpInd1)[0][0]
        inds_of_closest[tmpInd0] = np.delete(inds_of_closest[tmpInd0],index) #we must remove tmpInd1 from inds_of_closest[tmpInd0]
        index = np.argwhere(inds_of_closest[tmpInd1]==tmpInd0)[0][0]
        inds_of_closest[tmpInd1] = np.delete(inds_of_closest[tmpInd1],index) #we must remove tmpInd1 from inds_of_closest[tmpInd0]
    return inds_of_closest


def calculateTriangles(inds_of_closest,out1kv):
    """Calculate Corners of each triangle, area of each triangle, and center of each triangle
    """
    #### Calculate Centroid of Each Area #################################
    #Calculate averate of the 3 inds
    triangleCornerIndList = list() # note these are inds of out1kv
    triangleAreaList = list()
    triangleCenterList = list()
    for ind0_1 in np.arange(len(inds_of_closest)):#iterate over all points
        for ind0_2 in inds_of_closest[ind0_1]:#also iterate over all points
            if ind0_1 == ind0_2: # Skip if inds are the same
                continue

            # # Skip if ind0_1 and ind0_2 are not connected
            if  not ind0_1 in set(inds_of_closest[ind0_2]) or \
                not ind0_2 in set(inds_of_closest[ind0_1]):
                continue

            #Get intersection between ind0_1 and ind0_2
            intSet = set(inds_of_closest[ind0_1]).intersection(set(inds_of_closest[ind0_2]))
            intSet.remove(ind0_1)#remove inds that are directly connected
            intSet.remove(ind0_2)#remove inds that are directly connected
            #pick ind0_1, pick ind0_2, iterate over the other two
            remainingInds = list(intSet)
            for ind in remainingInds:
                cornerIndSet = set([ind0_1, ind0_2, ind])
                if not cornerIndSet in triangleCornerIndList:
                    triangleCornerIndList.append(cornerIndSet)
                    centroid =  1./3.*out1kv[list(cornerIndSet)[0]] + \
                                1./3.*out1kv[list(cornerIndSet)[1]] + \
                                1./3.*out1kv[list(cornerIndSet)[2]]
                    triangleCenterList.append(centroid)
                    A, B, C = sphericalAngles(out1kv[list(cornerIndSet)[0]],out1kv[list(cornerIndSet)[1]],out1kv[list(cornerIndSet)[2]])
                    area = sphericalArea(A,B,C,out1kv[list(cornerIndSet)[0]],out1kv[list(cornerIndSet)[1]],out1kv[list(cornerIndSet)[2]])
                    #DELETE area = 0.5*np.abs(np.linalg.norm(np.cross(out1kv[list(cornerIndSet)[1]]-out1kv[list(cornerIndSet)[0]],\
                    #DELETE                                            out1kv[list(cornerIndSet)[2]]-out1kv[list(cornerIndSet)[0]])))
                    triangleAreaList.append(area)
                else:
                    continue
    sumTriangleArea = sum(np.asarray(triangleAreaList))
    #print sumTriangleArea
    triangleAreaList = list(np.asarray(triangleAreaList)*4.*np.pi/sumTriangleArea)
    #print sum(np.asarray(triangleAreaList))
    #print len(triangleCenterList)

    assert len([True for tri in triangleCornerIndList if len(tri) < 3]) < 1, 'At least One Triangle Corner List Item has fewer than 3 inds'
    ######################################################################
    return sumTriangleArea, triangleCornerIndList, triangleAreaList, triangleCenterList

def createTriangleCornerSets(sInds,comp,hEclipLat,hEclipLon,out1kv,triangleCornerIndList):
    """Create List of Corner Sets
    Args:
        sInds
        comp
        hEclipLat
        hEclipLon
        out1kv
        triangleCornerIndList
    Return:
        starAssignedTriangleCorners (list) - list of "corner sets" each star is assigned to
    """
    tmpsInds = sInds[comp>0.]
    starxyz = np.zeros([len(tmpsInds),3])
    starAssignedTriangleCorners = list() # this is the list of "corner sets" each star is assigned to
    for ind in np.arange(len(sInds[comp>0.])): # iterate over all stars
        starxyz[ind] = latlonToxyz(hEclipLat[ind],hEclipLon[ind]) #get star in xyz points

        # find closest triangle corner
        diff_star_corner = out1kv - starxyz[ind] #difference between corner and star location
        indOfMin = np.argmin(np.abs(np.linalg.norm(diff_star_corner,axis=1))) #grab minimum distance indicie
        #print indOfMin

        # Calculate r and project r onto plane defined by closest corner
        r = starxyz[ind]*(1. - np.dot(starxyz[ind],out1kv[indOfMin])) # vector from triangle corner to starxyz point
        projected_r = r - out1kv[indOfMin]*np.dot(out1kv[indOfMin]/np.linalg.norm(out1kv[indOfMin]),r/np.linalg.norm(r)) # project r onto plane defined by corner
        projected_r = projected_r/np.linalg.norm(projected_r) # normalize that projected vector

        # Find relevant triangle corners
        relevantTriangles = deepcopy([tri for tri in triangleCornerIndList if indOfMin in tri])
        assert len([True for tri in relevantTriangles if len(tri) < 3]) == 0, 'All sets in relevantTriangles must have length 3'
        #print 'rT: ' + str(relevantTriangles)
        tmp = set([])
        relevantCorners = set.union(*relevantTriangles)#[tmp tri for tri in relevantTriangles] # this is a set of all corners
        relevantCorners.remove(indOfMin) #remove the center corner
        relevantCornersArr = np.asarray(list(relevantCorners)) #turn into array

        # Calculate projected center-to-corner vectors
        r0i = list() # list of vectors from center to different triangle corners
        projected_r0i = list()
        for corrInd in np.arange(len(relevantCornersArr)):
            r0i.append(out1kv[relevantCornersArr[corrInd]]-out1kv[indOfMin]) # lines from center to corners of each triangle containing center
            projected_r0i.append(r0i[corrInd]/np.linalg.norm(r0i[corrInd]) - starxyz[ind]/np.linalg.norm(starxyz[ind])*np.dot(starxyz[ind]/np.linalg.norm(starxyz[ind]),r0i[corrInd]/np.linalg.norm(r0i[corrInd]))) #projects each r0i onto plane of closest star
            projected_r0i[corrInd] = projected_r0i[corrInd]/np.linalg.norm(projected_r0i[corrInd]) # normalizes this into a unit vector
        
        # Get vector from triangle edges to star
        r0iToPTDist = list() # contains vectors from triangle edges to star Point *all in projected plane
        for corrInd in np.arange(len(relevantCornersArr)):
            r0iToPTDist.append(projected_r - projected_r0i[corrInd]*np.dot(projected_r,projected_r0i[corrInd])) # vect from line from center to corner to star
        r0iToPTDist = np.asarray(r0iToPTDist)
        indClosestEdge = np.argmin(np.abs(np.linalg.norm(r0iToPTDist,axis=1))) #finds the distance from the star to the closest edge

        # Find Corners connected to indClosestEdge
        edgePT = relevantCornersArr[indClosestEdge]
        set(relevantCornersArr).intersection(set())
        relevantTriangles2 = deepcopy([tri for tri in relevantTriangles if len(tri.intersection(set([edgePT]))) > 0]) # sort out triangles with center and edgePT
        assert len([True for tri in relevantTriangles2 if len(tri) < 3]) == 0, 'All sets in relevantTriangles must have length 3'
        #print 'rT2: ' + str(relevantTriangles2)
        relevantTriangles2[0].remove(edgePT)
        relevantTriangles2[1].remove(edgePT)
        relevantTriangles2[0].remove(indOfMin)
        relevantTriangles2[1].remove(indOfMin)

        # Want vectors from edgePT to both items in relvantTriangles2
        cornerInd0 = np.where(relevantCornersArr==list(relevantTriangles2[0])[0])[0][0] #projected_r0i are tied to relevantCornersArr, need that index
        cornerInd1 = np.where(relevantCornersArr==list(relevantTriangles2[1])[0])[0][0]
        oc0 = projected_r0i[cornerInd0] - projected_r0i[indClosestEdge]
        oc1 = projected_r0i[cornerInd1] - projected_r0i[indClosestEdge]

        # 3rd Corner is whichever vector dot perpendicular is positive (should only be 1)
        if np.dot(oc0,r0iToPTDist[indClosestEdge]) > 0:
            starAssignedTriangleCorners.append(set([indOfMin, edgePT, relevantCornersArr[cornerInd0]]))
            #return
        elif np.dot(oc1,r0iToPTDist[indClosestEdge]) > 0:
            starAssignedTriangleCorners.append(set([indOfMin, edgePT, relevantCornersArr[cornerInd1]]))
            #return
        else:
            print saltyburrito # there was some kind of error I didn't  anticipate
    return starAssignedTriangleCorners
    ######################################################################

def createtDict(triangleCornerIndList,triangleAreaList,triangleCenterList,out1kv):
    """Creates tDict, the triangle dictionary of triangle knowledge
    Args:
        triangleCornerIndList () - 
        triangleAreaList () - 
        triangleCenterList () - 
        out1kv () - 
    Returns:
        tDict (dict) - triangle dictionary of triangle knowledge. Indexed first by str(sort(list(cornerInds)))
            then contains ['count','triangleCornerInds','triangleArea','triangleCenter',
            'triangleCornerPointsXYZ','triangleCornerPointsXYZlatlon']
    """
    tDict = {}
    for ind in np.arange(len(triangleCornerIndList)):
        tset = triangleCornerIndList[ind]
        ta = triangleAreaList[ind]
        tc = triangleCenterList[ind]
        tcptsxyz = [out1kv[ind2] for ind2 in sort(list(tset))]
        tcptslatlon = np.asarray([xyzTolonlat(tcptsxyz[ind2]) for ind2 in np.arange(len(tcptsxyz))])

        #Ensure lon aren't on other sides of skymap MOVE TRIANGLES AROUND
        if sum(np.sign(tcptslatlon[:,0]) > 0) == 3 or sum(np.sign(tcptslatlon[:,0]) < 0) == 3:
            pass
        elif sum(np.sign(tcptslatlon[:,0]) > 0) == 1\
            and (max(tcptslatlon[:,0]) - min(tcptslatlon[:,0])) > np.pi:
            # One 1 of the values positive and the others are negative
            # AND the distance between max and min is > pi (this avoids problems with triangles intersecting 0 lon line)

            #then make the positive number into a really negative one
            tcptslatlon[np.argmax(tcptslatlon[:,0]),0] = -np.pi - (np.pi - tcptslatlon[np.argmax(tcptslatlon[:,0]),0])
            tcptslatlon2 = deepcopy(tcptslatlon)
            tcptslatlon2[:,0] = tcptslatlon2[:,0]+2.*np.pi #trying this
        elif sum(np.sign(tcptslatlon[:,0]) < 0) == 1\
            and (max(tcptslatlon[:,0]) - min(tcptslatlon[:,0])) > np.pi:
            # One 1 of the values is negative and the others are positive
            # AND the distance between max and min is > pi (this avoids problems with triangles intersecting 0 lon line)
            #then make the negative number into a really positive one
            tcptslatlon[np.argmin(tcptslatlon[:,0]),0] = np.pi + (np.pi + tcptslatlon[np.argmin(tcptslatlon[:,0]),0])
            tcptslatlon2 = deepcopy(tcptslatlon)
            tcptslatlon2[:,0] = tcptslatlon2[:,0]-2.*np.pi #trying this
        else:
            pass

        try:
            tDict[str(sort(list(tset)))] = {'count':0,\
                'sIndsWithin':[],\
                'triangleComp':0,\
                'triangleIntTime':0,\
                'triangleMaxComp':0,\
                'triangleCornerInds':tset,\
                'triangleArea':ta,\
                'triangleCenter':tc,\
                'triangleCornerPointsXYZ':tcptsxyz,\
                'triangleCornerPointsXYZlatlon':tcptslatlon}
        except:
            pass

        try:
            tDict[str(sort(list(tset)))]['triangleCornerPointsXYZlatlon2'] = tcptslatlon2
            del tcptslatlon2
        except:
            pass
    return tDict
    #########################################################################

def distributeStarsIntoBins(tDict,starAssignedTriangleCorners,sInds):
    """Count number of each type of "corner set"/triangle. Effectively updates count field in tDict
    Args:
        tDict
        starAssignedTriangleCorners (list of sets) - a list, with length len(sInds), 
            containing sets of the indices of the corners of the triangle to which the star belongs
        sInds (list) - a list of indices of the stars
    Returns:
        tDict
    """
    #Count Stars In Each Triangle Bin
    for tset in starAssignedTriangleCorners:
        try:
            tDict[str(sort(list(tset)))]['count'] += 1
        except:
            pass#?
            #tDict[str(sort(list(tset)))] = {'count':1}
    countsForColoring = list() # this is a list of the number of stars in each bin
    for key in tDict.keys():
        countsForColoring.append(tDict[key])
    
    #Append sInds In Each Triangle Bin
    for i in np.arange(len(starAssignedTriangleCorners)):
        tset = starAssignedTriangleCorners[i]
        try:
            tDict[str(sort(list(tset)))]['sIndsWithin'].append(sInds[i])
        except:
            pass#?
            #tDict[str(sort(list(tset)))] = {'sIndsWithin':[sInds[i]]}

    return tDict
    ###########################################

def prettifyPlot():
    """ A method to change default plot parameters and make them prettier (bold axes and fold etc...)
    Args:
    Returns:
    """
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')

def plotSkyScheduledObservationCountDistribution(tDict,fignum=96993, PPoutpath='./'):
    """ Plots Distribution of Stars Scheduled to be Observed on Sky
    Args:
        tDict () - 
        fignum (integer) - 
    Returns:
        fig
    """
    #Each Triangle on a 2D plot with Hammer Projection
    plt.close(fignum)
    fig = plt.figure(num=fignum, figsize=(7,2.5))
    gs = GridSpec(1,1, width_ratios=[4,], height_ratios=[1])
    gs.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 
    ax = plt.subplot(gs[0],projection='mollweide')#2D histogram of planet pop
    #ax = fig.add_subplot(111, projection="mollweide")#"hammer")
    prettifyPlot()
    #grid(axis='both',which='major') # Dmitry says this makes it look too crowded
    ymin = min([min(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,1]) for ind in np.arange(len(tDict.keys()))])
    ymax = max([max(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,1]) for ind in np.arange(len(tDict.keys()))])
    xmin = min([min(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,0]) for ind in np.arange(len(tDict.keys()))])
    xmax = max([max(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,0]) for ind in np.arange(len(tDict.keys()))])
    #ax.set_xlim(left=xmin,right=xmax) #used for error checking on non-projected plot
    #ax.set_ylim(bottom=ymin,top=ymax)
    cmap = cm.viridis
    norm = mpl.colors.Normalize(vmin=0,vmax=max([tDict[key]['count']/tDict[key]['triangleArea'] for key in tDict.keys()]))
    #### Plot Each Surface with specific color scaled based on max(countsForColoring)
    for ind in np.arange(len(tDict.keys())):
        #Make Exceptions and Spoof North Pole Plotting ()
        if np.any(np.asarray(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZ'])[:,2] > 0.95): #determine if triangle at pole
            try:
                tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon2'])
                #1 Find index where lat is maximum
                dindex = np.argmax(tmp[:,1])
                #2 Remove From list of points
                tmp2 = np.delete(tmp,dindex,axis=0)
                #3 Reassign Maximum Values to that Value
                tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],np.pi/2.]]),axis=0)
                tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],np.pi/2.]]),axis=0)
                t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['count']/tDict[tDict.keys()[ind]]['triangleArea'])))
                ax.add_patch(t3)
            except:
                pass

            tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'])
            #1 Find index where lat is maximum
            dindex = np.argmax(tmp[:,1])
            #2 Remove From list of points
            tmp2 = np.delete(tmp,dindex,axis=0)
            #3 Reassign Maximum Values to that Value
            tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],np.pi/2.]]),axis=0)
            tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],np.pi/2.]]),axis=0)
            t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['count']/tDict[tDict.keys()[ind]]['triangleArea'])))
            ax.add_patch(t3)
            continue

        #Make Exceptions and Spoof South Pole Plotting
        if np.any(np.asarray(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZ'])[:,2] < -0.95): #determine if triangle at pole
            try:
                tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon2'])
                #1 Find index where lat is maximum
                dindex = np.argmin(tmp[:,1])
                #2 Remove From list of points
                tmp2 = np.delete(tmp,dindex,axis=0)
                #3 Reassign Maximum Values to that Value
                tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],-np.pi/2.]]),axis=0)
                tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],-np.pi/2.]]),axis=0)
                t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['count']/tDict[tDict.keys()[ind]]['triangleArea'])))
                ax.add_patch(t3)
            except:
                pass

            tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'])
            #1 Find index where lat is maximum
            dindex = np.argmin(tmp[:,1])
            #2 Remove From list of points
            tmp2 = np.delete(tmp,dindex,axis=0)
            #3 Reassign Maximum Values to that Value
            tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],-np.pi/2.]]),axis=0)
            tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],-np.pi/2.]]),axis=0)
            t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['count']/tDict[tDict.keys()[ind]]['triangleArea'])))
            ax.add_patch(t3)
            continue

        t1 = plt.Polygon(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'], color=cmap(norm(tDict[tDict.keys()[ind]]['count']/tDict[tDict.keys()[ind]]['triangleArea'])))
        ax.add_patch(t1)
        del t1
        #### Add mirror patch
        if 'triangleCornerPointsXYZlatlon2' in tDict[tDict.keys()[ind]].keys():
            #DELETE print 'HasKey!'
            t2 = plt.Polygon(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon2'], color=cmap(norm(tDict[tDict.keys()[ind]]['count'])/tDict[tDict.keys()[ind]]['triangleArea']))
            ax.add_patch(t2)
            del t2

    cnt = 0#tally total number of stars plotted
    for ind in np.arange(len(tDict.keys())):
        cnt += tDict[tDict.keys()[ind]]['count']

    sc = ax.scatter([-1000,-1000],[-1000,-1000],c=[norm.vmin,norm.vmax],cmap=cmap,vmin=0.,vmax=norm.vmax) #spoof colorbar
    cbar = fig.colorbar(sc) #spoof colorbar
    cbar.set_label('Star Count per Fraction Of Sky',weight='bold')
    fig.text(0.60,0.09,r'$\sum$# stars='+str(cnt))
    #Save Plots
    # Save to a File
    date = unicode(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date

    fname = 'skyObsCNTdistribution_' + folder.split('/')[-1] + '_' + date
    plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
    plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='png', dpi=500)
    plt.show(block=False)
    return fig

def plotSkyScheduledObservationCompletenessDistribution(tDict,fignum=96994, PPoutpath='./'):
    """ Plots Distribution of Star Completeness Scheduled to be Observed on Sky
    Args:
        tDict () - 
        fignum (integer) - 
    Returns:
        fig
    """
    #Each Triangle on a 2D plot with Hammer Projection
    plt.close(fignum)
    fig = plt.figure(num=fignum, figsize=(7,2.5))
    gs = GridSpec(1,1, width_ratios=[4,], height_ratios=[1])
    gs.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 
    ax = plt.subplot(gs[0],projection='mollweide')#2D histogram of planet pop
    #ax = fig.add_subplot(111, projection="mollweide")#"hammer")
    prettifyPlot()
    #grid(axis='both',which='major') # Dmitry says this makes it look too crowded
    ymin = min([min(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,1]) for ind in np.arange(len(tDict.keys()))])
    ymax = max([max(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,1]) for ind in np.arange(len(tDict.keys()))])
    xmin = min([min(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,0]) for ind in np.arange(len(tDict.keys()))])
    xmax = max([max(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,0]) for ind in np.arange(len(tDict.keys()))])
    #ax.set_xlim(left=xmin,right=xmax) #used for error checking on non-projected plot
    #ax.set_ylim(bottom=ymin,top=ymax)
    cmap = cm.viridis
    norm = mpl.colors.Normalize(vmin=0,vmax=max([tDict[key]['triangleComp']/tDict[key]['triangleArea'] for key in tDict.keys()]))
    #### Plot Each Surface with specific color scaled based on max(countsForColoring)
    for ind in np.arange(len(tDict.keys())):
        #Make Exceptions and Spoof North Pole Plotting ()
        if np.any(np.asarray(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZ'])[:,2] > 0.95): #determine if triangle at pole
            try:
                tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon2'])
                #1 Find index where lat is maximum
                dindex = np.argmax(tmp[:,1])
                #2 Remove From list of points
                tmp2 = np.delete(tmp,dindex,axis=0)
                #3 Reassign Maximum Values to that Value
                tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],np.pi/2.]]),axis=0)
                tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],np.pi/2.]]),axis=0)
                t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['triangleComp']/tDict[tDict.keys()[ind]]['triangleArea'])))
                ax.add_patch(t3)
            except:
                pass

            tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'])
            #1 Find index where lat is maximum
            dindex = np.argmax(tmp[:,1])
            #2 Remove From list of points
            tmp2 = np.delete(tmp,dindex,axis=0)
            #3 Reassign Maximum Values to that Value
            tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],np.pi/2.]]),axis=0)
            tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],np.pi/2.]]),axis=0)
            t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['triangleComp']/tDict[tDict.keys()[ind]]['triangleArea'])))
            ax.add_patch(t3)
            continue

        #Make Exceptions and Spoof South Pole Plotting
        if np.any(np.asarray(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZ'])[:,2] < -0.95): #determine if triangle at pole
            try:
                tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon2'])
                #1 Find index where lat is maximum
                dindex = np.argmin(tmp[:,1])
                #2 Remove From list of points
                tmp2 = np.delete(tmp,dindex,axis=0)
                #3 Reassign Maximum Values to that Value
                tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],-np.pi/2.]]),axis=0)
                tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],-np.pi/2.]]),axis=0)
                t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['triangleComp']/tDict[tDict.keys()[ind]]['triangleArea'])))
                ax.add_patch(t3)
            except:
                pass

            tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'])
            #1 Find index where lat is maximum
            dindex = np.argmin(tmp[:,1])
            #2 Remove From list of points
            tmp2 = np.delete(tmp,dindex,axis=0)
            #3 Reassign Maximum Values to that Value
            tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],-np.pi/2.]]),axis=0)
            tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],-np.pi/2.]]),axis=0)
            t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['triangleComp']/tDict[tDict.keys()[ind]]['triangleArea'])))
            ax.add_patch(t3)
            continue

        t1 = plt.Polygon(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'], color=cmap(norm(tDict[tDict.keys()[ind]]['triangleComp']/tDict[tDict.keys()[ind]]['triangleArea'])))
        ax.add_patch(t1)
        del t1
        #### Add mirror patch
        if 'triangleCornerPointsXYZlatlon2' in tDict[tDict.keys()[ind]].keys():
            #DELETE print 'HasKey!'
            t2 = plt.Polygon(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon2'], color=cmap(norm(tDict[tDict.keys()[ind]]['triangleComp'])/tDict[tDict.keys()[ind]]['triangleArea']))
            ax.add_patch(t2)
            del t2

    cntComp = 0#total comp of stars observed
    for ind in np.arange(len(tDict.keys())):
        cntComp += tDict[tDict.keys()[ind]]['triangleComp']

    sc = ax.scatter([-1000,-1000],[-1000,-1000],c=[norm.vmin,norm.vmax],cmap=cmap,vmin=0.,vmax=norm.vmax) #spoof colorbar
    cbar = fig.colorbar(sc) #spoof colorbar
    cbar.set_label(r'$\sum$ C per Fraction Of Sky',weight='bold')
    fig.text(0.625,0.11,r'$\sum$C='+str(round(cntComp,2)))
    #Save Plots
    # Save to a File
    date = unicode(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date

    fname = 'skyObsCompDistribution_' + folder.split('/')[-1] + '_' + date
    plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
    plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='png', dpi=500)
    plt.show(block=False)
    return fig

def plotSkyScheduledObservationIntegrationDistribution(tDict,fignum=96995, PPoutpath='./'):
    """ Plots Distribution of Stars Scheduled to be Observed on Sky
    Args:
        tDict () - 
    Returns:
        fig
    """
    #Each Triangle on a 2D plot with Hammer Projection
    plt.close(fignum)
    fig = plt.figure(num=fignum, figsize=(7,2.5))
    gs = GridSpec(1,1, width_ratios=[4,], height_ratios=[1])
    gs.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 
    ax = plt.subplot(gs[0],projection='mollweide')#2D histogram of planet pop
    #ax = fig.add_subplot(111, projection="mollweide")#"hammer")
    prettifyPlot()
    #grid(axis='both',which='major') # Dmitry says this makes it look too crowded
    ymin = min([min(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,1]) for ind in np.arange(len(tDict.keys()))])
    ymax = max([max(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,1]) for ind in np.arange(len(tDict.keys()))])
    xmin = min([min(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,0]) for ind in np.arange(len(tDict.keys()))])
    xmax = max([max(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,0]) for ind in np.arange(len(tDict.keys()))])
    #ax.set_xlim(left=xmin,right=xmax) #used for error checking on non-projected plot
    #ax.set_ylim(bottom=ymin,top=ymax)
    cmap = cm.viridis
    norm = mpl.colors.Normalize(vmin=0,vmax=max([tDict[key]['triangleIntTime']/tDict[key]['triangleArea'] for key in tDict.keys()]))
    #### Plot Each Surface with specific color scaled based on max(countsForColoring)
    for ind in np.arange(len(tDict.keys())):
        #Make Exceptions and Spoof North Pole Plotting ()
        if np.any(np.asarray(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZ'])[:,2] > 0.95): #determine if triangle at pole
            try:
                tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon2'])
                #1 Find index where lat is maximum
                dindex = np.argmax(tmp[:,1])
                #2 Remove From list of points
                tmp2 = np.delete(tmp,dindex,axis=0)
                #3 Reassign Maximum Values to that Value
                tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],np.pi/2.]]),axis=0)
                tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],np.pi/2.]]),axis=0)
                t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['triangleIntTime']/tDict[tDict.keys()[ind]]['triangleArea'])))
                ax.add_patch(t3)
            except:
                pass

            tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'])
            #1 Find index where lat is maximum
            dindex = np.argmax(tmp[:,1])
            #2 Remove From list of points
            tmp2 = np.delete(tmp,dindex,axis=0)
            #3 Reassign Maximum Values to that Value
            tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],np.pi/2.]]),axis=0)
            tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],np.pi/2.]]),axis=0)
            t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['triangleIntTime']/tDict[tDict.keys()[ind]]['triangleArea'])))
            ax.add_patch(t3)
            continue

        #Make Exceptions and Spoof South Pole Plotting
        if np.any(np.asarray(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZ'])[:,2] < -0.95): #determine if triangle at pole
            try:
                tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon2'])
                #1 Find index where lat is maximum
                dindex = np.argmin(tmp[:,1])
                #2 Remove From list of points
                tmp2 = np.delete(tmp,dindex,axis=0)
                #3 Reassign Maximum Values to that Value
                tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],-np.pi/2.]]),axis=0)
                tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],-np.pi/2.]]),axis=0)
                t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['triangleIntTime']/tDict[tDict.keys()[ind]]['triangleArea'])))
                ax.add_patch(t3)
            except:
                pass

            tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'])
            #1 Find index where lat is maximum
            dindex = np.argmin(tmp[:,1])
            #2 Remove From list of points
            tmp2 = np.delete(tmp,dindex,axis=0)
            #3 Reassign Maximum Values to that Value
            tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],-np.pi/2.]]),axis=0)
            tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],-np.pi/2.]]),axis=0)
            t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['triangleIntTime']/tDict[tDict.keys()[ind]]['triangleArea'])))
            ax.add_patch(t3)
            continue

        t1 = plt.Polygon(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'], color=cmap(norm(tDict[tDict.keys()[ind]]['triangleIntTime']/tDict[tDict.keys()[ind]]['triangleArea'])))
        ax.add_patch(t1)
        del t1
        #### Add mirror patch
        if 'triangleCornerPointsXYZlatlon2' in tDict[tDict.keys()[ind]].keys():
            #DELETE print 'HasKey!'
            t2 = plt.Polygon(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon2'], color=cmap(norm(tDict[tDict.keys()[ind]]['triangleIntTime'])/tDict[tDict.keys()[ind]]['triangleArea']))
            ax.add_patch(t2)
            del t2

    cntIntTime = 0#tally total number of stars plotted
    for ind in np.arange(len(tDict.keys())):
        cntIntTime += tDict[tDict.keys()[ind]]['triangleIntTime']

    sc = ax.scatter([-1000,-1000],[-1000,-1000],c=[norm.vmin,norm.vmax],cmap=cmap,vmin=0.,vmax=norm.vmax) #spoof colorbar
    cbar = fig.colorbar(sc) #spoof colorbar
    cbar.set_label(r'$\tau$ (d) per Fraction Of Sky',weight='bold')
    fig.text(0.62,0.11,r'$\sum \tau$='+str(np.round(cntIntTime,1)) + ' (d)')
    #Save Plots
    # Save to a File
    date = unicode(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date

    fname = 'skyObsIntTimeDistribution_' + folder.split('/')[-1] + '_' + date
    plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
    plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='png', dpi=500)
    plt.show(block=False)
    return fig

#NEED TO REDO BECAUSE THIS DOES NOT PLOT MAX COMPLETENESS YET
def plotSkyMaximumCompletenessDistribution(starDict,fignum=96996, PPoutpath='./'):
    """ Plots Distribution of Star Completeness Scheduled to be Observed on Sky
    Args:
        tDict () - 
    Returns:
        fig
    """
    #Each Triangle on a 2D plot with Hammer Projection
    plt.close(fignum)
    fig = plt.figure(num=fignum, figsize=(7,2.5))
    gs = GridSpec(1,1, width_ratios=[4,], height_ratios=[1])
    gs.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 
    ax = plt.subplot(gs[0],projection='mollweide')#2D histogram of planet pop
    #ax = fig.add_subplot(111, projection="mollweide")#"hammer")
    prettifyPlot()
    #grid(axis='both',which='major') # Dmitry says this makes it look too crowded
    ymin = min([min(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,1]) for ind in np.arange(len(tDict.keys()))])
    ymax = max([max(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,1]) for ind in np.arange(len(tDict.keys()))])
    xmin = min([min(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,0]) for ind in np.arange(len(tDict.keys()))])
    xmax = max([max(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'][:,0]) for ind in np.arange(len(tDict.keys()))])
    #ax.set_xlim(left=xmin,right=xmax) #used for error checking on non-projected plot
    #ax.set_ylim(bottom=ymin,top=ymax)
    cmap = cm.viridis
    norm = mpl.colors.Normalize(vmin=0,vmax=max([tDict[key]['triangleMaxComp']/tDict[key]['triangleArea'] for key in tDict.keys()]))
    #### Plot Each Surface with specific color scaled based on max(countsForColoring)
    for ind in np.arange(len(tDict.keys())):
        #Make Exceptions and Spoof North Pole Plotting ()
        if np.any(np.asarray(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZ'])[:,2] > 0.95): #determine if triangle at pole
            try:
                tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon2'])
                #1 Find index where lat is maximum
                dindex = np.argmax(tmp[:,1])
                #2 Remove From list of points
                tmp2 = np.delete(tmp,dindex,axis=0)
                #3 Reassign Maximum Values to that Value
                tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],np.pi/2.]]),axis=0)
                tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],np.pi/2.]]),axis=0)
                t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['triangleMaxComp']/tDict[tDict.keys()[ind]]['triangleArea'])))
                ax.add_patch(t3)
            except:
                pass

            tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'])
            #1 Find index where lat is maximum
            dindex = np.argmax(tmp[:,1])
            #2 Remove From list of points
            tmp2 = np.delete(tmp,dindex,axis=0)
            #3 Reassign Maximum Values to that Value
            tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],np.pi/2.]]),axis=0)
            tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],np.pi/2.]]),axis=0)
            t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['triangleMaxComp']/tDict[tDict.keys()[ind]]['triangleArea'])))
            ax.add_patch(t3)
            continue

        #Make Exceptions and Spoof South Pole Plotting
        if np.any(np.asarray(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZ'])[:,2] < -0.95): #determine if triangle at pole
            try:
                tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon2'])
                #1 Find index where lat is maximum
                dindex = np.argmin(tmp[:,1])
                #2 Remove From list of points
                tmp2 = np.delete(tmp,dindex,axis=0)
                #3 Reassign Maximum Values to that Value
                tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],-np.pi/2.]]),axis=0)
                tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],-np.pi/2.]]),axis=0)
                t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['triangleMaxComp']/tDict[tDict.keys()[ind]]['triangleArea'])))
                ax.add_patch(t3)
            except:
                pass

            tmp = deepcopy(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'])
            #1 Find index where lat is maximum
            dindex = np.argmin(tmp[:,1])
            #2 Remove From list of points
            tmp2 = np.delete(tmp,dindex,axis=0)
            #3 Reassign Maximum Values to that Value
            tmp3 = np.append(tmp2,np.asarray([[tmp2[1,0],-np.pi/2.]]),axis=0)
            tmp4 = np.append(tmp3,np.asarray([[tmp2[0,0],-np.pi/2.]]),axis=0)
            t3 = plt.Polygon(tmp4,color=cmap(norm(tDict[tDict.keys()[ind]]['triangleMaxComp']/tDict[tDict.keys()[ind]]['triangleArea'])))
            ax.add_patch(t3)
            continue

        t1 = plt.Polygon(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon'], color=cmap(norm(tDict[tDict.keys()[ind]]['triangleMaxComp']/tDict[tDict.keys()[ind]]['triangleArea'])))
        ax.add_patch(t1)
        del t1
        #### Add mirror patch
        if 'triangleCornerPointsXYZlatlon2' in tDict[tDict.keys()[ind]].keys():
            #DELETE print 'HasKey!'
            t2 = plt.Polygon(tDict[tDict.keys()[ind]]['triangleCornerPointsXYZlatlon2'], color=cmap(norm(tDict[tDict.keys()[ind]]['triangleMaxComp'])/tDict[tDict.keys()[ind]]['triangleArea']))
            ax.add_patch(t2)
            del t2

    cntMaxC = 0#tally total number of stars plotted
    for ind in np.arange(len(tDict.keys())):
        cntMaxC += tDict[tDict.keys()[ind]]['triangleMaxComp']

    sc = ax.scatter([-1000,-1000],[-1000,-1000],c=[norm.vmin,norm.vmax],cmap=cmap,vmin=0.,vmax=norm.vmax) #spoof colorbar
    cbar = fig.colorbar(sc) #spoof colorbar
    cbar.set_label(r'$\sum C_{max}$ per Fraction Of Sky',weight='bold')
    fig.text(0.62,0.11,r'$\sum C_{max}$ stars='+str(cntMaxC))
    #Save Plots
    # Save to a File
    date = unicode(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date

    fname = 'skyObsMaxCdistribution_' + folder.split('/')[-1] + '_' + date
    plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
    plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='png', dpi=500)
    plt.show(block=False)
    return fig

def generatePreferentiallyDistributedOB(barout, numOB, OBdur, exoplanetObsTime, numYears, loadingPreference='even'):
    """ Generates a distribution of OB start and end times which distributes observing blocks
    such that they approximately match the Planned Total Time Histogram 
    (amount of time needed in each portion of sky is met)
    Args:
        barout (dict) - 
        numOB (integer) - 
        OBdur (float) - 

        loadingPreference (string) - options are 'even','front','end' 
            'even' tries to evenly distribute OB with preference for overflow to be placed in first year
            'front' will load as many Observing blocks into the first year as possible
            'end' will load as many Observing Blocks into the last year as possible
    Returns:
        numOBassignedToBin (numpy array) - has length number of bins. Contains the number of OB to put into each bin
        OBstartTimes (numpy array) - observing block start times (automatically merges OB if they overlap)
        OBendTimes (numpy array) - observing block end times
    """
    daysInYear = 365.25 #Days in a year
    numOBperYear = np.ceil(numOB/numYears)#Number of Observing blocks I need to distribute into 1 year
    #REQUIRES OB size <= daysInYear/numBins # hmmmmm.... I could make the algorithm better than that
    centers = (barout['centers']+np.pi)*(daysInYear/(2*np.pi)) # centers of bins in units of days from start of year
    binWidths = barout['binWidths']*(daysInYear/(2*np.pi)) # widths of bins in units of days
    timeInBins = barout['timeInBins'] # amount of time to be placed into each bin
    p = timeInBins/np.sum(timeInBins) # set of "probabilities" of landing in each bin
    if daysInYear/len(centers) < OBdur: # If we cannot shove an OB into a bin
        #DO STUFF
        pass
    else: # distribute the OB into bins for each year
        numOBassignedToBin = np.floor(numOB*p) # this is nominally the number of observing blocks to assign to a given bin
        numOBleftToAssign = numOB - np.sum(np.floor(numOB*p)) # number of OB left to assign
        tic = True
        while numOBleftToAssign > 0:
            if tic == True: #alternate between max time in bin
                numOBassignedToBin[np.argsort(timeInBins - numOBassignedToBin*OBdur)[-1]] += 1 #assign next OB to one which needs it most
                numOBleftToAssign -= 1 # decrement 1 from number of OB to assign
                tic = False
                continue
            if tic == False: # and max cum time of nearby bins
                lrbins = int(np.floor(len(centers)/4.)) # number of bins left or right of current bin to consider
                nearby = list() # total amount of time available in nearby bins
                for ind in np.arange(len(centers)):#iterate over all bins
                    msumnearby = 0.
                    for ind2 in  range(ind-lrbins,ind+lrbins):
                        if ind2 > len(centers):
                            #print 'ind2 > 0'
                            #print ind2
                            tmpind2 = ind2 - int(len(centers)*np.floor((ind2+1)/float(len(centers))))#correction if bins would exceed indexing
                            #print tmpind2
                        elif ind2 < 0:
                            #print 'ind2 < 0'
                            #print ind2
                            tmpind2 = ind2 + int(len(centers)*np.floor((-ind2-1)/float(len(centers))))
                            #print tmpind2
                        else:
                            pass
                            #print ind2
                        msumnearby += timeInBins[tmpind2] - numOBassignedToBin[tmpind2]*OBdur
                    nearby.append(msumnearby)
                numOBassignedToBin[np.argmax(msumnearby)] += 1 #assign next OB to one which has the most nearby bins that need it
                numOBleftToAssign -= 1 # decrement 1 from number of OB to assign
                tic = True
                continue
    # we now have numOBassignedToBin

    #### Assign OB to years #################################################################
    numOBinYearsBins = np.zeros([int(np.ceil(numYears)), len(centers)]) # contains number of OB in each bin over each portion of year
    #1 approximately even
    numOBInBinperYearAVG = numOBassignedToBin/int(np.ceil(numYears))
    for yrInd in np.arange(int(np.ceil(numYears))): # Nominal Assignment of OB
        numOBinYearsBins[yrInd,:] = np.asarray(numOBInBinperYearAVG).astype(int)
    for binInd in np.arange(len(centers)): # distributing remaining OB
        binCount = int(np.sum(numOBinYearsBins[:,binInd]))
        remaingindBinsToDistribute = int(numOBassignedToBin[binInd] - binCount)
        for i in np.arange(remaingindBinsToDistribute):
            numOBinYearsBins[i,binInd] += 1
    assert numOBassignedToBin[binInd] == np.sum(numOBinYearsBins[:,binInd],axis=0), \
        'The number of observing blocks assigned is not equivalent to the number of observing blocks that should be assigned'
    #2 TODO front load in first year
    #3 TODO end load in final year

    #### Pick Final OBstartTimes and OBendTimes###############################################
    binStarts = centers - binWidths/2.
    binEnds = centers + binWidths/2.
    OBstartTimes = list()
    OBendTimes = list()
    for yrInd in np.arange(int(np.ceil(numYears))): # iterate over years
        for binInd in np.arange(len(centers)): # iterate over bins in years
            assert numOBinYearsBins[yrInd,binInd]*OBdur <= binEnds[binInd] - binStarts[binInd], \
                'There is not enough time in the bin to distribute the given observing blocks'
            tmpOBstartTimes = yrInd*daysInYear + np.sort(np.random.uniform(low=binStarts[binInd],high=binEnds[binInd]-OBdur,size=int(numOBinYearsBins[yrInd,binInd])))
            tmpOBendTimes = tmpOBstartTimes + OBdur # create end times
            while True:
                intersectionBool, vInd = obIntersectionViolationCheck(tmpOBstartTimes,tmpOBendTimes)
                if not intersectionBool: # there were no intersections
                    break
                #There is a violation at least 1 OB intersects #we merge OB[i] wit OB[i+1]
                i = vInd
                tmpOBendTimes[i] = tmpOBendTimes[i] + (tmpOBendTimes[i+1] - tmpOBstartTimes[i+1])
                tmpOBstartTimes = np.delete(tmpOBstartTimes,i+1)
                tmpOBendTimes = np.delete(tmpOBendTimes,i+1)
                #Check if current OB exceeds bin end
                if tmpOBendTimes[i] > binEnds[i]: #If so, shift left
                    tmpOBstartTimes[i] = binEnds[i] - (tmpOBendTimes[i] - tmpOBstartTimes[i]) # OB start time starts perfectly so the Ob ends at the end of the bin
                    tmpOBendTimes[i] = binEnds[i] #OB end time is the end of the bin

            OBstartTimes.append(tmpOBstartTimes)
            OBendTimes.append(tmpOBendTimes)
    OBstartTimes = np.concatenate(OBstartTimes)
    OBendTimes = np.concatenate(OBendTimes)
    return numOBassignedToBin, OBstartTimes, OBendTimes

def generatePlannedObsTimeHistHEL2(edges,tDict,fignum=2003,fname='HistogramPlannedTotalTimes', PPoutpath='./'):
    """ Finds bin centers, binwidths, and total time in each bin, plots time vs HEL histograms
    Args:
        edges (numpy array) - edges of the histogram to use
        tDict (dict) - dictionary of triangles, on the celestial sphere, stars in those triangles
        fignum (integer) - 
    """
    edges[0] = -np.pi#Force edges to -pi and pi
    edges[-1] = np.pi

    #### Calculate totalTime vs HEL and intTime vs HEL#################
    lon = list() # contains traingel longitudes
    lonIntTime = list() # contains intTimes in each traingle
    lonTotalTime = list() # contains totalTimes in each triangle
    starCount = list() # contains starCount
    starComps = list()
    for key in tDict.keys():
        lon.append((xyzTolonlat(tDict[key]['triangleCenter']))[0])
        lonIntTime.append(tDict[key]['triangleIntTime'])
        lonTotalTime.append(tDict[key]['triangleIntTime'] + 1.*tDict[key]['count'])
        starCount.append(1.*tDict[key]['count'])
        starComps.append(tDict[key]['triangleComp'])
    lonTotalTime = np.asarray(lonTotalTime)
    lonIntTime = np.asarray(lonIntTime)
    starCount = np.asarray(starCount)
    starComps = np.asarray(starComps)
    #####################################################################

    #### distribute totalTime and intTime into bins #####################
    t_bins = list()
    t_bins2 = list()
    scBins = list()
    scompBins = list()
    for i in np.arange(len(edges)-1):
        t_bins.append(sum(lonTotalTime[np.where((edges[i] <= lon)*(lon < edges[i+1]))[0]]))
        t_bins2.append(sum(lonIntTime[np.where((edges[i] <= lon)*(lon < edges[i+1]))[0]]))
        scBins.append(sum(starCount[np.where((edges[i] <= lon)*(lon < edges[i+1]))[0]]))
        scompBins.append(sum(starComps[np.where((edges[i] <= lon)*(lon < edges[i+1]))[0]]))
    #####################################################################

    #### intTime + target overhead
    left_edges = edges[:-1]
    right_edges = edges[1:]
    centers = (left_edges+right_edges)/2.
    t_bins = np.asarray(t_bins)
    widths = np.diff(edges)
    plt.close(fignum)
    fig = plt.figure(num=fignum,figsize=(10,2))
    plt.bar(centers,t_bins,width=widths,color='black')
    plt.xlabel('Heliocentric Ecliptic Longitude of Targets (rad)',weight='bold')
    plt.ylabel('Integration\nTime (days)',weight='bold')
    plt.xlim([-np.pi,np.pi])
    plt.title('Histogram of Planned Int Time and Overhead Time',weight='bold')
    fig.tight_layout()
    #Save Plots
    date = unicode(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
    fname = 'HistogramPlannedIntTimeandOHtime_' + folder.split('/')[-1] + '_' + date
    plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
    plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='png', dpi=500)
    

    #### Planned Int Time only
    plt.close(fignum+1)
    fig = plt.figure(num=fignum+1,figsize=(10,2))
    out = plt.bar(centers,t_bins2,width=widths,color='black')
    plt.xlabel('Heliocentric Ecliptic Longitude of Targets (rad)',weight='bold')
    plt.ylabel('Integration\nTime (days)',weight='bold')
    plt.xlim([-np.pi,np.pi])
    plt.title('Histogram of Planned IntTime',weight='bold')
    fig.tight_layout()
    #Save Plots
    date = unicode(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
    fname = 'HistogramPlannedintTime_' + folder.split('/')[-1] + '_' + date
    plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
    plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='png', dpi=500)

    #### Plot Target Count
    plt.close(fignum+2)
    fig = plt.figure(num=fignum+2,figsize=(10,2))
    out = plt.bar(centers,scBins,width=widths,color='black')
    plt.xlabel('Heliocentric Ecliptic Longitude of Targets (rad)',weight='bold')
    plt.ylabel('Targets',weight='bold')
    plt.xlim([-np.pi,np.pi])
    plt.title('Histogram of Planned Targets',weight='bold')
    fig.tight_layout()
    #Save Plots
    date = unicode(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
    fname = 'HistogramPlannedTargets_' + folder.split('/')[-1] + '_' + date
    plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
    plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='png', dpi=500)
    
    #### Plot Target Comp
    plt.close(fignum+3)
    fig = plt.figure(num=fignum+3,figsize=(10,2))
    out = plt.bar(centers,scompBins,width=widths,color='black')
    plt.xlabel('Heliocentric Ecliptic Longitude of Targets (rad)',weight='bold')
    plt.ylabel('Completeness',weight='bold')
    plt.xlim([-np.pi,np.pi])
    plt.title('Histogram of Planned Completeness',weight='bold')
    fig.tight_layout()
    #Save Plots
    date = unicode(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
    fname = 'HistogramPlannedComp_' + folder.split('/')[-1] + '_' + date
    plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
    plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='png', dpi=500)

    ####

    plt.show(block=False)

    return {'centers':centers, 'timeInBins':np.asarray(t_bins2), 'binWidths':widths}

def obIntersectionViolationCheck(OBstartTimes,OBendTimes):
    """ Simply checks if the next OB start time starts before the current ob ends
    Args:
        OBstartTimes (numpy array) - sorted from smallest to largest
        OBendTimes (numpy array) - follows OBstartTimes
    Returns:
        intersectionBool (bool) - True if there is an intersection of the OB
        violatingIndex (integer) - Integer of OBstartTimes (left violator) where violation occurs
    """
    for i in np.arange(len(OBstartTimes)-1): #from left to right
        if OBstartTimes[i+1] < OBendTimes[i]:
            return True, i
    return False, 0

def periodicDist(numOB, OBdur, maxNumDays):#, missionPortion):
    """ Creates a set of observing blocks which start at the same time every 
    Args:
        numOB (integer) - number of observing blocks to schedule
        OBdur (float) - observing block duration
        maxNumDays (float) - number of days in the whole mission
    Returns:
        OBstartTimes
        OBendTimes
    """
    daysInYear = 365.25 #Days in a year
    numYears = maxNumDays/daysInYear#Number of years

    numOBperYear = int(np.ceil(numOB/numYears))#Number of Observing blocks that can fit into 1 year

    OBstartTimes = np.asarray([])
    OBendTimes = np.asarray([])
    for i in range(int(np.ceil(numYears))):#Note it should be fine if we fully fill out the remaining year
        OBstartTimes = np.append(OBstartTimes,np.linspace(0.,365.25,num=numOBperYear, endpoint=False)+i*daysInYear)
        OBendTimes = np.append(OBendTimes,np.linspace(0.,365.25,num=numOBperYear, endpoint=False)+float(i*daysInYear+OBdur))
    return OBstartTimes, OBendTimes


prettifyPlot()
plt.close('all')

#### Inputs ##########################################################
exoplanetObsTime = 365.25#Maximum amount of observing time
maxNumYears = 6.#maximum number of years we will allow
maxNumDays = maxNumYears*365.25

OBdur = 1.
#####################################################################

def maxNumRepInTime(OBdur,Time):
    #Time in days
    #OBdur in days
    maxNumRepInTime = np.asarray([math.ceil(Time/OBduri) for OBduri in OBdur])#.astype('int')
    maxRepsPerYear = np.asarray([math.ceil(365.25/OBduri) for OBduri in OBdur])#.astype('int')
    return maxNumRepInTime, maxRepsPerYear


missionPortion = exoplanetObsTime/(maxNumYears*365.25) # This is constrained


#Create List of OB durations
OBdur2 = list(set(np.logspace(np.log10(0.1),np.log10(365.),num=50,base=10.).astype(int)))
tmp = list(np.asarray(range(10))+1.5)
OBdur2.remove(0)
OBdur2 = sort(np.asarray(OBdur2 + tmp))


#Calculate Maximum number of repetitions within exoplanetObsTime
maxNumRepTot2, maxRepsPerYear2 = maxNumRepInTime(OBdur2,exoplanetObsTime)
fig = plt.figure(1)
#loglog(OBdur2,maxNumRepTot2,marker='o')
plt.semilogx(OBdur2,maxNumRepTot2,marker='o')
plt.xlabel('Num Days',weight='bold')
plt.ylabel('Max Num Reps',weight='bold')
plt.show(block=False)


#### Create Periodic Distribution of OB ##################################
periodicDistOB = list()
for i in np.arange(len(OBdur2)):
    tmpStart, tmpEnd = periodicDist(maxNumRepTot2[i], OBdur2[i], maxNumDays)
    periodicDistOB.append([tmpStart, tmpEnd])

writeHarmonicToOutputFiles = False#Change this to true to create each of these start and end time Observing Blocks as .csv files
if writeHarmonicToOutputFiles == True:
    path = '/home/dean/Documents/exosims/Scripts/'
    tmp = ''
    for i in np.arange(len(OBdur2)):
        myList = list()
        for j in range(len(periodicDistOB[i][0])):
            myList.append(str(periodicDistOB[i][0][j]) + ',' + str(periodicDistOB[i][1][j]) + '\n')
        outString = ''.join(myList)
        #print outString
        fname = path + 'periodicDistOB' + str(i) + '.csv'
        f = open(fname, "w")
        f.write(outString)
        print '"' + fname.split('/')[-1] + '",'
#####################################################################


#######################################################################################
#maxNumReps = maxNumYears*maxRepsPerYear2#number of Reps/ number of years #The minimum number of repetitions to go into 1 year in order to finish before 6 years

plt.figure(2)
num = np.linspace(0,50,num=50)
tmp = np.geomspace(0.0001,maxNumDays,num=50)
frac = 0.6
tmp2 = frac*np.geomspace(0.0001,maxNumDays,num=50) + (1-frac)*np.linspace(0,maxNumDays,num=50)
tmp3 = frac*np.geomspace(0.0001,maxNumDays,num=50) + (1-frac)*np.linspace(0,maxNumDays,num=50)*np.geomspace(0.0001,maxNumDays,num=50)/maxNumDays
tmp4 = np.linspace(0,maxNumDays,num=50)
def func(x,valMax):
    m=200.
    frac = 0.8
    val1 = m*x
    val2 = x**3#12*x**2#np.exp(x)
    val = frac*val1 + (1-frac)*val2
    val = val*valMax/max(val)
    return val
def dfunc(x):
    dval = 0.8*200 + 3*(1-0.8)*x**2.
    return dval
tmp5 = func(num,maxNumDays)

plt.plot(tmp,num,marker='o',color='blue')
plt.plot(tmp2,num,marker='o',color='black')
plt.plot(tmp3,num,marker='o',color='red')
plt.plot(tmp4,num,marker='o',color='green')
plt.plot(tmp5,num,marker='o',color='orange')
plt.ylabel('Points Number',weight='bold')
plt.xlabel('Start Times',weight='bold')
plt.show(block=False)


### DELETE THIS ?????????########################
# import scipy.integrate as integrate
# tmp5L = integrate.quad(dfunc,0,max(num))#This is the total length of the path from (0,0) to (num,maxNumDays)

# minNumReps = 0

# numRep = 10#number of repetitions in 1 year
# assert numRep <= 365.25/OBdur - 365.25%OBdur, 'numRep too large'
# missionPortion = numRep*OBdur/365.25
# missionLife = exoplanetObsTime/missionPortion

# def isoMissionDuration(mL,mP,mdur):
#     #missionLife,missionPortion,mission duration
#     #mdur = mL*mP #total amount of time to elapse during the mission
#     if mL is None:
#         mL = mdur/mP
#     elif mP is None:
#         mP = mdur/mL
#     return mL, mP


# tmp = np.asarray(range(30))*12.
# tmp1 = np.asarray(range(30))
# tmp2 = np.asarray(range(12))+0.5
# denom = np.asarray(range(30),)+1.
# tmp3 = 365.25/denom

# OBdurs = list()
# [OBdurs.append(x) for x in tmp.tolist()]
# [OBdurs.append(x) for x in tmp1.tolist()]
# [OBdurs.append(x) for x in tmp2.tolist()]
# [OBdurs.append(x) for x in tmp3.tolist()]


#### Calculate the Maximum Star Completeness of all 651 Targets under Consideration #####################
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import astropy.units as u
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))#EXOSIMS/EXOSIMS/Scripts'))#EXOSIMS/EXOSIMS/Scripts'))
filename = 'HabEx_4m_TSDD_pop100DD_revisit_20180424.json'##'WFIRSTcycle6core.json'#'Dean3June18RS26CXXfZ01OB66PP01SU01.json'#'Dean1June18RS26CXXfZ01OB56PP01SU01.json'#'./TestScripts/04_KeplerLike_Occulter_linearJScheduler.json'#'Dean13May18RS09CXXfZ01OB01PP03SU01.json'#'sS_AYO7.json'#'ICDcontents.json'###'sS_protoTimeKeeping.json'#'sS_AYO3.json'#sS_SLSQPstatic_parallel_ensembleJTWIN.json'#'sS_JTwin.json'#'sS_AYO4.json'#'sS_AYO3.json'
#filename = 'sS_intTime6_KeplerLike2.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile,nopar=True)

TL = sim.TargetList#OK
ZL = sim.ZodiacalLight#OK
sInds = np.arange(TL.nStars)#OK
fZ = sim.ZodiacalLight.fEZ0#fZmin/u.arcsec**2#
fEZ = ZL.fEZ0
OS = sim.OpticalSystem
WA = OS.WA0
mode = sim.SurveySimulation.detmode
dMagLim = sim.Completeness.dMagLim#OK #This is set based off the 10**-9 contrast of WFIRST
dmag = np.linspace(1, dMagLim, num=1500,endpoint=True)
Cp = np.zeros([sInds.shape[0],dmag.shape[0]])
Cb = np.zeros(sInds.shape[0])/u.s
Csp = np.zeros(sInds.shape[0])/u.s
for i in xrange(dmag.shape[0]):
    Cp[:,i], Cb[:], Csp[:] = OS.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dmag[i], WA, mode)
#Cb = np.zeros(sInds.shape[0])/u.s#Technically, forcing this to be zero results in the dMag limit that can be achieved with inf time

#Look 
t_dets_inf = np.asarray([t.value + np.inf if t.value > 0. else t.value for t in sim.SurveySimulation.t0])*u.d#sim.SurveySimulation.t0+np.inf*u.d#OK
comp_inf = sim.Completeness.comp_per_intTime(t_dets_inf, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)
sum_comp_inf = sum(comp_inf[comp_inf>0.])
t_dets_BIG = np.asarray([t.value + 1000. if t.value > 0. else t.value for t in sim.SurveySimulation.t0])*u.d#sim.SurveySimulation.t0+1000.*u.d#OK
comp_BIG = sim.Completeness.comp_per_intTime(t_dets_BIG, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)
sum_comp_BIG = sum(comp_inf[comp_BIG>0.])
t_dets = sim.SurveySimulation.t0#OK
comp = sim.Completeness.comp_per_intTime(t_dets, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)
sum_comp = sum(comp[comp>0.])

Cb2 = np.zeros(sInds.shape[0])/u.s#Technically, forcing this to be zero results in the dMag limit that can be achieved with inf time
comp_Cb0 = sim.Completeness.comp_per_intTime(t_dets, TL, sInds, fZ, fEZ, WA, mode, Cb2, Csp)

sum_comp_Cb0 = sum(comp_Cb0[comp_Cb0>0.])
#########################################################################################################



plt.close('all')
#### Calculate Ecliptic Latitude and Longitude of Stars
ra = sim.TargetList.coords.ra.value
dec = sim.TargetList.coords.dec.value
ra = ra*np.pi/180. -np.pi#The right ascension of the stars in the heliocentric ecliptic fixed frame
ra2 = ra[comp > 0.]
dec = dec*np.pi/180.#The declinations of the stars in the heliocentric ecliptic fixed frame
dec2 = dec[comp > 0.]
x = np.cos(dec2)*np.cos(ra2)#When dec2 =0, x/y=1
y = np.cos(dec2)*np.sin(ra2)
z = np.sin(dec2)
r_stars_equat = np.asarray([[x[i],y[i],z[i]] for i in np.arange(len(x))])
r_stars_equat = np.divide(r_stars_equat,np.asarray([np.linalg.norm(r_stars_equat,axis=1).tolist()]).T)*u.AU#Target stars in the equatorial coordinate frame
r_stars_eclip = sim.Observatory.equat2eclip(r_stars_equat,sim.TimeKeeping.currentTimeAbs,rotsign=1).value#target stars in the heliocentric ecliptic frame
hEclipLat = np.arcsin(r_stars_eclip[:,2])
hEclipLon = np.arctan2(r_stars_eclip[:,1],r_stars_eclip[:,0])
#######

#### Create Set of Evenly distributed Points on Sphere ##############
# From EXOSIMS/util/evenlyDistributePointsOnSphere.py
from EXOSIMS.util.evenlyDistributePointsOnSphere import splitOut, nlcon2, f, pt_pt_distances, secondSmallest, setupConstraints, initialXYZpoints
from scipy.optimize import minimize
x, y, z, v = initialXYZpoints(num_pts=30) # Generate Initial Set of XYZ Points
con = setupConstraints(v,nlcon2) # Define constraints on each point of the sphere
x0 = v.flatten() # takes v and converts it into [x0,y0,z0,x1,y1,z1,...,xn,yn,zn]
out1k = minimize(f,x0, method='SLSQP',constraints=(con), options={'ftol':1e-4, 'maxiter':1000}) # run optimization problem for 1000 iterations
out1kx, out1ky, out1kz = splitOut(out1k)
out1kv = np.asarray([[out1kx[i], out1ky[i], out1kz[i]] for i in np.arange(len(out1kx))]) #These are the points of each ind
dist1k = pt_pt_distances(out1kv) # for informational purposes: distances between points on sphere
####################################################################

#### Calculate closest points to each point on unit sphere ################
d_diff_pts_array, inds_of_closest, diff_closest = calculateClosestPoints(out1kv)
fig, ax = plotClosestPoints(inds_of_closest, out1kv)
###########################################################################

#### Remove Coplanar Connecting Lines (lines that cross)
inds_of_closest = removeCoplanarConnectingLines(inds_of_closest, out1kv)
########################################################

#### Calculate Triangle Areas, Corners indices, and triangle centers #####
sumTriangleArea, triangleCornerIndList, triangleAreaList, triangleCenterList = calculateTriangles(inds_of_closest,out1kv)
##########################################################################

#### Re plot Sphere ################################
plt.close(500672)
fig = plt.figure(num=500672)
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(out1kv[:,0], out1kv[:,1], out1kv[:,2], color='black',zorder=1)
plt.title('Plot of all point-to-point connections on sphere Corrected')
plt.show(block=False)

#Plot Edges
plotted = list()
for i in np.arange(len(out1kv)):
    xyzpoint = out1kv[i] # extract a single xyz point on sphere
    #plotted = list() #keeps track of index-to-index lines plotted
    for j in inds_of_closest[i]:#np.delete(inds_of_closest[i],0):
        if [i,j] in plotted or [j,i] in plotted:
            continue
        ax.plot([xyzpoint[0],out1kv[j,0]],[xyzpoint[1],out1kv[j,1]],[xyzpoint[2],out1kv[j,2]],color='red',zorder=1)
        plotted.append([i,j])
#Plot Centroids
for i in np.arange(len(triangleCenterList)):
    ax.scatter(triangleCenterList[i][0],triangleCenterList[i][1],triangleCenterList[i][2],color='purple', zorder=2)
ax.set_xlabel('X',weight='bold')
ax.set_ylabel('Y',weight='bold')
ax.set_zlabel('Z',weight='bold')
plt.show(block=False)
######################################################################


#### Create sets of triangle corners #################################
starAssignedTriangleCorners = createTriangleCornerSets(sInds,comp,hEclipLat,hEclipLon,out1kv,triangleCornerIndList)
######################################################################

#### Create tDict, the triangleDictionary of triangleKnowledge ############
tDict = createtDict(triangleCornerIndList,triangleAreaList,triangleCenterList,out1kv)
###########################################################################

#### Distribute Stars into Bins ######################################
tDict = distributeStarsIntoBins(tDict,starAssignedTriangleCorners,sInds[comp>0])
######################################################################

#### Plot Observation Schedule Sky Count Distribution ################
fig = plotSkyScheduledObservationCountDistribution(tDict)
######################################################################

#### Distribut Optimized Star Completeness Into Bins #################
for key in tDict.keys():#Iterate over triangles
    for sInd in tDict[key]['sIndsWithin']:#iterate over sIndsWithin
        try:
            tDict[key]['triangleComp'] += comp[sInd]
        except:
            tDict[key]['triangleComp'] = comp[sInd]
        try:
            tDict[key]['triangleIntTime'] += t_dets[sInd].value
        except:
            tDict[key]['triangleIntTime'] = t_dets[sInd].value
        try:
            tDict[key]['triangleMaxComp'] += comp_inf[sInd]
        except:
            tDict[key]['triangleMaxComp'] = comp_inf[sInd]
######################################################################

#### Plot Observation Schedule Sky Completeness Distribution #########
fig = plotSkyScheduledObservationCompletenessDistribution(tDict)
######################################################################

#### Plot Observation Schedule Sky Integration Time Distribution #####
fig = plotSkyScheduledObservationIntegrationDistribution(tDict)
######################################################################

#### Plot Sky Maximum Completeness Distribution ######################
#fig = plotSkyMaximumCompletenessDistribution(tDict)
starDict = createtDict(triangleCornerIndList,triangleAreaList,triangleCenterList,out1kv)
starDict = distributeStarsIntoBins(starDict,starAssignedTriangleCorners,sInds)
fig = plotSkyScheduledObservationCountDistribution(starDict,fignum=1124)
######################################################################


#### Histograms of Star Count vs Heliocentric Ecliptic Longitude ##################################
histInterp, targUnderSpline, sumh, xdiff, edges = generateHistHEL(hEclipLon)
#histIntep is a periodic interpolant from -pi to pi of planned observation planet occurence frequency
#targUnderSpline and sumh are here to ensure consistency of the integral
###################################################################################

#### Generate Random OB distributions #####################
# 1 Use Observing Block Durations Previously Defined OBdur2
# 2 Determine 
generatePlannedObsTimeHistHEL(edges,t_dets,comp,hEclipLon)
###########################################################





#lonGrabPts = np.linspace(-np.pi,np.pi,num=100) # the points to grab along longitude
lon = list()
lonIntTime = list()
lonTotalTime = list()
for key in tDict.keys():
    lon.append((xyzTolonlat(tDict[key]['triangleCenter']))[0])
    lonIntTime.append(tDict[key]['triangleIntTime'])
    lonTotalTime.append(tDict[key]['triangleIntTime'] + 1.*tDict[key]['count'])
plt.close(2356)
fig = plt.figure(num=2356)
plt.scatter(lon,lonTotalTime)
plt.show(block=False)



#### Generat Preferentially Distributed Integration Times ####################
barout = generatePlannedObsTimeHistHEL2(edges,tDict)
numOBassignedToBin, OBstartTimes, OBendTimes = generatePreferentiallyDistributedOB(barout, maxNumRepTot2[12], OBdur2[12], exoplanetObsTime, maxNumYears, loadingPreference='even')



prefDistOB = list()
for i in np.arange(len(OBdur2)):
    numOBassignedToBin, tmpStart, tmpEnd = generatePreferentiallyDistributedOB(barout, maxNumRepTot2[12], OBdur2[12], exoplanetObsTime, maxNumYears, loadingPreference='even')
    #DELETE tmpStart, tmpEnd = periodicDist(maxNumRepTot2[i], OBdur2[i], maxNumDays)
    prefDistOB.append([tmpStart, tmpEnd])

writePrefToOutputFiles = False#Change this to true to create each of these start and end time Observing Blocks as .csv files
if writePrefToOutputFiles == True:
    path = '/home/dean/Documents/exosims/Scripts/'
    tmp = ''
    for i in np.arange(len(OBdur2)):
        myList = list()
        for j in range(len(prefDistOB[i][0])):
            myList.append(str(prefDistOB[i][0][j]) + ',' + str(prefDistOB[i][1][j]) + '\n')
        outString = ''.join(myList)
        #print outString
        fname = path + 'prefDistOB' + str(i) + '.csv'
        f = open(fname, "w")
        f.write(outString)
        print '"' + fname.split('/')[-1] + '",'

###############################################################################


####





# #### Plottung cumulative sum of integration time needed
# sortInds = np.argsort(lon)
# lonSorted = [lon[ind] for ind in sortInds]
# lonIntTimeSorted = [lonIntTime[ind] for ind in sortInds]
# f2 = np.cumsum(lonIntTimeSorted)
# close(2358)
# fig = figure(num=2358)
# plot(lonSorted,f2)
# plot(np.linspace(-np.pi,np.pi),sum(lonIntTime)/(2*np.pi)*np.linspace(0.,2.*np.pi))
# plot(np.linspace(-np.pi,np.pi),365.25/(2*np.pi)*np.linspace(0.,2.*np.pi))
# show(block=False)

# #### Plotting cumulative sum of total time needed 
# sortInds = np.argsort(lon)
# lonSorted = [lon[ind] for ind in sortInds]
# lonTotalTimeSorted = [lonTotalTime[ind] for ind in sortInds]
# f1 = np.cumsum(lonTotalTimeSorted)
# close(2357)
# fig = figure(num=2357)
# plot(lonSorted,f1)
# plot(np.linspace(-np.pi,np.pi),365.25/(2*np.pi)*np.linspace(0.,2.*np.pi))
# show(block=False)

def plotClosestDistanceBetweenTwoSkewLines():
    """ An example plot demonstrating our ability to find the closest distance between two arbitrary
    skew lines
    Args:
    Returns:
    """
    #### Find Closest Distance between two arbitrary lines ############## #See method line2linev2
    #### Test Distance between two arbitrary lines ##########################
    plt.close(2055121)
    fig = plt.figure(num=2055121)
    ax = fig.add_subplot(111, projection='3d')
    prettifyPlot()
    p0 = np.asarray([0., 0., 0.])
    p1 = np.asarray([1., 1., 2.])
    v0 = np.asarray([1., 0., 1.])
    v1 = np.asarray([0., 1., 0.])
    out = line2linev2(p0,v0,p1,v1)
    t0 = out[4]
    t1 = out[5]
    q0 = out[2]
    q1 = out[3]
    dP = out[1]
    ax.plot([p0[0],p0[0]-t0*v0[0]],[p0[1],p0[1]-t0*v0[1]],[p0[2],p0[2]-t0*v0[2]],color='red')
    ax.plot([p1[0],p1[0]-t1*v1[0]],[p1[1],p1[1]-t1*v1[1]],[p1[2],p1[2]-t1*v1[2]],color='blue')
    ax.scatter(p0[0],p0[1],p0[2],color='black')#starting points
    ax.scatter(p1[0],p1[1],p1[2],color='black')#starting points
    ax.plot([q0[0],q1[0]],[q0[1],q1[1]],[q0[2],q1[2]],color='purple')
    ax.scatter(q0[0],q0[1],q0[2],color='purple')#ending points
    ax.scatter(q1[0],q1[1],q1[2],color='purple')#ending points
    ax.set_xlabel('X',weight='bold')
    ax.set_ylabel('Y',weight='bold')
    ax.set_zlabel('Z',weight='bold')
    ax.scatter(-1,-1,-1,color='white')
    ax.scatter(3,3,3,color='white')
    plt.show(block=False)
    #####################################################################




