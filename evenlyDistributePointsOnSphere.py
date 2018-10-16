"""evenlyDistributePointsOnSphere.py
This code creates a set of n points on a unit sphere which are approximately spaced as far as possible from each other.

#Written By: Dean Keithly
#Written On: 10/16/2018
"""

from numpy import pi, cos, sin, arccos, arange
import mpl_toolkits.mplot3d
import matplotlib.pyplot as pp
import numpy as np
from scipy.optimize import minimize


#### Generate Initial Set of XYZ Points ###############
num_pts = 30#1000
indices = arange(0, num_pts, dtype=float) + 0.5
phi = arccos(1 - 2*indices/num_pts)
theta = pi * (1 + 5**0.5) * indices
x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
v = np.asarray([[x[i], y[i], z[i]] for i in np.arange(len(x))]) # an array of each point on the sphere
d = np.linalg.norm(v,axis=1) # used to ensure the length of each vector is 1
#######################################################

#### Plot Initial XYZ Points ##########################
pp.close(5006)
fig = pp.figure(num=5006)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='b');
pp.show(block=False)
#######################################################

def secondSmallest(d_diff_pts):
    """For a list of points, return the value and ind of the second smallest
    args:
        d_diff_pts - numy array of floats of distances between points
    returns:
        secondSmallest_value - 
        secondSmallest_ind - 
    """
    tmp_inds = np.arange(len(d_diff_pts))
    tmp_inds_min0 = np.argmin(d_diff_pts)
    tmp_inds = np.delete(tmp_inds, tmp_inds_min0)
    tmp_d_diff_pts =np.delete(d_diff_pts, tmp_inds_min0)
    secondSmallest_value = min(tmp_d_diff_pts)
    secondSmallest_ind = np.argmin(np.abs(d_diff_pts - secondSmallest_value))
    return secondSmallest_value, secondSmallest_ind

def f(vv):
    # This is the optimization problem objective function
    # We calculate the sum of all distances between points and 
    xx = vv[::3]
    yy = vv[1::3]
    zz = vv[2::3]
    xyzpoints = np.asarray([[xx[i], yy[i], zz[i]]for i in np.arange(len(zz))])
    #Calculates the sum(min(dij)**3.)
    distances = list()
    closest_point_inds = list() # list of numpy arrays containing closest points to a given ind
    for i in np.arange(len(xyzpoints)):
        xyzpoint = xyzpoints[i] # extract a single xyz point on sphere
        diff_pts = xyzpoints - xyzpoint # calculate linear difference between point spacing
        d_diff_pts = np.linalg.norm(diff_pts,axis=1) # calculate linear distance between points
        ss_d, ss_ind = secondSmallest(d_diff_pts) #we must get the second smallest because the smallest is the point itself
        distances.append(ss_d)
    return -sum(np.asarray(distances)**2.) #squares and sums each point-to-closest point distances

def nlcon(vvv):
    # This is the nonlinear constraint on each "point" of the sphere
    # We require that the center-to-point distance be ~1.
    xxx = vvv[::3] # this decodes the x vars, vvv[0] and every 3rd element after that
    yyy = vvv[1::3] # this decodes the y vars, vvv[1] and every 3rd element after that
    zzz = vvv[2::3] # this decodes the z vars, vvv[2] and every 3rd element after that
    xyzpoints = np.asarray([[xxx[i], yyy[i], zzz[i]]for i in np.arange(len(zzz))])
    return np.linalg.norm(xyzpoints,axis=1) - np.ones(len(xxx)) #I just want the length to be 1

x0 = v.flatten() # takes v and converts it into [x0,y0,z0,x1,y1,z1,...,xn,yn,zn]
#nlc = NonlinearConstraint(nlcon,np.ones(len(v)),np.ones(len(v)))

def nlcon2(vvv,ind):
    xxx = vvv[::3][ind]
    yyy = vvv[1::3][ind]
    zzz = vvv[2::3][ind]
    xyzpoint = np.asarray([xxx,yyy,zzz])#[[xxx[i], yyy[i], zzz[i]]for i in np.arange(len(zzz))])
    return np.linalg.norm(xyzpoint) - 1. #I just want the length to be 1

con = list()
for i in np.arange(len(v)):
    ctemp = {'type':'eq','fun':nlcon2,'args':(i,)}
    con.append(ctemp) 
#{'type':'eq','fun':nlcon}
out1k = minimize(f,x0, method='SLSQP',constraints=(con), options={'ftol':1e-4, 'maxiter':1000})
# out1k = fmin(f,x0, xtol=1e-4, ftol=1e-4, maxiter=1000)#disp=True, full_output=True, retall=True)
print out1k['fun']
print out1k['success']
print out1k['x']

out1kx = out1k['x'][::3]
out1ky = out1k['x'][1::3]
out1kz = out1k['x'][2::3]
out1kv = np.asarray([[out1kx[i], out1ky[i], out1kz[i]] for i in np.arange(len(out1kx))])

ax.scatter(out1kx, out1ky, out1kz,color='r')
pp.show(block=False)
out2k = minimize(f,x0, method='SLSQP',constraints=(con), options={'ftol':1e-4, 'maxiter':2000})
out2kx = out2k['x'][::3]
out2ky = out2k['x'][1::3]
out2kz = out2k['x'][2::3]
ax.scatter(out2kx, out2ky, out2kz,color='g')
pp.show(block=False)
out4k = minimize(f,x0, method='SLSQP',constraints=(con), options={'ftol':1e-4, 'maxiter':4000})
out4kx = out4k['x'][::3]
out4ky = out4k['x'][1::3]
out4kz = out4k['x'][2::3]
ax.scatter(out4kx, out4ky, out4kz,color='c')
pp.legend(['Initial','1k iter.','2k iter.','4k iter.'])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
pp.title('Points Distributed on a Sphere')
pp.show(block=False)


