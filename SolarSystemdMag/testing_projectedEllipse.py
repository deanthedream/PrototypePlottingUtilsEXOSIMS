import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as random
from sys import getsizeof
import time

#### Randomly Generate Orbits
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))
filename = 'HabEx_CKL2_PPKL2.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
PPop = sim.PlanetPopulation
n = 10**4 #Dean's nice computer can go up to 10**8 what can atuin go up to?
inc, W, w = PPop.gen_angles(n,None)
inc = inc.to('rad').value
inc[np.where(inc>np.pi/2)[0]] = np.pi - inc[np.where(inc>np.pi/2)[0]]
W = W.to('rad').value
w = w.to('rad').value
sma, e, p, Rp = PPop.gen_plan_params(n)
sma = sma.to('AU').value

#### SAVED PLANET FOR Plot 3D Ellipse to 2D Ellipse Projection Diagram
ind = 23 #22
sma[ind] = 1.2164387563540457
e[ind] = 0.531071885292766
w[ind] = 3.477496280463054
W[ind] = 5.333215834002414
inc[ind] = 1.025093642138022
####

#### Calculate Projected Ellipse Angles and Minor Axis
start0 = time.time()
dmajorp, dminorp, Psi, psi, theta_OpQ_X, theta_OpQp_X, dmajorp_v2, dminorp_v2, Psi_v2, psi_v2 = projected_apbpPsipsi(sma,e,W,w,inc)
stop0 = time.time()
print('stop0: ' + str(stop0-start0))
#DELETEO = projected_Op(sma,e,W,w,inc)
#DELETEc_3D_projected = projected_projectedLinearEccentricity(sma,e,W,w,inc)
#3D Ellipse Center
start1 = time.time()
Op = projected_Op(sma,e,W,w,inc)
stop1 = time.time()
print('stop1: ' + str(stop1-start1))

# Checks
assert np.all(dmajorp < sma), "Not all Semi-major axis of the projected ellipse are less than the original 3D ellipse"
assert np.all(dminorp < dmajorp), "All projected Semi-minor axes are less than all projected semi-major axes"

#### Plotting Projected Ellipse
def plotProjectedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, num):
    plt.close(num)
    fig = plt.figure(num=num)
    ca = plt.gca()
    ca.axis('equal')
    ## Central Sun
    plt.scatter([0],[0],color='orange')
    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
    x_3Dellipse = r[0,0,:]
    y_3Dellipse = r[1,0,:]
    plt.plot(x_3Dellipse,y_3Dellipse,color='black')

    #plot 3D Ellipse Center
    plt.scatter(Op[0][ind],Op[1][ind],color='black')
    print('a: ' + str(np.round(sma[ind],2)) + ' e: ' + str(np.round(e[ind],2)) + ' W: ' + str(np.round(W[ind],2)) + ' w: ' + str(np.round(w[ind],2)) + ' i: ' + str(np.round(inc[ind],2)) +\
         ' Psi: ' + str(np.round(Psi[ind],2)) + ' psi: ' + str(np.round(psi[ind],2)))# + ' theta: ' + str(np.round(theta[ind],2)))
    #print(dmajorp[ind]*np.cos(theta[ind]))#print(dmajorp[ind]*np.cos(theta[ind]))#print(dminorp[ind]*np.cos(theta[ind]+np.pi/2))#print(dminorp[ind]*np.sin(theta[ind]+np.pi/2))

    ang2 = (theta_OpQ_X[ind]+theta_OpQp_X[ind])/2
    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='purple',linestyle='-')
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='purple',linestyle='-')
    plt.show(block=False)
    ####

ind = random.randint(low=0,high=n)
plotProjectedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, num=877)



#### Plot 3D Ellipse to 2D Ellipse Projection Diagram

num = 666999888777
plt.close(num)
fig = plt.figure(num)
ax = fig.add_subplot(111, projection='3d')

## 3D Ellipse
vs = np.linspace(start=0,stop=2*np.pi,num=300)
r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
x_3Dellipse = r[0,0,:]
y_3Dellipse = r[1,0,:]
z_3Dellipse = r[2,0,:]
ax.plot(x_3Dellipse,y_3Dellipse,z_3Dellipse,color='black',label='Planet Orbit')
min_z = np.min(z_3Dellipse)

## Central Sun
ax.scatter(0,0,0,color='orange',marker='o',s=25) #of 3D ellipse
ax.text(0,0,0.15*np.abs(min_z), 'F', None)
ax.plot([0,0],[0,0],[0,1.3*min_z],color='orange',linestyle=':') #connecting line
ax.scatter(0,0,1.3*min_z,color='orange',marker='x',s=25) #of 2D ellipse
ax.text(0,0,1.5*min_z, 'F\'', None)

## Plot 3D Ellipse semi-major/minor axis
rper = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],0.) #planet position perigee
rapo = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],np.pi) #planet position apogee
ax.plot([rper[0][0],rapo[0][0]],[rper[1][0],rapo[1][0]],[rper[2][0],rapo[2][0]],color='purple', linestyle='-') #3D Ellipse Semi-major axis
ax.scatter(rper[0][0],rper[1][0],rper[2][0],color='black',marker='D',s=25) #3D Ellipse Perigee Diamond
ax.text(1.2*rper[0][0],1.2*rper[1][0],rper[2][0], 'A', None)#(rper[0][0],rper[1][0],0))
ax.scatter(rper[0][0],rper[1][0],1.3*min_z,color='red',marker='D',s=25) #2D Ellipse Perigee Diamond
ax.text(1.1*rper[0][0],1.1*rper[1][0],1.3*min_z, 'A\'', None)#(rper[0][0],rper[1][0],0))
ax.plot([rper[0][0],rper[0][0]],[rper[1][0],rper[1][0]],[rper[2][0],1.3*min_z],color='black',linestyle=':') #3D to 2D Ellipse Perigee Diamond
ax.scatter(rapo[0][0],rapo[1][0],rapo[2][0],color='black', marker='D',s=25) #3D Ellipse Apogee Diamond
ax.text(1.1*rapo[0][0],1.1*rapo[1][0],1.2*rapo[2][0], 'B', None)#(rapo[0][0],rapo[1][0],0))
ax.scatter(rapo[0][0],rapo[1][0],1.3*min_z,color='red',marker='D',s=25) #2D Ellipse Perigee Diamond
ax.text(1.1*rapo[0][0],1.1*rapo[1][0],1.3*min_z, 'B\'', None)#(rapo[0][0],rapo[1][0],0))

ax.plot([rapo[0][0],rapo[0][0]],[rapo[1][0],rapo[1][0]],[rapo[2][0],1.3*min_z],color='black',linestyle=':') #3D to 2D Ellipse Apogee Diamond
rbp = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],np.arccos((np.cos(np.pi/2)-e[ind])/(1-e[ind]*np.cos(np.pi/2)))) #3D Ellipse E=90
rbm = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],-np.arccos((np.cos(-np.pi/2)-e[ind])/(1-e[ind]*np.cos(-np.pi/2)))) #3D Ellipse E=-90
ax.plot([rbp[0][0],rbm[0][0]],[rbp[1][0],rbm[1][0]],[rbp[2][0],rbm[2][0]],color='purple', linestyle='-') #
ax.scatter(rbp[0][0],rbp[1][0],rbp[2][0],color='black',marker='D',s=25) #3D ellipse minor +
ax.text(1.1*rbp[0][0],1.1*rbp[1][0],1.2*rbp[2][0], 'C', None)#(rbp[0][0],rbp[1][0],0))
ax.scatter(rbp[0][0],rbp[1][0],1.3*min_z,color='red',marker='D',s=25) #2D ellipse minor+ projection
ax.text(1.1*rbp[0][0],1.1*rbp[1][0],1.3*min_z, 'C\'', None)#(rbp[0][0],rbp[1][0],0))
ax.plot([rbp[0][0],rbp[0][0]],[rbp[1][0],rbp[1][0]],[rbp[2][0],1.3*min_z],color='black',linestyle=':') #3D to 2D Ellipse minor + Diamond
ax.scatter(rbm[0][0],rbm[1][0],rbm[2][0],color='black', marker='D',s=25) #3D ellipse minor -
ax.text(1.1*rbm[0][0],0.5*(rbm[1][0]-Op[1][ind]),rbm[2][0], 'D', None)#(rbm[0][0],rbm[1][0],0))
ax.scatter(rbm[0][0],rbm[1][0],1.3*min_z,color='red', marker='D',s=25) #2D ellipse minor- projection
ax.text(1.1*rbm[0][0],0.5*(rbm[1][0]-Op[1][ind]),1.3*min_z, 'D\'', None)#(rbm[0][0],rbm[1][0],0))
ax.plot([rbm[0][0],rbm[0][0]],[rbm[1][0],rbm[1][0]],[rbm[2][0],1.3*min_z],color='black',linestyle=':') #3D to 2D Ellipse minor - Diamond

## Plot Conjugate Diameters
ax.plot([rbp[0][0],rbm[0][0]],[rbp[1][0],rbm[1][0]],[1.3*min_z,1.3*min_z],color='blue',linestyle='-') #2D ellipse minor+ projection
ax.plot([rper[0][0],rapo[0][0]],[rper[1][0],rapo[1][0]],[1.3*min_z,1.3*min_z],color='blue',linestyle='-') #2D Ellipse Perigee Diamond

## Plot Ellipse Center
ax.scatter((rper[0][0] + rapo[0][0])/2,(rper[1][0] + rapo[1][0])/2,(rper[2][0] + rapo[2][0])/2,color='black',marker='o',s=36) #3D Ellipse
ax.text(1.2*(rper[0][0] + rapo[0][0])/2,1.2*(rper[1][0] + rapo[1][0])/2,1.31*(rper[2][0] + rapo[2][0])/2, 'O', None)
ax.scatter(Op[0][ind],Op[1][ind], 1.3*min_z, color='black', marker='o',s=25) #2D Ellipse Center
ax.text(1.2*(rper[0][0] + rapo[0][0])/2,1.2*(rper[1][0] + rapo[1][0])/2,1.4*min_z, 'O\'', None)
ax.plot([(rper[0][0] + rapo[0][0])/2,Op[0][ind]],[(rper[1][0] + rapo[1][0])/2,Op[1][ind]],[(rper[2][0] + rapo[2][0])/2,1.3*min_z],color='black',linestyle=':')
print('a: ' + str(np.round(sma[ind],2)) + ' e: ' + str(np.round(e[ind],2)) + ' W: ' + str(np.round(W[ind],2)) + ' w: ' + str(np.round(w[ind],2)) + ' i: ' + str(np.round(inc[ind],2)) +\
     ' Psi: ' + str(np.round(Psi[ind],2)) + ' psi: ' + str(np.round(psi[ind],2)))# + ' theta: ' + str(np.round(theta[ind],2)))


ang2 = (theta_OpQ_X[ind]+theta_OpQp_X[ind])/2
dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
ax.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],[1.3*min_z,1.3*min_z],color='purple',linestyle='-')
ax.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],[1.3*min_z,1.3*min_z],color='purple',linestyle='-')
dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
ax.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],[1.3*min_z,1.3*min_z],color='purple',linestyle='-')
ax.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],[1.3*min_z,1.3*min_z],color='purple',linestyle='-')

dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
ax.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],[1.3*min_z,1.3*min_z],color='purple',linestyle='-')
ax.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],[1.3*min_z,1.3*min_z],color='purple',linestyle='-')
ax.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],[1.3*min_z,1.3*min_z],color='purple',linestyle='-')
ax.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],[1.3*min_z,1.3*min_z],color='purple',linestyle='-')
ax.scatter([dmajorpx1,dmajorpx2,dminorpx1,dminorpx2],[dmajorpy1,dmajorpy2,dminorpy1,dminorpy2],[1.3*min_z,1.3*min_z,1.3*min_z,1.3*min_z],color='grey',marker='o',s=25,zorder=2)
ax.text(1.05*dmajorpx1,1.05*dmajorpy1,1.3*min_z, 'I', None)#(dmajorpx1,dmajorpy1,0))
ax.text(1.1*dmajorpx2,1.1*dmajorpy2,1.3*min_z, 'R', None)#(dmajorpx2,dmajorpy2,0))
ax.text(1.05*dminorpx1,0.1*(dminorpy1-Op[1][ind]),1.3*min_z, 'S', None)#(dminorpx1,dminorpy1,0))
ax.text(1.05*dminorpx2,1.05*dminorpy2,1.3*min_z, 'T', None)#(dminorpx2,dminorpy2,0))
#ax.text(x,y,z, label, zdir)
x_projEllipse = Op[0][ind] + dmajorp[ind]*np.cos(vs)*np.cos(ang2) - dminorp[ind]*np.sin(vs)*np.sin(ang2)
y_projEllipse = Op[1][ind] + dmajorp[ind]*np.cos(vs)*np.sin(ang2) + dminorp[ind]*np.sin(vs)*np.cos(ang2)
ax.plot(x_projEllipse,y_projEllipse,1.3*min_z*np.ones(len(vs)), color='red', linestyle='-',zorder=7)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(False)
#artificial box
xmax = np.max([np.abs(rper[0][0]),np.abs(rapo[0][0]),np.abs(1.3*min_z)])
ax.scatter([-xmax,xmax],[-xmax,xmax],[-0.2-np.abs(1.3*min_z),0.2+1.3*min_z],color=None,alpha=0)
ax.set_xlim3d(-0.99*xmax+Op[0][ind],0.99*xmax+Op[0][ind])
ax.set_ylim3d(-0.99*xmax+Op[1][ind],0.99*xmax+Op[1][ind])
ax.set_zlim3d(-0.99*xmax,0.99*xmax)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) #remove background color
ax.set_axis_off() #removes axes
plt.show(block=False)
####

#### Create Projected Ellipse Conjugate Diameters and QQ' construction diagram
####


#### Derotate Ellipse
start2 = time.time()
x, y, Phi = derotatedEllipse(theta_OpQ_X, theta_OpQp_X, Op)
stop2 = time.time()
print('stop2: ' + str(stop2-start2))
a = dmajorp
b = dminorp
mx = np.abs(x) #x converted to a strictly positive value
my = np.abs(y) #y converted to a strictly positive value

def plotDerotatedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, a, b, num=879):
    plt.close(num)
    fig = plt.figure(num=num)
    ca = plt.gca()
    ca.axis('equal')
    plt.scatter([0],[0],color='orange')
    ## Plot 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
    x_3Dellipse = r[0,0,:]
    y_3Dellipse = r[1,0,:]
    plt.plot(x_3Dellipse,y_3Dellipse,color='black')
    ## Plot 3D Ellipse Center
    plt.scatter(Op[0][ind],Op[1][ind],color='black')
    ## Plot Rotated Ellipse
    ang2 = (theta_OpQ_X[ind]+theta_OpQp_X[ind])/2
    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='purple',linestyle='-')
    #new plot stuff
    Erange = np.linspace(start=0.,stop=2*np.pi,num=400)
    plt.plot([-a[ind],a[ind]],[0,0],color='purple',linestyle='--') #major
    plt.plot([0,0],[-b[ind],b[ind]],color='purple',linestyle='--') #minor
    xellipsetmp = a[ind]*np.cos(Erange)
    yellipsetmp = b[ind]*np.sin(Erange)
    plt.plot(xellipsetmp,yellipsetmp,color='black')
    plt.scatter(x[ind],y[ind],color='orange',marker='x')

    c_ae = a[ind]*np.sqrt(1-b[ind]**2/a[ind]**2)
    plt.scatter([-c_ae,c_ae],[0,0],color='blue')

    plt.show(block=False)

start3 = time.time()
plotDerotatedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, a, b, num=880)
stop3 = time.time()
print('stop3: ' + str(stop3-start3))

#### Calculate X,Y Position of Minimum and Maximums with Quartic
# start4 = time.time()
# #OLD METHOD USING NUMPY ROOT
# xreal, imag = quarticSolutions(a, b, mx, my)
# stop4 = time.time()
# print('stop4: ' + str(stop4-start4))
#NEW METHOD USING ANALYTICAL
start4_new = time.time()
A, B, C, D = quarticCoefficients_smin_smax_lmin_lmax(a.astype('complex128'), b, mx, my)
xreal, delta, P, D2, R, delta_0 = quarticSolutions_ellipse_to_Quarticipynb(A.astype('complex128'), B, C, D)
del A, B, C, D #delting for memory efficiency
assert np.max(np.nanmin(np.abs(np.imag(xreal)),axis=1)) < 1e-15, 'At least one row has min > 1e-15' #this ensures each row has a solution
xreal.real = np.abs(xreal)
stop4_new = time.time()
print('stop4_new: ' + str(stop4_new-start4_new))


#Technically, each row must have at least 2 solutions, but whatever
yreal = ellipseYFromX(xreal.astype('complex128'), a, b)

#### Calculate Minimum, Maximum, Local Minimum, Local Maximum Separations
minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds = smin_smax_slmin_slmax(n, xreal, yreal, mx, my, x, y)

# #Note: the solving method breaks down when the inclination is nearly zero and the star 
# #Correction for 0 inclination planets where star is nearly centers in x and y
# zeroIncCentralStarPlanets = np.where((np.abs(inc-np.pi/2) < 1e-3)*(mx < 5*1e-2)*(my < 1e-5))[0]
# minSep2[zeroIncCentralStarPlanets] = s_mplminSeps2[zeroIncCentralStarPlanets]
# minSepPoints2_x[zeroIncCentralStarPlanets] = lminSepPoints2_x[zeroIncCentralStarPlanets]
# minSepPoints2_y[zeroIncCentralStarPlanets] = -lminSepPoints2_y[zeroIncCentralStarPlanets]

#### Old Method
# start5 = time.time()
# yreal = ellipseYFromX(xreal, a, b)
# stop5 = time.time()
# print('stop5: ' + str(stop5-start5))

#### Calculate Separations
# start6 = time.time()
# s_mp, s_pm, minSep, maxSep = calculateSeparations(xreal, yreal, mx, my)
# stop6 = time.time()
# print('stop6: ' + str(stop6-start6))

#### Calculate Min Max Separation Points
# start7 = time.time()
# minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, s_mplminSeps, s_mplmaxSeps = sepsMinMaxLminLmax(minSep, maxSep, s_mp, xreal, yreal, x, y)
# stop7 = time.time()
# print('stop7: ' + str(stop7-start7))
#################################################################################


#### Memory Usage
memories = [getsizeof(inc),getsizeof(W),getsizeof(w),getsizeof(sma),getsizeof(e),getsizeof(p),getsizeof(Rp),getsizeof(dmajorp),getsizeof(dminorp),getsizeof(Psi),getsizeof(psi),getsizeof(theta_OpQ_X),\
getsizeof(theta_OpQp_X),getsizeof(dmajorp_v2),getsizeof(dminorp_v2),getsizeof(Psi_v2),getsizeof(psi_v2),getsizeof(Op),getsizeof(x),getsizeof(y),getsizeof(Phi),getsizeof(a),getsizeof(b),\
getsizeof(mx),getsizeof(my),getsizeof(xreal),getsizeof(yreal),getsizeof(minSepPoints_x),getsizeof(minSepPoints_y),\
getsizeof(maxSepPoints_x),getsizeof(maxSepPoints_y),getsizeof(lminSepPoints_x),getsizeof(lminSepPoints_y),getsizeof(lmaxSepPoints_x),getsizeof(lmaxSepPoints_y),getsizeof(minSep),\
getsizeof(maxSep)]#,getsizeof(s_mplminSeps),getsizeof(s_mplmaxSeps)]
totalMemoryUsage = np.sum(memories)
print('Total Data Used: ' + str(totalMemoryUsage/10**9) + ' GB')
####

#### Plot separation vs vs parameter
num=961
plt.close(num)
fig = plt.figure(num=num)
Erange = np.linspace(start=0.,stop=2*np.pi,num=400)
xellipsetmp = a[ind]*np.cos(Erange)
yellipsetmp = b[ind]*np.sin(Erange)
septmp = np.sqrt((xellipsetmp - x[ind])**2 + (yellipsetmp - y[ind])**2)
plt.plot(Erange,septmp,color='black')
plt.plot([0,2.*np.pi],[0,0],color='black',linestyle='--') #0 sep line
plt.plot([0,2*np.pi],[minSep[ind],minSep[ind]],color='cyan')
plt.plot([0,2*np.pi],[maxSep[ind],maxSep[ind]],color='red')
if ind in yrealAllRealInds:
    tind = np.where(yrealAllRealInds == ind)[0]
    plt.plot([0,2*np.pi],[lminSep[tind],lminSep[tind]],color='magenta')
    plt.plot([0,2*np.pi],[lmaxSep[tind],lmaxSep[tind]],color='gold')
plt.plot([0,2*np.pi],[1,1],color='green')
plt.xlim([0,2.*np.pi])
plt.ylabel('Projected Separation in AU')
plt.xlabel('Projected Ellipse E (rad)')
plt.show(block=False)
####



#### Ellipse Circle Intersection
#### Testing ellipse_to_Quartic solution
r = np.ones(len(a),dtype='complex128')
a.astype('complex128')
b.astype('complex128')
mx.astype('complex128')
my.astype('complex128')
r.astype('complex128')
A, B, C, D = quarticCoefficients_ellipse_to_Quarticipynb(a, b, mx, my, r)
xreal2, delta, P, D2, R, delta_0 = quarticSolutions_ellipse_to_Quarticipynb(A, B, C, D)
del A, B, C, D #delting for memory efficiency
yreal2 = ellipseYFromX(xreal2.astype('complex128'), a, b)

#### All Real Inds
#Where r is...
gtMinSepBool = (minSep[yrealAllRealInds] < r[yrealAllRealInds])
ltMaxSepBool = (maxSep[yrealAllRealInds] >= r[yrealAllRealInds])
gtLMaxSepBool = (lmaxSep < r[yrealAllRealInds])
ltLMaxSepBool = (lmaxSep > r[yrealAllRealInds])
gtLMinSepBool = (lminSep <= r[yrealAllRealInds])
ltLMinSepBool = (lminSep > r[yrealAllRealInds])

#Two intersections on same y-side of projected ellipse
twoIntSameYInds = np.where(gtMinSepBool*ltLMinSepBool)[0]
#Four intersections total
fourIntInds = np.where(gtLMinSepBool*ltLMaxSepBool)[0]
#Two intersections opposite x-side
twoIntOppositeXInds = np.where(ltMaxSepBool*gtLMaxSepBool)[0]
del gtMinSepBool, ltMaxSepBool, gtLMaxSepBool, ltLMaxSepBool, gtLMinSepBool, ltLMinSepBool #for memory efficiency

#Solution Checks
assert np.max(np.imag(xreal2[yrealAllRealInds[fourIntInds]])) < 1e-7, 'an Imag component of the all reals is too low!'


#### Four Intersection Points
fourInt_dx = (np.real(xreal2[yrealAllRealInds[fourIntInds]]).T - mx[yrealAllRealInds[fourIntInds]]).T
fourIntSortInds = np.argsort(fourInt_dx, axis=1)
sameYOppositeXInds = fourIntSortInds[:,0]
assert np.all(sameYOppositeXInds==0), 'not all 0'
sameYXInds = fourIntSortInds[:,3]
assert np.all(sameYXInds==3), 'not all 3'
oppositeYOppositeXInds = fourIntSortInds[:,1]
assert np.all(oppositeYOppositeXInds==1), 'not all 1'
oppositeYSameXInds = fourIntSortInds[:,2]
assert np.all(oppositeYSameXInds==2), 'not all 2'
fourInt_y = np.zeros((len(fourIntInds),4))
fourInt_x = np.zeros((len(fourIntInds),4))
fourInt_x[:,0] = xreal2[yrealAllRealInds[fourIntInds],sameYOppositeXInds]
fourInt_x[:,1] = xreal2[yrealAllRealInds[fourIntInds],sameYXInds]
fourInt_x[:,2] = xreal2[yrealAllRealInds[fourIntInds],oppositeYOppositeXInds]
fourInt_x[:,3] = xreal2[yrealAllRealInds[fourIntInds],oppositeYSameXInds]
fourInt_y = ellipseYFromX(np.abs(fourInt_x), a[yrealAllRealInds[fourIntInds]], b[yrealAllRealInds[fourIntInds]])
fourInt_y[:,2] = -fourInt_y[:,2]
fourInt_y[:,3] = -fourInt_y[:,3]
#Quadrant Star Belongs to
bool1 = x > 0
bool2 = y > 0
#Quadrant 1 if T,T
#Quadrant 2 if F,T
#Quadrant 3 if F,F
#Quadrant 4 if T,F
#### Four Intercept Points
fourInt_x = (fourInt_x.T*(2*bool1[yrealAllRealInds[fourIntInds]]-1)).T
fourInt_y = (fourInt_y.T*(2*bool2[yrealAllRealInds[fourIntInds]]-1)).T
####
#### Two Intersection Points twoIntSameYInds
twoIntSameY_x = np.zeros((len(twoIntSameYInds),2))
twoIntSameY_y = np.zeros((len(twoIntSameYInds),2))
assert np.max(np.imag(xreal2[yrealAllRealInds[twoIntSameYInds],0])) < 1e-12, ''
twoIntSameY_x[:,0] = np.real(xreal2[yrealAllRealInds[twoIntSameYInds],0])
smallImagInds = np.where(np.abs(np.imag(xreal2[yrealAllRealInds[twoIntSameYInds],1])) < 1e-10)[0]
largeImagInds = np.where(np.abs(np.imag(xreal2[yrealAllRealInds[twoIntSameYInds],1])) > 1e-10)[0]
twoIntSameY_x[smallImagInds,1] = np.real(xreal2[yrealAllRealInds[twoIntSameYInds[smallImagInds]],1])
twoIntSameY_x[largeImagInds,1] = np.real(xreal2[yrealAllRealInds[twoIntSameYInds[largeImagInds]],3])
twoIntSameY_y = np.asarray([np.sqrt(b[yrealAllRealInds[twoIntSameYInds]]**2*(1-twoIntSameY_x[:,0]**2/a[yrealAllRealInds[twoIntSameYInds]]**2)),\
        np.sqrt(b[yrealAllRealInds[twoIntSameYInds]]**2*(1-twoIntSameY_x[:,1]**2/a[yrealAllRealInds[twoIntSameYInds]]**2))]).T
twoIntSameY_y = (twoIntSameY_y.T*(2*bool2[yrealAllRealInds[twoIntSameYInds]]-1)).T
#Quadrant Star Belongs to
bool1 = x > 0
bool2 = y > 0
#Quadrant 1 if T,T
#Quadrant 2 if F,T
#Quadrant 3 if F,F
#Quadrant 4 if T,F
twoIntSameY_x = (twoIntSameY_x.T*(2*bool1[yrealAllRealInds[twoIntSameYInds]]-1)).T
twoIntSameY_y = (twoIntSameY_y.T*(2*bool2[yrealAllRealInds[twoIntSameYInds]]-1)).T
####
#### Two Intersection Points twoIntOppositeXInds
twoIntOppositeX_x = np.zeros((len(twoIntOppositeXInds),2))
twoIntOppositeX_y = np.zeros((len(twoIntOppositeXInds),2))
assert np.max(np.imag(xreal2[yrealAllRealInds[twoIntOppositeXInds],0])) < 1e-12, ''
twoIntOppositeX_x[:,0] = np.real(xreal2[yrealAllRealInds[twoIntOppositeXInds],0])
twoIntOppositeX_x[:,1] = np.real(xreal2[yrealAllRealInds[twoIntOppositeXInds],1])
twoIntOppositeX_y = np.asarray([np.sqrt(b[yrealAllRealInds[twoIntOppositeXInds]]**2*(1-np.abs(twoIntOppositeX_x[:,0])**2/a[yrealAllRealInds[twoIntOppositeXInds]]**2)),\
        np.sqrt(b[yrealAllRealInds[twoIntOppositeXInds]]**2*(1-np.abs(twoIntOppositeX_x[:,1])**2/a[yrealAllRealInds[twoIntOppositeXInds]]**2))]).T
twoIntOppositeX_x = (twoIntOppositeX_x.T*(-2*bool1[yrealAllRealInds[twoIntOppositeXInds]]+1)).T
twoIntOppositeX_y[:,1] = -twoIntOppositeX_y[:,1]
#Quadrant Star Belongs to
bool1 = x > 0
bool2 = y > 0
#Quadrant 1 if T,T
#Quadrant 2 if F,T
#Quadrant 3 if F,F
#Quadrant 4 if T,F
twoIntOppositeX_x = (twoIntOppositeX_x.T*(2*bool1[yrealAllRealInds[twoIntOppositeXInds]]-1)).T
twoIntOppositeX_y = (twoIntOppositeX_y.T*(2*bool2[yrealAllRealInds[twoIntOppositeXInds]]-1)).T
####


#Testing a hypothesis
xIntercept = a/2 - b**2/(2*a)
#yline = mx*a/b - a**2/(2*b) + b/2 #between first quadrant a,b
yline = -mx*a/b + a**2/(2*b) - b/2 #between 4th quadrant a,b
xltXIntercept = np.where(mx <= xIntercept)[0]
xgtXIntercept = np.where(mx > xIntercept)[0]
yaboveInds = np.where((my > yline)*(mx > xIntercept))[0]
ybelowInds = np.where((my < yline)*(mx > xIntercept))[0]
intersect1 = np.intersect1d(np.concatenate((xltXIntercept,yaboveInds)),yrealAllRealInds)
intersect2 = np.intersect1d(ybelowInds,yrealAllRealInds)
print(len(intersect1))
print(len(yrealAllRealInds))
print(len(np.concatenate((xltXIntercept,yaboveInds))))
print(len(intersect2))
yaboveInds2 = np.where((my > yline)*(mx < xIntercept))[0]
ybelowInds2 = np.where((my < yline)*(mx < xIntercept))[0]
inter3 = np.intersect1d(yaboveInds2,yrealAllRealInds)
inter4 = np.intersect1d(ybelowInds2,yrealAllRealInds)
inter5 = np.intersect1d(xgtXIntercept,yrealAllRealInds)
inter6 = np.intersect1d(yaboveInds2,yrealImagInds)
inter7 = np.intersect1d(ybelowInds2,yrealImagInds)
inter8 = np.intersect1d(xgtXIntercept,yrealImagInds)
print(len(inter3))
print(len(inter4))
print(len(inter5))
print(len(inter6))
print(len(inter7))
print(len(inter8))


plt.close(1000)
plt.figure(1000)
# plt.scatter(x[yaboveInds],yline[yaboveInds] - y[yaboveInds],color='blue')
# plt.scatter(x[ybelowInds],yline[ybelowInds] - y[ybelowInds],color='red')
# yrealAllRealInds
# yrealImagInds
# plt.scatter(mx,my,color='blue')
# plt.scatter(mx[yrealAllRealInds]-xIntercept[yrealAllRealInds],my[yrealAllRealInds]-yline[yrealAllRealInds],color='blue',s=1)
# plt.scatter(mx[yrealImagInds]-xIntercept[yrealImagInds],my[yrealImagInds] - yline[yrealImagInds],color='red',s=1)
#plt.scatter(mx[xltXIntercept]/a[xltXIntercept],my[xltXIntercept]/b[xltXIntercept],color='blue',s=1)
# plt.scatter(mx[yrealAllRealInds]/a[yrealAllRealInds],my[yrealAllRealInds]-yline[yrealAllRealInds],color='blue',s=1)
# plt.scatter(mx[yrealImagInds]/a[yrealImagInds],my[yrealImagInds] - yline[yrealImagInds],color='red',s=1)

# ## GOOD KEEP
# plt.scatter(mx[xgtXIntercept]/a[xgtXIntercept],my[xgtXIntercept]-yline[xgtXIntercept],color='blue',s=1)
# plt.scatter(mx[yaboveInds2]/a[yaboveInds2],my[yaboveInds2]-yline[yaboveInds2],color='blue',s=1)
# plt.scatter(mx[ybelowInds2]/a[ybelowInds2],my[ybelowInds2]-yline[ybelowInds2],color='red',s=1)
# ##

## GOOD KEEP
plt.scatter(mx[yrealAllRealInds]/a[yrealAllRealInds],my[yrealAllRealInds]-yline[yrealAllRealInds],color='blue',s=1)
#DELETEplt.scatter(mx[yaboveInds2]/a[yaboveInds2],my[yaboveInds2]-yline[yaboveInds2],color='blue',s=1)
plt.scatter(mx[yrealImagInds]/a[yrealImagInds],my[yrealImagInds]-yline[yrealImagInds],color='red',s=1)
##

plt.plot([0,np.max(x)],[0,0],color='black')
#plt.yscale('log')
plt.xlim([0,0.7])
plt.ylim([-6,4])
plt.show(block=False)


#### ONLY 2 Real Inds (No Local Min/Max)
sepsInsideInds = np.where((maxSep[yrealImagInds] >= r[yrealImagInds]) & (r[yrealImagInds] >= minSep[yrealImagInds]))[0] #inds where r is within the minimum and maximum separations
only2RealInds = yrealImagInds[sepsInsideInds] #indicies of planets with only 2 real interesections
#lets try usnig separation bounds
#We will calculate separation at [0,+/-b] and [+/-a,0]
sepbp = np.sqrt(mx[only2RealInds]**2+(b[only2RealInds]+my[only2RealInds])**2)
sepbm = np.sqrt(mx[only2RealInds]**2+(b[only2RealInds]-my[only2RealInds])**2)
sepap = np.sqrt((a[only2RealInds]+mx[only2RealInds])**2+my[only2RealInds]**2)
sepam = np.sqrt((a[only2RealInds]-mx[only2RealInds])**2+my[only2RealInds]**2)

#Types of Star Locations In Projected Ellipse
typeInds0 = np.where(sepap < sepbp)[0]
typeInds1 = np.where(sepbp < sepam)[0]
typeInds2 = np.where((sepam < sepbp)*(sepbp < sepap)*(sepbm < sepam))[0]
typeInds3 = np.where(sepam < sepbm)[0]
print(len(typeInds0))
print(len(typeInds1))
print(len(typeInds2))
print(len(typeInds3))

xIntersectionsOnly2 = np.zeros((len(only2RealInds),2))
yIntersectionsOnly2 = np.zeros((len(only2RealInds),2))
#Separation Order For Each Location Type with Inds
#Type0
type0_0Inds = np.where((sepap < sepbp)*(r[only2RealInds] < sepbm))[0]
xIntersectionsOnly2[type0_0Inds] = np.real(xreal2[only2RealInds[type0_0Inds],0:2])
yIntersectionsOnly2[type0_0Inds] = np.real(yreal2[only2RealInds[type0_0Inds],0:2])
type0_1Inds = np.where((sepap < sepbp)*(sepbm < r[only2RealInds])*(r[only2RealInds] < sepam))[0]
xIntersectionsOnly2[type0_1Inds] = np.real(np.asarray([xreal2[only2RealInds[type0_1Inds],0],xreal2[only2RealInds[type0_1Inds],1]]).T) #-x is already in solution
yIntersectionsOnly2[type0_1Inds] = np.real(yreal2[only2RealInds[type0_1Inds],0:2])
type0_2Inds = np.where((sepap < sepbp)*(sepam < r[only2RealInds])*(r[only2RealInds] < sepap))[0]
xIntersectionsOnly2[type0_2Inds] = np.real(np.asarray([xreal2[only2RealInds[type0_2Inds],0],xreal2[only2RealInds[type0_2Inds],1]]).T) #-x is already in solution
yIntersectionsOnly2[type0_2Inds] = np.real(np.asarray([yreal2[only2RealInds[type0_2Inds],0],-yreal2[only2RealInds[type0_2Inds],1]]).T)
type0_3Inds = np.where((sepap < sepbp)*(sepap < r[only2RealInds])*(r[only2RealInds] < sepbp))[0]
xIntersectionsOnly2[type0_3Inds] = np.real(np.asarray([xreal2[only2RealInds[type0_3Inds],0],xreal2[only2RealInds[type0_3Inds],1]]).T) #-x is already in solution
yIntersectionsOnly2[type0_3Inds] = np.real(np.asarray([-yreal2[only2RealInds[type0_3Inds],0],-yreal2[only2RealInds[type0_3Inds],1]]).T)
type0_4Inds = np.where((sepap < sepbp)*(sepbp < r[only2RealInds]))[0]
xIntersectionsOnly2[type0_4Inds] = np.real(np.asarray([xreal2[only2RealInds[type0_4Inds],0],xreal2[only2RealInds[type0_4Inds],1]]).T) #-x is already in solution
yIntersectionsOnly2[type0_4Inds] = np.real(np.asarray([-yreal2[only2RealInds[type0_4Inds],0],-yreal2[only2RealInds[type0_4Inds],1]]).T)
#TODO FIX ALL THE STUFF HERE. FIRST FIND WHEN TYPE 1 Situations Occur (Should they have 4 real solutions always?)
type1_0Inds = np.where((sepbp < sepam)*(r[only2RealInds] < sepbm))[0]
xIntersectionsOnly2[type1_0Inds] = np.real(xreal2[only2RealInds[type1_0Inds],0:2])
yIntersectionsOnly2[type1_0Inds] = np.real(yreal2[only2RealInds[type1_0Inds],0:2])
type1_1Inds = np.where((sepbp < sepam)*(sepbm < r[only2RealInds])*(r[only2RealInds] < sepbp))[0]
xIntersectionsOnly2[type1_1Inds] = np.real(np.asarray([-xreal2[only2RealInds[type1_1Inds],0],xreal2[only2RealInds[type1_1Inds],1]]).T)
yIntersectionsOnly2[type1_1Inds] = np.real(np.asarray([yreal2[only2RealInds[type1_1Inds],0],yreal2[only2RealInds[type1_1Inds],1]]).T)
type1_2Inds = np.where((sepbp < sepam)*(sepbp < r[only2RealInds])*(r[only2RealInds] < sepam))[0]
xIntersectionsOnly2[type1_2Inds] = np.real(np.asarray([-xreal2[only2RealInds[type1_2Inds],0],xreal2[only2RealInds[type1_2Inds],1]]).T)
yIntersectionsOnly2[type1_2Inds] = np.real(np.asarray([yreal2[only2RealInds[type1_2Inds],0],yreal2[only2RealInds[type1_2Inds],1]]).T)
type1_3Inds = np.where((sepbp < sepam)*(sepam < r[only2RealInds])*(r[only2RealInds] < sepap))[0]
xIntersectionsOnly2[type1_3Inds] = np.real(np.asarray([-xreal2[only2RealInds[type1_3Inds],0],-xreal2[only2RealInds[type1_3Inds],1]]).T)
yIntersectionsOnly2[type1_3Inds] = np.real(np.asarray([yreal2[only2RealInds[type1_3Inds],0],-yreal2[only2RealInds[type1_3Inds],1]]).T)
type1_4Inds = np.where((sepbp < sepam)*(sepap < r[only2RealInds]))[0]
xIntersectionsOnly2[type1_4Inds] = np.real(np.asarray([-xreal2[only2RealInds[type1_4Inds],0],-xreal2[only2RealInds[type1_4Inds],1]]).T)
yIntersectionsOnly2[type1_4Inds] = np.real(np.asarray([-yreal2[only2RealInds[type1_4Inds],0],-yreal2[only2RealInds[type1_4Inds],1]]).T)
#Type1 sepbm, sepbp, sepam, sepap #NOTE: Type1 should not be yrealImagInds
#Type2 sepbm, sepam, sepbp, sepap
type2_0Inds = np.where((sepam < sepbp)*(sepbp < sepap)*(sepbm < sepam)*(r[only2RealInds] < sepbm))[0]
xIntersectionsOnly2[type2_0Inds] = np.real(np.asarray([xreal2[only2RealInds[type2_0Inds],0],xreal2[only2RealInds[type2_0Inds],1]]).T)
yIntersectionsOnly2[type2_0Inds] = np.real(np.asarray([yreal2[only2RealInds[type2_0Inds],0],yreal2[only2RealInds[type2_0Inds],1]]).T)
type2_1Inds = np.where((sepam < sepbp)*(sepbp < sepap)*(sepbm < sepam)*(sepbm < r[only2RealInds])*(r[only2RealInds] < sepam))[0]
xIntersectionsOnly2[type2_1Inds] = np.real(np.asarray([xreal2[only2RealInds[type2_1Inds],0],xreal2[only2RealInds[type2_1Inds],1]]).T)#-x is already in solution
yIntersectionsOnly2[type2_1Inds] = np.real(np.asarray([yreal2[only2RealInds[type2_1Inds],0],yreal2[only2RealInds[type2_1Inds],1]]).T)
type2_2Inds = np.where((sepam < sepbp)*(sepbp < sepap)*(sepbm < sepam)*(sepam < r[only2RealInds])*(r[only2RealInds] < sepbp))[0]
xIntersectionsOnly2[type2_2Inds] = np.real(np.asarray([xreal2[only2RealInds[type2_2Inds],0],xreal2[only2RealInds[type2_2Inds],1]]).T)#-x is already in solution
yIntersectionsOnly2[type2_2Inds] = np.real(np.asarray([yreal2[only2RealInds[type2_2Inds],0],-yreal2[only2RealInds[type2_2Inds],1]]).T)
type2_3Inds = np.where((sepam < sepbp)*(sepbp < sepap)*(sepbm < sepam)*(sepbp < r[only2RealInds])*(r[only2RealInds] < sepap))[0]
xIntersectionsOnly2[type2_3Inds] = np.real(np.asarray([xreal2[only2RealInds[type2_3Inds],0],xreal2[only2RealInds[type2_3Inds],1]]).T)#-x is already in solution
yIntersectionsOnly2[type2_3Inds] = np.real(np.asarray([yreal2[only2RealInds[type2_3Inds],0],-yreal2[only2RealInds[type2_3Inds],1]]).T)
type2_4Inds = np.where((sepam < sepbp)*(sepbp < sepap)*(sepbm < sepam)*(sepap < r[only2RealInds]))[0]
xIntersectionsOnly2[type2_4Inds] = np.real(np.asarray([xreal2[only2RealInds[type2_4Inds],0],xreal2[only2RealInds[type2_4Inds],1]]).T)#-x is already in solution
yIntersectionsOnly2[type2_4Inds] = np.real(np.asarray([-yreal2[only2RealInds[type2_4Inds],0],-yreal2[only2RealInds[type2_4Inds],1]]).T)
#Type3 sepam, sepbm, sepbp, sepap
type3_0Inds = np.where((sepam < sepbm)*(r[only2RealInds] < sepam))[0]
xIntersectionsOnly2[type3_0Inds] = np.real(np.asarray([xreal2[only2RealInds[type3_0Inds],0],xreal2[only2RealInds[type3_0Inds],1]]).T)
yIntersectionsOnly2[type3_0Inds] = np.real(np.asarray([yreal2[only2RealInds[type3_0Inds],0],yreal2[only2RealInds[type3_0Inds],1]]).T)
type3_1Inds = np.where((sepam < sepbm)*(sepam < r[only2RealInds])*(r[only2RealInds] < sepbm))[0]
xIntersectionsOnly2[type3_1Inds] = np.real(np.asarray([xreal2[only2RealInds[type3_1Inds],0],xreal2[only2RealInds[type3_1Inds],1]]).T)
yIntersectionsOnly2[type3_1Inds] = np.real(np.asarray([yreal2[only2RealInds[type3_1Inds],0],-yreal2[only2RealInds[type3_1Inds],1]]).T)
type3_2Inds = np.where((sepam < sepbm)*(sepbm < r[only2RealInds])*(r[only2RealInds] < sepbp))[0]
xIntersectionsOnly2[type3_2Inds] = np.real(np.asarray([xreal2[only2RealInds[type3_2Inds],0],xreal2[only2RealInds[type3_2Inds],1]]).T)#-x is already in solution
yIntersectionsOnly2[type3_2Inds] = np.real(np.asarray([yreal2[only2RealInds[type3_2Inds],0],-yreal2[only2RealInds[type3_2Inds],1]]).T)
type3_3Inds = np.where((sepam < sepbm)*(sepbp < r[only2RealInds])*(r[only2RealInds] < sepap))[0]
xIntersectionsOnly2[type3_3Inds] = np.real(np.asarray([xreal2[only2RealInds[type3_3Inds],0],xreal2[only2RealInds[type3_3Inds],1]]).T)#-x is already in solution
yIntersectionsOnly2[type3_3Inds] = np.real(np.asarray([yreal2[only2RealInds[type3_3Inds],0],-yreal2[only2RealInds[type3_3Inds],1]]).T)
type3_4Inds = np.where((sepam < sepbm)*(sepap < r[only2RealInds]))[0]
xIntersectionsOnly2[type3_4Inds] = np.real(np.asarray([xreal2[only2RealInds[type3_4Inds],0],xreal2[only2RealInds[type3_4Inds],1]]).T)#-x is already in solution
yIntersectionsOnly2[type3_4Inds] = np.real(np.asarray([-yreal2[only2RealInds[type3_4Inds],0],-yreal2[only2RealInds[type3_4Inds],1]]).T)
#Quadrant Star Belongs to
bool1 = x > 0
bool2 = y > 0
#Quadrant 1 if T,T
#Quadrant 2 if F,T
#Quadrant 3 if F,F
#Quadrant 4 if T,F
xIntersectionsOnly2 = (xIntersectionsOnly2.T*(2*bool1[only2RealInds]-1)).T
yIntersectionsOnly2 = (yIntersectionsOnly2.T*(2*bool2[only2RealInds]-1)).T
################################################
allIndsUsed = np.concatenate((type0_0Inds,type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,
        type2_0Inds,type2_1Inds,type2_2Inds,type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds))

#works
ind = yrealAllRealInds[fourIntInds[0]]
#works
ind = yrealAllRealInds[twoIntSameYInds[0]]
#works
ind = yrealAllRealInds[twoIntOppositeXInds[1]]
#in progress
#GOOD KEEP ind = only2RealInds[4]
ind = only2RealInds[4]



#type0 checks out
ind = only2RealInds[typeInds0[0]]
# #type1 checks out
# ind = only2RealInds[typeInds1[0]]
# #type2 checks out
# ind = only2RealInds[typeInds2[0]]
# #type3 checks out
# ind = only2RealInds[typeInds3[0]]

#type0_0 #OK
ind = only2RealInds[type0_0Inds[0]]#works
ind = only2RealInds[type0_1Inds[0]]#works
ind = only2RealInds[type0_2Inds[0]]#works
ind = only2RealInds[type0_3Inds[0]]#works
ind = only2RealInds[type0_4Inds[0]]#works
#type1 skipping since empty
#type2 #OK
ind = only2RealInds[type2_0Inds[0]]#works
ind = only2RealInds[type2_1Inds[0]]#works
ind = only2RealInds[type2_2Inds[0]]#works
ind = only2RealInds[type2_3Inds[0]]#works
ind = only2RealInds[type2_4Inds[0]]#works
#type3
#ind = only2RealInds[type3_0Inds[0]]#works
ind = only2RealInds[type3_1Inds[0]]#works
#ind = only2RealInds[type3_2Inds[0]]#works
#ind = only2RealInds[type3_3Inds[0]]#works
#ind = only2RealInds[type3_4Inds[0]]#works


# tmp = (mx[only2RealInds]**2 + my[only2RealInds]**2)/a[only2RealInds]
# plt.figure()
# plt.scatter(mx[only2RealInds],my[only2RealInds],s=1,color='blue')
# plt.show(block=False)


num=960
plt.close(num)
fig = plt.figure(num=num)
ca = plt.gca()
ca.axis('equal')
#DELETEplt.scatter([xreal[ind,0],xreal[ind,1],xreal[ind,2],xreal[ind,3]], [yreal[ind,0],yreal[ind,1],yreal[ind,2],yreal[ind,3]], color='purple')
plt.scatter([0],[0],color='orange')
## 3D Ellipse
vs = np.linspace(start=0,stop=2*np.pi,num=300)
#new plot stuff
Erange = np.linspace(start=0.,stop=2*np.pi,num=400)
plt.plot([-a[ind],a[ind]],[0,0],color='purple',linestyle='--') #major
plt.plot([0,0],[-b[ind],b[ind]],color='purple',linestyle='--') #minor
xellipsetmp = a[ind]*np.cos(Erange)
yellipsetmp = b[ind]*np.sin(Erange)
plt.plot(xellipsetmp,yellipsetmp,color='black')
plt.scatter(x[ind],y[ind],color='orange',marker='x')
if ind in only2RealInds[typeInds0]:
    plt.scatter(x[ind],y[ind],edgecolors='cyan',marker='o',s=64,facecolors='none')
if ind in only2RealInds[typeInds1]:
    plt.scatter(x[ind],y[ind],edgecolors='red',marker='o',s=64,facecolors='none')
if ind in only2RealInds[typeInds2]:
    plt.scatter(x[ind],y[ind],edgecolors='blue',marker='o',s=64,facecolors='none')
if ind in only2RealInds[typeInds3]:
    plt.scatter(x[ind],y[ind],edgecolors='magenta',marker='o',s=64,facecolors='none')

c_ae = a[ind]*np.sqrt(1-b[ind]**2/a[ind]**2)
plt.scatter([-c_ae,c_ae],[0,0],color='blue')

# #Plot Min Sep Circle
# x_circ = minSep[ind]*np.cos(vs)
# y_circ = minSep[ind]*np.sin(vs)
# plt.plot(x[ind]+x_circ,y[ind]+y_circ,color='cyan')
# #Plot Max Sep Circle
# x_circ2 = maxSep[ind]*np.cos(vs)
# y_circ2 = maxSep[ind]*np.sin(vs)
# plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='red')
#Plot Min Sep Ellipse Intersection
plt.scatter(minSepPoints_x[ind],minSepPoints_y[ind],color='cyan',marker='D')
#Plot Max Sep Ellipse Intersection
plt.scatter(maxSepPoints_x[ind],maxSepPoints_y[ind],color='red',marker='D')

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
    plt.scatter(lminSepPoints_x[tind], lminSepPoints_y[tind],color='magenta',marker='D')
    #### Plot Local Max Points
    plt.scatter(lmaxSepPoints_x[tind], lmaxSepPoints_y[tind],color='gold',marker='D')

#### r Intersection test
x_circ2 = np.cos(vs)
y_circ2 = np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='green')
#### Intersection Points
if ind in yrealAllRealInds[fourIntInds]:
    yind = np.where(yrealAllRealInds[fourIntInds] == ind)[0]
    plt.scatter(fourInt_x[yind],fourInt_y[yind], color='green',marker='o')
elif ind in yrealAllRealInds[twoIntSameYInds]: #Same Y
    yind = np.where(yrealAllRealInds[twoIntSameYInds] == ind)[0]
    plt.scatter(twoIntSameY_x[yind],twoIntSameY_y[yind], color='green',marker='o')
elif ind in yrealAllRealInds[twoIntOppositeXInds]: #Same X
    yind = np.where(yrealAllRealInds[twoIntOppositeXInds] == ind)[0]
    plt.scatter(twoIntOppositeX_x[yind],twoIntOppositeX_y[yind], color='green',marker='o')
    #### Type0
elif ind in only2RealInds[type0_0Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    print('plotted')
elif ind in only2RealInds[type0_1Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type0_2Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type0_3Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type0_4Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    #### Type1
elif ind in only2RealInds[type1_0Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type1_1Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type1_2Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type1_3Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type1_4Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    #### Type2
elif ind in only2RealInds[type2_0Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type2_1Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type2_2Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type2_3Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type2_4Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    #### Type3
elif ind in only2RealInds[type3_0Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type3_1Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type3_2Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type3_3Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
elif ind in only2RealInds[type3_4Inds]:
    gind = np.where(only2RealInds == ind)[0]
    plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')


#Plot Star Location Type Dividers
xran = np.linspace(start=(a[ind]*(a[ind]**2*(a[ind] - b[ind])*(a[ind] + b[ind]) - b[ind]**2*np.sqrt(3*a[ind]**4 + 2*a[ind]**2*b[ind]**2 + 3*b[ind]**4))/(2*(a[ind]**4 + b[ind]**4))),\
    stop=(a[ind]*(a[ind]**2*(a[ind] - b[ind])*(a[ind] + b[ind]) + b[ind]**2*np.sqrt(3*a[ind]**4 + 2*a[ind]**2*b[ind]**2 + 3*b[ind]**4))/(2*(a[ind]**4 + b[ind]**4))), num=3, endpoint=True)
ylineQ1 = xran*a[ind]/b[ind] - a[ind]**2/(2*b[ind]) + b[ind]/2 #between first quadrant a,b
ylineQ4 = -xran*a[ind]/b[ind] + a[ind]**2/(2*b[ind]) - b[ind]/2 #between 4th quadrant a,b
plt.plot(xran, ylineQ1, color='brown', linestyle='-.', )
plt.plot(-xran, ylineQ4, color='grey', linestyle='-.')
plt.plot(-xran, ylineQ1, color='orange', linestyle='-.')
plt.plot(xran, ylineQ4, color='red', linestyle='-.')


plt.xlim([-1.2*a[ind],1.2*a[ind]])
plt.ylim([-1.2*b[ind],1.2*b[ind]])

plt.show(block=False)











# xreal2[np.abs(np.imag(xreal2)) > 1e-4] = np.nan #There is evidence from below that the residual resulting from entiring solutions with 3e-5j results in 0+1e-20j therefore we will nan above 1e-4
# xreal2 = np.real(xreal2)
yreal2 = ellipseYFromX(xreal2, a, b)
seps2_0 = np.sqrt((xreal2[:,0]-x)**2 + (yreal2[:,0]-y)**2)
seps2_1 = np.sqrt((xreal2[:,1]-x)**2 + (yreal2[:,1]-y)**2)
seps2_2 = np.sqrt((xreal2[:,2]-x)**2 + (yreal2[:,2]-y)**2)
seps2_3 = np.sqrt((xreal2[:,3]-x)**2 + (yreal2[:,3]-y)**2)

# seps2_0 = np.sqrt((xreal2[:,0]-mx)**2 + (yreal2[:,0]-my)**2)
# seps2_1 = np.sqrt((xreal2[:,1]-mx)**2 + (yreal2[:,1]-my)**2)
# seps2_2 = np.sqrt((xreal2[:,2]-mx)**2 + (yreal2[:,2]-my)**2)
# seps2_3 = np.sqrt((xreal2[:,3]-mx)**2 + (yreal2[:,3]-my)**2)
seps2 = np.asarray([seps2_0,seps2_1,seps2_2,seps2_3]).T

#we are currently omitting all of these potential calculations so-long-as the following assert is never true
#assert ~np.any(p2+p3**2/12 == 0), 'Oops, looks like the sympy piecewise was true once!'

#### Root Types For Each Planet #######################################################
#ORDER HAS BEEN CHANGED
# If delta > 0 and P < 0 and D < 0 four roots all real or none
#allRealDistinctInds = np.where((delta > 0)*(P < 0)*(D2 < 0))[0] #METHOD 1, out of 10000, this found 1638, missing ~54
allRealDistinctInds = np.where(np.all(np.abs(np.imag(xreal2)) < 2.5*1e-5, axis=1))[0]#1e-9, axis=1))[0] #This found 1692
residual_allreal, isAll_allreal, maxRealResidual_allreal, maxImagResidual_allreal = checkResiduals(A,B,C,D,xreal2,allRealDistinctInds,4)
assert maxRealResidual_allreal < 1e-9, 'At least one all real residual is too large'
# If delta < 0, two distinct real roots, two complex
#DELETEtwoRealDistinctInds = np.where(delta < 0)[0]
#DELETE UNNECESSARYxrealsImag = np.abs(np.imag(xreal2))
xrealsImagInds = np.argsort(np.abs(np.imag(xreal2)),axis=1)
xrealsImagInds2 = np.asarray([xrealsImagInds[:,0],xrealsImagInds[:,1]])
xrealOfSmallest2Imags = np.real(xreal2[np.arange(len(a)),xrealsImagInds2]).T
ximagOfSmallest2Imags = np.imag(xreal2[np.arange(len(a)),xrealsImagInds2]).T
#~np.all(np.abs(np.imag(xreal2)) < 1e-9, axis=1) removes solutions with 4 distinct real roots
#The other two are thresholds that happend to work well once
indsOf2RealSols = np.where((np.abs(ximagOfSmallest2Imags[:,0]) < 2.5*1e-5)*(np.abs(ximagOfSmallest2Imags[:,1]) < 2.5*1e-5)*~np.all(np.abs(np.imag(xreal2)) < 2.5*1e-5, axis=1))[0]
#DELETElen(indsOf2RealSols) - len(allRealDistinctInds)
xrealsTwoRealSols = np.real(np.asarray([xreal2[indsOf2RealSols,xrealsImagInds2[0,indsOf2RealSols]],xreal2[indsOf2RealSols,xrealsImagInds2[1,indsOf2RealSols]]]).T)
residual_TwoRealSols, isAll_TwoRealSols, maxRealResidual_TwoRealSols, maxImagResidual_TwoRealSols = checkResiduals(A[indsOf2RealSols],B[indsOf2RealSols],C[indsOf2RealSols],D[indsOf2RealSols],xrealsTwoRealSols,np.arange(len(xrealsTwoRealSols)),2)
assert len(np.intersect1d(allRealDistinctInds,indsOf2RealSols)) == 0, 'There is intersection between Two Real Distinct and the 4 real solution inds, investigate'

#DELETE cruft
# twoRealDistinctInds2 = np.where(np.all(np.abs(np.imag(xreal2)) < 1e-9, axis=1))[0] #This found 1692
# twoRealSorting = np.argsort(np.abs(np.imag(xreal2[twoRealDistinctInds,:])),axis=1)
# tmpxReals = np.asarray([xreal2[np.arange(len(twoRealDistinctInds)),twoRealSorting[:,0]], xreal2[np.arange(len(twoRealDistinctInds)),twoRealSorting[:,1]]]).T

# If delta > 0 and (P < 0 or D < 0)
#allImagInds = np.where((delta > 0)*((P > 0)|(D2 > 0)))[0]
#allImagInds = np.where(np.all(np.abs(np.imag(xreal2)) >= 1e-9, axis=1))[0]
allImagInds = np.where(np.all(np.abs(np.imag(xreal2)) >= 2.5*1e-5, axis=1))[0]
assert len(np.intersect1d(allRealDistinctInds,allImagInds)) == 0, 'There is intersection between All Imag and the 4 real solution inds, investigate'
assert len(np.intersect1d(indsOf2RealSols,allImagInds)) == 0, 'There is intersection between All Imag and the Two Real Distinct solution inds, investigate'

# If delta == 0, multiple root
realDoubleRootTwoRealRootsInds = np.where((delta == 0)*(P < 0)*(D2 < 0)*(delta_0 != 0))[0] #delta=0 and P<0 and D2<0
realDoubleRootTwoComplexInds = np.where((delta == 0)*((D2 > 0)|((P > 0)*((D2 != 0)|(R != 0)))))[0] #delta=0 and (D>0 or (P>0 and (D!=0 or R!=0)))
tripleRootSimpleRootInds = np.where((delta == 0)*(delta_0 == 0)*(D2 !=0))[0]
twoRealDoubleRootsInds = np.where((delta == 0)*(D2 == 0)*(P < 0))[0]
twoComplexDoubleRootsInds = np.where((delta == 0)*(D2 == 0)*(P > 0)*(R == 0))[0]
fourIdenticalRealRootsInds = np.where((delta == 0)*(D2 == 0)*(delta_0 == 0))[0]

#DELETE cruft?
# #### Double checking root classification
# #twoRealDistinctInds #check that 2 of the 4 imags are below thresh
# numUnderThresh = np.sum(np.abs(np.imag(xreal2[twoRealDistinctInds])) > 1e-11, axis=1)
# indsUnderThresh = np.where(numUnderThresh != 2)[0]
# indsThatDontBelongIntwoRealDistinctInds = twoRealDistinctInds[indsUnderThresh]
# twoRealDistinctInds = np.delete(twoRealDistinctInds,indsThatDontBelongIntwoRealDistinctInds) #deletes the desired inds from aray
# #np.count_nonzero(numUnderThresh < 2)

#DELETE IN FUTURE
#The 1e-5 here gave me the number as the Imag count
#allRealDistinctInds2 = np.where(np.all(np.abs(np.imag(xreal2)) > 1e-5, axis=1))[0]
#allRealDistinctInds2 = np.where(np.all(np.abs(np.imag(xreal2)) > 1e-9, axis=1))[0]


#Number of Solutions of Each Type
numRootInds = [indsOf2RealSols,allRealDistinctInds,allImagInds,realDoubleRootTwoRealRootsInds,realDoubleRootTwoComplexInds,\
    tripleRootSimpleRootInds,twoRealDoubleRootsInds,twoComplexDoubleRootsInds,fourIdenticalRealRootsInds]

#Number of Roots of Each Type
lenNumRootsInds = [len(numRootInds[i]) for i in np.arange(len(numRootInds))]
assert len(indsOf2RealSols)+len(allRealDistinctInds)+len(allImagInds)-len(realDoubleRootTwoRealRootsInds), 'Number of roots does not add up, investigate'
########################################################################


# Calculate Residuals
# residual_0 = xreal2[:,0]**4 + A*xreal2[:,0]**3 + B*xreal2[:,0]**2 + C*xreal2[:,0] + D
# residual_1 = xreal2[:,1]**4 + A*xreal2[:,1]**3 + B*xreal2[:,1]**2 + C*xreal2[:,1] + D
# residual_2 = xreal2[:,2]**4 + A*xreal2[:,2]**3 + B*xreal2[:,2]**2 + C*xreal2[:,2] + D
# residual_3 = xreal2[:,3]**4 + A*xreal2[:,3]**3 + B*xreal2[:,3]**2 + C*xreal2[:,3] + D
# residual = np.asarray([residual_0, residual_1, residual_2, residual_3]).T
# #assert np.all((np.real(residual) < 1e-7)*(np.imag(residual) < 1e-7)), 'All residual are not less than 1e-7'
# del residual_0, residual_1, residual_2, residual_3
residual_all, isAll_all, maxRealResidual_all, maxImagResidual_all = checkResiduals(A,B,C,D,xreal2,np.arange(len(A)),4)



xfinal = np.zeros(xreal2.shape) + np.nan
# case 1 Two Real Distinct Inds
#find 2 xsols with smallest imag part
#xreal2[indsOf2RealSols[0]]
#ximags2 = np.imag(xreal2[indsOf2RealSols])
#ximags2smallImagInds = np.argsort(np.abs(ximags2),axis=1)[:,0:2] #sorts from smallest magnitude to largest magnitude
#xrealsTwoRealDistinct = np.asarray([xreal2[indsOf2RealSols,ximags2smallImagInds[:,0]], xreal2[indsOf2RealSols,ximags2smallImagInds[:,1]]]).T
xfinal[indsOf2RealSols,0:2] = xrealOfSmallest2Imags[indsOf2RealSols]#np.real(xrealsTwoRealDistinct)
#Check residuals
residual_case1, isAll_case1, maxRealResidual_case1, maxImagResidual_case1 = checkResiduals(A,B,C,D,xfinal,indsOf2RealSols,2)
#The following does not work
# residual_0 = np.real(xrealsTwoRealDistinct[:,0])**4 + A[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,0])**3 + B[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,0])**2 + C[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,0]) + D[twoRealDistinctInds]
# residual_1 = np.real(xrealsTwoRealDistinct[:,1])**4 + A[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,1])**3 + B[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,1])**2 + C[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,1]) + D[twoRealDistinctInds]
# residual = np.asarray([residual_0, residual_1]).T
# assert np.all((np.real(residual) < 1e-8)*(np.imag(residual) < 1e-8)), 'All residual are not less than 1e-8'
# del residual_0, residual_1
indsOfRebellious_0 = np.where(np.real(residual_case1[:,0]) > 1e-1)[0]
indsOfRebellious_1 = np.where(np.real(residual_case1[:,1]) > 1e-1)[0]
indsOfRebellious = np.unique(np.concatenate((indsOfRebellious_0,indsOfRebellious_1)))
xrealIndsOfRebellious = indsOf2RealSols[indsOfRebellious]

#residual = tmpxreal2[twoRealDistinctInds[0]]**4 + A[twoRealDistinctInds[0]]*tmpxreal2[twoRealDistinctInds[0]]**3 + B[twoRealDistinctInds[0]]*tmpxreal2[twoRealDistinctInds[0]]**2 + C[twoRealDistinctInds[0]]*tmpxreal2[twoRealDistinctInds[0]] + D[twoRealDistinctInds[0]]
#residual2 = xreal2[twoRealDistinctInds[0]]**4 + A[twoRealDistinctInds[0]]*xreal2[twoRealDistinctInds[0]]**3 + B[twoRealDistinctInds[0]]*xreal2[twoRealDistinctInds[0]]**2 + C[twoRealDistinctInds[0]]*xreal2[twoRealDistinctInds[0]] + D[twoRealDistinctInds[0]]
#residual3 = np.real(xreal2[twoRealDistinctInds[0]])**4 + A[twoRealDistinctInds[0]]*np.real(xreal2[twoRealDistinctInds[0]])**3 + B[twoRealDistinctInds[0]]*np.real(xreal2[twoRealDistinctInds[0]])**2 + C[twoRealDistinctInds[0]]*np.real(xreal2[twoRealDistinctInds[0]]) + D[twoRealDistinctInds[0]]

#currently getting intersection points that are not physically possible

# case 2 All Real Distinct Inds
xfinal[allRealDistinctInds] = np.real(xreal2[allRealDistinctInds])
# residual_0 = xfinal[allRealDistinctInds,0]**4 + A[allRealDistinctInds]*xfinal[allRealDistinctInds,0]**3 + B[allRealDistinctInds]*xfinal[allRealDistinctInds,0]**2 + C[allRealDistinctInds]*xfinal[allRealDistinctInds,0] + D[allRealDistinctInds]
# residual_1 = xfinal[allRealDistinctInds,1]**4 + A[allRealDistinctInds]*xfinal[allRealDistinctInds,1]**3 + B[allRealDistinctInds]*xfinal[allRealDistinctInds,1]**2 + C[allRealDistinctInds]*xfinal[allRealDistinctInds,1] + D[allRealDistinctInds]
# residual_2 = xfinal[allRealDistinctInds,2]**4 + A[allRealDistinctInds]*xfinal[allRealDistinctInds,2]**3 + B[allRealDistinctInds]*xfinal[allRealDistinctInds,2]**2 + C[allRealDistinctInds]*xfinal[allRealDistinctInds,2] + D[allRealDistinctInds]
# residual_3 = xfinal[allRealDistinctInds,3]**4 + A[allRealDistinctInds]*xfinal[allRealDistinctInds,3]**3 + B[allRealDistinctInds]*xfinal[allRealDistinctInds,3]**2 + C[allRealDistinctInds]*xfinal[allRealDistinctInds,3] + D[allRealDistinctInds]
# residual = np.asarray([residual_0, residual_1, residual_2, residual_3]).T
# assert np.all((np.real(residual) < 1e-7)*(np.imag(residual) < 1e-7)), 'All residual, All Real Distinct, are not less than 1e-7'
# del residual_0, residual_1, residual_2, residual_3
residual_case2, isAll_case2, maxRealResidual_case2, maxImagResidual_case2 = checkResiduals(A,B,C,D,xfinal,allRealDistinctInds,4)

# case 3 All Imag Inds
#NO REAL ROOTS
#xreal2[allImagInds[0]]

# # case 4 a real double root and 2 real solutions (2 real solutions which are identical and 2 other real solutions)
# #xreal2[realDoubleRootTwoRealRootsInds[0]]
# ximags2 = np.imag(xreal2[realDoubleRootTwoRealRootsInds])
# ximags2smallImagInds = np.argsort(np.abs(ximags2),axis=1)[:,0:2] #sorts from smallest magnitude to largest magnitude
# xrealDoubleRootTwoRealRoots = np.asarray([xreal2[realDoubleRootTwoRealRootsInds,ximags2smallImagInds[:,0]], xreal2[realDoubleRootTwoRealRootsInds,ximags2smallImagInds[:,1]]]).T
# xfinal[realDoubleRootTwoRealRootsInds,0:2] = np.real(xrealDoubleRootTwoRealRoots)



# # case 5 a real double root
# #xreal2[realDoubleRootTwoComplexInds[0]]
# ximags2 = np.imag(xreal2[realDoubleRootTwoComplexInds])
# ximags2smallImagInds = np.argsort(np.abs(ximags2),axis=1)[:,0:2] #sorts from smallest magnitude to largest magnitude
# xrealDoubleRootTwoComplex = np.asarray([xreal2[realDoubleRootTwoComplexInds,ximags2smallImagInds[:,0]], xreal2[realDoubleRootTwoComplexInds,ximags2smallImagInds[:,1]]]).T
# xfinal[realDoubleRootTwoComplexInds,0:2] = np.real(xrealDoubleRootTwoComplex)

yfinal = ellipseYFromX(xfinal, a, b)
s_mpr, s_pmr, minSepr, maxSepr = calculateSeparations(xfinal, yfinal, mx, my)
#TODO need to do what I did for the sepsMinMaxLminLmax function for x, y coordinate determination

# #s_pm = np.asarray([np.sqrt((xreal[:,0]+mx)**2 + (yreal[:,0]-my)**2), np.sqrt((xreal[:,1]+mx)**2 + (yreal[:,1]-my)**2), np.sqrt((xreal[:,2]+mx)**2 + (yreal[:,2]-my)**2), np.sqrt((xreal[:,3]+mx)**2 + (yreal[:,3]-my)**2)]).T
# s_pmr = np.asarray([np.sqrt((xfinal[:,0]+mx)**2 + (yfinal[:,0]-my)**2), np.sqrt((xfinal[:,1]+mx)**2 + (yfinal[:,1]-my)**2), np.sqrt((xfinal[:,2]+mx)**2 + (yfinal[:,2]-my)**2), np.sqrt((xfinal[:,3]+mx)**2 + (yfinal[:,3]-my)**2)]).T


print('nanmin')
print(np.nanmin(s_mpr[allRealDistinctInds]))
print(np.nanmin(s_pmr[allRealDistinctInds]))
print(np.nanmin(minSepr[allRealDistinctInds]))
print(np.nanmin(maxSepr[allRealDistinctInds]))
print(np.nanmin(s_mpr[indsOf2RealSols]))
print(np.nanmin(s_pmr[indsOf2RealSols]))
print(np.nanmin(minSepr[indsOf2RealSols]))
print(np.nanmin(maxSepr[indsOf2RealSols]))
print('nanmax')
print(np.nanmax(s_mpr[allRealDistinctInds]))
print(np.nanmax(s_pmr[allRealDistinctInds]))
print(np.nanmax(minSepr[allRealDistinctInds]))
print(np.nanmax(maxSepr[allRealDistinctInds]))
print(np.nanmax(s_mpr[indsOf2RealSols]))
print(np.nanmax(s_pmr[indsOf2RealSols]))
print(np.nanmax(minSepr[indsOf2RealSols]))
print(np.nanmax(maxSepr[indsOf2RealSols]))

PMRsols = np.asarray([np.nanmin(s_mpr[allRealDistinctInds],axis=1),np.nanmin(s_pmr[allRealDistinctInds],axis=1),np.nanmin(minSepr[allRealDistinctInds],axis=1),np.nanmin(maxSepr[allRealDistinctInds],axis=1)]).T

bool1 = x > 0
bool2 = y > 0
s_mpr2, s_pmr2, minSepr2, maxSepr2 = calculateSeparations(xfinal, yfinal, x, y)


np.nanmin(minSepr,axis=1)

#### Notes
#If r < smin, then all imag
#if r > smin and r > slmin, then 2 real.
#if r > slmin and r < slmax, then 4 real.
#if r < smax and r > slmax, then 2 real.
#if r > smax, then all imag.



#DELETEminSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, s_mplminSeps, s_mplmaxSeps = sepsMinMaxLminLmax(minSep, maxSep, s_mp, xreal, yreal, x, y)

