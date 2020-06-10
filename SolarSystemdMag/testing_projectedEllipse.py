import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as random
from sys import getsizeof
import time
from astropy import constants as const
import astropy.units as u

#### Randomly Generate Orbits
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))
filename = 'HabEx_CKL2_PPKL2.json'
filename = 'WFIRSTcycle6core.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
PPop = sim.PlanetPopulation
n = 10**5 #Dean's nice computer can go up to 10**8 what can atuin go up to?
inc, W, w = PPop.gen_angles(n,None)
inc = inc.to('rad').value
inc[np.where(inc>np.pi/2)[0]] = np.pi - inc[np.where(inc>np.pi/2)[0]]
W = W.to('rad').value
w = w.to('rad').value
sma, e, p, Rp = PPop.gen_plan_params(n)
sma = sma.to('AU').value
####

#### SAVED PLANET FOR Plot 3D Ellipse to 2D Ellipse Projection Diagram
ind = 23 #22
sma[ind] = 1.2164387563540457
e[ind] = 0.531071885292766
w[ind] = 3.477496280463054
W[ind] = 5.333215834002414
inc[ind] = 1.025093642138022
####
#### SAVED PLANET FOR Plot Projected, derotated, centered ellipse 
derotatedInd = 33
sma[ind] = 5.738800898338014
e[ind] = 0.29306873405223816
w[ind] = 4.436383063578559
W[ind] = 4.240810639711751
inc[ind] = 1.072680736014668
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
del start1, stop1

# Checks
if not np.all(dmajorp < sma):
    print("Not all Semi-major axis of the projected ellipse are less than the original 3D ellipse, caused by circular orbits required for circular orbits")
    assert np.all(sma - dmajorp > -1e-12), "Not all Semi-major axis of the projected ellipse are less than the original 3D ellipse" #required for circular orbits
assert np.all(dminorp < dmajorp), "All projected Semi-minor axes are less than all projected semi-major axes"

#### Plotting Projected Ellipse
start2 = time.time()
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
stop2 = time.time()
print('stop2: ' + str(stop2-start2))
del start2, stop2
plt.close(877)
####


#### Plot 3D Ellipse to 2D Ellipse Projection Diagram
start3 = time.time()
def plot3DEllipseto2DEllipseProjectionDiagram(ind, sma, e, W, w, inc, num):
    """
    """
    plt.close(num)
    fig = plt.figure(num)
    ax = fig.add_subplot(111, projection='3d')

    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
    x_3Dellipse = r[0,0,:]
    y_3Dellipse = r[1,0,:]
    z_3Dellipse = r[2,0,:]
    ax.plot(x_3Dellipse,y_3Dellipse,z_3Dellipse,color='black',label='Planet Orbit',linewidth=2)
    min_z = np.min(z_3Dellipse)

    ## Central Sun
    ax.scatter(0,0,0,color='orange',marker='o',s=25) #of 3D ellipse
    ax.text(0,0,0.15*np.abs(min_z), 'F', None)
    ax.plot([0,0],[0,0],[0,1.3*min_z],color='orange',linestyle='--',linewidth=2) #connecting line
    ax.scatter(0,0,1.3*min_z,color='orange',marker='x',s=25) #of 2D ellipse
    ax.text(0,0,1.5*min_z, 'F\'', None)

    ## Plot 3D Ellipse semi-major/minor axis
    rper = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],0.) #planet position perigee
    rapo = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],np.pi) #planet position apogee
    ax.plot([rper[0][0],rapo[0][0]],[rper[1][0],rapo[1][0]],[rper[2][0],rapo[2][0]],color='purple', linestyle='-',linewidth=2) #3D Ellipse Semi-major axis
    ax.scatter(rper[0][0],rper[1][0],rper[2][0],color='grey',marker='D',s=25) #3D Ellipse Perigee Diamond
    ax.text(1.2*rper[0][0],1.2*rper[1][0],rper[2][0], 'A', None)
    ax.scatter(rper[0][0],rper[1][0],1.3*min_z,color='blue',marker='D',s=25) #2D Ellipse Perigee Diamond
    ax.text(1.1*rper[0][0],1.1*rper[1][0],1.3*min_z, 'A\'', None)
    ax.plot([rper[0][0],rper[0][0]],[rper[1][0],rper[1][0]],[rper[2][0],1.3*min_z],color='grey',linestyle='--',linewidth=2) #3D to 2D Ellipse Perigee Diamond
    ax.scatter(rapo[0][0],rapo[1][0],rapo[2][0],color='grey', marker='D',s=25) #3D Ellipse Apogee Diamond
    ax.text(1.1*rapo[0][0],1.1*rapo[1][0],1.2*rapo[2][0], 'B', None)
    ax.scatter(rapo[0][0],rapo[1][0],1.3*min_z,color='blue',marker='D',s=25) #2D Ellipse Perigee Diamond
    ax.text(1.1*rapo[0][0],1.1*rapo[1][0],1.3*min_z, 'B\'', None)

    ax.plot([rapo[0][0],rapo[0][0]],[rapo[1][0],rapo[1][0]],[rapo[2][0],1.3*min_z],color='grey',linestyle='--',linewidth=2) #3D to 2D Ellipse Apogee Diamond
    rbp = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],np.arccos((np.cos(np.pi/2)-e[ind])/(1-e[ind]*np.cos(np.pi/2)))) #3D Ellipse E=90
    rbm = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],-np.arccos((np.cos(-np.pi/2)-e[ind])/(1-e[ind]*np.cos(-np.pi/2)))) #3D Ellipse E=-90
    ax.plot([rbp[0][0],rbm[0][0]],[rbp[1][0],rbm[1][0]],[rbp[2][0],rbm[2][0]],color='purple', linestyle='-',linewidth=2) #
    ax.scatter(rbp[0][0],rbp[1][0],rbp[2][0],color='grey',marker='D',s=25) #3D ellipse minor +
    ax.text(1.1*rbp[0][0],1.1*rbp[1][0],1.2*rbp[2][0], 'C', None)
    ax.scatter(rbp[0][0],rbp[1][0],1.3*min_z,color='blue',marker='D',s=25) #2D ellipse minor+ projection
    ax.text(1.1*rbp[0][0],1.1*rbp[1][0],1.3*min_z, 'C\'', None)
    ax.plot([rbp[0][0],rbp[0][0]],[rbp[1][0],rbp[1][0]],[rbp[2][0],1.3*min_z],color='grey',linestyle='--',linewidth=2) #3D to 2D Ellipse minor + Diamond
    ax.scatter(rbm[0][0],rbm[1][0],rbm[2][0],color='grey', marker='D',s=25) #3D ellipse minor -
    ax.text(1.1*rbm[0][0],0.5*(rbm[1][0]-Op[1][ind]),rbm[2][0]+0.05, 'D', None)
    ax.scatter(rbm[0][0],rbm[1][0],1.3*min_z,color='blue', marker='D',s=25) #2D ellipse minor- projection
    ax.text(1.1*rbm[0][0],0.5*(rbm[1][0]-Op[1][ind]),1.3*min_z, 'D\'', None)
    ax.plot([rbm[0][0],rbm[0][0]],[rbm[1][0],rbm[1][0]],[rbm[2][0],1.3*min_z],color='grey',linestyle='--',linewidth=2) #3D to 2D Ellipse minor - Diamond

    ## Plot K, H, P
    ax.scatter(0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            0.6*(rapo[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2,color='green', marker='x',s=36) #Point along OB, point H
    ax.text(0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            0.6*(rapo[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2+0.1,'H', None) #Point along OB, point H
    xscaletmp = np.sqrt(1-.6**2)
    ax.scatter(xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            xscaletmp*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2,color='green',marker='x',s=36) #point along OC, point K
    ax.text(xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            xscaletmp*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2+0.1,'K',None) #point along OC, point K
    angtmp = np.arctan2(0.6,xscaletmp)
    ax.scatter(np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2,\
        np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2,\
        np.cos(angtmp)*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rapo[2][0] - (rper[2][0] + rapo[2][0])/2)*np.sin(angtmp) + (rper[2][0] + rapo[2][0])/2,color='green',marker='o',s=25) #Point P on 3D Ellipse
    ax.text(np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2,\
        np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2,\
        np.cos(angtmp)*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rapo[2][0] - (rper[2][0] + rapo[2][0])/2)*np.sin(angtmp) + (rper[2][0] + rapo[2][0])/2+0.1,'P',None) #Point P on 3D Ellipse
    ## Plot KP, HP
    ax.plot([0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2],\
            [0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2],\
            [0.6*(rapo[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2,np.cos(angtmp)*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rapo[2][0] - (rper[2][0] + rapo[2][0])/2)*np.sin(angtmp) + (rper[2][0] + rapo[2][0])/2],linestyle=':',color='black') #H to P line
    ax.plot([xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2],\
            [xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2],\
            [xscaletmp*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2,np.cos(angtmp)*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rapo[2][0] - (rper[2][0] + rapo[2][0])/2)*np.sin(angtmp) + (rper[2][0] + rapo[2][0])/2],linestyle=':',color='black') #K to P line
    ## Plot K', H', P'
    ax.scatter(0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            1.3*min_z,color='magenta', marker='x',s=36) #Point along O'B', point H'
    ax.text(0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            1.3*min_z-0.1,'H\'',None) #Point along O'B', point H'
    xscaletmp = np.sqrt(1-.6**2)
    ax.scatter(xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            1.3*min_z,color='magenta',marker='x',s=36) #point along O'C', point K'
    ax.text(xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            1.3*min_z-0.1,'K\'',None) #point along O'C', point K'
    angtmp = np.arctan2(0.6,xscaletmp)
    ax.scatter(np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2,\
        np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2,\
        1.3*min_z,color='magenta',marker='o',s=25) #Point P' on 2D Ellipse
    ax.text(np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2,\
        np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2,\
        1.3*min_z-0.1,'P\'',None) #Point P' on 2D Ellipse
    ## Plot K'P', H'P'
    ax.plot([0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2],\
            [0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2],\
            [1.3*min_z,1.3*min_z],linestyle=':',color='black') #H to P line
    ax.plot([xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2],\
            [xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2],\
            [1.3*min_z,1.3*min_z],linestyle=':',color='black') #K to P line
    ## Plot PP', KK', HH'
    ax.plot([np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2,np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2],\
        [np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2,np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2],\
        [np.cos(angtmp)*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rapo[2][0] - (rper[2][0] + rapo[2][0])/2)*np.sin(angtmp) + (rper[2][0] + rapo[2][0])/2,1.3*min_z],color='black',linestyle=':') #PP'
    ax.plot([0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2],\
            [0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2],\
            [0.6*(rapo[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2,1.3*min_z],color='black',linestyle=':') #HH'
    ax.plot([xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2],\
            [xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2],\
            [xscaletmp*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2,1.3*min_z],color='black',linestyle=':') #KK'



    ## Plot Conjugate Diameters
    ax.plot([rbp[0][0],rbm[0][0]],[rbp[1][0],rbm[1][0]],[1.3*min_z,1.3*min_z],color='blue',linestyle='-',linewidth=2) #2D ellipse minor+ projection
    ax.plot([rper[0][0],rapo[0][0]],[rper[1][0],rapo[1][0]],[1.3*min_z,1.3*min_z],color='blue',linestyle='-',linewidth=2) #2D Ellipse Perigee Diamond

    ## Plot Ellipse Center
    ax.scatter((rper[0][0] + rapo[0][0])/2,(rper[1][0] + rapo[1][0])/2,(rper[2][0] + rapo[2][0])/2,color='grey',marker='o',s=36) #3D Ellipse
    ax.text(1.2*(rper[0][0] + rapo[0][0])/2,1.2*(rper[1][0] + rapo[1][0])/2,1.31*(rper[2][0] + rapo[2][0])/2, 'O', None)
    ax.scatter(Op[0][ind],Op[1][ind], 1.3*min_z, color='grey', marker='o',s=25) #2D Ellipse Center
    ax.text(1.2*(rper[0][0] + rapo[0][0])/2,1.2*(rper[1][0] + rapo[1][0])/2,1.4*min_z, 'O\'', None)
    ax.plot([(rper[0][0] + rapo[0][0])/2,Op[0][ind]],[(rper[1][0] + rapo[1][0])/2,Op[1][ind]],[(rper[2][0] + rapo[2][0])/2,1.3*min_z],color='grey',linestyle='--',linewidth=2) #Plot ) to )''
    print('a: ' + str(np.round(sma[ind],2)) + ' e: ' + str(np.round(e[ind],2)) + ' W: ' + str(np.round(W[ind],2)) + ' w: ' + str(np.round(w[ind],2)) + ' i: ' + str(np.round(inc[ind],2)) +\
         ' Psi: ' + str(np.round(Psi[ind],2)) + ' psi: ' + str(np.round(psi[ind],2)))# + ' theta: ' + str(np.round(theta[ind],2)))


    ang2 = (theta_OpQ_X[ind]+theta_OpQp_X[ind])/2
    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    ax.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    ax.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)

    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    ax.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)
    ax.scatter([dmajorpx1,dmajorpx2,dminorpx1,dminorpx2],[dmajorpy1,dmajorpy2,dminorpy1,dminorpy2],[1.3*min_z,1.3*min_z,1.3*min_z,1.3*min_z],color='black',marker='o',s=25,zorder=6)
    ax.text(1.05*dmajorpx1,1.05*dmajorpy1,1.3*min_z, 'I', None)#(dmajorpx1,dmajorpy1,0))
    ax.text(1.1*dmajorpx2,1.1*dmajorpy2,1.3*min_z, 'R', None)#(dmajorpx2,dmajorpy2,0))
    ax.text(1.05*dminorpx1,0.1*(dminorpy1-Op[1][ind]),1.3*min_z, 'S', None)#(dminorpx1,dminorpy1,0))
    ax.text(1.05*dminorpx2,1.05*dminorpy2,1.3*min_z, 'T', None)#(dminorpx2,dminorpy2,0))
    #ax.text(x,y,z, label, zdir)
    x_projEllipse = Op[0][ind] + dmajorp[ind]*np.cos(vs)*np.cos(ang2) - dminorp[ind]*np.sin(vs)*np.sin(ang2)
    y_projEllipse = Op[1][ind] + dmajorp[ind]*np.cos(vs)*np.sin(ang2) + dminorp[ind]*np.sin(vs)*np.cos(ang2)
    ax.plot(x_projEllipse,y_projEllipse,1.3*min_z*np.ones(len(vs)), color='red', linestyle='-',zorder=5,linewidth=2)

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

num = 666999888777
plot3DEllipseto2DEllipseProjectionDiagram(ind, sma, e, W, w, inc, num=num)
stop3 = time.time()
print('stop3: ' + str(stop3-start3))
del start3, stop3
plt.close(num)
####


#### Create Projected Ellipse Conjugate Diameters and QQ' construction diagram
start4 = time.time()
def plotEllipseMajorAxisFromConjugate(ind, sma, e, W, w, inc, num):
    """ Plots the Q and Q' points as well as teh line 
    """
    plt.close(num)
    fig = plt.figure(num)
    ax = plt.gca()

    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
    x_3Dellipse = r[0,0,:]
    y_3Dellipse = r[1,0,:]
    z_3Dellipse = r[2,0,:]
    ax.plot(x_3Dellipse,y_3Dellipse,color='black',label='Planet Orbit',linewidth=2)
    min_z = np.min(z_3Dellipse)

    ## Central Sun
    ax.scatter(0,0,color='orange',marker='x',s=25,zorder=20) #of 2D ellipse
    ax.text(0-.1,0-.1, 'F\'', None)

    ## Plot 3D Ellipse semi-major/minor axis
    rper = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],0.) #planet position perigee
    rapo = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],np.pi) #planet position apogee
    ax.scatter(rper[0][0],rper[1][0],color='blue',marker='D',s=25,zorder=25) #2D Ellipse Perigee Diamond
    ax.text(1.1*rper[0][0],1.1*rper[1][0], 'A\'', None)
    ax.scatter(rapo[0][0],rapo[1][0],color='blue',marker='D',s=25,zorder=25) #2D Ellipse Perigee Diamond
    ax.text(1.1*rapo[0][0]-0.1,1.1*rapo[1][0], 'B\'', None)

    rbp = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],np.arccos((np.cos(np.pi/2)-e[ind])/(1-e[ind]*np.cos(np.pi/2)))) #3D Ellipse E=90
    rbm = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],-np.arccos((np.cos(-np.pi/2)-e[ind])/(1-e[ind]*np.cos(-np.pi/2)))) #3D Ellipse E=-90
    ax.plot([rbp[0][0],rbm[0][0]],[rbp[1][0],rbm[1][0]],color='purple', linestyle='-',linewidth=2) #
    ax.scatter(rbp[0][0],rbp[1][0],color='blue',marker='D',s=25,zorder=20) #2D ellipse minor+ projection
    ax.text(1.1*rbp[0][0]-.01,1.1*rbp[1][0]-.05, 'C\'', None)
    ax.scatter(rbm[0][0],rbm[1][0],color='blue', marker='D',s=25,zorder=20) #2D ellipse minor- projection
    ax.text(1.1*rbm[0][0],0.5*(rbm[1][0]-Op[1][ind])-.05, 'D\'', None)

    ## Plot QQ' Line
    #rapo[0][0],rapo[1][0] #B'
    #rbp[0][0],rbp[1][0] #C'
    #Op[0][ind],Op[1][ind] #O'
    tmp = np.asarray([-(rbp[1][0]-Op[1][ind]),(rbp[0][0]-Op[0][ind])])
    QQp_hat = tmp/np.linalg.norm(tmp)
    dOpCp = np.sqrt((rbp[0][0]-Op[0][ind])**2 + (rbp[1][0]-Op[1][ind])**2)
    #Q = Bp - dOpCp*QQp_hat
    Qx = rapo[0][0] - dOpCp*QQp_hat[0]
    Qy = rapo[1][0] - dOpCp*QQp_hat[1]
    #Qp = Bp + DOpCp*QQp_hat
    Qpx = rapo[0][0] + dOpCp*QQp_hat[0]
    Qpy = rapo[1][0] + dOpCp*QQp_hat[1]
    ax.plot([Op[0][ind],Qx],[Op[1][ind],Qy],color='black',linestyle='-',linewidth=2,zorder=29) #OpQ
    ax.plot([Op[0][ind],Qpx],[Op[1][ind],Qpy],color='black',linestyle='-',linewidth=2,zorder=29) #OpQp
    ax.plot([Qx,Qpx],[Qy,Qpy],color='grey',linestyle='-',linewidth=2,zorder=29)
    ax.scatter([Qx,Qpx],[Qy,Qpy],color='grey',marker='s',s=36,zorder=30)
    ax.text(Qx,Qy-0.1,'Q', None)
    ax.text(Qpx,Qpy+0.05,'Q\'', None)

    ## Plot Conjugate Diameters
    ax.plot([rbp[0][0],rbm[0][0]],[rbp[1][0],rbm[1][0]],color='blue',linestyle='-',linewidth=2) #2D ellipse minor+ projection
    ax.plot([rper[0][0],rapo[0][0]],[rper[1][0],rapo[1][0]],color='blue',linestyle='-',linewidth=2) #2D Ellipse Perigee Diamond

    ## Plot Ellipse Center
    ax.scatter(Op[0][ind],Op[1][ind], color='grey', marker='o',s=25,zorder=30) #2D Ellipse Center
    ax.text(1.2*(rper[0][0] + rapo[0][0])/2,1.2*(rper[1][0] + rapo[1][0])/2+0.05, 'O\'', None)
    print('a: ' + str(np.round(sma[ind],2)) + ' e: ' + str(np.round(e[ind],2)) + ' W: ' + str(np.round(W[ind],2)) + ' w: ' + str(np.round(w[ind],2)) + ' i: ' + str(np.round(inc[ind],2)) +\
         ' Psi: ' + str(np.round(Psi[ind],2)) + ' psi: ' + str(np.round(psi[ind],2)))# + ' theta: ' + str(np.round(theta[ind],2)))

    ang2 = (theta_OpQ_X[ind]+theta_OpQp_X[ind])/2
    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    ax.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='purple',linestyle='-',linewidth=2)
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    ax.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='purple',linestyle='-',linewidth=2)

    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    ax.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='purple',linestyle='-',linewidth=2)
    ax.scatter([dmajorpx1,dmajorpx2,dminorpx1,dminorpx2],[dmajorpy1,dmajorpy2,dminorpy1,dminorpy2],color='black',marker='o',s=25,zorder=6)
    ax.text(1.05*dmajorpx1,1.05*dmajorpy1, 'I', None)
    ax.text(1.1*dmajorpx2,1.1*dmajorpy2, 'R', None)
    ax.text(1.05*dminorpx1,0.1*(dminorpy1-Op[1][ind])-.05, 'S', None)
    ax.text(1.05*dminorpx2-0.1,1.05*dminorpy2-.075, 'T', None)
    x_projEllipse = Op[0][ind] + dmajorp[ind]*np.cos(vs)*np.cos(ang2) - dminorp[ind]*np.sin(vs)*np.sin(ang2)
    y_projEllipse = Op[1][ind] + dmajorp[ind]*np.cos(vs)*np.sin(ang2) + dminorp[ind]*np.sin(vs)*np.cos(ang2)
    ax.plot(x_projEllipse,y_projEllipse, color='red', linestyle='-',zorder=5,linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(False)
    xmax = np.max([np.abs(rper[0][0]),np.abs(rapo[0][0]),np.abs(1.3*min_z), np.abs(Qpx), np.abs(Qx)])
    ax.scatter([-xmax,xmax],[-xmax,xmax],color=None,alpha=0)
    ax.set_xlim(-0.99*xmax+Op[0][ind],0.99*xmax+Op[0][ind])
    ax.set_ylim(-0.99*xmax+Op[1][ind],0.99*xmax+Op[1][ind])
    ax.set_axis_off() #removes axes
    ax.axis('equal')
    plt.show(block=False)

num = 3335555888
plotEllipseMajorAxisFromConjugate(ind, sma, e, W, w, inc, num)
stop4 = time.time()
print('stop4: ' + str(stop4-start4))
del start4, stop4
plt.close(num)
####


#### Derotate Ellipse
start5 = time.time()
x, y, Phi = derotatedEllipse(theta_OpQ_X, theta_OpQp_X, Op)
stop5 = time.time()
print('stop5: ' + str(stop5-start5))
del start5, stop5
a = dmajorp
b = dminorp
mx = np.abs(x) #x converted to a strictly positive value
my = np.abs(y) #y converted to a strictly positive value

start6 = time.time()
def plotDerotatedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, a, b, x, y, num=879):
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

num=880
plotDerotatedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, a, b, x, y, num)
stop6 = time.time()
print('stop6: ' + str(stop6-start6))
del start6, stop6
plt.close(num)

#### Calculate X,Y Position of Minimum and Maximums with Quartic
start7 = time.time()
A, B, C, D = quarticCoefficients_smin_smax_lmin_lmax(a.astype('complex128'), b, mx, my)
xreal, delta, P, D2, R, delta_0 = quarticSolutions_ellipse_to_Quarticipynb(A.astype('complex128'), B, C, D)
del A, B, C, D #delting for memory efficiency
assert np.max(np.nanmin(np.abs(np.imag(xreal)),axis=1)) < 1e-5, 'At least one row has min > 1e-5' #this ensures each row has a solution
#print(w[np.argmax(np.nanmin(np.abs(np.imag(xreal)),axis=1))]) #prints the argument of perigee (assert above fails on 1.57 or 1.5*pi)
#Failure of the above occured where w=4.712 which is approx 1.5pi
#NOTE: originally 1e-15 but there were some with x=1e-7 and w=pi/2, 5e-6 from 
tind = np.argmax(np.nanmin(np.abs(np.imag(xreal)),axis=1)) #DELETE
tinds = np.argsort(np.nanmin(np.abs(np.imag(xreal)),axis=1)) #DELETE
del tind, tinds #DELETE
xreal.real = np.abs(xreal)
stop7 = time.time()
print('stop7: ' + str(stop7-start7))
del stop7, start7
#DELETEprintKOE(ind,a,e,W,w,inc)

#Technically, each row must have at least 2 solutions, but whatever
start8 = time.time()
yreal = ellipseYFromX(xreal.astype('complex128'), a, b)
stop8 = time.time()
print('stop8: ' + str(stop8-start8))
del start8, stop8


#### Calculate Minimum, Maximum, Local Minimum, Local Maximum Separations
start9 = time.time()
minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds = smin_smax_slmin_slmax(n, xreal, yreal, mx, my, x, y)
lminSepPoints_x = np.real(lminSepPoints_x)
lminSepPoints_y = np.real(lminSepPoints_y)
lmaxSepPoints_x = np.real(lmaxSepPoints_x)
lmaxSepPoints_y = np.real(lmaxSepPoints_y)
stop9 = time.time()
print('stop9: ' + str(stop9-start9))
del start9, stop9

##### Plot Proving Rerotation method works
start10 = time.time()
def plotReorientationMethod(ind, sma, e, W, w, inc, Op, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, num):
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
    plt.scatter(minSepPoints_x[ind],minSepPoints_y[ind],color='magenta')
    ux = np.cos(Phi[ind])*minSepPoints_x[ind] - np.sin(Phi[ind])*minSepPoints_y[ind] + Op[0][ind] 
    uy = np.sin(Phi[ind])*minSepPoints_x[ind] + np.cos(Phi[ind])*minSepPoints_y[ind] + Op[1][ind] 
    plt.scatter(ux,uy,color='green')

    plt.show(block=False)

num=883
plotReorientationMethod(ind, sma, e, W, w, inc, Op, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, num)
stop10 = time.time()
print('stop10: ' + str(stop10-start10))
del start10, stop10
plt.close(num)
####


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
####


#### Memory Usage
memories = [getsizeof(inc),getsizeof(W),getsizeof(w),getsizeof(sma),getsizeof(e),getsizeof(p),getsizeof(Rp),getsizeof(dmajorp),getsizeof(dminorp),getsizeof(Psi),getsizeof(psi),getsizeof(theta_OpQ_X),\
getsizeof(theta_OpQp_X),getsizeof(dmajorp_v2),getsizeof(dminorp_v2),getsizeof(Psi_v2),getsizeof(psi_v2),getsizeof(Op),getsizeof(x),getsizeof(y),getsizeof(Phi),getsizeof(a),getsizeof(b),\
getsizeof(mx),getsizeof(my),getsizeof(xreal),getsizeof(yreal),getsizeof(minSepPoints_x),getsizeof(minSepPoints_y),\
getsizeof(maxSepPoints_x),getsizeof(maxSepPoints_y),getsizeof(lminSepPoints_x),getsizeof(lminSepPoints_y),getsizeof(lmaxSepPoints_x),getsizeof(lmaxSepPoints_y),getsizeof(minSep),\
getsizeof(maxSep)]#,getsizeof(s_mplminSeps),getsizeof(s_mplmaxSeps)]
totalMemoryUsage = np.sum(memories)
print('Total Data Used: ' + str(totalMemoryUsage/10**9) + ' GB')
####

#### Ellipse Circle Intersection #######################################################################
start11 = time.time()
r = np.ones(len(a))
def ellipseCircleIntersections(r, a, b, mx, my, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds):
    #### Testing ellipse_to_Quartic solution
    if r == None:
        r = np.ones(len(a),dtype='complex128')
    #ARE THESE NECESSARY?
    a.astype('complex128')
    b.astype('complex128')
    mx.astype('complex128')
    my.astype('complex128')
    r.astype('complex128')
    A, B, C, D = quarticCoefficients_ellipse_to_Quarticipynb(a, b, mx, my, r)
    xreal2, delta, P, D2, R, delta_0 = quarticSolutions_ellipse_to_Quarticipynb(A, B, C, D)
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
    assert np.max(np.imag(xreal2[yrealAllRealInds[fourIntInds]])) < 1e-5, 'an Imag component of the all reals is too high!' #uses to be 1e-7 but would occasionally get errors so relaxing

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
    #DELETEassert np.max(np.imag(xreal2[yrealAllRealInds[twoIntSameYInds],0])) < 1e-8, 'An Imaginary component was too large' #Was 1e-12, but is now 1e-8
    if len(twoIntSameYInds) > 0:
        assert np.max(np.imag(xreal2[yrealAllRealInds[twoIntSameYInds],0])) < 1e-8, 'An Imaginary component was too large' #Was 1e-12, but is now 1e-8
    twoIntSameY_x[:,0] = np.real(xreal2[yrealAllRealInds[twoIntSameYInds],0])
    smallImagInds = np.where(np.abs(np.imag(xreal2[yrealAllRealInds[twoIntSameYInds],1])) < 1e-9)[0]
    largeImagInds = np.where(np.abs(np.imag(xreal2[yrealAllRealInds[twoIntSameYInds],1])) > 1e-9)[0]
    twoIntSameY_x[smallImagInds,1] = np.real(xreal2[yrealAllRealInds[twoIntSameYInds[smallImagInds]],1])
    twoIntSameY_x[largeImagInds,1] = np.real(xreal2[yrealAllRealInds[twoIntSameYInds[largeImagInds]],3])
    twoIntSameY_y = np.asarray([np.sqrt(b[yrealAllRealInds[twoIntSameYInds]]**2*(1-twoIntSameY_x[:,0]**2/a[yrealAllRealInds[twoIntSameYInds]]**2)),\
            np.sqrt(b[yrealAllRealInds[twoIntSameYInds]]**2*(1-twoIntSameY_x[:,1]**2/a[yrealAllRealInds[twoIntSameYInds]]**2))]).T
    #Adjust for Quadrant Star Belongs to
    twoIntSameY_x = (twoIntSameY_x.T*(2*bool1[yrealAllRealInds[twoIntSameYInds]]-1)).T
    twoIntSameY_y = (twoIntSameY_y.T*(2*bool2[yrealAllRealInds[twoIntSameYInds]]-1)).T
    ####
    #### Two Intersection Points twoIntOppositeXInds
    twoIntOppositeX_x = np.zeros((len(twoIntOppositeXInds),2))
    twoIntOppositeX_y = np.zeros((len(twoIntOppositeXInds),2))
    assert np.max(np.imag(xreal2[yrealAllRealInds[twoIntOppositeXInds],0])) < 1e-8, '' #was 1e-12 but caused problems
    twoIntOppositeX_x[:,0] = np.real(xreal2[yrealAllRealInds[twoIntOppositeXInds],0])
    twoIntOppositeX_x[:,1] = np.real(xreal2[yrealAllRealInds[twoIntOppositeXInds],1])
    twoIntOppositeX_y = np.asarray([np.sqrt(b[yrealAllRealInds[twoIntOppositeXInds]]**2*(1-np.abs(twoIntOppositeX_x[:,0])**2/a[yrealAllRealInds[twoIntOppositeXInds]]**2)),\
            np.sqrt(b[yrealAllRealInds[twoIntOppositeXInds]]**2*(1-np.abs(twoIntOppositeX_x[:,1])**2/a[yrealAllRealInds[twoIntOppositeXInds]]**2))]).T
    #twoIntOppositeX_x = (twoIntOppositeX_x.T*(-2*bool1[yrealAllRealInds[twoIntOppositeXInds]]+1)).T
    twoIntOppositeX_y[:,1] = -twoIntOppositeX_y[:,1]
    #Adjust for Quadrant Star Belongs to
    twoIntOppositeX_x = (twoIntOppositeX_x.T*(2*bool1[yrealAllRealInds[twoIntOppositeXInds]]-1)).T
    twoIntOppositeX_y = (twoIntOppositeX_y.T*(2*bool2[yrealAllRealInds[twoIntOppositeXInds]]-1)).T
    ####

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
    #Adjust for Quadrant Star Belongs to
    xIntersectionsOnly2 = (xIntersectionsOnly2.T*(2*bool1[only2RealInds]-1)).T
    yIntersectionsOnly2 = (yIntersectionsOnly2.T*(2*bool2[only2RealInds]-1)).T
    ################################################
    allIndsUsed = np.concatenate((type0_0Inds,type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,
            type2_0Inds,type2_1Inds,type2_2Inds,type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds))


    return a, b, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
        minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
        twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2, twoIntSameYInds,\
        type0_0Inds,type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,\
        type2_0Inds,type2_1Inds,type2_2Inds,type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,\
        allIndsUsed

a, b, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
        minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
        twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2, twoIntSameYInds,\
        type0_0Inds,type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,\
        type2_0Inds,type2_1Inds,type2_2Inds,type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,\
        allIndsUsed = ellipseCircleIntersections(None, a, b, mx, my, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds)
stop11 = time.time()
print('stop11: ' + str(stop11-start11))
del start11, stop11
####

#DELETE This was a test to see if there was any correlation between number of solutions and star location type
# #### Calculate Star Location Types for 
# def classifyStarTypeInds(mx,my,a,b):
#     sepbp = np.sqrt(mx**2+(b+my)**2)
#     sepbm = np.sqrt(mx**2+(b-my)**2)
#     sepap = np.sqrt((a+mx)**2+my**2)
#     sepam = np.sqrt((a-mx)**2+my**2)

#     #Types of Star Locations In Projected Ellipse
#     typeInds0 = np.where(sepap < sepbp)[0]
#     typeInds1 = np.where(sepbp < sepam)[0]
#     typeInds2 = np.where((sepam < sepbp)*(sepbp < sepap)*(sepbm < sepam))[0]
#     typeInds3 = np.where(sepam < sepbm)[0]
#     return typeInds0, typeInds1, typeInds2, typeInds3
# alltypeInds0, alltypeInds1, alltypeInds2, alltypeInds3 = classifyStarTypeInds(mx,my,a,b)
# print(len(alltypeInds0) + len(alltypeInds1) + len(alltypeInds2) + len(alltypeInds3))
# print(np.intersect1d(alltypeInds0,only2RealInds))
# print(np.intersect1d(alltypeInds1,only2RealInds))
# print(np.intersect1d(alltypeInds2,only2RealInds))
# print(np.intersect1d(alltypeInds3,only2RealInds))

# print(len(np.intersect1d(alltypeInds0,yrealAllRealInds)))
# print(len(np.intersect1d(alltypeInds1,yrealAllRealInds)))
# print(len(np.intersect1d(alltypeInds2,yrealAllRealInds)))
# print(len(np.intersect1d(alltypeInds3,yrealAllRealInds)))

# print(len(np.intersect1d(alltypeInds0,yrealImagInds)))
# print(len(np.intersect1d(alltypeInds1,yrealImagInds)))
# print(len(np.intersect1d(alltypeInds2,yrealImagInds)))
# print(len(np.intersect1d(alltypeInds3,yrealImagInds)))

# print(len(alltypeInds0))
# print(len(alltypeInds1))
# print(len(alltypeInds2))
# print(len(alltypeInds3))
# ####

#### Generalized Correct Ellipse Circle Intersection Fixer
def intersectionFixer_pm(x, y, sep_xlocs, sep_ylocs, afflictedIndsxy, rs):
    """
    """
    # seps = np.sqrt((sep_xlocs-x[afflictedIndsxy])**2 + (sep_ylocs-y[afflictedIndsxy])**2) #calculate error for all TwoIntSameY
    # error = np.abs(np.sort(-np.abs(np.ones(len(seps)) - seps))) #calculate error for all TwoIntSameY
    # largeErrorInds = np.where(error > 1e-7)[0] #get inds of large errors
    # indsToFix = np.argsort(-np.abs(np.ones(len(seps)) - seps))[largeErrorInds] #inds of TwoIndSameY
    # seps_deciding = np.sqrt((sep_xlocs[indsToFix]-x[afflictedIndsxy[indsToFix]])**2 + (-sep_ylocs[indsToFix]-y[afflictedIndsxy[indsToFix]])**2) #calculate error for indsToFix
    # error_deciding = -np.abs(np.ones(len(seps_deciding)) - seps_deciding) #calculate errors for swapping y of the candidated to swap y for
    # indsToSwap = np.where(np.abs(error_deciding) < np.abs(error[indsToFix]))[0] #find where the errors produced by swapping y is lowered
    # sep_ylocs[indsToFix[indsToSwap]] = -sep_ylocs[indsToFix[indsToSwap]] #here we fix the y values where they should be fixed by swapping y values
    # seps = np.sqrt((sep_xlocs-x[afflictedIndsxy])**2 + (sep_ylocs-y[afflictedIndsxy])**2)
    # error = np.abs(np.sort(-np.abs(np.ones(len(seps)) - seps)))
    # indsToFix = np.argsort(-np.abs(np.ones(len(seps)) - seps))[np.where(error > 1e-7)[0]]

    seps = np.sqrt((sep_xlocs-x[afflictedIndsxy])**2 + (sep_ylocs-y[afflictedIndsxy])**2) #calculate error for all TwoIntSameY
    error = np.abs(rs - seps) #calculate error for all TwoIntSameY
    indsToFix = np.where(error > 1e-7)[0] #get inds of large errors
    #DELETEerror = np.abs(np.sort(-np.abs(rs[afflictedIndsxy] - seps))) #calculate error for all TwoIntSameY
    #DELETElargeErrorInds = np.where(error > 1e-7)[0] #get inds of large errors
    #DELETEindsToFix = np.argsort(-np.abs(rs[afflictedIndsxy] - seps))[largeErrorInds] #inds of TwoIndSameY
    if len(indsToFix) == 0: #There are no inds to fix
        return sep_xlocs, sep_ylocs

    seps_decidingpm = np.sqrt((sep_xlocs[indsToFix]-x[afflictedIndsxy[indsToFix]])**2 + (-sep_ylocs[indsToFix]-y[afflictedIndsxy[indsToFix]])**2) #calculate error for indsToFix
    seps_decidingmp = np.sqrt((-sep_xlocs[indsToFix]-x[afflictedIndsxy[indsToFix]])**2 + (sep_ylocs[indsToFix]-y[afflictedIndsxy[indsToFix]])**2) #calculate error for indsToFix
    seps_decidingmm = np.sqrt((-sep_xlocs[indsToFix]-x[afflictedIndsxy[indsToFix]])**2 + (-sep_ylocs[indsToFix]-y[afflictedIndsxy[indsToFix]])**2) #calculate error for indsToFix

    #error_decidingpm = -np.abs(np.ones(len(seps_decidingpm)) - seps_decidingpm) #calculate errors for swapping y of the candidated to swap y for
    #error_decidingmp = -np.abs(np.ones(len(seps_decidingmp)) - seps_decidingmp) #calculate errors for swapping y of the candidated to swap y for
    #error_decidingmm = -np.abs(np.ones(len(seps_decidingmm)) - seps_decidingmm) #calculate errors for swapping y of the candidated to swap y for
    #DELETEerror_deciding = np.array([error,-np.abs(np.ones(len(seps_decidingpm)) - seps_decidingpm),-np.abs(np.ones(len(seps_decidingmp)) - seps_decidingmp),-np.abs(np.ones(len(seps_decidingmm)) - seps_decidingmm)])
    error_deciding = np.stack((error[indsToFix],np.abs(rs[indsToFix] - seps_decidingpm),np.abs(rs[indsToFix] - seps_decidingmp),np.abs(rs[indsToFix] - seps_decidingmm)),axis=1)


    minErrorInds = np.argmin(error_deciding,axis=1)

    tmpxlocs = np.asarray([sep_xlocs,sep_xlocs,-sep_xlocs,-sep_xlocs]).T
    sep_xlocs[indsToFix] = tmpxlocs[indsToFix,minErrorInds]
    tmpylocs = np.asarray([sep_ylocs,-sep_ylocs,sep_ylocs,-sep_ylocs]).T
    sep_ylocs[indsToFix] = tmpylocs[indsToFix,minErrorInds]

    #indsToSwap = np.where(np.abs(error_deciding) < np.abs(error[indsToFix]))[0] #find where the errors produced by swapping y is lowered
    #sep_ylocs[indsToFix[indsToSwap]] = -sep_ylocs[indsToFix[indsToSwap]] #here we fix the y values where they should be fixed by swapping y values
    #seps = np.sqrt((sep_xlocs-x[afflictedIndsxy])**2 + (sep_ylocs-y[afflictedIndsxy])**2)
    #error = np.abs(np.sort(-np.abs(np.ones(len(seps)) - seps)))
    #indsToFix = np.argsort(-np.abs(np.ones(len(seps)) - seps))[np.where(error > 1e-7)[0]]

    return sep_xlocs, sep_ylocs


#### Correct Ellipse Circle Intersections fourInt1 ####################################
fourInt_x[:,0], fourInt_y[:,0] = intersectionFixer_pm(x, y, fourInt_x[:,0], fourInt_y[:,0], yrealAllRealInds[fourIntInds], np.ones(len(fourIntInds)))
fourInt_x[:,1], fourInt_y[:,1] = intersectionFixer_pm(x, y, fourInt_x[:,1], fourInt_y[:,1], yrealAllRealInds[fourIntInds], np.ones(len(fourIntInds)))
#### Correct Ellipse Circle Intersections twoIntSameY0
twoIntSameY_x[:,0], twoIntSameY_y[:,0] = intersectionFixer_pm(x, y, twoIntSameY_x[:,0], twoIntSameY_y[:,0], yrealAllRealInds[twoIntSameYInds], np.ones(len(twoIntSameYInds)))
#### Correct Ellipse Circle Intersections twoIntSameY1 
twoIntSameY_x[:,1], twoIntSameY_y[:,1] = intersectionFixer_pm(x, y, twoIntSameY_x[:,1], twoIntSameY_y[:,1], yrealAllRealInds[twoIntSameYInds], np.ones(len(twoIntSameYInds)))
#### Correct Ellipse Circle Intersections twoIntOppositeX0
twoIntOppositeX_x[:,0], twoIntOppositeX_y[:,0] = intersectionFixer_pm(x, y, twoIntOppositeX_x[:,0], twoIntOppositeX_y[:,0], yrealAllRealInds[twoIntOppositeXInds], np.ones(len(twoIntOppositeXInds)))
#### Correct Ellipse Circle Intersections twoIntOppositeX1 
twoIntOppositeX_x[:,1], twoIntOppositeX_y[:,1] = intersectionFixer_pm(x, y, twoIntOppositeX_x[:,1], twoIntOppositeX_y[:,1], yrealAllRealInds[twoIntOppositeXInds], np.ones(len(twoIntOppositeXInds)))
#### COULD RUN ON OTHER CASES #########################################################



# ind = yrealAllRealInds[twoIntOppositeXInds[indsToFix[0]]]
# plotRerotatedFromNus(ind, sma[ind], e[ind], W[ind], w[ind], inc[ind], Op[:,ind], yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds,\
#     nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
#     twoIntSameY_x, twoIntSameY_y, num=8001)
#del seps_TwoIntSameY1, errors_TwoIntSameY1, indsToFix
####


#Testing plotting inds
ind = yrealAllRealInds[fourIntInds[0]]#works
if len(twoIntSameYInds) > 0:
    ind = yrealAllRealInds[twoIntSameYInds[0]]#works
ind = yrealAllRealInds[twoIntOppositeXInds[1]]#works
#GOOD KEEP ind = only2RealInds[4]
ind = only2RealInds[4]

#type0_0 #OK
# ind = only2RealInds[type0_0Inds[0]]#works
# ind = only2RealInds[type0_1Inds[0]]#works
# ind = only2RealInds[type0_2Inds[0]]#works
# ind = only2RealInds[type0_3Inds[0]]#works
# ind = only2RealInds[type0_4Inds[0]]#works
#type1 skipping since empty
#type2 #OK
# ind = only2RealInds[type2_0Inds[0]]#works
# ind = only2RealInds[type2_1Inds[0]]#works
# ind = only2RealInds[type2_2Inds[0]]#works
# ind = only2RealInds[type2_3Inds[0]]#works
# ind = only2RealInds[type2_4Inds[0]]#works
#type3
#ind = only2RealInds[type3_0Inds[0]]#works
if len(only2RealInds) > 0:
    ind = only2RealInds[type3_1Inds[0]]#works
#ind = only2RealInds[type3_2Inds[0]]#works
#ind = only2RealInds[type3_3Inds[0]]#works
#ind = only2RealInds[type3_4Inds[0]]#works

#### Plot Derotated Intersections, Min/Max, and Star Location Type Bounds
start12 = time.time()
def plotDerotatedIntersectionsMinMaxStarLocBounds(ind, a, b, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
    minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
    twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
    type0_0Inds, type0_1Inds, type0_2Inds, type0_3Inds, type0_4Inds, type1_0Inds, type1_1Inds, type1_2Inds, type1_3Inds, type1_4Inds,\
    type2_0Inds, type2_1Inds, type2_2Inds, type2_3Inds, type2_4Inds, type3_0Inds, type3_1Inds, type3_2Inds, type3_3Inds, type3_4Inds, num):
    """
    """
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

    # Plot Star Location Type Dividers
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

num = 960
plotDerotatedIntersectionsMinMaxStarLocBounds(ind, a, b, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
        minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
        twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
        type0_0Inds, type0_1Inds, type0_2Inds, type0_3Inds, type0_4Inds, type1_0Inds, type1_1Inds, type1_2Inds, type1_3Inds, type1_4Inds,\
        type2_0Inds, type2_1Inds, type2_2Inds, type2_3Inds, type2_4Inds, type3_0Inds, type3_1Inds, type3_2Inds, type3_3Inds, type3_4Inds, num)
stop12 = time.time()
print('stop12: ' + str(stop12-start12))
del start12, stop12
plt.close(num)
####

#### Plot Derotated Ellipse Separation Extrema
start12_1 = time.time()
def plotDerotatedExtrema(ind, a, b, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
    minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
    twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2, num):
    """
    """
    plt.close(num)
    fig = plt.figure(num=num)
    ca = plt.gca()
    ca.axis('equal')
    plt.scatter([0],[0],color='orange',zorder=25)
    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    #new plot stuff
    Erange = np.linspace(start=0.,stop=2*np.pi,num=400)
    plt.plot([-a[ind],a[ind]],[0,0],color='purple',linestyle='--') #major
    plt.plot([0,0],[-b[ind],b[ind]],color='purple',linestyle='--') #minor
    xellipsetmp = a[ind]*np.cos(Erange)
    yellipsetmp = b[ind]*np.sin(Erange)
    plt.plot(xellipsetmp,yellipsetmp,color='black')
    plt.scatter(x[ind],y[ind],color='orange',marker='x',zorder=30)
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
    plt.scatter(minSepPoints_x[ind],minSepPoints_y[ind],color='cyan',marker='D',zorder=25)
    #Plot Max Sep Ellipse Intersection
    plt.scatter(maxSepPoints_x[ind],maxSepPoints_y[ind],color='red',marker='D',zorder=25)
    #### Plot star to min line
    plt.plot([x[ind],minSepPoints_x[ind]], [y[ind],minSepPoints_y[ind]],color='cyan',zorder=25)
    #### Plot star to max line
    plt.plot([x[ind],maxSepPoints_x[ind]], [y[ind],maxSepPoints_y[ind]],color='red',zorder=25)

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
        plt.scatter(lminSepPoints_x[tind], lminSepPoints_y[tind],color='magenta',marker='D',zorder=25)
        #### Plot Local Max Points
        plt.scatter(lmaxSepPoints_x[tind], lmaxSepPoints_y[tind],color='gold',marker='D',zorder=25)
        #### Plot star to local min line
        plt.plot([x[ind],lminSepPoints_x[tind]], [y[ind],lminSepPoints_y[tind]],color='magenta',zorder=25)
        #### Plot star to local max line
        plt.plot([x[ind],lmaxSepPoints_x[tind]], [y[ind],lmaxSepPoints_y[tind]],color='gold',zorder=25)

    plt.xlim([-1.2*a[ind],1.2*a[ind]])
    plt.ylim([-1.2*b[ind],1.2*b[ind]])
    plt.xlabel('X Position In Image Plane (AU)', weight='bold')
    plt.ylabel('Y Position In Image Plane (AU)', weight='bold')
    plt.show(block=False)

num = 961
plotDerotatedExtrema(derotatedInd, a, b, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
        minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
        twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2, num)
stop12_1 = time.time()
print('stop12_1: ' + str(stop12_1-start12_1))
del start12_1, stop12_1
####


#### Rerotate Extrema and Intersection Points
start13 = time.time()
def rerotateExtremaAndIntersectionPoints(minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y,\
    fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
    Phi, Op, yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds):
    """
    Rotate the intersection points to the original projected ellipse
    """
    minSepPoints_x_dr = np.zeros(len(minSepPoints_x))
    minSepPoints_y_dr = np.zeros(len(minSepPoints_y))
    maxSepPoints_x_dr = np.zeros(len(maxSepPoints_x))
    maxSepPoints_y_dr = np.zeros(len(maxSepPoints_y))
    lminSepPoints_x_dr = np.zeros(len(lminSepPoints_x))
    lminSepPoints_y_dr = np.zeros(len(lminSepPoints_y))
    lmaxSepPoints_x_dr = np.zeros(len(lmaxSepPoints_x))
    lmaxSepPoints_y_dr = np.zeros(len(lmaxSepPoints_y))
    fourInt_x_dr = np.zeros((len(fourInt_x),4))
    fourInt_y_dr = np.zeros((len(fourInt_y),4))
    twoIntSameY_x_dr = np.zeros((len(twoIntSameY_x),2))
    twoIntSameY_y_dr = np.zeros((len(twoIntSameY_y),2))
    twoIntOppositeX_x_dr = np.zeros((len(twoIntOppositeX_x),2))
    twoIntOppositeX_y_dr = np.zeros((len(twoIntOppositeX_y),2))
    xIntersectionsOnly2_dr = np.zeros((len(xIntersectionsOnly2),2))
    yIntersectionsOnly2_dr = np.zeros((len(yIntersectionsOnly2),2))


    minSepPoints_x_dr, minSepPoints_y_dr = rerotateEllipsePoints(minSepPoints_x, minSepPoints_y,Phi,Op[0],Op[1])
    maxSepPoints_x_dr, maxSepPoints_y_dr = rerotateEllipsePoints(maxSepPoints_x, maxSepPoints_y,Phi,Op[0],Op[1])
    lminSepPoints_x_dr, lminSepPoints_y_dr = rerotateEllipsePoints(lminSepPoints_x, lminSepPoints_y,Phi[yrealAllRealInds],Op[0][yrealAllRealInds],Op[1][yrealAllRealInds])
    lmaxSepPoints_x_dr, lmaxSepPoints_y_dr = rerotateEllipsePoints(lmaxSepPoints_x, lmaxSepPoints_y,Phi[yrealAllRealInds],Op[0][yrealAllRealInds],Op[1][yrealAllRealInds])
    fourInt_x_dr[:,0], fourInt_y_dr[:,0] = rerotateEllipsePoints(fourInt_x[:,0], fourInt_y[:,0],Phi[yrealAllRealInds[fourIntInds]],Op[0][yrealAllRealInds[fourIntInds]],Op[1][yrealAllRealInds[fourIntInds]])
    fourInt_x_dr[:,1], fourInt_y_dr[:,1] = rerotateEllipsePoints(fourInt_x[:,1], fourInt_y[:,1],Phi[yrealAllRealInds[fourIntInds]],Op[0][yrealAllRealInds[fourIntInds]],Op[1][yrealAllRealInds[fourIntInds]])
    fourInt_x_dr[:,2], fourInt_y_dr[:,2] = rerotateEllipsePoints(fourInt_x[:,2], fourInt_y[:,2],Phi[yrealAllRealInds[fourIntInds]],Op[0][yrealAllRealInds[fourIntInds]],Op[1][yrealAllRealInds[fourIntInds]])
    fourInt_x_dr[:,3], fourInt_y_dr[:,3] = rerotateEllipsePoints(fourInt_x[:,3], fourInt_y[:,3],Phi[yrealAllRealInds[fourIntInds]],Op[0][yrealAllRealInds[fourIntInds]],Op[1][yrealAllRealInds[fourIntInds]])
    twoIntSameY_x_dr[:,0], twoIntSameY_y_dr[:,0] = rerotateEllipsePoints(twoIntSameY_x[:,0], twoIntSameY_y[:,0],Phi[yrealAllRealInds[twoIntSameYInds]],Op[0][yrealAllRealInds[twoIntSameYInds]],Op[1][yrealAllRealInds[twoIntSameYInds]])
    twoIntSameY_x_dr[:,1], twoIntSameY_y_dr[:,1] = rerotateEllipsePoints(twoIntSameY_x[:,1], twoIntSameY_y[:,1],Phi[yrealAllRealInds[twoIntSameYInds]],Op[0][yrealAllRealInds[twoIntSameYInds]],Op[1][yrealAllRealInds[twoIntSameYInds]])
    twoIntOppositeX_x_dr[:,0], twoIntOppositeX_y_dr[:,0] = rerotateEllipsePoints(twoIntOppositeX_x[:,0], twoIntOppositeX_y[:,0],Phi[yrealAllRealInds[twoIntOppositeXInds]],Op[0][yrealAllRealInds[twoIntOppositeXInds]],Op[1][yrealAllRealInds[twoIntOppositeXInds]])
    twoIntOppositeX_x_dr[:,1], twoIntOppositeX_y_dr[:,1] = rerotateEllipsePoints(twoIntOppositeX_x[:,1], twoIntOppositeX_y[:,1],Phi[yrealAllRealInds[twoIntOppositeXInds]],Op[0][yrealAllRealInds[twoIntOppositeXInds]],Op[1][yrealAllRealInds[twoIntOppositeXInds]])
    xIntersectionsOnly2_dr[:,0], yIntersectionsOnly2_dr[:,0] = rerotateEllipsePoints(xIntersectionsOnly2[:,0], yIntersectionsOnly2[:,0],Phi[only2RealInds],Op[0][only2RealInds],Op[1][only2RealInds])
    xIntersectionsOnly2_dr[:,1], yIntersectionsOnly2_dr[:,1] = rerotateEllipsePoints(xIntersectionsOnly2[:,1], yIntersectionsOnly2[:,1],Phi[only2RealInds],Op[0][only2RealInds],Op[1][only2RealInds])
    return minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,\
            fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr

minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,\
    fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr = \
    rerotateExtremaAndIntersectionPoints(minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y,\
    fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
    Phi, Op, yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds)
stop13 = time.time()
print('stop13: ' + str(stop13-start13))
del start13, stop13
####

#### Calculate True Anomalies of Points
start14 = time.time()
def trueAnomaliesOfPoints(minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,\
    fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr,\
    yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, W, w, inc):
    nu_minSepPoints = trueAnomalyFromXY(minSepPoints_x_dr, minSepPoints_y_dr,W,w,inc)
    nu_maxSepPoints = trueAnomalyFromXY(maxSepPoints_x_dr, maxSepPoints_y_dr,W,w,inc)
    nu_lminSepPoints = trueAnomalyFromXY(lminSepPoints_x_dr, lminSepPoints_y_dr,W[yrealAllRealInds],w[yrealAllRealInds],inc[yrealAllRealInds])
    nu_lmaxSepPoints = trueAnomalyFromXY(lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,W[yrealAllRealInds],w[yrealAllRealInds],inc[yrealAllRealInds])
    nu_fourInt = np.zeros(fourInt_x_dr.shape)
    nu_fourInt[:,0] = trueAnomalyFromXY(fourInt_x_dr[:,0], fourInt_y_dr[:,0],W[yrealAllRealInds[fourIntInds]],w[yrealAllRealInds[fourIntInds]],inc[yrealAllRealInds[fourIntInds]])
    nu_fourInt[:,1] = trueAnomalyFromXY(fourInt_x_dr[:,1], fourInt_y_dr[:,1],W[yrealAllRealInds[fourIntInds]],w[yrealAllRealInds[fourIntInds]],inc[yrealAllRealInds[fourIntInds]])
    nu_fourInt[:,2] = trueAnomalyFromXY(fourInt_x_dr[:,2], fourInt_y_dr[:,2],W[yrealAllRealInds[fourIntInds]],w[yrealAllRealInds[fourIntInds]],inc[yrealAllRealInds[fourIntInds]])
    nu_fourInt[:,3] = trueAnomalyFromXY(fourInt_x_dr[:,3], fourInt_y_dr[:,3],W[yrealAllRealInds[fourIntInds]],w[yrealAllRealInds[fourIntInds]],inc[yrealAllRealInds[fourIntInds]])
    nu_twoIntSameY = np.zeros(twoIntSameY_x_dr.shape)
    nu_twoIntSameY[:,0] = trueAnomalyFromXY(twoIntSameY_x_dr[:,0], twoIntSameY_y_dr[:,0],W[yrealAllRealInds[twoIntSameYInds]],w[yrealAllRealInds[twoIntSameYInds]],inc[yrealAllRealInds[twoIntSameYInds]])
    nu_twoIntSameY[:,1] = trueAnomalyFromXY(twoIntSameY_x_dr[:,1], twoIntSameY_y_dr[:,1],W[yrealAllRealInds[twoIntSameYInds]],w[yrealAllRealInds[twoIntSameYInds]],inc[yrealAllRealInds[twoIntSameYInds]])
    nu_twoIntOppositeX = np.zeros(twoIntOppositeX_x_dr.shape)
    nu_twoIntOppositeX[:,0] = trueAnomalyFromXY(twoIntOppositeX_x_dr[:,0], twoIntOppositeX_y_dr[:,0],W[yrealAllRealInds[twoIntOppositeXInds]],w[yrealAllRealInds[twoIntOppositeXInds]],inc[yrealAllRealInds[twoIntOppositeXInds]])
    nu_twoIntOppositeX[:,1] = trueAnomalyFromXY(twoIntOppositeX_x_dr[:,1], twoIntOppositeX_y_dr[:,1],W[yrealAllRealInds[twoIntOppositeXInds]],w[yrealAllRealInds[twoIntOppositeXInds]],inc[yrealAllRealInds[twoIntOppositeXInds]])
    nu_IntersectionsOnly2 = np.zeros(xIntersectionsOnly2_dr.shape)
    nu_IntersectionsOnly2[:,0] = trueAnomalyFromXY(xIntersectionsOnly2_dr[:,0], yIntersectionsOnly2_dr[:,0],W[only2RealInds],w[only2RealInds],inc[only2RealInds])
    nu_IntersectionsOnly2[:,1] = trueAnomalyFromXY(xIntersectionsOnly2_dr[:,1], yIntersectionsOnly2_dr[:,1],W[only2RealInds],w[only2RealInds],inc[only2RealInds])
    return nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2

nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2\
     = trueAnomaliesOfPoints(minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,\
    fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr,\
    yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, W, w, inc)
stop14 = time.time()
print('stop14: ' + str(stop14-start14))
del start14, stop14
#Now can I delete the x,y points?
#del minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, fourInt_x, fourInt_y
#del twoIntSameY_x, twoIntSameY_y, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2
####

#### Plot Rerotated Points 
def plotRerotatedFromNus(ind, sma, e, W, w, inc, Op, yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds,\
    nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
    twoIntSameY_x_dr, twoIntSameY_y_dr, num):
    """
    """
    plt.close(num)
    fig = plt.figure(num=num)
    ca = plt.gca()
    ca.axis('equal')
    plt.scatter([0],[0],color='orange')
    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    rs = xyz_3Dellipse(sma,e,W,w,inc,vs)
    plt.plot(rs[0,0],rs[1,0],color='black')

    ## Plot Intersection circle
    plt.plot(1*np.cos(vs),1*np.sin(vs),color='green')

    ## Plot Intersections
    if ind in yrealAllRealInds[fourIntInds]:
        yind = np.where(yrealAllRealInds[fourIntInds] == ind)[0]
        r_int0 = xyz_3Dellipse(sma,e,W,w,inc,nu_fourInt[yind,0])
        plt.scatter(r_int0[0],r_int0[1],color='green',marker='o')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,nu_fourInt[yind,1])
        plt.scatter(r_int1[0],r_int1[1],color='green',marker='o')
        r_int2 = xyz_3Dellipse(sma,e,W,w,inc,nu_fourInt[yind,2])
        plt.scatter(r_int2[0],r_int2[1],color='green',marker='o')
        r_int3 = xyz_3Dellipse(sma,e,W,w,inc,nu_fourInt[yind,3])
        plt.scatter(r_int3[0],r_int3[1],color='green',marker='o')


        # r_int0 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntSameY[yind,0]+np.pi)
        # plt.scatter(r_int0[0],r_int0[1],color='magenta',marker='o')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,nu_fourInt[yind,1]+np.pi)
        plt.scatter(r_int1[0],r_int1[1],color='magenta',marker='o')

        # r_int0 = xyz_3Dellipse(sma,e,W,w,inc,np.pi - nu_twoIntSameY[yind,0])
        # plt.scatter(r_int0[0],r_int0[1],color='green',marker='x')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,np.pi - nu_fourInt[yind,1])
        plt.scatter(r_int1[0],r_int1[1],color='green',marker='x')

        # r_int0 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntSameY[yind,0]+np.pi/6)
        # plt.scatter(r_int0[0],r_int0[1],color='red',marker='x')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,nu_fourInt[yind,1]+np.pi/6)
        plt.scatter(r_int1[0],r_int1[1],color='red',marker='x')

        # r_int0 = xyz_3Dellipse(sma,e,W,w,inc,-nu_twoIntSameY[yind,0])
        # plt.scatter(r_int0[0],r_int0[1],color='blue',marker='x')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,-nu_fourInt[yind,1])
        plt.scatter(r_int1[0],r_int1[1],color='blue',marker='x')


    if ind in yrealAllRealInds[twoIntSameYInds]:
        yind = np.where(yrealAllRealInds[twoIntSameYInds] == ind)[0]
        r_int0 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntSameY[yind,0])
        plt.scatter(r_int0[0],r_int0[1],color='green',marker='o')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntSameY[yind,1])
        plt.scatter(r_int1[0],r_int1[1],color='green',marker='o')



        #plt.scatter(twoIntSameY_x[yind], twoIntSameY_y[yind],color='blue',marker='o')

    if ind in yrealAllRealInds[twoIntOppositeXInds]:
        yind = np.where(yrealAllRealInds[twoIntOppositeXInds] == ind)[0]
        r_int0 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntOppositeX[yind,0])
        plt.scatter(r_int0[0],r_int0[1],color='green',marker='o')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntOppositeX[yind,1])
        plt.scatter(r_int1[0],r_int1[1],color='green',marker='o')

        r_int0 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntOppositeX[yind,0]+np.pi)
        plt.scatter(r_int0[0],r_int0[1],color='red',marker='D')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntOppositeX[yind,1]+np.pi)
        plt.scatter(r_int1[0],r_int1[1],color='blue',marker='D')

        r_int0 = xyz_3Dellipse(sma,e,W,w,inc,-nu_twoIntOppositeX[yind,0])
        plt.scatter(r_int0[0],r_int0[1],color='red',marker='x')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,-nu_twoIntOppositeX[yind,1])
        plt.scatter(r_int1[0],r_int1[1],color='blue',marker='x')

        r_int0 = xyz_3Dellipse(sma,e,W,w,inc,-nu_twoIntOppositeX[yind,0]+np.pi)
        plt.scatter(r_int0[0],r_int0[1],color='red',marker='^')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,-nu_twoIntOppositeX[yind,1]+np.pi)
        plt.scatter(r_int1[0],r_int1[1],color='blue',marker='^')

    if ind in only2RealInds:
        yind = np.where(only2RealInds == ind)[0]
        r_int0 = xyz_3Dellipse(sma,e,W,w,inc,nu_IntersectionsOnly2[yind,0])
        plt.scatter(r_int0[0],r_int0[1],color='green',marker='o')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,nu_IntersectionsOnly2[yind,1])
        plt.scatter(r_int1[0],r_int1[1],color='green',marker='o')

    ## Plot Smin Smax Diamonds
    r_min = xyz_3Dellipse(sma,e,W,w,inc,nu_minSepPoints[ind])
    plt.scatter(r_min[0],r_min[1],color='cyan',marker='D',s=64)
    r_max = xyz_3Dellipse(sma,e,W,w,inc,nu_maxSepPoints[ind])
    plt.scatter(r_max[0],r_max[1],color='red',marker='D',s=64)

    ## Plot Slmin Slmax Diamonds
    if ind in yrealAllRealInds:
        tind = np.where(yrealAllRealInds == ind)[0]
        r_lmin = xyz_3Dellipse(sma,e,W,w,inc,nu_lminSepPoints[tind])
        plt.scatter(r_lmin[0],r_lmin[1],color='magenta',marker='D',s=64)
        r_lmax = xyz_3Dellipse(sma,e,W,w,inc,nu_lmaxSepPoints[tind])
        plt.scatter(r_lmax[0],r_lmax[1],color='gold',marker='D',s=64)

    plt.show(block=False)


#### START ERROR PLOT
def nuCorrections_extrema(sma,e,W,w,inc,nus,mainInds,seps):
    r_extrema = xyz_3Dellipse(sma[mainInds],e[mainInds],W[mainInds],w[mainInds],inc[mainInds],nus)
    s_extrema = np.sqrt(r_extrema[0,0]**2 + r_extrema[1,0]**2)
    error0 = np.abs(seps - s_extrema)
    nus_extrema_ppi = nus + np.pi
    r_extrema_ppi = xyz_3Dellipse(sma[mainInds],e[mainInds],W[mainInds],w[mainInds],inc[mainInds],nus_extrema_ppi)
    s_extrema_ppi = np.sqrt(r_extrema_ppi[0,0]**2 + r_extrema_ppi[1,0]**2)
    error1 = np.abs(seps - s_extrema_ppi)

    error_deciding = np.stack((error0,error1),axis=1)
    minErrorInds = np.argmin(error_deciding,axis=1)

    tmpnus = np.asarray([nus,nus_extrema_ppi]).T
    nus = tmpnus[np.arange(len(nus)),minErrorInds]
    error = error_deciding[np.arange(len(nus)),minErrorInds]
    nus = np.mod(nus,2.*np.pi)
    return nus, error

#### Fix minSep True Anomalies
nu_minSepPoints, error_numinSep = nuCorrections_extrema(sma,e,W,w,inc,nu_minSepPoints,np.arange(len(sma)),minSep)
####
#### Fix maxSep True Anomalies
nu_maxSepPoints, error_numaxSep = nuCorrections_extrema(sma,e,W,w,inc,nu_maxSepPoints,np.arange(len(sma)),maxSep)
####
#### Fix lminSep True Anomalies
nu_lminSepPoints, error_nulminSep = nuCorrections_extrema(sma,e,W,w,inc,nu_lminSepPoints,yrealAllRealInds,lminSep)
####
#### Fix lmaxSep True Anomalies
nu_lmaxSepPoints, error_nulmaxSep = nuCorrections_extrema(sma,e,W,w,inc,nu_lmaxSepPoints,yrealAllRealInds,lmaxSep)
####

#### Correcting nu for ellipse-circle intersections
def nuCorrections_int(sma,e,W,w,inc,r,nus,mainInds,subInds):
    r_fourInt0 = xyz_3Dellipse(sma[mainInds[subInds]],e[mainInds[subInds]],W[mainInds[subInds]],w[mainInds[subInds]],inc[mainInds[subInds]],nus)
    tmp_fourInt0Seps = np.sqrt(r_fourInt0[0,0]**2 + r_fourInt0[1,0]**2)
    wrong_fourInt0Inds = np.where(np.abs(r[mainInds[subInds]] - tmp_fourInt0Seps) > 1e-6)[0]#1e-6)[0]
    if len(wrong_fourInt0Inds) > 0:
        nus[wrong_fourInt0Inds] = nus[wrong_fourInt0Inds] + np.pi
        r_fourInt0 = xyz_3Dellipse(sma[mainInds[subInds]],e[mainInds[subInds]],W[mainInds[subInds]],w[mainInds[subInds]],inc[mainInds[subInds]],nus)
        tmp_fourInt0Seps = np.sqrt(r_fourInt0[0,0]**2 + r_fourInt0[1,0]**2)
        wrong_fourInt0Inds = np.where(np.abs(r[mainInds[subInds]] - tmp_fourInt0Seps) > 1e-6)[0]
    print(len(wrong_fourInt0Inds))
    print(wrong_fourInt0Inds)
    if len(wrong_fourInt0Inds) > 0: #now choose the smaller error of the two
        r_fourInt0_2 = xyz_3Dellipse(sma[mainInds[subInds[wrong_fourInt0Inds]]],e[mainInds[subInds[wrong_fourInt0Inds]]],W[mainInds[subInds[wrong_fourInt0Inds]]],w[mainInds[subInds[wrong_fourInt0Inds]]],inc[mainInds[subInds[wrong_fourInt0Inds]]],nus[wrong_fourInt0Inds])
        tmp_fourInt0Seps_2 = np.sqrt(r_fourInt0_2[0,0]**2 + r_fourInt0_2[1,0]**2)
        r_fourInt0_3 = xyz_3Dellipse(sma[mainInds[subInds[wrong_fourInt0Inds]]],e[mainInds[subInds[wrong_fourInt0Inds]]],W[mainInds[subInds[wrong_fourInt0Inds]]],w[mainInds[subInds[wrong_fourInt0Inds]]],inc[mainInds[subInds[wrong_fourInt0Inds]]],nus[wrong_fourInt0Inds]+np.pi)
        tmp_fourInt0Seps_3 = np.sqrt(r_fourInt0_3[0,0]**2 + r_fourInt0_3[1,0]**2)
        indsToSwap = np.where(np.abs(r[mainInds[subInds[wrong_fourInt0Inds]]] - tmp_fourInt0Seps_2) > np.abs(r[mainInds[fourIntInds[wrong_fourInt0Inds]]] - tmp_fourInt0Seps_3))[0]
        if len(indsToSwap) > 0:
            nus[wrong_fourInt0Inds[indsToSwap]] = nus[wrong_fourInt0Inds[indsToSwap]] + np.pi
    r_fourInt0 = xyz_3Dellipse(sma[mainInds[subInds]],e[mainInds[subInds]],W[mainInds[subInds]],w[mainInds[subInds]],inc[mainInds[subInds]],nus)
    tmp_fourInt0Seps = np.sqrt(r_fourInt0[0,0]**2 + r_fourInt0[1,0]**2)
    errors = np.abs(r[mainInds[subInds]] - tmp_fourInt0Seps)
    maxError_fourInt0 = np.max(errors)
    print(maxError_fourInt0)
    nus = np.mod(nus,2.*np.pi)
    return nus, errors

#### yrealAllRealInds[fourIntInds]
nu_fourInt[:,0], errors_fourInt0 = nuCorrections_int(sma,e,W,w,inc,r,nu_fourInt[:,0],yrealAllRealInds,fourIntInds)
nu_fourInt[:,1], errors_fourInt1 = nuCorrections_int(sma,e,W,w,inc,r,nu_fourInt[:,1],yrealAllRealInds,fourIntInds)
nu_fourInt[:,2], errors_fourInt2 = nuCorrections_int(sma,e,W,w,inc,r,nu_fourInt[:,2],yrealAllRealInds,fourIntInds)
nu_fourInt[:,3], errors_fourInt3 = nuCorrections_int(sma,e,W,w,inc,r,nu_fourInt[:,3],yrealAllRealInds,fourIntInds)
####
#### yrealAllRealInds[twoIntSameYInds]
nu_twoIntSameY[:,0], errors_twoIntSameY0 = nuCorrections_int(sma,e,W,w,inc,r,nu_twoIntSameY[:,0],yrealAllRealInds,twoIntSameYInds)
nu_twoIntSameY[:,1], errors_twoIntSameY1 = nuCorrections_int(sma,e,W,w,inc,r,nu_twoIntSameY[:,1],yrealAllRealInds,twoIntSameYInds)
####
#### yrealAllRealInds[twoIntOppositeXInds]
nu_twoIntOppositeX[:,0], errors_twoIntOppositeX0 = nuCorrections_int(sma,e,W,w,inc,r,nu_twoIntOppositeX[:,0],yrealAllRealInds,twoIntOppositeXInds)
nu_twoIntOppositeX[:,1], errors_twoIntOppositeX1 = nuCorrections_int(sma,e,W,w,inc,r,nu_twoIntOppositeX[:,1],yrealAllRealInds,twoIntOppositeXInds)
####
#### only2RealInds
nu_IntersectionsOnly2[:,0], errors_IntersectionsOnly2X0 = nuCorrections_int(sma,e,W,w,inc,r,nu_IntersectionsOnly2[:,0],np.arange(len(sma)),only2RealInds)
nu_IntersectionsOnly2[:,1], errors_IntersectionsOnly2X1 = nuCorrections_int(sma,e,W,w,inc,r,nu_IntersectionsOnly2[:,1],np.arange(len(sma)),only2RealInds)
####

#### Error Plot ####
num=822
plt.close(num)
plt.figure(num=num)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
plt.yscale('log')
plt.xscale('log')
# yrealAllRealInds[fourIntInds]
plt.plot(np.arange(len(fourIntInds)),np.abs(np.sort(-errors_fourInt0)[np.arange(len(fourIntInds))]),label='Four Int 0')
plt.plot(np.arange(len(fourIntInds)),np.abs(np.sort(-errors_fourInt1)[np.arange(len(fourIntInds))]),label='Four Int 1')
plt.plot(np.arange(len(fourIntInds)),np.abs(np.sort(-errors_fourInt2)[np.arange(len(fourIntInds))]),label='Four Int 2')
plt.plot(np.arange(len(fourIntInds)),np.abs(np.sort(-errors_fourInt3)[np.arange(len(fourIntInds))]),label='Four Int 3')
# yrealAllRealInds[twoIntSameYInds]
plt.plot(np.arange(len(twoIntSameYInds)),np.abs(np.sort(-errors_twoIntSameY0)),label='Two Int Same Y 0')
plt.plot(np.arange(len(twoIntSameYInds)),np.abs(np.sort(-errors_twoIntSameY1)),label='Two Int Same Y 1')
# yrealAllRealInds[twoIntOppositeXInds]
plt.plot(np.arange(len(twoIntOppositeXInds)),np.abs(np.sort(-errors_twoIntOppositeX0)),label='Two Int Opposite X 0')
plt.plot(np.arange(len(twoIntOppositeXInds)),np.abs(np.sort(-errors_twoIntOppositeX1)),label='Two Int Opposite X 1')
# only2RealInds
plt.plot(np.arange(len(only2RealInds)),np.abs(np.sort(-errors_IntersectionsOnly2X0)),label='Only 2 Int 0')
plt.plot(np.arange(len(only2RealInds)),np.abs(np.sort(-errors_IntersectionsOnly2X1)),label='Only 2 Int 1')

plt.legend()
plt.ylabel('Absolute Separation Error (AU)', weight='bold')
plt.xlabel('Planet Orbit Index', weight='bold')
plt.show(block=False)
plt.close(num)
######################

# ind = yrealAllRealInds[fourIntInds[np.argsort(-errors_fourInt1)[0]]]
# plotRerotatedFromNus(ind, sma[ind], e[ind], W[ind], w[ind], inc[ind], Op[:,ind], yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds,\
#     nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
#     twoIntSameY_x, twoIntSameY_y, num=8001)

ind = yrealAllRealInds[twoIntSameYInds[np.argsort(-errors_twoIntSameY1)[0]]]
plotRerotatedFromNus(ind, sma[ind], e[ind], W[ind], w[ind], inc[ind], Op[:,ind], yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds,\
    nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
    twoIntSameY_x, twoIntSameY_y, num=8001)

# ind = only2RealInds[np.argsort(-errors_IntersectionsOnly2X0)[0]]
# plotRerotatedFromNus(ind, sma[ind], e[ind], W[ind], w[ind], inc[ind], Op[:,ind], yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds,\
#     nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
#     twoIntSameY_x, twoIntSameY_y, num=8001)

###### DONE FIXING NU

#### Plot Histogram of Error
plt.close(823)
plt.figure(num=823)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
plt.yscale('log')
plt.xscale('log')

#counts, bins = np.histogram(np.abs(errors_fourInt0)+1e-17,bins=10**np.linspace(1e-17,1e-1,16))
#plt.hist(bins[:-1], bins, weights=counts)#bins=10**np.linspace(1e-17,1e-1,16),label='Four Int 0')
plt.hist(np.abs(errors_fourInt0)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Four Int 0',alpha=0.2)
plt.hist(np.abs(errors_fourInt1)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Four Int 1',alpha=0.2)
plt.hist(np.abs(errors_fourInt2)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Four Int 2',alpha=0.2)
plt.hist(np.abs(errors_fourInt3)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Four Int 1',alpha=0.2)

plt.hist(np.abs(errors_twoIntSameY0)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Two Int Same Y 0',alpha=0.2)
plt.hist(np.abs(errors_twoIntSameY1)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Two Int Same Y 1',alpha=0.2)

plt.hist(np.abs(errors_twoIntOppositeX0)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Two Int Opposite X 0',alpha=0.2)
plt.hist(np.abs(errors_twoIntOppositeX1)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Two Int Opposite X 1',alpha=0.2)

plt.hist(np.abs(errors_IntersectionsOnly2X0)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Only 2 Int 0',alpha=0.2)
plt.hist(np.abs(errors_IntersectionsOnly2X1)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Only 2 Int 1',alpha=0.2)
plt.xlabel('Absolute Error (AU)', weight='bold')
plt.ylabel('Number of Planets', weight='bold') #Switch to fraction
plt.legend()
plt.show(block=False)
plt.close(823) #thinking the above plot is relativly useless
####

#### Plot Histogram of Error
num=824
plt.close(num)
plt.figure(num=num)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
plt.yscale('log')
plt.xscale('log')

#counts, bins = np.histogram(np.abs(errors_fourInt0)+1e-17,bins=10**np.linspace(1e-17,1e-1,16))
#plt.hist(bins[:-1], bins, weights=counts)#bins=10**np.linspace(1e-17,1e-1,16),label='Four Int 0')
plt.hist(np.concatenate((np.abs(errors_fourInt0)+1e-17, np.abs(errors_fourInt1)+1e-17,np.abs(errors_fourInt2)+1e-17,np.abs(errors_fourInt3)+1e-17,\
        np.abs(errors_twoIntSameY0)+1e-17, np.abs(errors_twoIntSameY1)+1e-17,np.abs(errors_twoIntOppositeX0)+1e-17,\
        np.abs(errors_twoIntOppositeX1)+1e-17,np.abs(errors_IntersectionsOnly2X0)+1e-17,np.abs(errors_IntersectionsOnly2X1)+1e-17)), bins=np.logspace(start=-17.,stop=-1,num=17),color='purple')
plt.xlabel('Absolute Error (AU)', weight='bold')
plt.ylabel('Number of Planets', weight='bold') #Switch to fraction
plt.show(block=False)

plt.close(num)
####



#DELETE code for finding correct nu values in stuff above
# ind = yrealAllRealInds[twoIntSameYInds[np.argsort(-errors_twoIntSameY0)[0]]]
# #ind = yrealAllRealInds[twoIntSameYInds[0]]
# plotRerotatedFromNus(ind, sma[ind], e[ind], W[ind], w[ind], inc[ind], Op[:,ind], yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds,\
#     nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
#     twoIntSameY_x, twoIntSameY_y, num=8000)

# from scipy.optimize import fsolve
# def errorFunc(x,sma,e,W,w,inc):
#     """
#     """
#     r_0 = xyz_3Dellipse(sma,e,W,w,inc,x)
#     error = np.abs(1-np.sqrt(r_0[0,0]**2 + r_0[1,0]**2))
#     return error
# yind = np.where(ind == yrealAllRealInds[twoIntSameYInds])[0]
# out0 = fsolve(errorFunc,nu_twoIntSameY[yind,0],args=(sma[ind],e[ind],W[ind],w[ind],inc[ind]))
# out1 = fsolve(errorFunc,nu_twoIntSameY[yind,1],args=(sma[ind],e[ind],W[ind],w[ind],inc[ind]))
# print('Goal0: ' + str(out0))
# print('Goal0: ' + str(out0 - 2*np.pi))
# print('Goal0: ' + str(out0 + np.pi))
# print('Goal1: ' + str(out1))
# print('Goal1: ' + str(out1 - 2*np.pi))
# print('Goal1: ' + str(out1 + np.pi))
# print('Goal1: ' + str(np.pi - out1))
# print(nu_twoIntSameY[yind,0])
# print(2*np.pi - nu_twoIntSameY[yind,0])
# print(np.pi - nu_twoIntSameY[yind,0])
# print(nu_twoIntSameY[yind,1])
# print(2*np.pi - nu_twoIntSameY[yind,1])
# print(np.pi - nu_twoIntSameY[yind,1])
# rout = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],0.372)
# sepout = np.sqrt(rout[0]**2 + rout[1]**2)
# print(sepout)





#### Redo Significant Point plot Using these Nu
num=3690
plt.close(num)
fig = plt.figure(num=num)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')

## Central Sun
plt.scatter([0],[0],color='orange')
## 3D Ellipse
vs = np.linspace(start=0,stop=2*np.pi,num=300)
r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
x_3Dellipse = r[0,0,:]
y_3Dellipse = r[1,0,:]
plt.plot(x_3Dellipse,y_3Dellipse,color='black')

#Plot Separation Limits
r_minSep = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_minSepPoints[ind])
tmp_minSep = np.sqrt(r_minSep[0]**2 + r_minSep[1]**2)
r_maxSep = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_maxSepPoints[ind])
tmp_maxSep = np.sqrt(r_maxSep[0]**2 + r_maxSep[1]**2)
plt.scatter(r_minSep[0],r_minSep[1],color='cyan',marker='D')
plt.scatter(r_maxSep[0],r_maxSep[1],color='red',marker='D')
if ind in yrealAllRealInds:
    print('All Real')
    tind = np.where(yrealAllRealInds == ind)[0]
    r_lminSep = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_lminSepPoints[tind])#[yrealAllRealInds[tind]])
    tmp_lminSep = np.sqrt(r_lminSep[0]**2 + r_lminSep[1]**2)
    r_lmaxSep = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_lmaxSepPoints[tind])#[yrealAllRealInds[tind]]+np.pi)
    tmp_lmaxSep = np.sqrt(r_lmaxSep[0]**2 + r_lmaxSep[1]**2)
    plt.scatter(r_lminSep[0],r_lminSep[1],color='magenta',marker='D')
    plt.scatter(r_lmaxSep[0],r_lmaxSep[1],color='gold',marker='D')

if ind in yrealAllRealInds[fourIntInds]:
    #WORKING
    print('All Real 4 Int')
    yind = np.where(yrealAllRealInds[fourIntInds] == ind)[0]
    r_fourInt0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,0])
    r_fourInt1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,1])
    r_fourInt2 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,2])
    r_fourInt3 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,3])
    plt.scatter(r_fourInt0[0],r_fourInt0[1],color='green',marker='o')
    plt.scatter(r_fourInt1[0],r_fourInt1[1],color='green',marker='o')
    plt.scatter(r_fourInt2[0],r_fourInt2[1],color='green',marker='o')
    plt.scatter(r_fourInt3[0],r_fourInt3[1],color='green',marker='o')
elif ind in yrealAllRealInds[twoIntSameYInds]: #Same Y
    print('All Real 2 Int Same Y')
    yind = np.where(yrealAllRealInds[twoIntSameYInds] == ind)[0]
    r_twoIntSameY0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntSameY[yind,0])
    r_twoIntSameY1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntSameY[yind,1])
    plt.scatter(r_twoIntSameY0[0],r_twoIntSameY0[1],color='green',marker='o')
    plt.scatter(r_twoIntSameY1[0],r_twoIntSameY1[1],color='green',marker='o')
elif ind in yrealAllRealInds[twoIntOppositeXInds]: #Same X
    print('All Real 2 Int Opposite X')
    yind = np.where(yrealAllRealInds[twoIntOppositeXInds] == ind)[0]
    r_twoIntOppositeX0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntOppositeX[yind,0])
    r_twoIntOppositeX1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntOppositeX[yind,1])
    plt.scatter(r_twoIntOppositeX0[0],r_twoIntOppositeX0[1],color='green',marker='o')
    plt.scatter(r_twoIntOppositeX1[0],r_twoIntOppositeX1[1],color='green',marker='o')
elif ind in only2RealInds:
    print('All Real 2 Int')
    yind = np.where(only2RealInds == ind)[0]
    r_IntersectionOnly20 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,0])
    r_IntersectionOnly21 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,1])
    plt.scatter(r_IntersectionOnly20[0],r_IntersectionOnly20[1],color='green',marker='o')
    plt.scatter(r_IntersectionOnly21[0],r_IntersectionOnly21[1],color='green',marker='o')

    r_IntersectionOnly20 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,0])
    plt.scatter(r_IntersectionOnly20[0],r_IntersectionOnly20[1],color='grey',marker='x')

    r_IntersectionOnly21 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,1])
    plt.scatter(r_IntersectionOnly21[0],r_IntersectionOnly21[1],color='blue',marker='x')

#Plot lmaxSep Circle
x_circ2 = 1.*np.cos(vs)
y_circ2 = 1.*np.sin(vs)
plt.plot(x_circ2,y_circ2,color='green')
ca = plt.gca()
ca.axis('equal')
plt.show(block=False)
####





#### Plot separation vs nu
def plotSeparationvsnu(ind, sma, e, W, w, inc, minSep, maxSep, lminSep, lmaxSep, \
        nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints,\
        nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
        yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, num):
    plt.close(num)
    fig = plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    nurange = np.linspace(start=0.,stop=2.*np.pi,num=100)
    prs = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nurange)
    pseps = np.sqrt(prs[0,0]**2+prs[1,0]**2)
    plt.plot(nurange,pseps,color='black')
    plt.plot([0,2.*np.pi],[0,0],color='black',linestyle='--') #0 sep line
    plt.plot([0,2*np.pi],[minSep[ind],minSep[ind]],color='cyan')
    plt.plot([0,2*np.pi],[maxSep[ind],maxSep[ind]],color='red')
    if ind in yrealAllRealInds:
        tind = np.where(yrealAllRealInds == ind)[0]
        plt.plot([0,2*np.pi],[lminSep[tind],lminSep[tind]],color='magenta')
        plt.plot([0,2*np.pi],[lmaxSep[tind],lmaxSep[tind]],color='gold')
    plt.plot([0,2*np.pi],[1,1],color='green') #the plot intersection line

    #Plot Separation Limits
    plt.scatter(nu_minSepPoints[ind],minSep[ind],color='cyan',marker='D')
    plt.scatter(nu_maxSepPoints[ind],maxSep[ind],color='red',marker='D')
    if ind in yrealAllRealInds:
        tind = np.where(yrealAllRealInds == ind)[0]
        plt.scatter(nu_lminSepPoints[tind],lminSep[tind],color='magenta',marker='D')
        plt.scatter(nu_lmaxSepPoints[tind],lmaxSep[tind],color='gold',marker='D')

    if ind in yrealAllRealInds[fourIntInds]:
        yind = np.where(yrealAllRealInds[fourIntInds] == ind)[0]
        # t_fourInt0 = timeFromTrueAnomaly(nu_fourInt[yind,0],periods[ind],e[ind])
        # t_fourInt1 = timeFromTrueAnomaly(nu_fourInt[yind,1],periods[ind],e[ind])
        # t_fourInt2 = timeFromTrueAnomaly(nu_fourInt[yind,2],periods[ind],e[ind])
        # t_fourInt3 = timeFromTrueAnomaly(nu_fourInt[yind,3],periods[ind],e[ind])
        r_fourInt0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,0])
        r_fourInt1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,1])
        r_fourInt2 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,2])
        r_fourInt3 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,3])
        plt.scatter(nu_fourInt[yind,0],np.sqrt(r_fourInt0[0]**2 + r_fourInt0[1]**2),color='green',marker='o')
        plt.scatter(nu_fourInt[yind,1],np.sqrt(r_fourInt1[0]**2 + r_fourInt1[1]**2),color='green',marker='o')
        plt.scatter(nu_fourInt[yind,2],np.sqrt(r_fourInt2[0]**2 + r_fourInt2[1]**2),color='green',marker='o')
        plt.scatter(nu_fourInt[yind,3],np.sqrt(r_fourInt3[0]**2 + r_fourInt3[1]**2),color='green',marker='o')
    elif ind in yrealAllRealInds[twoIntSameYInds]: #Same Y
        yind = np.where(yrealAllRealInds[twoIntSameYInds] == ind)[0]
        #t_twoIntSameY0 = timeFromTrueAnomaly(nu_twoIntSameY[yind,0],periods[ind],e[ind])
        #t_twoIntSameY1 = timeFromTrueAnomaly(nu_twoIntSameY[yind,1],periods[ind],e[ind])
        r_twoIntSameY0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntSameY[yind,0])
        r_twoIntSameY1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntSameY[yind,1])
        plt.scatter(nu_twoIntSameY[yind,0],np.sqrt(r_twoIntSameY0[0]**2 + r_twoIntSameY0[1]**2),color='green',marker='o')
        plt.scatter(nu_twoIntSameY[yind,1],np.sqrt(r_twoIntSameY1[0]**2 + r_twoIntSameY1[1]**2),color='green',marker='o')
    elif ind in yrealAllRealInds[twoIntOppositeXInds]: #Same X
        yind = np.where(yrealAllRealInds[twoIntOppositeXInds] == ind)[0]
        #t_twoIntOppositeX0 = timeFromTrueAnomaly(nu_twoIntOppositeX[yind,0],periods[ind],e[ind])
        #t_twoIntOppositeX1 = timeFromTrueAnomaly(nu_twoIntOppositeX[yind,1],periods[ind],e[ind])
        r_twoIntOppositeX0 = xyz_3Dellipse(a[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntOppositeX[yind,0])
        r_twoIntOppositeX1 = xyz_3Dellipse(a[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntOppositeX[yind,1])
        plt.scatter(nu_twoIntOppositeX[yind,0],np.sqrt(r_twoIntOppositeX0[0]**2 + r_twoIntOppositeX0[1]**2),color='green',marker='o')
        plt.scatter(nu_twoIntOppositeX[yind,1],np.sqrt(r_twoIntOppositeX1[0]**2 + r_twoIntOppositeX1[1]**2),color='green',marker='o')
    elif ind in only2RealInds:
        yind = np.where(only2RealInds == ind)[0]
        #t_IntersectionOnly20 = timeFromTrueAnomaly(nu_IntersectionsOnly2[yind,0],periods[ind],e[ind])
        #t_IntersectionOnly21 = timeFromTrueAnomaly(nu_IntersectionsOnly2[yind,1],periods[ind],e[ind])
        r_IntersectionOnly20 = xyz_3Dellipse(a[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,0])
        r_IntersectionOnly21 = xyz_3Dellipse(a[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,1])
        plt.scatter(nu_IntersectionsOnly2[yind,0],np.sqrt(r_IntersectionOnly20[0]**2 + r_IntersectionOnly20[1]**2),color='green',marker='o')
        plt.scatter(nu_IntersectionsOnly2[yind,1],np.sqrt(r_IntersectionOnly21[0]**2 + r_IntersectionOnly21[1]**2),color='green',marker='o')


    plt.xlim([0,2.*np.pi])
    #plt.xlim([0,periods[ind]])
    plt.ylabel('Projected Separation, s, in AU',weight='bold')
    #plt.xlabel('Projected Ellipse E (rad)',weight='bold')
    plt.xlabel(r'$True Anomaly, \nu, (rad)$',weight='bold')
    plt.show(block=False)

num=962
plotSeparationvsnu(ind, sma, e, W, w, inc, minSep, maxSep, lminSep, lmaxSep, \
    nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints,\
    nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
    yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, num)
####





#### Plot separation vs time
def plotSeparationVsTime(ind, sma, e, W, w, inc, minSep, maxSep, lminSep, lmaxSep, \
    nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints,\
    nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
    yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, num):
    plt.close(num)
    fig = plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    # Erange = np.linspace(start=0.,stop=2*np.pi,num=400)
    # Mrange = Erange - e[ind]*np.sin(Erange)
    periods = (2*np.pi*np.sqrt((sma*u.AU)**3/(const.G.to('AU3 / (kg s2)')*const.M_sun))).to('year').value
    # xellipsetmp = a[ind]*np.cos(Erange)
    # yellipsetmp = b[ind]*np.sin(Erange)
    # septmp = np.sqrt((xellipsetmp - x[ind])**2 + (yellipsetmp - y[ind])**2)
    #plt.plot(Erange,septmp,color='black')
    nurange = np.linspace(start=0.,stop=2.*np.pi,num=400)
    trange = timeFromTrueAnomaly(nurange,periods[ind],e[ind])
    rs = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nurange)
    seps = np.sqrt(rs[0,0]**2+rs[1,0]**2)

    plt.plot(trange,seps,color='black')
    plt.plot([0,periods[ind]],[0,0],color='black',linestyle='--') #0 sep line
    plt.plot([0,periods[ind]],[minSep[ind],minSep[ind]],color='cyan')
    plt.plot([0,periods[ind]],[maxSep[ind],maxSep[ind]],color='red')
    if ind in yrealAllRealInds:
        tind = np.where(yrealAllRealInds == ind)[0]
        plt.plot([0,periods[ind]],[lminSep[tind],lminSep[tind]],color='magenta')
        plt.plot([0,periods[ind]],[lmaxSep[tind],lmaxSep[tind]],color='gold')
    plt.plot([0,periods[ind]],[1,1],color='green') #the plot intersection line

    #Plot Separation Limits
    t_minSep = timeFromTrueAnomaly(nu_minSepPoints[ind],periods[ind],e[ind])
    t_maxSep = timeFromTrueAnomaly(nu_maxSepPoints[ind],periods[ind],e[ind])
    plt.scatter(t_minSep,minSep[ind],color='cyan',marker='D')
    plt.scatter(t_maxSep,maxSep[ind],color='red',marker='D')
    if ind in yrealAllRealInds:
        tind = np.where(yrealAllRealInds == ind)[0]
        t_lminSep = timeFromTrueAnomaly(nu_lminSepPoints[tind],periods[ind],e[ind])
        t_lmaxSep = timeFromTrueAnomaly(nu_lmaxSepPoints[tind],periods[ind],e[ind])
        plt.scatter(t_lminSep,lminSep[tind],color='magenta',marker='D')
        plt.scatter(t_lmaxSep,lmaxSep[tind],color='gold',marker='D')

    if ind in yrealAllRealInds[fourIntInds]:
        yind = np.where(yrealAllRealInds[fourIntInds] == ind)[0]
        t_fourInt0 = timeFromTrueAnomaly(nu_fourInt[yind,0],periods[ind],e[ind])
        t_fourInt1 = timeFromTrueAnomaly(nu_fourInt[yind,1],periods[ind],e[ind])
        t_fourInt2 = timeFromTrueAnomaly(nu_fourInt[yind,2],periods[ind],e[ind])
        t_fourInt3 = timeFromTrueAnomaly(nu_fourInt[yind,3],periods[ind],e[ind])
        r_fourInt0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,0])
        r_fourInt1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,1])
        r_fourInt2 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,2])
        r_fourInt3 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,3])
        plt.scatter(t_fourInt0,np.sqrt(r_fourInt0[0]**2 + r_fourInt0[1]**2),color='green',marker='o')
        plt.scatter(t_fourInt1,np.sqrt(r_fourInt1[0]**2 + r_fourInt1[1]**2),color='green',marker='o')
        plt.scatter(t_fourInt2,np.sqrt(r_fourInt2[0]**2 + r_fourInt2[1]**2),color='green',marker='o')
        plt.scatter(t_fourInt3,np.sqrt(r_fourInt3[0]**2 + r_fourInt3[1]**2),color='green',marker='o')
    elif ind in yrealAllRealInds[twoIntSameYInds]: #Same Y
        yind = np.where(yrealAllRealInds[twoIntSameYInds] == ind)[0]
        t_twoIntSameY0 = timeFromTrueAnomaly(nu_twoIntSameY[yind,0],periods[ind],e[ind])
        t_twoIntSameY1 = timeFromTrueAnomaly(nu_twoIntSameY[yind,1],periods[ind],e[ind])
        r_twoIntSameY0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntSameY[yind,0])
        r_twoIntSameY1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntSameY[yind,1])
        plt.scatter(t_twoIntSameY0,np.sqrt(r_twoIntSameY0[0]**2 + r_twoIntSameY0[1]**2),color='green',marker='o')
        plt.scatter(t_twoIntSameY1,np.sqrt(r_twoIntSameY1[0]**2 + r_twoIntSameY1[1]**2),color='green',marker='o')
    elif ind in yrealAllRealInds[twoIntOppositeXInds]: #Same X
        yind = np.where(yrealAllRealInds[twoIntOppositeXInds] == ind)[0]
        t_twoIntOppositeX0 = timeFromTrueAnomaly(nu_twoIntOppositeX[yind,0],periods[ind],e[ind])
        t_twoIntOppositeX1 = timeFromTrueAnomaly(nu_twoIntOppositeX[yind,1],periods[ind],e[ind])
        r_twoIntOppositeX0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntOppositeX[yind,0])
        r_twoIntOppositeX1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntOppositeX[yind,1])
        plt.scatter(t_twoIntOppositeX0,np.sqrt(r_twoIntOppositeX0[0]**2 + r_twoIntOppositeX0[1]**2),color='green',marker='o')
        plt.scatter(t_twoIntOppositeX1,np.sqrt(r_twoIntOppositeX1[0]**2 + r_twoIntOppositeX1[1]**2),color='green',marker='o')
    elif ind in only2RealInds:
        yind = np.where(only2RealInds == ind)[0]
        t_IntersectionOnly20 = timeFromTrueAnomaly(nu_IntersectionsOnly2[yind,0],periods[ind],e[ind])
        t_IntersectionOnly21 = timeFromTrueAnomaly(nu_IntersectionsOnly2[yind,1],periods[ind],e[ind])
        r_IntersectionOnly20 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,0])
        r_IntersectionOnly21 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,1])
        plt.scatter(t_IntersectionOnly20,np.sqrt(r_IntersectionOnly20[0]**2 + r_IntersectionOnly20[1]**2),color='green',marker='o')
        plt.scatter(t_IntersectionOnly21,np.sqrt(r_IntersectionOnly21[0]**2 + r_IntersectionOnly21[1]**2),color='green',marker='o')

    plt.xlim([0,periods[ind]])
    plt.ylabel('Projected Separation, s, in AU',weight='bold')
    plt.xlabel('Time Past Periastron, t, (years)',weight='bold')
    plt.show(block=False)

num=963
plotSeparationVsTime(ind, sma, e, W, w, inc, minSep, maxSep, lminSep, lmaxSep, \
        nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints,\
        nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
        yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, num)
####






####  Plot Derotate Ellipse
tinds = np.argsort(-np.abs(errors_fourInt1))
ind = yrealAllRealInds[fourIntInds[tinds[1]]]
num=55670
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

# Plot Star Location Type Dividers
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
plt.close(num)
####





# #IS ANY OF THE FOLLOWING NECESSARY. MAY BE DELETABLE

# # xreal2[np.abs(np.imag(xreal2)) > 1e-4] = np.nan #There is evidence from below that the residual resulting from entiring solutions with 3e-5j results in 0+1e-20j therefore we will nan above 1e-4
# # xreal2 = np.real(xreal2)
# yreal2 = ellipseYFromX(xreal2, a, b)
# seps2_0 = np.sqrt((xreal2[:,0]-x)**2 + (yreal2[:,0]-y)**2)
# seps2_1 = np.sqrt((xreal2[:,1]-x)**2 + (yreal2[:,1]-y)**2)
# seps2_2 = np.sqrt((xreal2[:,2]-x)**2 + (yreal2[:,2]-y)**2)
# seps2_3 = np.sqrt((xreal2[:,3]-x)**2 + (yreal2[:,3]-y)**2)

# # seps2_0 = np.sqrt((xreal2[:,0]-mx)**2 + (yreal2[:,0]-my)**2)
# # seps2_1 = np.sqrt((xreal2[:,1]-mx)**2 + (yreal2[:,1]-my)**2)
# # seps2_2 = np.sqrt((xreal2[:,2]-mx)**2 + (yreal2[:,2]-my)**2)
# # seps2_3 = np.sqrt((xreal2[:,3]-mx)**2 + (yreal2[:,3]-my)**2)
# seps2 = np.asarray([seps2_0,seps2_1,seps2_2,seps2_3]).T

# #we are currently omitting all of these potential calculations so-long-as the following assert is never true
# #assert ~np.any(p2+p3**2/12 == 0), 'Oops, looks like the sympy piecewise was true once!'

# #### Root Types For Each Planet #######################################################
# #ORDER HAS BEEN CHANGED
# # If delta > 0 and P < 0 and D < 0 four roots all real or none
# #allRealDistinctInds = np.where((delta > 0)*(P < 0)*(D2 < 0))[0] #METHOD 1, out of 10000, this found 1638, missing ~54
# allRealDistinctInds = np.where(np.all(np.abs(np.imag(xreal2)) < 2.5*1e-5, axis=1))[0]#1e-9, axis=1))[0] #This found 1692
# residual_allreal, isAll_allreal, maxRealResidual_allreal, maxImagResidual_allreal = checkResiduals(A,B,C,D,xreal2,allRealDistinctInds,4)
# del A, B, C, D #delting for memory efficiency
# assert maxRealResidual_allreal < 1e-9, 'At least one all real residual is too large'
# # If delta < 0, two distinct real roots, two complex
# #DELETEtwoRealDistinctInds = np.where(delta < 0)[0]
# #DELETE UNNECESSARYxrealsImag = np.abs(np.imag(xreal2))
# xrealsImagInds = np.argsort(np.abs(np.imag(xreal2)),axis=1)
# xrealsImagInds2 = np.asarray([xrealsImagInds[:,0],xrealsImagInds[:,1]])
# xrealOfSmallest2Imags = np.real(xreal2[np.arange(len(a)),xrealsImagInds2]).T
# ximagOfSmallest2Imags = np.imag(xreal2[np.arange(len(a)),xrealsImagInds2]).T
# #~np.all(np.abs(np.imag(xreal2)) < 1e-9, axis=1) removes solutions with 4 distinct real roots
# #The other two are thresholds that happend to work well once
# indsOf2RealSols = np.where((np.abs(ximagOfSmallest2Imags[:,0]) < 2.5*1e-5)*(np.abs(ximagOfSmallest2Imags[:,1]) < 2.5*1e-5)*~np.all(np.abs(np.imag(xreal2)) < 2.5*1e-5, axis=1))[0]
# #DELETElen(indsOf2RealSols) - len(allRealDistinctInds)
# xrealsTwoRealSols = np.real(np.asarray([xreal2[indsOf2RealSols,xrealsImagInds2[0,indsOf2RealSols]],xreal2[indsOf2RealSols,xrealsImagInds2[1,indsOf2RealSols]]]).T)
# residual_TwoRealSols, isAll_TwoRealSols, maxRealResidual_TwoRealSols, maxImagResidual_TwoRealSols = checkResiduals(A[indsOf2RealSols],B[indsOf2RealSols],C[indsOf2RealSols],D[indsOf2RealSols],xrealsTwoRealSols,np.arange(len(xrealsTwoRealSols)),2)
# assert len(np.intersect1d(allRealDistinctInds,indsOf2RealSols)) == 0, 'There is intersection between Two Real Distinct and the 4 real solution inds, investigate'

# #DELETE cruft
# # twoRealDistinctInds2 = np.where(np.all(np.abs(np.imag(xreal2)) < 1e-9, axis=1))[0] #This found 1692
# # twoRealSorting = np.argsort(np.abs(np.imag(xreal2[twoRealDistinctInds,:])),axis=1)
# # tmpxReals = np.asarray([xreal2[np.arange(len(twoRealDistinctInds)),twoRealSorting[:,0]], xreal2[np.arange(len(twoRealDistinctInds)),twoRealSorting[:,1]]]).T

# # If delta > 0 and (P < 0 or D < 0)
# #allImagInds = np.where((delta > 0)*((P > 0)|(D2 > 0)))[0]
# #allImagInds = np.where(np.all(np.abs(np.imag(xreal2)) >= 1e-9, axis=1))[0]
# allImagInds = np.where(np.all(np.abs(np.imag(xreal2)) >= 2.5*1e-5, axis=1))[0]
# assert len(np.intersect1d(allRealDistinctInds,allImagInds)) == 0, 'There is intersection between All Imag and the 4 real solution inds, investigate'
# assert len(np.intersect1d(indsOf2RealSols,allImagInds)) == 0, 'There is intersection between All Imag and the Two Real Distinct solution inds, investigate'

# # If delta == 0, multiple root
# realDoubleRootTwoRealRootsInds = np.where((delta == 0)*(P < 0)*(D2 < 0)*(delta_0 != 0))[0] #delta=0 and P<0 and D2<0
# realDoubleRootTwoComplexInds = np.where((delta == 0)*((D2 > 0)|((P > 0)*((D2 != 0)|(R != 0)))))[0] #delta=0 and (D>0 or (P>0 and (D!=0 or R!=0)))
# tripleRootSimpleRootInds = np.where((delta == 0)*(delta_0 == 0)*(D2 !=0))[0]
# twoRealDoubleRootsInds = np.where((delta == 0)*(D2 == 0)*(P < 0))[0]
# twoComplexDoubleRootsInds = np.where((delta == 0)*(D2 == 0)*(P > 0)*(R == 0))[0]
# fourIdenticalRealRootsInds = np.where((delta == 0)*(D2 == 0)*(delta_0 == 0))[0]

# #DELETE cruft?
# # #### Double checking root classification
# # #twoRealDistinctInds #check that 2 of the 4 imags are below thresh
# # numUnderThresh = np.sum(np.abs(np.imag(xreal2[twoRealDistinctInds])) > 1e-11, axis=1)
# # indsUnderThresh = np.where(numUnderThresh != 2)[0]
# # indsThatDontBelongIntwoRealDistinctInds = twoRealDistinctInds[indsUnderThresh]
# # twoRealDistinctInds = np.delete(twoRealDistinctInds,indsThatDontBelongIntwoRealDistinctInds) #deletes the desired inds from aray
# # #np.count_nonzero(numUnderThresh < 2)

# #DELETE IN FUTURE
# #The 1e-5 here gave me the number as the Imag count
# #allRealDistinctInds2 = np.where(np.all(np.abs(np.imag(xreal2)) > 1e-5, axis=1))[0]
# #allRealDistinctInds2 = np.where(np.all(np.abs(np.imag(xreal2)) > 1e-9, axis=1))[0]


# #Number of Solutions of Each Type
# numRootInds = [indsOf2RealSols,allRealDistinctInds,allImagInds,realDoubleRootTwoRealRootsInds,realDoubleRootTwoComplexInds,\
#     tripleRootSimpleRootInds,twoRealDoubleRootsInds,twoComplexDoubleRootsInds,fourIdenticalRealRootsInds]

# #Number of Roots of Each Type
# lenNumRootsInds = [len(numRootInds[i]) for i in np.arange(len(numRootInds))]
# assert len(indsOf2RealSols)+len(allRealDistinctInds)+len(allImagInds)-len(realDoubleRootTwoRealRootsInds), 'Number of roots does not add up, investigate'
# ########################################################################


# # Calculate Residuals
# # residual_0 = xreal2[:,0]**4 + A*xreal2[:,0]**3 + B*xreal2[:,0]**2 + C*xreal2[:,0] + D
# # residual_1 = xreal2[:,1]**4 + A*xreal2[:,1]**3 + B*xreal2[:,1]**2 + C*xreal2[:,1] + D
# # residual_2 = xreal2[:,2]**4 + A*xreal2[:,2]**3 + B*xreal2[:,2]**2 + C*xreal2[:,2] + D
# # residual_3 = xreal2[:,3]**4 + A*xreal2[:,3]**3 + B*xreal2[:,3]**2 + C*xreal2[:,3] + D
# # residual = np.asarray([residual_0, residual_1, residual_2, residual_3]).T
# # #assert np.all((np.real(residual) < 1e-7)*(np.imag(residual) < 1e-7)), 'All residual are not less than 1e-7'
# # del residual_0, residual_1, residual_2, residual_3
# residual_all, isAll_all, maxRealResidual_all, maxImagResidual_all = checkResiduals(A,B,C,D,xreal2,np.arange(len(A)),4)



# xfinal = np.zeros(xreal2.shape) + np.nan
# # case 1 Two Real Distinct Inds
# #find 2 xsols with smallest imag part
# #xreal2[indsOf2RealSols[0]]
# #ximags2 = np.imag(xreal2[indsOf2RealSols])
# #ximags2smallImagInds = np.argsort(np.abs(ximags2),axis=1)[:,0:2] #sorts from smallest magnitude to largest magnitude
# #xrealsTwoRealDistinct = np.asarray([xreal2[indsOf2RealSols,ximags2smallImagInds[:,0]], xreal2[indsOf2RealSols,ximags2smallImagInds[:,1]]]).T
# xfinal[indsOf2RealSols,0:2] = xrealOfSmallest2Imags[indsOf2RealSols]#np.real(xrealsTwoRealDistinct)
# #Check residuals
# residual_case1, isAll_case1, maxRealResidual_case1, maxImagResidual_case1 = checkResiduals(A,B,C,D,xfinal,indsOf2RealSols,2)
# #The following does not work
# # residual_0 = np.real(xrealsTwoRealDistinct[:,0])**4 + A[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,0])**3 + B[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,0])**2 + C[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,0]) + D[twoRealDistinctInds]
# # residual_1 = np.real(xrealsTwoRealDistinct[:,1])**4 + A[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,1])**3 + B[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,1])**2 + C[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,1]) + D[twoRealDistinctInds]
# # residual = np.asarray([residual_0, residual_1]).T
# # assert np.all((np.real(residual) < 1e-8)*(np.imag(residual) < 1e-8)), 'All residual are not less than 1e-8'
# # del residual_0, residual_1
# indsOfRebellious_0 = np.where(np.real(residual_case1[:,0]) > 1e-1)[0]
# indsOfRebellious_1 = np.where(np.real(residual_case1[:,1]) > 1e-1)[0]
# indsOfRebellious = np.unique(np.concatenate((indsOfRebellious_0,indsOfRebellious_1)))
# xrealIndsOfRebellious = indsOf2RealSols[indsOfRebellious]

# #residual = tmpxreal2[twoRealDistinctInds[0]]**4 + A[twoRealDistinctInds[0]]*tmpxreal2[twoRealDistinctInds[0]]**3 + B[twoRealDistinctInds[0]]*tmpxreal2[twoRealDistinctInds[0]]**2 + C[twoRealDistinctInds[0]]*tmpxreal2[twoRealDistinctInds[0]] + D[twoRealDistinctInds[0]]
# #residual2 = xreal2[twoRealDistinctInds[0]]**4 + A[twoRealDistinctInds[0]]*xreal2[twoRealDistinctInds[0]]**3 + B[twoRealDistinctInds[0]]*xreal2[twoRealDistinctInds[0]]**2 + C[twoRealDistinctInds[0]]*xreal2[twoRealDistinctInds[0]] + D[twoRealDistinctInds[0]]
# #residual3 = np.real(xreal2[twoRealDistinctInds[0]])**4 + A[twoRealDistinctInds[0]]*np.real(xreal2[twoRealDistinctInds[0]])**3 + B[twoRealDistinctInds[0]]*np.real(xreal2[twoRealDistinctInds[0]])**2 + C[twoRealDistinctInds[0]]*np.real(xreal2[twoRealDistinctInds[0]]) + D[twoRealDistinctInds[0]]

# #currently getting intersection points that are not physically possible

# # case 2 All Real Distinct Inds
# xfinal[allRealDistinctInds] = np.real(xreal2[allRealDistinctInds])
# # residual_0 = xfinal[allRealDistinctInds,0]**4 + A[allRealDistinctInds]*xfinal[allRealDistinctInds,0]**3 + B[allRealDistinctInds]*xfinal[allRealDistinctInds,0]**2 + C[allRealDistinctInds]*xfinal[allRealDistinctInds,0] + D[allRealDistinctInds]
# # residual_1 = xfinal[allRealDistinctInds,1]**4 + A[allRealDistinctInds]*xfinal[allRealDistinctInds,1]**3 + B[allRealDistinctInds]*xfinal[allRealDistinctInds,1]**2 + C[allRealDistinctInds]*xfinal[allRealDistinctInds,1] + D[allRealDistinctInds]
# # residual_2 = xfinal[allRealDistinctInds,2]**4 + A[allRealDistinctInds]*xfinal[allRealDistinctInds,2]**3 + B[allRealDistinctInds]*xfinal[allRealDistinctInds,2]**2 + C[allRealDistinctInds]*xfinal[allRealDistinctInds,2] + D[allRealDistinctInds]
# # residual_3 = xfinal[allRealDistinctInds,3]**4 + A[allRealDistinctInds]*xfinal[allRealDistinctInds,3]**3 + B[allRealDistinctInds]*xfinal[allRealDistinctInds,3]**2 + C[allRealDistinctInds]*xfinal[allRealDistinctInds,3] + D[allRealDistinctInds]
# # residual = np.asarray([residual_0, residual_1, residual_2, residual_3]).T
# # assert np.all((np.real(residual) < 1e-7)*(np.imag(residual) < 1e-7)), 'All residual, All Real Distinct, are not less than 1e-7'
# # del residual_0, residual_1, residual_2, residual_3
# residual_case2, isAll_case2, maxRealResidual_case2, maxImagResidual_case2 = checkResiduals(A,B,C,D,xfinal,allRealDistinctInds,4)

# # case 3 All Imag Inds
# #NO REAL ROOTS
# #xreal2[allImagInds[0]]

# # # case 4 a real double root and 2 real solutions (2 real solutions which are identical and 2 other real solutions)
# # #xreal2[realDoubleRootTwoRealRootsInds[0]]
# # ximags2 = np.imag(xreal2[realDoubleRootTwoRealRootsInds])
# # ximags2smallImagInds = np.argsort(np.abs(ximags2),axis=1)[:,0:2] #sorts from smallest magnitude to largest magnitude
# # xrealDoubleRootTwoRealRoots = np.asarray([xreal2[realDoubleRootTwoRealRootsInds,ximags2smallImagInds[:,0]], xreal2[realDoubleRootTwoRealRootsInds,ximags2smallImagInds[:,1]]]).T
# # xfinal[realDoubleRootTwoRealRootsInds,0:2] = np.real(xrealDoubleRootTwoRealRoots)



# # # case 5 a real double root
# # #xreal2[realDoubleRootTwoComplexInds[0]]
# # ximags2 = np.imag(xreal2[realDoubleRootTwoComplexInds])
# # ximags2smallImagInds = np.argsort(np.abs(ximags2),axis=1)[:,0:2] #sorts from smallest magnitude to largest magnitude
# # xrealDoubleRootTwoComplex = np.asarray([xreal2[realDoubleRootTwoComplexInds,ximags2smallImagInds[:,0]], xreal2[realDoubleRootTwoComplexInds,ximags2smallImagInds[:,1]]]).T
# # xfinal[realDoubleRootTwoComplexInds,0:2] = np.real(xrealDoubleRootTwoComplex)

# yfinal = ellipseYFromX(xfinal, a, b)
# s_mpr, s_pmr, minSepr, maxSepr = calculateSeparations(xfinal, yfinal, mx, my)
# #TODO need to do what I did for the sepsMinMaxLminLmax function for x, y coordinate determination

# # #s_pm = np.asarray([np.sqrt((xreal[:,0]+mx)**2 + (yreal[:,0]-my)**2), np.sqrt((xreal[:,1]+mx)**2 + (yreal[:,1]-my)**2), np.sqrt((xreal[:,2]+mx)**2 + (yreal[:,2]-my)**2), np.sqrt((xreal[:,3]+mx)**2 + (yreal[:,3]-my)**2)]).T
# # s_pmr = np.asarray([np.sqrt((xfinal[:,0]+mx)**2 + (yfinal[:,0]-my)**2), np.sqrt((xfinal[:,1]+mx)**2 + (yfinal[:,1]-my)**2), np.sqrt((xfinal[:,2]+mx)**2 + (yfinal[:,2]-my)**2), np.sqrt((xfinal[:,3]+mx)**2 + (yfinal[:,3]-my)**2)]).T


# print('nanmin')
# print(np.nanmin(s_mpr[allRealDistinctInds]))
# print(np.nanmin(s_pmr[allRealDistinctInds]))
# print(np.nanmin(minSepr[allRealDistinctInds]))
# print(np.nanmin(maxSepr[allRealDistinctInds]))
# print(np.nanmin(s_mpr[indsOf2RealSols]))
# print(np.nanmin(s_pmr[indsOf2RealSols]))
# print(np.nanmin(minSepr[indsOf2RealSols]))
# print(np.nanmin(maxSepr[indsOf2RealSols]))
# print('nanmax')
# print(np.nanmax(s_mpr[allRealDistinctInds]))
# print(np.nanmax(s_pmr[allRealDistinctInds]))
# print(np.nanmax(minSepr[allRealDistinctInds]))
# print(np.nanmax(maxSepr[allRealDistinctInds]))
# print(np.nanmax(s_mpr[indsOf2RealSols]))
# print(np.nanmax(s_pmr[indsOf2RealSols]))
# print(np.nanmax(minSepr[indsOf2RealSols]))
# print(np.nanmax(maxSepr[indsOf2RealSols]))




