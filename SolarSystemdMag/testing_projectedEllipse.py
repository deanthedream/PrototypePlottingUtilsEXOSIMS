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
ellipseProjection3Dto2DInd = 23 #22
sma[ellipseProjection3Dto2DInd] = 1.2164387563540457
e[ellipseProjection3Dto2DInd] = 0.531071885292766
w[ellipseProjection3Dto2DInd] = 3.477496280463054
W[ellipseProjection3Dto2DInd] = 5.333215834002414
inc[ellipseProjection3Dto2DInd] = 1.025093642138022
####
#### SAVED PLANET FOR Plot Projected, derotated, centered ellipse 
derotatedInd = 33
sma[derotatedInd] = 5.738800898338014
e[derotatedInd] = 0.29306873405223816
w[derotatedInd] = 4.436383063578559
W[derotatedInd] = 4.240810639711751
inc[derotatedInd] = 1.072680736014668
####
#### SAVED PLANET FOR Plot Sep vs nu
sepvsnuInd = 24
sma[sepvsnuInd] = 1.817006521549392
e[sepvsnuInd] = 0.08651509983996385
W[sepvsnuInd] = 3.3708439025758006
w[sepvsnuInd] = 4.862116908343989
inc[sepvsnuInd] = 1.2491324942585256
####
#### SAVED PLANET FOR Plot Sep vs t
sepvstInd = 25
sma[sepvstInd] = 2.204556035394906
e[sepvstInd] = 0.2898368164549611
W[sepvstInd] = 4.787284415551434
w[sepvstInd] = 2.71176523941224
inc[sepvstInd] = 1.447634036719772
####


#### Calculate Projected Ellipse Angles and Minor Axis
start0 = time.time()
dmajorp, dminorp, Psi, psi, theta_OpQ_X, theta_OpQp_X, dmajorp_v2, dminorp_v2, Psi_v2, psi_v2 = projected_apbpPsipsi(sma,e,W,w,inc)
stop0 = time.time()
print('stop0: ' + str(stop0-start0))
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

a, b, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3,\
        yrealAllRealInds, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
        twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2, twoIntSameYInds,\
        type0_0Inds,type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,\
        type2_0Inds,type2_1Inds,type2_2Inds,type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,\
        allIndsUsed = ellipseCircleIntersections(None, a, b, mx, my, x, y, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds)
stop11 = time.time()
print('stop11: ' + str(stop11-start11))
del start11, stop11
####


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
        r_twoIntSameY0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntSameY[yind,0])
        r_twoIntSameY1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntSameY[yind,1])
        plt.scatter(nu_twoIntSameY[yind,0],np.sqrt(r_twoIntSameY0[0]**2 + r_twoIntSameY0[1]**2),color='green',marker='o')
        plt.scatter(nu_twoIntSameY[yind,1],np.sqrt(r_twoIntSameY1[0]**2 + r_twoIntSameY1[1]**2),color='green',marker='o')
    elif ind in yrealAllRealInds[twoIntOppositeXInds]: #Same X
        yind = np.where(yrealAllRealInds[twoIntOppositeXInds] == ind)[0]
        r_twoIntOppositeX0 = xyz_3Dellipse(a[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntOppositeX[yind,0])
        r_twoIntOppositeX1 = xyz_3Dellipse(a[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntOppositeX[yind,1])
        plt.scatter(nu_twoIntOppositeX[yind,0],np.sqrt(r_twoIntOppositeX0[0]**2 + r_twoIntOppositeX0[1]**2),color='green',marker='o')
        plt.scatter(nu_twoIntOppositeX[yind,1],np.sqrt(r_twoIntOppositeX1[0]**2 + r_twoIntOppositeX1[1]**2),color='green',marker='o')
    elif ind in only2RealInds:
        yind = np.where(only2RealInds == ind)[0]
        r_IntersectionOnly20 = xyz_3Dellipse(a[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,0])
        r_IntersectionOnly21 = xyz_3Dellipse(a[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,1])
        plt.scatter(nu_IntersectionsOnly2[yind,0],np.sqrt(r_IntersectionOnly20[0]**2 + r_IntersectionOnly20[1]**2),color='green',marker='o')
        plt.scatter(nu_IntersectionsOnly2[yind,1],np.sqrt(r_IntersectionOnly21[0]**2 + r_IntersectionOnly21[1]**2),color='green',marker='o')

    plt.xlim([0,2.*np.pi])
    plt.ylabel('Projected Separation, s, in AU',weight='bold')
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
#plt.close(num)
####

