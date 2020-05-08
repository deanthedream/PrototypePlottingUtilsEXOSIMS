import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
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
    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
    x_3Dellipse = r[0,0,:]
    y_3Dellipse = r[1,0,:]
    plt.plot(x_3Dellipse,y_3Dellipse,color='black')
    plt.scatter(Op[0][ind],Op[1][ind],color='black')
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
xreal_new, delta, P, D2, R, delta_0 = quarticSolutions_ellipse_to_Quarticipynb(A, B, C, D)
assert np.max(np.nanmin(np.abs(np.imag(xreal_new)),axis=1)) < 1e-15, 'At least one row has min > 1e-15' #this ensures each row has a solution
xreal_new = np.real(xreal_new)
# #xreal_new = np.sort(np.abs(xreal_new),axis=1) #trying this
# xreal_new = np.asarray([xreal_new[np.arange(xreal_new.shape[0]),np.argsort(np.abs(xreal_new),axis=1)[:,0]],xreal_new[np.arange(xreal_new.shape[0]),np.argsort(np.abs(xreal_new),axis=1)[:,1]],xreal_new[np.arange(xreal_new.shape[0]),np.argsort(np.abs(xreal_new),axis=1)[:,2]],xreal_new[np.arange(xreal_new.shape[0]),np.argsort(np.abs(xreal_new),axis=1)[:,3]]]).T #trying this

stop4_new = time.time()
print('stop4_new: ' + str(stop4_new-start4_new))


#Technically, each row must have at least 2 solutions, but whatever
yreal_new = ellipseYFromX(xreal_new.astype('complex128'), a, b)
yreal_newAllRealInds = np.where(np.all(np.abs(np.imag(yreal_new)) < 1e-5,axis=1))[0]
yreal_new[np.abs(np.imag(yreal_new)) < 1e-5] = np.real(yreal_new[np.abs(np.imag(yreal_new)) < 1e-5]) #eliminate any unreasonably small imaginary components
yreal_newImagInds = np.where(np.any(np.abs(np.imag(yreal_new)) >= 1e-5,axis=1))[0] #inds where any of the values are imaginary
assert len(yreal_newImagInds) + len(yreal_newAllRealInds) == n, 'For some reason, this sum does not account for all planets'
assert len(np.intersect1d(yreal_newImagInds,yreal_newAllRealInds)) == 0, 'For some reason, this sum does not account for all planets'
#The following 7 lines can be deleted. it just says the first 2 cols of yreal_new have the smallest imaginary component
yrealImagArgsortInds = np.argsort(np.imag(yreal_new[yreal_newImagInds]),axis=1)
assert len(yreal_newImagInds) == np.count_nonzero(yrealImagArgsortInds[:,0] == 0), "Not all first indicies have smallest Imag component"
assert len(yreal_newImagInds) == np.count_nonzero(yrealImagArgsortInds[:,1] == 1), "Not all first indicies have second smallest Imag component"
#maxImagFirstCol = np.max(np.imag(yreal_new[yreal_newImagInds,0]))
assert np.max(np.imag(yreal_new[yreal_newImagInds,0])) == 0, 'max y imag component of column 0 is not 0'
#maxImagSecondCol = np.max(np.imag(yreal_new[yreal_newImagInds,1]))
assert np.max(np.imag(yreal_new[yreal_newImagInds,1])) == 0, 'max y imag component of column 1 is not 0'
np.max(np.imag(yreal_new[yreal_newImagInds,2])) #this is quite large
np.max(np.imag(yreal_new[yreal_newImagInds,3])) #this is quite large


minSep = np.zeros(xreal_new.shape[0])
maxSep = np.zeros(xreal_new.shape[0])
###################################################################################
#### Smin and Smax Two Real Solutions Two Imaginary Solutions #####################
#Smin and Smax need to be calculated separately for x,y with imaginary solutions vs those without
#For yreal_newImagInds. Smin and Smax must be either first column of second column
assert np.all(np.real(xreal_new[yreal_newImagInds,0]) < 0), 'not all xreal components are strictly negative'
assert np.all(np.real(yreal_new[yreal_newImagInds,0]) > 0), 'not all yreal components are strictly positive'
assert np.all(np.real(xreal_new[yreal_newImagInds,1]) > 0), 'not all xreal components are strictly positive'
assert np.all(np.real(yreal_new[yreal_newImagInds,1]) > 0), 'not all yreal components are strictly positive'
smm0 = np.sqrt((np.real(xreal_new[yreal_newImagInds,0])-mx[yreal_newImagInds])**2 + (np.real(yreal_new[yreal_newImagInds,0])-mx[yreal_newImagInds])**2)
smp0 = np.sqrt((np.real(xreal_new[yreal_newImagInds,0])-mx[yreal_newImagInds])**2 + (np.real(yreal_new[yreal_newImagInds,0])+mx[yreal_newImagInds])**2)
spm0 = np.sqrt((np.real(xreal_new[yreal_newImagInds,0])+mx[yreal_newImagInds])**2 + (np.real(yreal_new[yreal_newImagInds,0])-mx[yreal_newImagInds])**2)
spp0 = np.sqrt((np.real(xreal_new[yreal_newImagInds,0])+mx[yreal_newImagInds])**2 + (np.real(yreal_new[yreal_newImagInds,0])+mx[yreal_newImagInds])**2)
smm1 = np.sqrt((np.real(xreal_new[yreal_newImagInds,1])-mx[yreal_newImagInds])**2 + (np.real(yreal_new[yreal_newImagInds,1])-mx[yreal_newImagInds])**2)
smp1 = np.sqrt((np.real(xreal_new[yreal_newImagInds,1])-mx[yreal_newImagInds])**2 + (np.real(yreal_new[yreal_newImagInds,1])+mx[yreal_newImagInds])**2)
spm1 = np.sqrt((np.real(xreal_new[yreal_newImagInds,1])+mx[yreal_newImagInds])**2 + (np.real(yreal_new[yreal_newImagInds,1])-mx[yreal_newImagInds])**2)
spp1 = np.sqrt((np.real(xreal_new[yreal_newImagInds,1])+mx[yreal_newImagInds])**2 + (np.real(yreal_new[yreal_newImagInds,1])+mx[yreal_newImagInds])**2)

#Search for Smallest
smm = np.asarray([smm0,smm1])
assert np.all(np.argmin(smm,axis=0) == 1), 'mins are not all are smm1'
smp = np.asarray([smp0,smp1])
assert np.all(np.argmin(smp,axis=0) == 1), 'mins are not all are smp1'
spm = np.asarray([spm0,spm1])
assert np.all(np.argmin(spm,axis=0) == 0), 'mins are not all are spm0'
spp = np.asarray([spp0,spp1])
assert np.all(np.argmin(spp,axis=0) == 0), 'mins are not all are spp0'
#above says smallest must be one of these: smm1, smp1, spm0, spp0
#The following are where each of these separations are 0
smm1Inds = np.where((smm1 < smp1)*(smm1 < spm0)*(smm1 < spp0))[0]
smp1Inds = np.where((smp1 < smm1)*(smp1 < spm0)*(smp1 < spp0))[0]
spm0Inds = np.where((spm0 < smp1)*(spm0 < smm1)*(spm0 < spp0))[0]
spp0Inds = np.where((spp0 < smp1)*(spp0 < spm0)*(spp0 < smm1))[0]
assert len(yreal_newImagInds) == len(smm1Inds) + len(smp1Inds) + len(spm0Inds) + len(spp0Inds), 'Have not covered all cases'
if len(smm1Inds) > 0:
    minSep[yreal_newImagInds[smm1Inds]] = smm1[smm1Inds]
if len(smp1Inds) > 0:
    minSep[yreal_newImagInds[smp1Inds]] = smp1[smp1Inds]
if len(spm0Inds) > 0:
    minSep[yreal_newImagInds[spm0Inds]] = smp0[spm0Inds]
if len(spp0Inds) > 0:
    minSep[yreal_newImagInds[spp0Inds]] = spp1[spp0Inds]
#above says largest must be one of these: smm0, smp0, spm1, spp1
smm0Inds = np.where((smm0 > smp0)*(smm0 > spm1)*(smm0 > spp1))[0]
smp0Inds = np.where((smp0 > smm0)*(smp0 > spm1)*(smp0 > spp1))[0]
spm1Inds = np.where((spm1 > smp0)*(spm1 > smm0)*(spm1 > spp1))[0]
spp1Inds = np.where((spp1 > smp0)*(spp1 > spm1)*(spp1 > smm0))[0]
if len(smm0Inds) > 0:
    maxSep[yreal_newImagInds[smm0Inds]] = smm0[smm0Inds]
if len(smp0Inds) > 0:
    maxSep[yreal_newImagInds[smp0Inds]] = smp0[smp0Inds]
if len(spm1Inds) > 0:
    maxSep[yreal_newImagInds[spm1Inds]] = smp1[spm1Inds]
if len(spp1Inds) > 0:
    maxSep[yreal_newImagInds[spp1Inds]] = spp1[spp1Inds]
#not currentyl assigning x,y values or lmin lmax for 2 solutions with 2 complex
########################################################
#### 4 Real Solutions ##################################
smm = np.zeros((4,len(yreal_newAllRealInds)))
smp = np.zeros((4,len(yreal_newAllRealInds)))
spm = np.zeros((4,len(yreal_newAllRealInds)))
spp = np.zeros((4,len(yreal_newAllRealInds)))
for i in [0,1,2,3]:
    smm[i] = np.sqrt((np.real(xreal_new[yreal_newAllRealInds,i])-mx[yreal_newAllRealInds])**2 + (np.real(yreal_new[yreal_newAllRealInds,i])-mx[yreal_newAllRealInds])**2)
    smp[i] = np.sqrt((np.real(xreal_new[yreal_newAllRealInds,i])-mx[yreal_newAllRealInds])**2 + (np.real(yreal_new[yreal_newAllRealInds,i])+mx[yreal_newAllRealInds])**2)
    spm[i] = np.sqrt((np.real(xreal_new[yreal_newAllRealInds,i])+mx[yreal_newAllRealInds])**2 + (np.real(yreal_new[yreal_newAllRealInds,i])-mx[yreal_newAllRealInds])**2)
    spp[i] = np.sqrt((np.real(xreal_new[yreal_newAllRealInds,i])+mx[yreal_newAllRealInds])**2 + (np.real(yreal_new[yreal_newAllRealInds,i])+mx[yreal_newAllRealInds])**2)
smm = smm.T
smp = smp.T
spm = spm.T
spp = spp.T
#### minSep
# Finds SXXY searching over Y
smmMinInds = np.argmin(smm,axis=1)
smpMinInds = np.argmin(smp,axis=1)
spmMinInds = np.argmin(spm,axis=1)
sppMinInds = np.argmin(spp,axis=1)
#All mins occur in SXX1 with the exception of 21. Do they come from another place? 2
#We can remove the mp and pp when searching for min. It appears none have this index for the minimum. found by np.unique(sXXMinInds)
sXXmins = np.asarray([smm[np.arange(len(yreal_newAllRealInds)),smmMinInds],smp[np.arange(len(yreal_newAllRealInds)),smpMinInds],spm[np.arange(len(yreal_newAllRealInds)),spmMinInds],spp[np.arange(len(yreal_newAllRealInds)),sppMinInds]]).T
sXXMinInds = np.argmin(sXXmins,axis=1)
minSep[yreal_newAllRealInds] = sXXmins[np.arange(len(yreal_newAllRealInds)),sXXMinInds]
#convert xreal used in smin to nan so I can use nanmin and nanmax
sXXInds1 = np.asarray([smmMinInds,smpMinInds,spmMinInds,sppMinInds]).T
elimInds1 = sXXInds1[np.arange(len(yreal_newAllRealInds)),sXXMinInds]
smm[np.arange(len(yreal_newAllRealInds)),elimInds1] = np.nan
smp[np.arange(len(yreal_newAllRealInds)),elimInds1] = np.nan
spm[np.arange(len(yreal_newAllRealInds)),elimInds1] = np.nan
spp[np.arange(len(yreal_newAllRealInds)),elimInds1] = np.nan
####
#### maxSep
# Finds SXXY searching over Y
smmMaxInds = np.nanargmax(smm,axis=1)
smpMaxInds = np.nanargmax(smp,axis=1)
spmMaxInds = np.nanargmax(spm,axis=1)
sppMaxInds = np.nanargmax(spp,axis=1)
#All mins occur in SXX1 with the exception of 21. Do they come from another place? 2
#We can remove the mp and pp when searching for min. It appears none have this index for the minimum. found by np.unique(sXXMinInds)
sXXmaxs = np.asarray([smm[np.arange(len(yreal_newAllRealInds)),smmMaxInds],smp[np.arange(len(yreal_newAllRealInds)),smpMaxInds],spm[np.arange(len(yreal_newAllRealInds)),spmMaxInds],spp[np.arange(len(yreal_newAllRealInds)),sppMaxInds]]).T
sXXMaxInds = np.nanargmax(sXXmaxs,axis=1) #CHOOSES BETWEEN smm, smp, spm, spp
maxSep[yreal_newAllRealInds] = sXXmaxs[np.arange(len(yreal_newAllRealInds)),sXXMaxInds]
#convert xreal used in smax to nan so I can use nanmin and nanmax
sXXInds0 = np.asarray([smmMaxInds,smpMaxInds,spmMaxInds,sppMaxInds]).T
elimInds0 = sXXInds0[np.arange(len(yreal_newAllRealInds)),sXXMaxInds]
smm[np.arange(len(yreal_newAllRealInds)),elimInds0] = np.nan
smp[np.arange(len(yreal_newAllRealInds)),elimInds0] = np.nan
spm[np.arange(len(yreal_newAllRealInds)),elimInds0] = np.nan
spp[np.arange(len(yreal_newAllRealInds)),elimInds0] = np.nan

np.unique(np.count_nonzero(np.isnan(smm),axis=1),return_counts=True)
assert ~np.any(elimInds0 == elimInds1), 'Oops, looks like some xreal were used for smin and smax...'
### slmin
#USE NANMIN on smm smp spm spp
### slmax
##################################################################################
assert ~np.any(minSep == 0), 'Oops, a minSep was missed'
assert ~np.any(maxSep == 0), 'Oops, a maxSep was missed'


#DELETE
#minSep = smm1 if the following are all true
# assert np.all(smm0 - smm1 < 1e-14), ''
# assert np.all(smm0 - smp1 < 1e-14), ''
# assert np.all(smm0 - spm1 < 1e-14), ''
# assert np.all(smm0 - spp1 < 1e-14), ''
# assert np.all(smm0 - smm0 < 1e-14), ''
# assert np.all(smm0 - smp0 < 1e-14), ''
# assert np.all(smm0 - spm0 < 1e-14), ''
# assert np.all(smm0 - spp0 < 1e-14), ''

#DELETE
# assert np.all(smm1 - smm1 < 1e-14), ''
# assert np.all(smm1 - smp1 < 1e-14), ''
# assert np.all(smm1 - spm1 < 1e-14), ''
# assert np.all(smm1 - spp1 < 1e-14), ''
# assert np.all(smm1 - smm0 < 1e-14), ''
# assert np.all(smm1 - smp0 < 1e-14), ''
# assert np.all(smm1 - spm0 < 1e-14), ''
# assert np.all(smm1 - spp0 < 1e-14), ''
# tinds = np.where(~(spm0 - smm1 < 1e-14))[0]
# smm0[tinds]

print(saltyburrito)

####
sepmm0 = np.sqrt((np.abs(xreal_new[:,0])-mx)**2 + (np.abs(np.real(yreal_new[:,0]))-my)**2) #minSep
sepmp0 = np.sqrt((np.abs(xreal_new[:,0])-mx)**2 + (np.abs(np.real(yreal_new[:,0]))+my)**2) 
seppm0 = np.sqrt((np.abs(xreal_new[:,0])+mx)**2 + (np.abs(np.real(yreal_new[:,0]))-my)**2) 
seppp0 = np.sqrt((np.abs(xreal_new[:,0])+mx)**2 + (np.abs(np.real(yreal_new[:,0]))+my)**2) 
sepmm1 = np.sqrt((np.abs(xreal_new[:,1])-mx)**2 + (np.abs(np.real(yreal_new[:,1]))-my)**2) 
sepmp1 = np.sqrt((np.abs(xreal_new[:,1])-mx)**2 + (np.abs(np.real(yreal_new[:,1]))+my)**2) 
seppm1 = np.sqrt((np.abs(xreal_new[:,1])+mx)**2 + (np.abs(np.real(yreal_new[:,1]))-my)**2) 
seppp1 = np.sqrt((np.abs(xreal_new[:,1])+mx)**2 + (np.abs(np.real(yreal_new[:,1]))+my)**2)
sepmm2 = np.sqrt((np.abs(xreal_new[:,2])-mx)**2 + (np.abs(np.real(yreal_new[:,2]))-my)**2)
sepmp2 = np.sqrt((np.abs(xreal_new[:,2])-mx)**2 + (np.abs(np.real(yreal_new[:,2]))+my)**2) 
seppm2 = np.sqrt((np.abs(xreal_new[:,2])+mx)**2 + (np.abs(np.real(yreal_new[:,2]))-my)**2) 
seppp2 = np.sqrt((np.abs(xreal_new[:,2])+mx)**2 + (np.abs(np.real(yreal_new[:,2]))+my)**2) 
sepmm3 = np.sqrt((np.abs(xreal_new[:,3])-mx)**2 + (np.abs(np.real(yreal_new[:,3]))-my)**2)
sepmp3 = np.sqrt((np.abs(xreal_new[:,3])-mx)**2 + (np.abs(np.real(yreal_new[:,3]))+my)**2) 
seppm3 = np.sqrt((np.abs(xreal_new[:,3])+mx)**2 + (np.abs(np.real(yreal_new[:,3]))-my)**2) 
seppp3 = np.sqrt((np.abs(xreal_new[:,3])+mx)**2 + (np.abs(np.real(yreal_new[:,3]))+my)**2) 

#####Sepmm0 is the minimum for each star IF
assert np.all(sepmm0 - sepmm0 < 1e-14), ''
assert np.all(sepmm0 - sepmp0 < 1e-14), ''
assert np.all(sepmm0 - seppm0 < 1e-14), ''
assert np.all(sepmm0 - seppp0 < 1e-14), ''
assert np.all(sepmm0 - sepmm1 < 1e-14), ''
assert np.all(sepmm0 - sepmp1 < 1e-14), ''
assert np.all(sepmm0 - seppm1 < 1e-14), ''
assert np.all(sepmm0 - seppp1 < 1e-14), ''
assert np.all(sepmm0 - sepmm2 < 1e-14), ''
assert np.all(sepmm0 - sepmp2 < 1e-14), ''
assert np.all(sepmm0 - seppm2 < 1e-14), ''
assert np.all(sepmm0 - seppp2 < 1e-14), ''
assert np.all(sepmm0 - sepmm3 < 1e-14), ''
assert np.all(sepmm0 - sepmp3 < 1e-14), ''
assert np.all(sepmm0 - seppm3 < 1e-14), ''
assert np.all(sepmm0 - seppp3 < 1e-14), ''

# #####Sepmm1 WAS the minimum for each star IF
# assert np.all(sepmm1 - sepmm0 < 1e-14), ''
# assert np.all(sepmm1 - sepmp0 < 1e-14), ''
# assert np.all(sepmm1 - seppm0 < 1e-14), ''
# assert np.all(sepmm1 - seppp0 < 1e-14), ''
# assert np.all(sepmm1 - sepmm1 < 1e-14), ''
# assert np.all(sepmm1 - sepmp1 < 1e-14), ''
# assert np.all(sepmm1 - seppm1 < 1e-14), ''
# assert np.all(sepmm1 - seppp1 < 1e-14), ''
# assert np.all(sepmm1 - sepmm2 < 1e-14), ''
# assert np.all(sepmm1 - sepmp2 < 1e-14), ''
# assert np.all(sepmm1 - seppm2 < 1e-14), ''
# assert np.all(sepmm1 - seppp2 < 1e-14), ''
# assert np.all(sepmm1 - sepmm3 < 1e-14), ''
# assert np.all(sepmm1 - sepmp3 < 1e-14), ''
# assert np.all(sepmm1 - seppm3 < 1e-14), ''
# assert np.all(sepmm1 - seppp3 < 1e-14), ''

#####Sepmm3 is the maximum for each star IF
assert np.all(seppp3 - sepmm0 > -1e-14), ''
assert np.all(seppp3 - sepmp0 > -1e-14), ''
assert np.all(seppp3 - seppm0 > -1e-14), ''
assert np.all(seppp3 - seppp0 > -1e-14), ''
assert np.all(seppp3 - sepmm1 > -1e-14), ''
assert np.all(seppp3 - sepmp1 > -1e-14), ''
assert np.all(seppp3 - seppm1 > -1e-14), ''
assert np.all(seppp3 - seppp1 > -1e-14), ''
assert np.all(seppp3 - sepmm2 > -1e-14), ''
assert np.all(seppp3 - sepmp2 > -1e-14), ''
assert np.all(seppp3 - seppm2 > -1e-14), ''
assert np.all(seppp3 - seppp2 > -1e-14), ''
assert np.all(seppp3 - sepmm3 > -1e-14), ''
assert np.all(seppp3 - sepmp3 > -1e-14), ''
assert np.all(seppp3 - seppm3 > -1e-14), ''
assert np.all(seppp3 - seppp3 > -1e-14), ''








#DELETEyreal_new2 = ellipseYFromX(np.real(xreal_new).astype('complex128'), a, b)
s_mp_new, s_pm_new, s_absmin_new, s_absmax_new = calculateSeparations(xreal_new, yreal_new, mx, my)
assert np.all(np.nanmin(s_absmin_new,axis=1) < np.nanmin(s_absmax_new,axis=1)), 'minimum of s_absmin_new < maximum of s_absmax_new'
np.count_nonzero(np.nanmin(s_absmin_new,axis=1) < np.nanmin(s_pm_new,axis=1))
np.count_nonzero(np.nanmin(s_absmin_new,axis=1) < np.nanmin(s_mp_new,axis=1))
residual_new, isAll_new, maxRealResidual_new, maxImagResidual_new = checkResiduals(A,B,C,D,xreal_new,np.arange(len(xreal_new)),4)

#### assign nans
#nanmask = (np.imag(xreal_new) > 2.5e-5) #where imaginary numbers are dominant

#### Smin, Smax, Lmin, Lmax Root Types For Each Planet #######################################################
# If delta > 0 and P < 0 and D < 0 four roots all real or none
allRealDistinctInds = np.where((delta > 0)*(P < 0)*(D2 < 0))[0] #METHOD 1, out of 10000, this found 1638, missing ~54
residual_allreal, isAll_allreal, maxRealResidual_allreal, maxImagResidual_allreal = checkResiduals(A,B,C,D,xreal_new,allRealDistinctInds,4)
#Check residual are sufficiently small
assert np.max(residual_allreal) < 1e-9, 'Not all residual_allreal detected using delta are sufficiently small'

# If delta < 0, two distinct real roots, two complex
xrealsImagInds = np.argsort(np.abs(np.imag(xreal_new)),axis=1)
xrealsImagInds2 = np.asarray([xrealsImagInds[:,0],xrealsImagInds[:,1]])
del xrealsImagInds
xrealOfSmallest2Imags = np.real(xreal_new[np.arange(len(a)),xrealsImagInds2]).T
ximagOfSmallest2Imags = np.imag(xreal_new[np.arange(len(a)),xrealsImagInds2]).T
twoRealTwoComplexInds = np.where((np.abs(ximagOfSmallest2Imags[:,0]) < 1e-9)*(np.abs(ximagOfSmallest2Imags[:,1]) < 1e-9)*~((delta > 0)*(P < 0)*(D2 < 0)))[0]
del ximagOfSmallest2Imags
xrealsTwoRealTwoComplex = np.real(np.asarray([xreal_new[twoRealTwoComplexInds,xrealsImagInds2[0,twoRealTwoComplexInds]],xreal_new[twoRealTwoComplexInds,xrealsImagInds2[1,twoRealTwoComplexInds]]]).T)
residual_twoRealTwoComplex, isAll_twoRealTwoComplex, maxRealResidual_twoRealTwoComplex, maxImagResidual_twoRealTwoComplex = checkResiduals(A[twoRealTwoComplexInds],B[twoRealTwoComplexInds],C[twoRealTwoComplexInds],D[twoRealTwoComplexInds],xrealsTwoRealTwoComplex,np.arange(len(xrealsTwoRealTwoComplex)),2)
#Check residual are sufficiently small
#assert np.max(residual_twoRealTwoComplex) < 1e-9, 'Not all residual_twoRealTwoComplex detected using delta are sufficiently small'
#Check ind intersection with allreal
assert len(np.intersect1d(allRealDistinctInds,twoRealTwoComplexInds)) == 0, 'There is intersection between Two Real Distinct and the 4 real solution inds, investigate'

tmp = np.zeros((len(twoRealTwoComplexInds), 2)) + np.nan
xreal_new[twoRealTwoComplexInds] = np.concatenate((xrealsTwoRealTwoComplex, tmp), axis=1)  
del tmp
minSepPoints2_x, minSepPoints2_y, maxSepPoints2_x, maxSepPoints2_y, lminSepPoints2_x, lminSepPoints2_y, lmaxSepPoints2_x, lmaxSepPoints2_y, minSep2, maxSep2, s_mplminSeps2, s_mplmaxSeps2 = sepsMinMaxLminLmax(s_absmin_new, s_absmax_new, s_mp_new, xreal_new, yreal_new, x, y)
np.count_nonzero(np.abs(np.imag(s_mplminSeps2))>1e-5) #finds large imaginary component values and counts how many there are
np.count_nonzero(np.abs(minSep2 - s_mplminSeps2) < 1e-5)


#Note: the solving method breaks down when the inclination is nearly zero and the star 
#Correction for 0 inclination planets where star is nearly centers in x and y
zeroIncCentralStarPlanets = np.where((np.abs(inc-np.pi/2) < 1e-3)*(mx < 5*1e-2)*(my < 1e-5))[0]
minSep2[zeroIncCentralStarPlanets] = s_mplminSeps2[zeroIncCentralStarPlanets]
minSepPoints2_x[zeroIncCentralStarPlanets] = lminSepPoints2_x[zeroIncCentralStarPlanets]
minSepPoints2_y[zeroIncCentralStarPlanets] = -lminSepPoints2_y[zeroIncCentralStarPlanets]

#### Old Method
# start5 = time.time()
# yreal = ellipseYFromX(xreal, a, b)
# stop5 = time.time()
# print('stop5: ' + str(stop5-start5))

#### Calculate Separations
# start6 = time.time()
# s_mp, s_pm, s_absmin, s_absmax = calculateSeparations(xreal, yreal, mx, my)
# stop6 = time.time()
# print('stop6: ' + str(stop6-start6))

#### Calculate Min Max Separation Points
# start7 = time.time()
# minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, s_mplminSeps, s_mplmaxSeps = sepsMinMaxLminLmax(s_absmin, s_absmax, s_mp, xreal, yreal, x, y)
# stop7 = time.time()
# print('stop7: ' + str(stop7-start7))
#################################################################################

print(saltyburrito)

#### Memory Usage
memories = [getsizeof(inc),getsizeof(W),getsizeof(w),getsizeof(sma),getsizeof(e),getsizeof(p),getsizeof(Rp),getsizeof(dmajorp),getsizeof(dminorp),getsizeof(Psi),getsizeof(psi),getsizeof(theta_OpQ_X),\
getsizeof(theta_OpQp_X),getsizeof(dmajorp_v2),getsizeof(dminorp_v2),getsizeof(Psi_v2),getsizeof(psi_v2),getsizeof(Op),getsizeof(x),getsizeof(y),getsizeof(Phi),getsizeof(a),getsizeof(b),\
getsizeof(mx),getsizeof(my),getsizeof(xreal),getsizeof(imag),getsizeof(yreal),getsizeof(s_mp),getsizeof(s_absmin),getsizeof(s_absmax),getsizeof(minSepPoints_x),getsizeof(minSepPoints_y),\
getsizeof(maxSepPoints_x),getsizeof(maxSepPoints_y),getsizeof(lminSepPoints_x),getsizeof(lminSepPoints_y),getsizeof(lmaxSepPoints_x),getsizeof(lmaxSepPoints_y),getsizeof(minSep),\
getsizeof(maxSep),getsizeof(s_mplminSeps),getsizeof(s_mplmaxSeps)]
totalMemoryUsage = np.sum(memories)
print('Total Data Used: ' + str(totalMemoryUsage/10**9) + ' GB')
####

#DELETE
#minSep
#np.where(np.abs(minSep - minSep2) > 1e-2)[0]



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

c_ae = a[ind]*np.sqrt(1-b[ind]**2/a[ind]**2)
plt.scatter([-c_ae,c_ae],[0,0],color='blue')

#Plot Min Sep Circle
x_circ = minSep2[ind]*np.cos(vs)
y_circ = minSep2[ind]*np.sin(vs)
plt.plot(x[ind]+x_circ,y[ind]+y_circ,color='cyan')

#Plot Max Sep Circle
x_circ2 = maxSep2[ind]*np.cos(vs)
y_circ2 = maxSep2[ind]*np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='red')

#Plot lminSep Circle
x_circ2 = s_mplminSeps2[ind]*np.cos(vs)
y_circ2 = s_mplminSeps2[ind]*np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='magenta')
#Plot lmaxSep Circle
x_circ2 = s_mplmaxSeps2[ind]*np.cos(vs)
y_circ2 = s_mplmaxSeps2[ind]*np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='gold')

#Plot Min Sep Ellipse Intersection
plt.scatter(minSepPoints2_x[ind],minSepPoints2_y[ind],color='cyan')
#Plot Max Sep Ellipse Intersection
plt.scatter(maxSepPoints2_x[ind],maxSepPoints2_y[ind],color='red')
#### Plot Local Min
plt.scatter(lminSepPoints2_x[ind], lminSepPoints2_y[ind],color='magenta')
#### Plot Local Max Points
plt.scatter(lmaxSepPoints2_x[ind], lmaxSepPoints2_y[ind],color='gold')

#### r Intersection test
x_circ2 = np.cos(vs)
y_circ2 = np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='green')

plt.show(block=False)




#### Plot separation vs vs parameter
num=961
plt.close(num)
fig = plt.figure(num=num)
xellipsetmp = a[ind]*np.cos(Erange)
yellipsetmp = b[ind]*np.sin(Erange)
septmp = (x[ind] - xellipsetmp)**2 + (y[ind] - yellipsetmp)**2
plt.plot(Erange,septmp,color='black')
plt.plot([0,2.*np.pi],[0,0],color='black',linestyle='--') #0 sep line
plt.xlim([0,2.*np.pi])
plt.ylabel('Projected Separation in AU')
plt.xlabel('Projected Ellipse E (rad)')
plt.show(block=False)
####



#### Testing ellipse_to_Quartic solution
r = np.ones(len(a),dtype='complex128')
a.astype('complex128')
b.astype('complex128')
mx.astype('complex128')
my.astype('complex128')
r.astype('complex128')
A, B, C, D = quarticCoefficients_ellipse_to_Quarticipynb(a, b, mx, my, r)
xreals2, delta, P, D2, R, delta_0 = quarticSolutions_ellipse_to_Quarticipynb(A, B, C, D)

# xreals2[np.abs(np.imag(xreals2)) > 1e-4] = np.nan #There is evidence from below that the residual resulting from entiring solutions with 3e-5j results in 0+1e-20j therefore we will nan above 1e-4
# xreals2 = np.real(xreals2)
yreals2 = ellipseYFromX(xreals2, a, b)
# seps2_0 = np.sqrt((xreals2[:,0]-x)**2 + (yreals2[:,0]-y)**2)
# seps2_1 = np.sqrt((xreals2[:,1]-x)**2 + (yreals2[:,1]-y)**2)
# seps2_2 = np.sqrt((xreals2[:,2]-x)**2 + (yreals2[:,2]-y)**2)
# seps2_3 = np.sqrt((xreals2[:,3]-x)**2 + (yreals2[:,3]-y)**2)
seps2_0 = np.sqrt((xreals2[:,0]-mx)**2 + (yreals2[:,0]-my)**2)
seps2_1 = np.sqrt((xreals2[:,1]-mx)**2 + (yreals2[:,1]-my)**2)
seps2_2 = np.sqrt((xreals2[:,2]-mx)**2 + (yreals2[:,2]-my)**2)
seps2_3 = np.sqrt((xreals2[:,3]-mx)**2 + (yreals2[:,3]-my)**2)
seps2 = np.asarray([seps2_0,seps2_1,seps2_2,seps2_3]).T

#we are currently omitting all of these potential calculations so-long-as the following assert is never true
#assert ~np.any(p2+p3**2/12 == 0), 'Oops, looks like the sympy piecewise was true once!'

#### Root Types For Each Planet #######################################################
#ORDER HAS BEEN CHANGED
# If delta > 0 and P < 0 and D < 0 four roots all real or none
#allRealDistinctInds = np.where((delta > 0)*(P < 0)*(D2 < 0))[0] #METHOD 1, out of 10000, this found 1638, missing ~54
allRealDistinctInds = np.where(np.all(np.abs(np.imag(xreals2)) < 2.5*1e-5, axis=1))[0]#1e-9, axis=1))[0] #This found 1692
residual_allreal, isAll_allreal, maxRealResidual_allreal, maxImagResidual_allreal = checkResiduals(A,B,C,D,xreals2,allRealDistinctInds,4)
assert maxRealResidual_allreal < 1e-9, 'At least one all real residual is too large'
# If delta < 0, two distinct real roots, two complex
#DELETEtwoRealDistinctInds = np.where(delta < 0)[0]
#DELETE UNNECESSARYxrealsImag = np.abs(np.imag(xreals2))
xrealsImagInds = np.argsort(np.abs(np.imag(xreals2)),axis=1)
xrealsImagInds2 = np.asarray([xrealsImagInds[:,0],xrealsImagInds[:,1]])
xrealOfSmallest2Imags = np.real(xreals2[np.arange(len(a)),xrealsImagInds2]).T
ximagOfSmallest2Imags = np.imag(xreals2[np.arange(len(a)),xrealsImagInds2]).T
#~np.all(np.abs(np.imag(xreals2)) < 1e-9, axis=1) removes solutions with 4 distinct real roots
#The other two are thresholds that happend to work well once
indsOf2RealSols = np.where((np.abs(ximagOfSmallest2Imags[:,0]) < 2.5*1e-5)*(np.abs(ximagOfSmallest2Imags[:,1]) < 2.5*1e-5)*~np.all(np.abs(np.imag(xreals2)) < 2.5*1e-5, axis=1))[0]
#DELETElen(indsOf2RealSols) - len(allRealDistinctInds)
xrealsTwoRealSols = np.real(np.asarray([xreals2[indsOf2RealSols,xrealsImagInds2[0,indsOf2RealSols]],xreals2[indsOf2RealSols,xrealsImagInds2[1,indsOf2RealSols]]]).T)
residual_TwoRealSols, isAll_TwoRealSols, maxRealResidual_TwoRealSols, maxImagResidual_TwoRealSols = checkResiduals(A[indsOf2RealSols],B[indsOf2RealSols],C[indsOf2RealSols],D[indsOf2RealSols],xrealsTwoRealSols,np.arange(len(xrealsTwoRealSols)),2)
assert len(np.intersect1d(allRealDistinctInds,indsOf2RealSols)) == 0, 'There is intersection between Two Real Distinct and the 4 real solution inds, investigate'

#DELETE cruft
# twoRealDistinctInds2 = np.where(np.all(np.abs(np.imag(xreals2)) < 1e-9, axis=1))[0] #This found 1692
# twoRealSorting = np.argsort(np.abs(np.imag(xreals2[twoRealDistinctInds,:])),axis=1)
# tmpxReals = np.asarray([xreals2[np.arange(len(twoRealDistinctInds)),twoRealSorting[:,0]], xreals2[np.arange(len(twoRealDistinctInds)),twoRealSorting[:,1]]]).T

# If delta > 0 and (P < 0 or D < 0)
#allImagInds = np.where((delta > 0)*((P > 0)|(D2 > 0)))[0]
#allImagInds = np.where(np.all(np.abs(np.imag(xreals2)) >= 1e-9, axis=1))[0]
allImagInds = np.where(np.all(np.abs(np.imag(xreals2)) >= 2.5*1e-5, axis=1))[0]
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
# numUnderThresh = np.sum(np.abs(np.imag(xreals2[twoRealDistinctInds])) > 1e-11, axis=1)
# indsUnderThresh = np.where(numUnderThresh != 2)[0]
# indsThatDontBelongIntwoRealDistinctInds = twoRealDistinctInds[indsUnderThresh]
# twoRealDistinctInds = np.delete(twoRealDistinctInds,indsThatDontBelongIntwoRealDistinctInds) #deletes the desired inds from aray
# #np.count_nonzero(numUnderThresh < 2)

#DELETE IN FUTURE
#The 1e-5 here gave me the number as the Imag count
#allRealDistinctInds2 = np.where(np.all(np.abs(np.imag(xreals2)) > 1e-5, axis=1))[0]
#allRealDistinctInds2 = np.where(np.all(np.abs(np.imag(xreals2)) > 1e-9, axis=1))[0]


#Number of Solutions of Each Type
numRootInds = [indsOf2RealSols,allRealDistinctInds,allImagInds,realDoubleRootTwoRealRootsInds,realDoubleRootTwoComplexInds,\
    tripleRootSimpleRootInds,twoRealDoubleRootsInds,twoComplexDoubleRootsInds,fourIdenticalRealRootsInds]

#Number of Roots of Each Type
lenNumRootsInds = [len(numRootInds[i]) for i in np.arange(len(numRootInds))]
assert len(indsOf2RealSols)+len(allRealDistinctInds)+len(allImagInds)-len(realDoubleRootTwoRealRootsInds), 'Number of roots does not add up, investigate'
########################################################################


# Calculate Residuals
# residual_0 = xreals2[:,0]**4 + A*xreals2[:,0]**3 + B*xreals2[:,0]**2 + C*xreals2[:,0] + D
# residual_1 = xreals2[:,1]**4 + A*xreals2[:,1]**3 + B*xreals2[:,1]**2 + C*xreals2[:,1] + D
# residual_2 = xreals2[:,2]**4 + A*xreals2[:,2]**3 + B*xreals2[:,2]**2 + C*xreals2[:,2] + D
# residual_3 = xreals2[:,3]**4 + A*xreals2[:,3]**3 + B*xreals2[:,3]**2 + C*xreals2[:,3] + D
# residual = np.asarray([residual_0, residual_1, residual_2, residual_3]).T
# #assert np.all((np.real(residual) < 1e-7)*(np.imag(residual) < 1e-7)), 'All residual are not less than 1e-7'
# del residual_0, residual_1, residual_2, residual_3
residual_all, isAll_all, maxRealResidual_all, maxImagResidual_all = checkResiduals(A,B,C,D,xreals2,np.arange(len(A)),4)



xfinal = np.zeros(xreals2.shape) + np.nan
# case 1 Two Real Distinct Inds
#find 2 xsols with smallest imag part
#xreals2[indsOf2RealSols[0]]
#ximags2 = np.imag(xreals2[indsOf2RealSols])
#ximags2smallImagInds = np.argsort(np.abs(ximags2),axis=1)[:,0:2] #sorts from smallest magnitude to largest magnitude
#xrealsTwoRealDistinct = np.asarray([xreals2[indsOf2RealSols,ximags2smallImagInds[:,0]], xreals2[indsOf2RealSols,ximags2smallImagInds[:,1]]]).T
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

#residual = tmpxreals2[twoRealDistinctInds[0]]**4 + A[twoRealDistinctInds[0]]*tmpxreals2[twoRealDistinctInds[0]]**3 + B[twoRealDistinctInds[0]]*tmpxreals2[twoRealDistinctInds[0]]**2 + C[twoRealDistinctInds[0]]*tmpxreals2[twoRealDistinctInds[0]] + D[twoRealDistinctInds[0]]
#residual2 = xreals2[twoRealDistinctInds[0]]**4 + A[twoRealDistinctInds[0]]*xreals2[twoRealDistinctInds[0]]**3 + B[twoRealDistinctInds[0]]*xreals2[twoRealDistinctInds[0]]**2 + C[twoRealDistinctInds[0]]*xreals2[twoRealDistinctInds[0]] + D[twoRealDistinctInds[0]]
#residual3 = np.real(xreals2[twoRealDistinctInds[0]])**4 + A[twoRealDistinctInds[0]]*np.real(xreals2[twoRealDistinctInds[0]])**3 + B[twoRealDistinctInds[0]]*np.real(xreals2[twoRealDistinctInds[0]])**2 + C[twoRealDistinctInds[0]]*np.real(xreals2[twoRealDistinctInds[0]]) + D[twoRealDistinctInds[0]]

#currently getting intersection points that are not physically possible

# case 2 All Real Distinct Inds
xfinal[allRealDistinctInds] = np.real(xreals2[allRealDistinctInds])
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
#xreals2[allImagInds[0]]

# # case 4 a real double root and 2 real solutions (2 real solutions which are identical and 2 other real solutions)
# #xreals2[realDoubleRootTwoRealRootsInds[0]]
# ximags2 = np.imag(xreals2[realDoubleRootTwoRealRootsInds])
# ximags2smallImagInds = np.argsort(np.abs(ximags2),axis=1)[:,0:2] #sorts from smallest magnitude to largest magnitude
# xrealDoubleRootTwoRealRoots = np.asarray([xreals2[realDoubleRootTwoRealRootsInds,ximags2smallImagInds[:,0]], xreals2[realDoubleRootTwoRealRootsInds,ximags2smallImagInds[:,1]]]).T
# xfinal[realDoubleRootTwoRealRootsInds,0:2] = np.real(xrealDoubleRootTwoRealRoots)



# # case 5 a real double root
# #xreals2[realDoubleRootTwoComplexInds[0]]
# ximags2 = np.imag(xreals2[realDoubleRootTwoComplexInds])
# ximags2smallImagInds = np.argsort(np.abs(ximags2),axis=1)[:,0:2] #sorts from smallest magnitude to largest magnitude
# xrealDoubleRootTwoComplex = np.asarray([xreals2[realDoubleRootTwoComplexInds,ximags2smallImagInds[:,0]], xreals2[realDoubleRootTwoComplexInds,ximags2smallImagInds[:,1]]]).T
# xfinal[realDoubleRootTwoComplexInds,0:2] = np.real(xrealDoubleRootTwoComplex)

yfinal = ellipseYFromX(xfinal, a, b)
s_mpr, s_pmr, s_absminr, s_absmaxr = calculateSeparations(xfinal, yfinal, mx, my)
#TODO need to do what I did for the sepsMinMaxLminLmax function for x, y coordinate determination

# #s_pm = np.asarray([np.sqrt((xreal[:,0]+mx)**2 + (yreal[:,0]-my)**2), np.sqrt((xreal[:,1]+mx)**2 + (yreal[:,1]-my)**2), np.sqrt((xreal[:,2]+mx)**2 + (yreal[:,2]-my)**2), np.sqrt((xreal[:,3]+mx)**2 + (yreal[:,3]-my)**2)]).T
# s_pmr = np.asarray([np.sqrt((xfinal[:,0]+mx)**2 + (yfinal[:,0]-my)**2), np.sqrt((xfinal[:,1]+mx)**2 + (yfinal[:,1]-my)**2), np.sqrt((xfinal[:,2]+mx)**2 + (yfinal[:,2]-my)**2), np.sqrt((xfinal[:,3]+mx)**2 + (yfinal[:,3]-my)**2)]).T


print('nanmin')
print(np.nanmin(s_mpr[allRealDistinctInds]))
print(np.nanmin(s_pmr[allRealDistinctInds]))
print(np.nanmin(s_absminr[allRealDistinctInds]))
print(np.nanmin(s_absmaxr[allRealDistinctInds]))
print(np.nanmin(s_mpr[indsOf2RealSols]))
print(np.nanmin(s_pmr[indsOf2RealSols]))
print(np.nanmin(s_absminr[indsOf2RealSols]))
print(np.nanmin(s_absmaxr[indsOf2RealSols]))
print('nanmax')
print(np.nanmax(s_mpr[allRealDistinctInds]))
print(np.nanmax(s_pmr[allRealDistinctInds]))
print(np.nanmax(s_absminr[allRealDistinctInds]))
print(np.nanmax(s_absmaxr[allRealDistinctInds]))
print(np.nanmax(s_mpr[indsOf2RealSols]))
print(np.nanmax(s_pmr[indsOf2RealSols]))
print(np.nanmax(s_absminr[indsOf2RealSols]))
print(np.nanmax(s_absmaxr[indsOf2RealSols]))

PMRsols = np.asarray([np.nanmin(s_mpr[allRealDistinctInds],axis=1),np.nanmin(s_pmr[allRealDistinctInds],axis=1),np.nanmin(s_absminr[allRealDistinctInds],axis=1),np.nanmin(s_absmaxr[allRealDistinctInds],axis=1)]).T

bool1 = x > 0
bool2 = y > 0
s_mpr2, s_pmr2, s_absminr2, s_absmaxr2 = calculateSeparations(xfinal, yfinal, x, y)


np.nanmin(s_absminr,axis=1)

#### Notes
#If r < smin, then all imag
#if r > smin and r > slmin, then 2 real.
#if r > slmin and r < slmax, then 4 real.
#if r < smax and r > slmax, then 2 real.
#if r > smax, then all imag.



minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, s_mplminSeps, s_mplmaxSeps = sepsMinMaxLminLmax(s_absmin, s_absmax, s_mp, xreal, yreal, x, y)

