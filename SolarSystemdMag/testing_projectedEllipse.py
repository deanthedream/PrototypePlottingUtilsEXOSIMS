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
xreal, delta, P, D2, R, delta_0 = quarticSolutions_ellipse_to_Quarticipynb(A.astype('complex128'), B, C, D)
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
x_circ = minSep[ind]*np.cos(vs)
y_circ = minSep[ind]*np.sin(vs)
plt.plot(x[ind]+x_circ,y[ind]+y_circ,color='cyan')
#Plot Max Sep Circle
x_circ2 = maxSep[ind]*np.cos(vs)
y_circ2 = maxSep[ind]*np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='red')
#Plot Min Sep Ellipse Intersection
plt.scatter(minSepPoints_x[ind],minSepPoints_y[ind],color='cyan')
#Plot Max Sep Ellipse Intersection
plt.scatter(maxSepPoints_x[ind],maxSepPoints_y[ind],color='red')

if ind in yrealAllRealInds:
    tind = np.where(yrealAllRealInds == ind)[0]
    #Plot lminSep Circle
    x_circ2 = lminSep[tind]*np.cos(vs)
    y_circ2 = lminSep[tind]*np.sin(vs)
    plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='magenta')
    #Plot lmaxSep Circle
    x_circ2 = lmaxSep[tind]*np.cos(vs)
    y_circ2 = lmaxSep[tind]*np.sin(vs)
    plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='gold')
    #### Plot Local Min
    plt.scatter(lminSepPoints_x[tind], lminSepPoints_y[tind],color='magenta')
    #### Plot Local Max Points
    plt.scatter(lmaxSepPoints_x[tind], lmaxSepPoints_y[tind],color='gold')

# #### r Intersection test
# x_circ2 = np.cos(vs)
# y_circ2 = np.sin(vs)
# plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='green')

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

