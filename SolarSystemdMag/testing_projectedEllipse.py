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

#### PLOT BOOL
plotBool = True
if plotBool == True:
    from plotProjectedEllipse import *

#### Randomly Generate Orbits
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))
filename = 'HabEx_CKL2_PPKL2.json'
filename = 'WFIRSTcycle6core.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
PPop = sim.PlanetPopulation
n = 10**5 #Dean's nice computer can go up to 10**8 what can atuin go up to?
inc, W, w = PPop.gen_angles(n,None)
W = W.to('rad').value
w = w.to('rad').value
#w correction caused in smin smax calcs
wReplacementInds = np.where(np.abs(w-1.5*np.pi)<1e-4)[0]
w[wReplacementInds] = w[wReplacementInds] - 0.001
wReplacementInds = np.where(np.abs(w-0.5*np.pi)<1e-4)[0]
w[wReplacementInds] = w[wReplacementInds] - 0.001
del wReplacementInds
inc = inc.to('rad').value
inc[np.where(inc>np.pi/2)[0]] = np.pi - inc[np.where(inc>np.pi/2)[0]]
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
dmajorp, dminorp, theta_OpQ_X, theta_OpQp_X = projected_apbpPsipsi(sma,e,W,w,inc)#dmajorp_v2, dminorp_v2, Psi_v2, psi_v2, Psi, psi,
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

#### Derotate Ellipse Calculations
start5 = time.time()
x, y, Phi = derotatedEllipse(theta_OpQ_X, theta_OpQp_X, Op)
stop5 = time.time()
print('stop5: ' + str(stop5-start5))
del start5, stop5
#DELETEa = dmajorp
#DELETEb = dminorp
####

#### Calculate X,Y Position of Minimum and Maximums with Quartic
start7 = time.time()
A, B, C, D = quarticCoefficients_smin_smax_lmin_lmax(dmajorp.astype('complex128'), dminorp, np.abs(x), np.abs(y))
#xreal, delta, P, D2, R, delta_0 = quarticSolutions_ellipse_to_Quarticipynb(A.astype('complex128'), B, C, D)
xreal, _, _, _, _, _ = quarticSolutions_ellipse_to_Quarticipynb(A.astype('complex128'), B, C, D)
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

#### Technically, each row must have at least 2 solutions, but whatever
start8 = time.time()
yreal = ellipseYFromX(xreal.astype('complex128'), dmajorp, dminorp)
stop8 = time.time()
print('stop8: ' + str(stop8-start8))
del start8, stop8
####

#### Calculate Minimum, Maximum, Local Minimum, Local Maximum Separations
start9 = time.time()
minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds = smin_smax_slmin_slmax(n, xreal, yreal, np.abs(x), np.abs(y), x, y)
lminSepPoints_x = np.real(lminSepPoints_x)
lminSepPoints_y = np.real(lminSepPoints_y)
lmaxSepPoints_x = np.real(lmaxSepPoints_x)
lmaxSepPoints_y = np.real(lmaxSepPoints_y)
stop9 = time.time()
print('stop9: ' + str(stop9-start9))
del start9, stop9
####

#### Ellipse Circle Intersection #######################################################################
start11 = time.time()
r = np.ones(len(sma))

dmajorp, dminorp, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3,\
        yrealAllRealInds, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
        twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2, twoIntSameYInds,\
        type0_0Inds,type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,\
        type2_0Inds,type2_1Inds,type2_2Inds,type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,\
        _ = ellipseCircleIntersections(None, dmajorp, dminorp, np.abs(x), np.abs(y), x, y, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds)
if plotBool == False:
    del typeInds0, typeInds1, typeInds2, typeInds3
    del type0_0Inds,type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds
    del type2_0Inds,type2_1Inds,type2_2Inds,type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds
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

#### Rerotate Extrema and Intersection Points
start13 = time.time()

minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,\
    fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr = \
    rerotateExtremaAndIntersectionPoints(minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y,\
    fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
    Phi, Op, yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds)
if plotBool == False:
    del minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y
    del fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2
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
del minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr
del fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr
stop14 = time.time()
print('stop14: ' + str(stop14-start14))
del start14, stop14
#Now can I delete the x,y points?
#del minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, fourInt_x, fourInt_y
#del twoIntSameY_x, twoIntSameY_y, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2
####

#### Fix minSep True Anomalies
nu_minSepPoints = nuCorrections_extrema(sma,e,W,w,inc,nu_minSepPoints,np.arange(len(sma)),minSep)
####
#### Fix maxSep True Anomalies
nu_maxSepPoints = nuCorrections_extrema(sma,e,W,w,inc,nu_maxSepPoints,np.arange(len(sma)),maxSep)
####
#### Fix lminSep True Anomalies
nu_lminSepPoints = nuCorrections_extrema(sma,e,W,w,inc,nu_lminSepPoints,yrealAllRealInds,lminSep)
####
#### Fix lmaxSep True Anomalies
nu_lmaxSepPoints = nuCorrections_extrema(sma,e,W,w,inc,nu_lmaxSepPoints,yrealAllRealInds,lmaxSep)
####

#### Correcting nu for ellipse-circle intersections
#### yrealAllRealInds[fourIntInds]
nu_fourInt[:,0], errors_fourInt0 = nuCorrections_int(sma,e,W,w,inc,r,nu_fourInt[:,0],yrealAllRealInds,fourIntInds)
nu_fourInt[:,1], errors_fourInt1 = nuCorrections_int(sma,e,W,w,inc,r,nu_fourInt[:,1],yrealAllRealInds,fourIntInds)
nu_fourInt[:,2], errors_fourInt2 = nuCorrections_int(sma,e,W,w,inc,r,nu_fourInt[:,2],yrealAllRealInds,fourIntInds)
nu_fourInt[:,3], errors_fourInt3 = nuCorrections_int(sma,e,W,w,inc,r,nu_fourInt[:,3],yrealAllRealInds,fourIntInds)
if plotBool == False:
    del errors_fourInt0, errors_fourInt1, errors_fourInt2, errors_fourInt3
####
#### yrealAllRealInds[twoIntSameYInds]
nu_twoIntSameY[:,0], errors_twoIntSameY0 = nuCorrections_int(sma,e,W,w,inc,r,nu_twoIntSameY[:,0],yrealAllRealInds,twoIntSameYInds)
nu_twoIntSameY[:,1], errors_twoIntSameY1 = nuCorrections_int(sma,e,W,w,inc,r,nu_twoIntSameY[:,1],yrealAllRealInds,twoIntSameYInds)
if plotBool == False:
    del errors_twoIntSameY0, errors_twoIntSameY1
####
#### yrealAllRealInds[twoIntOppositeXInds]
nu_twoIntOppositeX[:,0], errors_twoIntOppositeX0 = nuCorrections_int(sma,e,W,w,inc,r,nu_twoIntOppositeX[:,0],yrealAllRealInds,twoIntOppositeXInds)
nu_twoIntOppositeX[:,1], errors_twoIntOppositeX1 = nuCorrections_int(sma,e,W,w,inc,r,nu_twoIntOppositeX[:,1],yrealAllRealInds,twoIntOppositeXInds)
if plotBool == False:
    del errors_twoIntOppositeX0, errors_twoIntOppositeX1
####
#### only2RealInds
nu_IntersectionsOnly2[:,0], errors_IntersectionsOnly2X0 = nuCorrections_int(sma,e,W,w,inc,r,nu_IntersectionsOnly2[:,0],np.arange(len(sma)),only2RealInds)
nu_IntersectionsOnly2[:,1], errors_IntersectionsOnly2X1 = nuCorrections_int(sma,e,W,w,inc,r,nu_IntersectionsOnly2[:,1],np.arange(len(sma)),only2RealInds)
if plotBool == False:
    del errors_IntersectionsOnly2X0, errors_IntersectionsOnly2X1
####

#### Memory Calculations
#Necessary Variables
memory_necessary = [inc.nbytes,
w.nbytes,
W.nbytes,
sma.nbytes,
e.nbytes,
p.nbytes,
Rp.nbytes,
dmajorp.nbytes,
dminorp.nbytes,
theta_OpQ_X.nbytes,
theta_OpQp_X.nbytes,
Op.nbytes,
x.nbytes,
y.nbytes,
Phi.nbytes,
xreal.nbytes,
only2RealInds.nbytes,
yrealAllRealInds.nbytes,
fourIntInds.nbytes,
twoIntOppositeXInds.nbytes,
twoIntSameYInds.nbytes,
nu_minSepPoints.nbytes,
nu_maxSepPoints.nbytes,
nu_lminSepPoints.nbytes,
nu_lmaxSepPoints.nbytes,
nu_fourInt.nbytes,
nu_twoIntSameY.nbytes,
nu_twoIntOppositeX.nbytes,
nu_IntersectionsOnly2.nbytes]
print('memory_necessary Used: ' + str(np.sum(memory_necessary)/10**9) + ' GB')


#Things currently calculated, used, and later deleted
#A, B, C, D
#minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr
#fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr

# Vestigal Variables
#TODO a and b are duplicates of dmajorp and dminorp
memory_vestigal = [0]
#a.nbytes,b.nbytes,
#error_numinSep.nbytes,error_numaxSep.nbytes,error_nulminSep.nbytes,error_nulmaxSep.nbytes,
#dmajorp_v2.nbytes,dminorp_v2.nbytes,Psi_v2.nbytes,psi_v2.nbytes,Psi.nbytes,psi.nbytes,
#delta.nbytes,delta_0.nbytes,P.nbytes, #not 100% sureD2.nbytes,R.nbytes,
#allIndsUsed.nbytes
print('memory_vestigal Used: ' + str(np.sum(memory_vestigal)/10**9) + ' GB')

# Variables Only For Plotting
if plotBool == True:
    memory_plotting = [errors_fourInt0.nbytes,
    errors_fourInt1.nbytes,
    errors_fourInt2.nbytes,
    errors_fourInt3.nbytes,
    errors_twoIntSameY0.nbytes,
    errors_twoIntSameY1.nbytes,
    errors_twoIntOppositeX0.nbytes,
    errors_twoIntOppositeX1.nbytes,
    errors_IntersectionsOnly2X0.nbytes,
    errors_IntersectionsOnly2X1.nbytes,
    type0_0Inds.nbytes,
    type0_1Inds.nbytes,
    type0_2Inds.nbytes,
    type0_3Inds.nbytes,
    type0_4Inds.nbytes,
    type1_0Inds.nbytes,
    type1_1Inds.nbytes,
    type1_2Inds.nbytes,
    type1_3Inds.nbytes,
    type1_4Inds.nbytes,
    type2_0Inds.nbytes,
    type2_1Inds.nbytes,
    type2_2Inds.nbytes,
    type2_3Inds.nbytes,
    type2_4Inds.nbytes,
    type3_0Inds.nbytes,
    type3_1Inds.nbytes,
    type3_2Inds.nbytes,
    type3_3Inds.nbytes,
    type3_4Inds.nbytes,
    fourInt_x.nbytes,
    fourInt_y.nbytes,
    twoIntSameY_x.nbytes,
    twoIntSameY_y.nbytes,
    twoIntOppositeX_x.nbytes,
    twoIntOppositeX_y.nbytes,
    xIntersectionsOnly2.nbytes,
    yIntersectionsOnly2.nbytes,
    typeInds0.nbytes,
    typeInds1.nbytes,
    typeInds2.nbytes,
    typeInds3.nbytes]
    print('memory_plotting Used: ' + str(np.sum(memory_plotting)/10**9) + ' GB')

#### START ANALYSIS AND PLOTTING ######################################
#######################################################################

#### Plotting Projected Ellipse
start2 = time.time()
ind = random.randint(low=0,high=n)
plotProjectedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, num=877)
stop2 = time.time()
print('stop2: ' + str(stop2-start2))
del start2, stop2
plt.close(877)
####

#### Plot 3D Ellipse to 2D Ellipse Projection Diagram
start3 = time.time()
num = 666999888777
plot3DEllipseto2DEllipseProjectionDiagram(ind, sma, e, W, w, inc, Op, theta_OpQ_X, theta_OpQp_X,\
    dmajorp, dminorp, num=num)
stop3 = time.time()
print('stop3: ' + str(stop3-start3))
del start3, stop3
plt.close(num)
####

#### Create Projected Ellipse Conjugate Diameters and QQ' construction diagram
start4 = time.time()
num = 3335555888
plotEllipseMajorAxisFromConjugate(ind, sma, e, W, w, inc, Op, theta_OpQ_X, theta_OpQp_X,\
    dmajorp, dminorp, num)
stop4 = time.time()
print('stop4: ' + str(stop4-start4))
del start4, stop4
plt.close(num)
####

#### Plot Derotated Ellipse
start6 = time.time()
num=880
plotDerotatedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, x, y, num)
stop6 = time.time()
print('stop6: ' + str(stop6-start6))
del start6, stop6
plt.close(num)
####

##### Plot Proving Rerotation method works
start10 = time.time()

num=883
plotReorientationMethod(ind, sma, e, W, w, inc, x, y, Phi, Op, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp,\
    minSepPoints_x, minSepPoints_y, num)
stop10 = time.time()
print('stop10: ' + str(stop10-start10))
del start10, stop10
plt.close(num)
####

#### Plot Derotated Intersections, Min/Max, and Star Location Type Bounds
start12 = time.time()
num = 960
plotDerotatedIntersectionsMinMaxStarLocBounds(ind, x, y, dmajorp, dminorp, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
    minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
    lmaxSepPoints_x, lmaxSepPoints_y, twoIntSameYInds,\
    maxSepPoints_x, maxSepPoints_y, twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
    type0_0Inds, type0_1Inds, type0_2Inds, type0_3Inds, type0_4Inds, type1_0Inds, type1_1Inds, type1_2Inds, type1_3Inds, type1_4Inds,\
    type2_0Inds, type2_1Inds, type2_2Inds, type2_3Inds, type2_4Inds, type3_0Inds, type3_1Inds, type3_2Inds, type3_3Inds, type3_4Inds, num)
stop12 = time.time()
print('stop12: ' + str(stop12-start12))
del start12, stop12
plt.close(num)
####

#### Plot Derotated Ellipse Separation Extrema
start12_1 = time.time()
num = 961
plotDerotatedExtrema(derotatedInd, x, y, dmajorp, dminorp, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
    maxSepPoints_x, maxSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y,\
    minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
    twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2, num)
stop12_1 = time.time()
print('stop12_1: ' + str(stop12_1-start12_1))
del start12_1, stop12_1
####


#### Plot Rerotated Points 
#### Error Plot ####
num=822
errorLinePlot(fourIntInds,errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,\
    twoIntSameYInds,errors_twoIntSameY0,errors_twoIntSameY1,twoIntOppositeXInds,errors_twoIntOppositeX0,errors_twoIntOppositeX1,\
    only2RealInds,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,num)
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
num= 823
plotErrorHistogramAlpha(errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,errors_twoIntSameY0,errors_twoIntSameY1,\
    errors_twoIntOppositeX0,errors_twoIntOppositeX1,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,num)
plt.close(num) #thinking the above plot is relativly useless
####

#### Plot Histogram of Error
num=824
plotErrorHistogram(errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,\
    errors_twoIntSameY0,errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,\
    errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,num)
plt.close(num)
####

#### Redo Significant Point plot Using these Nu
num=3690
plotProjectedEllipseWithNu(ind,sma,e,W,w,inc,nu_minSepPoints,nu_maxSepPoints, yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds,\
    only2RealInds, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2, num)
####

#### Plot separation vs nu
num=962
plotSeparationvsnu(ind, sma, e, W, w, inc, minSep, maxSep, lminSep, lmaxSep, \
    nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints,\
    nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
    yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, num)
####


#### Plot separation vs time
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
plotDerotatedEllipseStarLocDividers(ind, x, y, dmajorp, dminorp, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
    minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
    lmaxSepPoints_x, lmaxSepPoints_y, twoIntSameYInds,\
    maxSepPoints_x, maxSepPoints_y, twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
    type0_0Inds, type0_1Inds, type0_2Inds, type0_3Inds, type0_4Inds, type1_0Inds, type1_1Inds, type1_2Inds, type1_3Inds, type1_4Inds,\
    type2_0Inds, type2_1Inds, type2_2Inds, type2_3Inds, type2_4Inds, type3_0Inds, type3_1Inds, type3_2Inds, type3_3Inds, type3_4Inds, num)
#plt.close(num)
####

