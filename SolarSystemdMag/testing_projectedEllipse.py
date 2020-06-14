import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as random
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

#Separations
s_circle = np.ones(len(sma))

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


dmajorp,dminorp,theta_OpQ_X,theta_OpQp_X,Op,x,y,Phi,xreal,only2RealInds,yrealAllRealInds,\
    fourIntInds,twoIntOppositeXInds,twoIntSameYInds,nu_minSepPoints,nu_maxSepPoints,nu_lminSepPoints,nu_lmaxSepPoints,nu_fourInt,\
    nu_twoIntSameY,nu_twoIntOppositeX,nu_IntersectionsOnly2, yrealImagInds,\
    minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep,\
    errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,errors_twoIntSameY0,\
    errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,type0_0Inds,\
    type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,type2_0Inds,type2_1Inds,type2_2Inds,\
    type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,fourInt_x,fourInt_y,twoIntSameY_x,twoIntSameY_y,twoIntOppositeX_x,\
    twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3 = calcMasterIntersections(sma,e,W,w,inc,s_circle,plotBool)

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

