import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as random
import time
from astropy import constants as const
import astropy.units as u
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.planet_star_separation import planet_star_separation

#### PLOT BOOL
plotBool = False
if plotBool == True:
    from plotProjectedEllipse import *

#### Randomly Generate Orbits
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))
filename = 'HabEx_CKL2_PPKL2.json'
filename = 'WFIRSTcycle6core.json'
filename = 'HabEx_CSAG13_PPSAG13_compSubtype.json'
filename = 'HabEx_CSAG13_PPSAG13_compSubtypeHighEccen.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
PPop = sim.PlanetPopulation
comp = sim.Completeness
TL = sim.TargetList
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

#### Classify Planets
bini, binj, earthLike = comp.classifyPlanets(Rp, TL, np.arange(len(sma)), sma, e)
sma = sma.to('AU').value
####

#Separations
s_circle = np.ones(len(sma))
dmag = 29.0
dmag_upper = 29.0
IWA_HabEx = 0.045*u.arcsec #taken from a Habex Script in units of mas
IWA2=0.150*u.arcsec #Suggested by dmitry as analahous to WFIRST
s_inner = np.ones(len(sma))*10.*u.pc.to('AU')*IWA_HabEx.to('rad').value
s_outer = np.ones(len(sma))*10.*u.pc.to('AU')*3.*IWA_HabEx.to('rad').value #RANDOMLY MULTIPLY BY 3 HERE

#starMass
starMass = const.M_sun

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
#### A NICE 4 INTERSECTION EXAMPLE
fourIntersectionInd = 2173 #33
sma[fourIntersectionInd] = 5.363760022304063
e[fourIntersectionInd] = 0.557679118292977
w[fourIntersectionInd] = 5.058312201296985
W[fourIntersectionInd] = 0.6867396268911974
inc[fourIntersectionInd] = 0.8122666711110185
####


dmajorp,dminorp,theta_OpQ_X,theta_OpQp_X,Op,x,y,Phi,xreal,only2RealInds,yrealAllRealInds,\
    fourIntInds,twoIntOppositeXInds,twoIntSameYInds,nu_minSepPoints,nu_maxSepPoints,nu_lminSepPoints,nu_lmaxSepPoints,nu_fourInt,\
    nu_twoIntSameY,nu_twoIntOppositeX,nu_IntersectionsOnly2, yrealImagInds,\
    t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,t_twoIntSameY0,\
    t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
    minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep,\
    errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,errors_twoIntSameY0,\
    errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,type0_0Inds,\
    type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,type2_0Inds,type2_1Inds,type2_2Inds,\
    type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,fourInt_x,fourInt_y,twoIntSameY_x,twoIntSameY_y,twoIntOppositeX_x,\
    twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3, periods = calcMasterIntersections(sma,e,W,w,inc,s_circle,starMass,plotBool)



#### START ANALYSIS AND PLOTTING ######################################
#######################################################################
if plotBool == True:
    #### Plotting Projected Ellipse
    start2 = time.time()
    ind = random.randint(low=0,high=n)
    ind = 24 #for testing purposes
    plotProjectedEllipse(ind, sma, e, W, w, inc, Phi, dmajorp, dminorp, Op, num=877)
    stop2 = time.time()
    print('stop2: ' + str(stop2-start2))
    del start2, stop2
    #plt.close(877)
    ####

    #### Plot 3D Ellipse to 2D Ellipse Projection Diagram
    start3 = time.time()
    num = 666999888777
    plot3DEllipseto2DEllipseProjectionDiagram(ind, sma, e, W, w, inc, Op, Phi,\
        dmajorp, dminorp, num=num)
    stop3 = time.time()
    print('stop3: ' + str(stop3-start3))
    del start3, stop3
    #plt.close(num)
    ####

    #### Create Projected Ellipse Conjugate Diameters and QQ' construction diagram
    start4 = time.time()
    num = 3335555888
    plotEllipseMajorAxisFromConjugate(ind, sma, e, W, w, inc, Op, Phi,\
        dmajorp, dminorp, num)
    stop4 = time.time()
    print('stop4: ' + str(stop4-start4))
    del start4, stop4
    #plt.close(num)
    ####

    #### Plot Derotated Ellipse
    start6 = time.time()
    num=880
    plotDerotatedEllipse(ind, sma, e, W, w, inc, Phi, dmajorp, dminorp, Op, x, y, num)
    stop6 = time.time()
    print('stop6: ' + str(stop6-start6))
    del start6, stop6
    #plt.close(num)
    ####

    ##### Plot Proving Rerotation method works
    start10 = time.time()
    num=883
    plotReorientationMethod(ind, sma, e, W, w, inc, x, y, Phi, Op, dmajorp, dminorp,\
        minSepPoints_x, minSepPoints_y, num)
    stop10 = time.time()
    print('stop10: ' + str(stop10-start10))
    del start10, stop10
    #plt.close(num)
    ####

    #### Plot Derotated Intersections, Min/Max, and Star Location Type Bounds
    start12 = time.time()
    num = 960
    plotDerotatedIntersectionsMinMaxStarLocBounds(ind, sma, e, W, w, inc, x, y, dmajorp, dminorp, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
        minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
        lmaxSepPoints_x, lmaxSepPoints_y, twoIntSameYInds,\
        maxSepPoints_x, maxSepPoints_y, twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
        type0_0Inds, type0_1Inds, type0_2Inds, type0_3Inds, type0_4Inds, type1_0Inds, type1_1Inds, type1_2Inds, type1_3Inds, type1_4Inds,\
        type2_0Inds, type2_1Inds, type2_2Inds, type2_3Inds, type2_4Inds, type3_0Inds, type3_1Inds, type3_2Inds, type3_3Inds, type3_4Inds, num)
    stop12 = time.time()
    print('stop12: ' + str(stop12-start12))
    del start12, stop12
    #plt.close(num)
    ####

    #### Plot Derotated Ellipse Separation Extrema
    start12_1 = time.time()
    num = 961
    plotDerotatedExtrema(derotatedInd, sma, e, W, w, inc, x, y, dmajorp, dminorp, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
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
    #plt.close(num)
    ######################

    # ind = yrealAllRealInds[fourIntInds[np.argsort(-errors_fourInt1)[0]]]
    # plotRerotatedFromNus(ind, sma[ind], e[ind], W[ind], w[ind], inc[ind], Op[:,ind], yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds,\
    #     nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
    #     twoIntSameY_x, twoIntSameY_y, num=8001)

    tmpind = yrealAllRealInds[twoIntSameYInds[np.argsort(-errors_twoIntSameY1)[0]]]
    plotRerotatedFromNus(tmpind, sma[tmpind], e[tmpind], W[tmpind], w[tmpind], inc[tmpind], Op[:,tmpind], yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds,\
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
    #plt.close(num) #thinking the above plot is relativly useless
    ####

    #### Plot Histogram of Error
    num=824
    plotErrorHistogram(errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,\
        errors_twoIntSameY0,errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,\
        errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,num)
    #plt.close(num)
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
    plotSeparationVsTime(ind, sma, e, W, w, inc, minSep, maxSep, lminSep, lmaxSep,\
        t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,\
        t_twoIntSameY0,t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
        nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
        yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, periods, num)
    ####

    ####  Plot Derotate Ellipse
    tinds = np.argsort(-np.abs(errors_fourInt1))
    tind2 = yrealAllRealInds[fourIntInds[tinds[1]]]
    num=55670
    plotDerotatedEllipseStarLocDividers(tind2, sma, e, W, w, inc, x, y, dmajorp, dminorp, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
        minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
        lmaxSepPoints_x, lmaxSepPoints_y, twoIntSameYInds,\
        maxSepPoints_x, maxSepPoints_y, twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
        type0_0Inds, type0_1Inds, type0_2Inds, type0_3Inds, type0_4Inds, type1_0Inds, type1_1Inds, type1_2Inds, type1_3Inds, type1_4Inds,\
        type2_0Inds, type2_1Inds, type2_2Inds, type2_3Inds, type2_4Inds, type3_0Inds, type3_1Inds, type3_2Inds, type3_3Inds, type3_4Inds, num)
    #plt.close(num)
    ####

    #### Min Seps Histogram
    num=9701
    plotSepsHistogram(minSep,maxSep,lminSep,lmaxSep,sma,yrealAllRealInds,num)


#### s_inner, s_upper 
def calc_t_sInnersOuter(sma,e,W,w,inc,s_inner,s_outer,starMass,plotBool):
    """ Collates the times where each planet crosses s_inner and s_outer
    Args:
    Returns:
        times (numpy array):
            the collective array of times when the planet crosses the separation circle size (n x 8)
    """
    times_o = np.zeros((sma.shape[0],4))*np.nan
    times_i = np.zeros((sma.shape[0],4))*np.nan

    _,_,_,_,_,_,_,_,_,only2RealInds_o,yrealAllRealInds_o,\
        fourIntInds_o,twoIntOppositeXInds_o,twoIntSameYInds_o,_,_,_,_,_,\
        _,_,_, yrealImagInds_o,\
        _,_,_,_,t_fourInt0_o,t_fourInt1_o,t_fourInt2_o,t_fourInt3_o,t_twoIntSameY0_o,\
        t_twoIntSameY1_o,t_twoIntOppositeX0_o,t_twoIntOppositeX1_o,t_IntersectionOnly20_o,t_IntersectionOnly21_o,\
        _, _, _, _, _, _, _, _, _, _, _, _,\
        _,_,_,_,_,\
        _,_,_,_,_,_,\
        _,_,_,_,_,_,_,_,_,_,_,_,\
        _,_,_,_,_,_,_,_,_,_,_,_,\
        _,_,_,_,_,_,_, _ = calcMasterIntersections(sma,e,W,w,inc,s_inner,starMass,False)

    #Combine them all into one storage array
    times_o[yrealAllRealInds_o[fourIntInds_o],0] = t_fourInt0_o
    times_o[yrealAllRealInds_o[fourIntInds_o],1] = t_fourInt1_o
    times_o[yrealAllRealInds_o[fourIntInds_o],2] = t_fourInt2_o
    times_o[yrealAllRealInds_o[fourIntInds_o],3] = t_fourInt3_o
    times_o[yrealAllRealInds_o[twoIntSameYInds_o],0] = t_twoIntSameY0_o
    times_o[yrealAllRealInds_o[twoIntSameYInds_o],1] = t_twoIntSameY1_o
    times_o[yrealAllRealInds_o[twoIntOppositeXInds_o],0] = t_twoIntOppositeX0_o
    times_o[yrealAllRealInds_o[twoIntOppositeXInds_o],1] = t_twoIntOppositeX1_o
    times_o[only2RealInds_o,0] = t_IntersectionOnly20_o
    times_o[only2RealInds_o,1] = t_IntersectionOnly21_o

    _,_,_,_,_,_,_,_,_,only2RealInds_i,yrealAllRealInds_i,\
        fourIntInds_i,twoIntOppositeXInds_i,twoIntSameYInds_i,_,_,_,_,_,\
        _,_,_, yrealImagInds_i,\
        _,_,_,_,t_fourInt0_i,t_fourInt1_i,t_fourInt2_i,t_fourInt3_i,t_twoIntSameY0_i,\
        t_twoIntSameY1_i,t_twoIntOppositeX0_i,t_twoIntOppositeX1_i,t_IntersectionOnly20_i,t_IntersectionOnly21_i,\
        _, _, _, _, _, _, _, _, _, _, _, _,\
        _,_,_,_,_,\
        _,_,_,_,_,_,\
        _,_,_,_,_,_,_,_,_,_,_,_,\
        _,_,_,_,_,_,_,_,_,_,_,_,\
        _,_,_,_,_,_,_, _ = calcMasterIntersections(sma,e,W,w,inc,s_outer,starMass,False)

    #Combine them all into one storage array
    times_i[yrealAllRealInds_i[fourIntInds_i],0] = t_fourInt0_i
    times_i[yrealAllRealInds_i[fourIntInds_i],1] = t_fourInt1_i
    times_i[yrealAllRealInds_i[fourIntInds_i],2] = t_fourInt2_i
    times_i[yrealAllRealInds_i[fourIntInds_i],3] = t_fourInt3_i
    times_i[yrealAllRealInds_i[twoIntSameYInds_i],0] = t_twoIntSameY0_i
    times_i[yrealAllRealInds_i[twoIntSameYInds_i],1] = t_twoIntSameY1_i
    times_i[yrealAllRealInds_i[twoIntOppositeXInds_i],0] = t_twoIntOppositeX0_i
    times_i[yrealAllRealInds_i[twoIntOppositeXInds_i],1] = t_twoIntOppositeX1_i
    times_i[only2RealInds_i,0] = t_IntersectionOnly20_i
    times_i[only2RealInds_i,1] = t_IntersectionOnly21_i

    times = np.concatenate((times_o,times_i),axis=1)
    return times
####


#### nu From dMag #####################################################
#### Solving for dmag_min and dmag_max for each planet ################
mindmag, maxdmag, dmaglminAll, dmaglmaxAll, indsWith2, indsWith4, nuMinDmag, nuMaxDmag, nulminAll, nulmaxAll = calc_planet_dmagmin_dmagmax(e,inc,w,sma*u.AU,p,Rp)
print('Num Planets with At Least 2 Int given dmag: ' + str(np.sum((mindmag < dmag)*(maxdmag > dmag))))
print('Num Planets with dmag local extrema: ' + str(len(indsWith4)))
print('Num Planets with given 4 Int given dmag: ' + str(np.sum((dmaglminAll < dmag)*(dmaglmaxAll > dmag))))
indsWith4Int = indsWith4[np.where((dmaglminAll < dmag)*(dmaglmaxAll > dmag))[0]]
indsWith2Int = list(set(np.where((mindmag < dmag)*(maxdmag > dmag))[0]) - set(indsWith4Int))
######################################################################

#### Dmag Extrema Times ##############################################
time_dmagmin = timeFromTrueAnomaly(nuMinDmag,periods,e)
time_dmagmax = timeFromTrueAnomaly(nuMaxDmag,periods,e)
time_dmaglmin = timeFromTrueAnomaly(nulminAll,periods[indsWith4],e[indsWith4])
time_dmaglmax = timeFromTrueAnomaly(nulmaxAll,periods[indsWith4],e[indsWith4])
######################################################################

######################################################################
#### Solving for nu, dmag intersections ##############################
nus2Int, nus4Int, dmag2Int, dmag4Int = calc_planetnu_from_dmag(dmag,e,inc,w,sma*u.AU,p,Rp,mindmag, maxdmag, indsWith2Int, indsWith4Int)
time_dmagInts = np.zeros((len(e),4))*np.nan
time_dmagInts[indsWith2Int,0] = timeFromTrueAnomaly(nus2Int[:,0],periods[indsWith2Int],e[indsWith2Int])
time_dmagInts[indsWith2Int,1] = timeFromTrueAnomaly(nus2Int[:,1],periods[indsWith2Int],e[indsWith2Int])
time_dmagInts[indsWith4Int,0] = timeFromTrueAnomaly(nus4Int[:,0],periods[indsWith4Int],e[indsWith4Int])
time_dmagInts[indsWith4Int,1] = timeFromTrueAnomaly(nus4Int[:,1],periods[indsWith4Int],e[indsWith4Int])
time_dmagInts[indsWith4Int,2] = timeFromTrueAnomaly(nus4Int[:,2],periods[indsWith4Int],e[indsWith4Int])
time_dmagInts[indsWith4Int,3] = timeFromTrueAnomaly(nus4Int[:,3],periods[indsWith4Int],e[indsWith4Int])
# t2Int = np.zeros((len(indsWith2Int),4))
# t2Int[:,0] = timeFromTrueAnomaly(nus2Int[:,0],periods[indsWith2Int],e[indsWith2Int])
# t2Int[:,1] = timeFromTrueAnomaly(nus2Int[:,1],periods[indsWith2Int],e[indsWith2Int])
# t4Int = np.zeros((len(indsWith4Int),4))
# t4Int[:,0] = timeFromTrueAnomaly(nus4Int[:,0],periods[indsWith4Int],e[indsWith4Int])
# t4Int[:,1] = timeFromTrueAnomaly(nus4Int[:,1],periods[indsWith4Int],e[indsWith4Int])
# t4Int[:,2] = timeFromTrueAnomaly(nus4Int[:,2],periods[indsWith4Int],e[indsWith4Int])
# t4Int[:,3] = timeFromTrueAnomaly(nus4Int[:,3],periods[indsWith4Int],e[indsWith4Int])
######################################################################

#### Bulking all Times Together ######################################
times_s = calc_t_sInnersOuter(sma,e,W,w,inc,s_inner,s_outer,starMass,plotBool)
times = np.concatenate((np.zeros((len(e),1)),times_s,time_dmagInts,np.reshape(periods,(len(periods),1))),axis=1)
timesSortInds = np.argsort(times,axis=1)
times2 = np.sort(times,axis=1) #sorted from smallest to largest
indsWithAnyInt = np.where(np.sum(~np.isnan(times2),axis=1))[0] #Finds the planets which have any intersections
#####################################################################




#Check visibility in all given bounds (For Completeness)
#NEED TO BE ABLE TO PUT BOUNDS INTO BOX WITH 4 SIDES
#AND BOX WITH 3 SIDES

#### dmag vs nu extrema and intersection Verification plot
num=88833543453218
plt.figure(num=num)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
ind = fourIntersectionInd#indsWith4Int[0]
nus = np.linspace(start=0,stop=2.*np.pi,num=100)
phis = (1.+np.sin(inc[ind])*np.sin(nus+w[ind]))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
ds = sma[ind]*(1.-e[ind]**2.)/(e[ind]*np.cos(nus)+1.)
dmags = deltaMag(p[ind],Rp[ind].to('AU'),ds*u.AU,phis) #calculate dmag of the specified x-value

plt.plot(nus,dmags,color='black',zorder=10)
#plt.plot([0.,2.*np.pi],[dmag,dmag],color='blue')
plt.scatter(nuMinDmag[ind],mindmag[ind],color='cyan',marker='d',zorder=20)
plt.plot([0.,2.*np.pi],[mindmag[ind],mindmag[ind]],color='cyan',zorder=20)
plt.scatter(nuMaxDmag[ind],maxdmag[ind],color='red',marker='d',zorder=20)
plt.plot([0.,2.*np.pi],[maxdmag[ind],maxdmag[ind]],color='red',zorder=20)
lind = np.where(ind == indsWith4)[0]
if  ind in indsWith2Int:
    mind = np.where(ind == indsWith2Int)[0]
    plt.scatter(nus2Int[mind],dmag2Int[mind],color='green',marker='o',zorder=20)
    plt.plot([0.,2.*np.pi],[dmag,dmag],color='green',zorder=10)
elif ind in indsWith4Int:
    nind = np.where(ind == indsWith4Int)[0]
    plt.scatter(nus4Int[nind],dmag4Int[nind],color='green',marker='o',zorder=20)
    plt.plot([0.,2.*np.pi],[dmag,dmag],color='green',zorder=10)
plt.scatter(nulminAll[lind],dmaglminAll[lind],color='magenta',marker='d',zorder=20)
plt.plot([0.,2.*np.pi],[dmaglminAll[lind],dmaglminAll[lind]],color='magenta',zorder=20)
plt.scatter(nulmaxAll[lind],dmaglmaxAll[lind],color='gold',marker='d',zorder=20)
plt.plot([0.,2.*np.pi],[dmaglmaxAll[lind],dmaglmaxAll[lind]],color='gold',zorder=20)
plt.xlim([0.,2.*np.pi])
plt.ylim([-0.05*(maxdmag[ind]-mindmag[ind])+mindmag[ind],0.05*(maxdmag[ind]-mindmag[ind])+maxdmag[ind]])
plt.ylabel(r'$\Delta \mathrm{mag}$',weight='bold')
plt.xlabel('True Anomaly, ' + r'$\nu$' + ', in (rad)', weight='bold')
plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + '\nw: ' + str(np.round(w[ind],4)) + ' inc: ' + str(np.round(inc[ind],4)))
plt.show(block=False)

#### Verification With Time
num=8883354345329
plt.figure(num=num)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
ind = fourIntersectionInd#indsWith4Int[0]
nus = np.linspace(start=0,stop=2.*np.pi,num=100)
tmp_times = timeFromTrueAnomaly(nus,periods[ind],e[ind])
phis = (1.+np.sin(inc[ind])*np.sin(nus+w[ind]))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
ds = sma[ind]*(1.-e[ind]**2.)/(e[ind]*np.cos(nus)+1.)
dmags = deltaMag(p[ind],Rp[ind].to('AU'),ds*u.AU,phis) #calculate dmag of the specified x-value

plt.plot(tmp_times,dmags,color='black',zorder=10)
#plt.plot([0.,2.*np.pi],[dmag,dmag],color='blue')
plt.scatter(time_dmagmin[ind],mindmag[ind],color='cyan',marker='d',zorder=20)
plt.plot([0.,periods[ind]],[mindmag[ind],mindmag[ind]],color='cyan',zorder=20)
plt.scatter(time_dmagmax[ind],maxdmag[ind],color='red',marker='d',zorder=20)
plt.plot([0.,periods[ind]],[maxdmag[ind],maxdmag[ind]],color='red',zorder=20)
lind = np.where(ind == indsWith4)[0]
if  ind in indsWith2Int:
    mind = np.where(ind == indsWith2Int)[0]
    plt.scatter(time_dmagInts[indsWith4Int[mind]],dmag2Int[mind],color='green',marker='o',zorder=20)
    plt.plot([0.,periods[ind]],[dmag,dmag],color='green',zorder=10)
elif ind in indsWith4Int:
    nind = np.where(ind == indsWith4Int)[0]
    plt.scatter(time_dmagInts[indsWith4Int[nind]],dmag4Int[nind],color='green',marker='o',zorder=20)
    plt.plot([0.,periods[ind]],[dmag,dmag],color='green',zorder=10)
plt.scatter(time_dmaglmin[lind],dmaglminAll[lind],color='magenta',marker='d',zorder=20)
plt.plot([0.,periods[ind]],[dmaglminAll[lind],dmaglminAll[lind]],color='magenta',zorder=20)
plt.scatter(time_dmaglmax[lind],dmaglmaxAll[lind],color='gold',marker='d',zorder=20)
plt.plot([0.,periods[ind]],[dmaglmaxAll[lind],dmaglmaxAll[lind]],color='gold',zorder=20)
plt.xlim([0.,periods[ind]])
plt.ylim([-0.05*(maxdmag[ind]-mindmag[ind])+mindmag[ind],0.05*(maxdmag[ind]-mindmag[ind])+maxdmag[ind]])
plt.ylabel(r'$\Delta \mathrm{mag}$',weight='bold')
plt.xlabel('Time Past Periastron, t, (years)',weight='bold')
plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + '\nw: ' + str(np.round(w[ind],4)) + ' inc: ' + str(np.round(inc[ind],4)))
plt.show(block=False)
#######################################################################

#### Calculate Completeness ###########################################
def planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,plotBool, s_inner, s_outer, dmag_upper, dmag_lower=None):
    """ Finds the nu values where the planet intersects the separations or dmags, subsequently checks whether the planet is visible in the specified time ranges
    Args:
    sma,e,W,w,inc,p,Rp,starMass,plotBool, s_inner, s_outer, dmag_upper, dmag_lower
    Returns:
    nus, planetIsVisibleBool
    """
    nus = np.zeros((len(sma),18))*np.nan #4 from s_inner, 4 from s_outer, 4 from dmag_upper, 4 from dmag_lower, 2 for previous orbit intersection and next orbit intersection
    #### nu from s_inner
    dmajorp,dminorp,theta_OpQ_X,theta_OpQp_X,Op,x,y,Phi,xreal,only2RealInds,yrealAllRealInds,\
        fourIntInds,twoIntOppositeXInds,twoIntSameYInds,nu_minSepPoints,nu_maxSepPoints,nu_lminSepPoints,nu_lmaxSepPoints,nu_fourInt,\
        nu_twoIntSameY,nu_twoIntOppositeX,nu_IntersectionsOnly2, yrealImagInds,\
        t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,t_twoIntSameY0,\
        t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
        minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep,\
        errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,errors_twoIntSameY0,\
        errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,type0_0Inds,\
        type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,type2_0Inds,type2_1Inds,type2_2Inds,\
        type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,fourInt_x,fourInt_y,twoIntSameY_x,twoIntSameY_y,twoIntOppositeX_x,\
        twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3, periods = calcMasterIntersections(sma,e,W,w,inc,s_inner,starMass,plotBool)
    nus[only2RealInds,0:2] = nu_IntersectionsOnly2
    nus[yrealAllRealInds[fourIntInds],[0:4]] = nu_fourInt
    nus[yrealAllRealInds[twoIntOppositeXInds],[0:2]] = nu_twoIntOppositeX
    nus[yrealAllRealInds[twoIntSameYInds],[0:2]] = nu_twoIntSameY
    #### nu from s_outer
    dmajorp,dminorp,theta_OpQ_X,theta_OpQp_X,Op,x,y,Phi,xreal,only2RealInds,yrealAllRealInds,\
        fourIntInds,twoIntOppositeXInds,twoIntSameYInds,nu_minSepPoints,nu_maxSepPoints,nu_lminSepPoints,nu_lmaxSepPoints,nu_fourInt,\
        nu_twoIntSameY,nu_twoIntOppositeX,nu_IntersectionsOnly2, yrealImagInds,\
        t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,t_twoIntSameY0,\
        t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
        minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep,\
        errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,errors_twoIntSameY0,\
        errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,type0_0Inds,\
        type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,type2_0Inds,type2_1Inds,type2_2Inds,\
        type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,fourInt_x,fourInt_y,twoIntSameY_x,twoIntSameY_y,twoIntOppositeX_x,\
        twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3, periods = calcMasterIntersections(sma,e,W,w,inc,s_outer,starMass,plotBool)
    nus[only2RealInds,4:6] = nu_IntersectionsOnly2
    nus[yrealAllRealInds[fourIntInds],[4:8]] = nu_fourInt
    nus[yrealAllRealInds[twoIntOppositeXInds],[4:6]] = nu_twoIntOppositeX
    nus[yrealAllRealInds[twoIntSameYInds],[4:6]] = nu_twoIntSameY
    #### Solving for dmag_min and dmag_max for each planet ################
    mindmag, maxdmag, dmaglminAll, dmaglmaxAll, indsWith2, indsWith4, nuMinDmag, nuMaxDmag, nulminAll, nulmaxAll = calc_planet_dmagmin_dmagmax(e,inc,w,sma*u.AU,p,Rp)
    #### nu From dmag_upper
    print('Num Planets with At Least 2 Int given dmag: ' + str(np.sum((mindmag < dmag_upper)*(maxdmag > dmag_upper))))
    print('Num Planets with dmag local extrema: ' + str(len(indsWith4)))
    print('Num Planets with given 4 Int given dmag: ' + str(np.sum((dmaglminAll < dmag_upper)*(dmaglmaxAll > dmag_upper))))
    indsWith4Int = indsWith4[np.where((dmaglminAll < dmag_upper)*(dmaglmaxAll > dmag_upper))[0]]
    indsWith2Int = list(set(np.where((mindmag < dmag_upper)*(maxdmag > dmag_upper))[0]) - set(indsWith4Int))
    nus2Int, nus4Int, dmag2Int, dmag4Int = calc_planetnu_from_dmag(dmag_upper,e,inc,w,sma*u.AU,p,Rp,mindmag, maxdmag, indsWith2Int, indsWith4Int)
    nus[indsWith2Int,8:10] = nus2Int
    nus[indsWith4Int,8:12] = nus4Int
    #### nu From dmag_lower
    if dmag_lower == None:
        #default case? 0s maybe idk empty stuff
        dmag_lower = 0.
    else:
        print('Num Planets with At Least 2 Int given dmag: ' + str(np.sum((mindmag < dmag_lower)*(maxdmag > dmag_lower))))
        print('Num Planets with dmag local extrema: ' + str(len(indsWith4)))
        print('Num Planets with given 4 Int given dmag: ' + str(np.sum((dmaglminAll < dmag_lower)*(dmaglmaxAll > dmag_lower))))
        indsWith4Int = indsWith4[np.where((dmaglminAll < dmag_lower)*(dmaglmaxAll > dmag_lower))[0]]
        indsWith2Int = list(set(np.where((mindmag < dmag_lower)*(maxdmag > dmag_lower))[0]) - set(indsWith4Int))
        nus2Int, nus4Int, dmag2Int, dmag4Int = calc_planetnu_from_dmag(dmag_lower,e,inc,w,sma*u.AU,p,Rp,mindmag, maxdmag, indsWith2Int, indsWith4Int)
        nus[indsWith2Int,12:14] = nus2Int
        nus[indsWith4Int,12:16] = nus4Int
    ########################################################################
    
    #Finding which planets are all nan for efficiency
    nanbool = np.isnan(nus)
    indsNotAllNan = np.where(np.logical_not(np.all(nus,axis=1)))[0]

    #Aded ranges above or below each nan (so I can simply do a midpoint evaluation with no fancy indexing)
    nus_min = np.nanmin(nus[indsNotAllNan],axis=1)
    nus_max = np.nanmax(nus[indsNotAllNan],axis=1)
    nus[indsNotAllNan,16] = 2.*np.pi + nus_min #append the next orbit to this bit
    nus[indsNotAllNan,17] = nus_max - 2.*np.pi #append the previous orbit intersection

    #sort
    nus[indsNotAllNan] = np.sort(nus[indsNotAllNan],axis=1)

    #calculate nus midpoints (for evaluating whether planets are visible within the range specified)
    nus_midpoints = nus[:,1:] - nus[:,:-1]

    #Calculate dmag and s for all midpoints
    Phi = (1.+np.sin(np.tile(inc,(18,1)).T)*np.sin(nus_midpoints+np.tile(w,(18,1)).T))**2./4.
    d = np.tile(sma.to('AU'),(18,1)).T*(1.-np.tile(e,(18,1)).T**2.)/(np.tile(e,(18,1)).T*np.cos(nus_midpoints)+1.)
    dmags = deltaMag(np.tile(p,(18,1)).T,np.tile(Rp.to('AU'),(18,1)).T,d,Phi) #calculate dmag of the specified x-value
    ss = planet_star_separation(np.tile(sma,(18,1)).T,np.tile(e,(18,1)).T,nus_midpoints,np.tile(w,(18,1)).T,np.tile(inc,(18,1)).T)

    #Determine ranges where the planet is visible
    planetIsVisibleBool = (ss < s_outer)*(ss > s_inner)*(dmags < dmag_upper)*(dmags > dmag_lower)

    return nus, planetIsVisibleBool

nus, planetIsVisibleBool = planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,plotBool, s_inner, s_outer, dmag_upper, dmag_lower=None)
ts = timeFromTrueAnomaly(nus,np.tile(periods,(18,1)).T,np.tile(e,(18,1)).T)

#### Dynamic Completeness Calculations ################################
#######################################################################

#### Dynamic Completeness By Subtype Calculations #####################
#######################################################################


