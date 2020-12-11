#### Completeness Convergence
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
import itertools
import csv


calcCompBool = False
plotBool = True


#### Randomly Generate Orbits #####################################################
if calcCompBool == True:
    folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))
    filename = 'HabEx_CKL2_PPKL2.json'
    filename = 'WFIRSTcycle6core.json'
    filename = 'HabEx_CSAG13_PPSAG13_compSubtype.json'
    #filename = 'HabEx_CSAG13_PPSAG13_compSubtypeHighEccen.json'
    scriptfile = os.path.join(folder,filename)
    sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
    PPop = sim.PlanetPopulation
    comp = sim.Completeness
    TL = sim.TargetList


    dmag = 25. #29.0
    dmag_upper = 25. #29.0
    IWA_HabEx = 0.045*u.arcsec #taken from a Habex Script in units of mas
    IWA2=0.150*u.arcsec #Suggested by dmitry as analahous to WFIRST
    OWA_HabEx = 6.*u.arcsec #from the HabEx Standards Team Final Report
    s_inner = 10.*u.pc.to('AU')*IWA_HabEx.to('rad').value
    s_outer = 10.*u.pc.to('AU')*OWA_HabEx.to('rad').value

    #starMass
    starMass = const.M_sun

    tmp = np.linspace(start=2,stop=7,num=50)
    ns = np.floor(10**tmp).astype('int')
    ns = [10**5]*1000
    #ns = np.append(ns,np.asarray([10**7,5*10**7,10**8,2*10**8]))
    executionTimeInHours = np.sum(ns)*352./7906043/60./60. #Uses a rate calc from the nu from dmag function so this is an underestimate
    # = [10**2,10**3,10**4,10**5,10**6,10**7,10**8,10**9]
    comps = list()
    #n = 10**5 #Dean's nice computer can go up to 10**8 what can atuin go up to?
    for i in np.arange(len(ns)):
        print('Working on completeness for ' + str(ns[i]) + ' planets')
        inc, W, w = PPop.gen_angles(ns[i],None)
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
        sma, e, p, Rp = PPop.gen_plan_params(ns[i])

        #Planet to be removed
        ar = 0.6840751713914676*u.AU #sma
        er = 0.12443160036480415 #e
        Wr = 6.1198652952593 #W
        wr = 2.661645323283813 #w
        incr = 0.8803680245150818 #inc
        sma,e,W,w,inc = nukeKOE(sma,e,W,w,inc,ar,er,Wr,wr,incr)
        ar = 1.1300859542315127*u.AU
        er = 0.23306811746716588
        Wr = 5.480292250277455
        wr = 2.4440871464730183
        incr = 1.197618937201339
        sma,e,W,w,inc = nukeKOE(sma,e,W,w,inc,ar,er,Wr,wr,incr)
        ar = 1.294036031163908*u.AU
        er = 0.21199750171689644
        Wr = 4.2133670225641655
        wr = 5.065937897601312
        incr = 1.1190687587235382
        sma,e,W,w,inc = nukeKOE(sma,e,W,w,inc,ar,er,Wr,wr,incr)
        ar = 1.066634748569753*u.AU
        er = 0.23799650191541077
        Wr = 3.448179965706161
        wr = 1.6569964041754257
        incr = 1.2153358668430527
        sma,e,W,w,inc = nukeKOE(sma,e,W,w,inc,ar,er,Wr,wr,incr)
        #Removed because large imag components
        ar = 12.006090070519544*u.AU
        er = 0.011103778198982318
        Wr = 2.1899766657830924
        wr = 1.5709200609443608
        incr = 0.027074317262287463
        sma,e,W,w,inc = nukeKOE(sma,e,W,w,inc,ar,er,Wr,wr,incr)
        ar = 1.551574519580606*u.AU
        er = 0.170910065686594
        Wr = 3.771587648556721
        wr = 2.6878240121964816
        incr = 1.2879427951683768
        sma,e,W,w,inc = nukeKOE(sma,e,W,w,inc,ar,er,Wr,wr,incr)
        ar = 0.861296088681235*u.AU
        er = 0.3114402152030745
        Wr = 2.3598509674140193
        wr = 4.2429329404990686
        incr = 1.1280958647535413
        sma,e,W,w,inc = nukeKOE(sma,e,W,w,inc,ar,er,Wr,wr,incr)
        ar = 12.148033510787947*u.AU
        er = 0.010235555197470866
        Wr = 4.722538539300574
        wr = 1.5705497614401225
        incr = 0.021803790927026074
        sma,e,W,w,inc = nukeKOE(sma,e,W,w,inc,ar,er,Wr,wr,incr)
        ar = 6.972239708414176*u.AU
        er = 0.3314991502307672
        Wr = 1.0597252167081723
        wr = 4.95574654788906
        incr = 1.2972537064559475e-05
        sma,e,W,w,inc = nukeKOE(sma,e,W,w,inc,ar,er,Wr,wr,incr)

        #### Classify Planets
        bini, binj, earthLike = comp.classifyPlanets(Rp, TL, np.arange(len(sma)), sma, e)
        sma = sma.to('AU').value
        ####

        #Separations
        s_circle = np.ones(len(sma))
        #Planet Periods
        periods = (2.*np.pi*np.sqrt((sma*u.AU)**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value

        nus, planetIsVisibleBool = planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,False, s_inner, s_outer, dmag_upper, dmag_lower=None) #Calculate planet-star nu edges and visible regions
        ts = timeFromTrueAnomaly(nus,np.tile(periods,(18,1)).T*u.year.to('day'),np.tile(e,(18,1)).T) #Calculate the planet-star intersection edges
        dt = ts[:,1:] - ts[:,:-1] #Calculate time region widths
        maxIntTime = 0.
        gtIntLimit = dt > maxIntTime #Create boolean array for inds
        totalVisibleTimePerTarget = np.nansum(np.multiply(np.multiply(dt-maxIntTime,planetIsVisibleBool.astype('int')),gtIntLimit),axis=1) #We subtract the int time from the fraction of observable time
        totalCompleteness = np.divide(totalVisibleTimePerTarget,periods*u.year.to('day')) # Fraction of time each planet is visible of its period
        #comps.append(list())
        with open("./keithlyCompConvergence.csv", "a") as myfile:
            myfile.write(str(ns[i]) + "," + str(np.mean(totalCompleteness)) + " \n")
###########################################################

# Plot Convergence Plot
if plotBool==True:
    #### Load Completeness CSV File 
    with open('./keithlyCompConvergence.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = np.asarray(data,dtype='float')
    #numPlans = np.cumsum(data[:,0])
    cumComp = [data[0,1]]
    cumPlans = [data[0,0]]
    for i in np.arange(len(data)-1):
        cumPlans.append(cumPlans[i]+data[i+1,0])
        cumComp.append((cumComp[i]*cumPlans[i] + data[i+1,0]*data[i+1,1])/(cumPlans[i+1]))
        #(numPlans[i]*cumComp[i] + data[i+1,1]*data[i+1,0])/numPlans[i]
    #averageCompletenesses = 
    print(saltyburrito)


    num=9999
    plt.figure(num)
    plt.plot(cumPlans,cumComp,color='purple')
    plt.xscale('log')
    plt.xlabel('Number of Planets')
    plt.ylabel('Completeness')
    plt.show(block=False)
