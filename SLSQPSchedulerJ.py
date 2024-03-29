from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
from ortools.linear_solver import pywraplp
from scipy.optimize import minimize,minimize_scalar
import hashlib
import inspect,os
from astropy.time import Time
try:
   import cPickle as pickle
except:
   import pickle
import copy

class SLSQPSchedulerJ(SurveySimulation):
    """SLSQPScheduler
    
    This class implements a continuous optimization of integration times
    using the scipy minimize function with method SLSQP.  ortools with the CBC 
    linear solver is used to find an initial solution consistent with the constraints.
    For details see Savransky et al. 2017 (SPIE).

    Args:         
        \*\*specs:
            user specified values

    Notes:
        Due to the time costs of the current comp_per_inttime calculation in GarrettCompleteness
        this should be used with BrownCompleteness.

        Requires ortools
    
    """

    def __init__(self, cacheOptTimes=False, staticOptTimes=False, **specs):
        
        #initialize the prototype survey
        SurveySimulation.__init__(self, **specs)

        #Calculate fZmax
        self.valfZmax, self.absTimefZmax = self.ZodiacalLight.calcfZmax(np.arange(self.TargetList.nStars), self.Observatory, self.TargetList, self.TimeKeeping, filter(lambda mode: mode['detectionMode'] == True, self.OpticalSystem.observingModes)[0], self.cachefname)
        #sInds = np.arange(self.TargetList.nStars)
        #del self.absTimefZmin
        #self.valfZmin, self.absTimefZmin = self.ZodiacalLight.calcfZmin(sInds, self.Observatory, self.TargetList, self.TimeKeeping, self.mode, self.cachefname) # find fZmin to use in intTimeFilter
        
        print 'Printing absTimefZmin in SS init: ' + str(max(self.absTimefZmin))
        print 'Printing currentTimeAbs in SS init: ' + str(self.TimeKeeping.currentTimeAbs)

        assert isinstance(staticOptTimes, bool), 'staticOptTimes must be boolean.'
        self.staticOptTimes = staticOptTimes
        self._outspec['staticOptTimes'] = self.staticOptTimes

        assert isinstance(cacheOptTimes, bool), 'cacheOptTimes must be boolean.'
        self._outspec['cacheOptTimes'] = cacheOptTimes


        #some global defs
        self.detmode = filter(lambda mode: mode['detectionMode'] == True, self.OpticalSystem.observingModes)[0]
        self.ohTimeTot = self.Observatory.settlingTime + self.detmode['syst']['ohTime']
        self.maxTime = self.TimeKeeping.missionLife*self.TimeKeeping.missionPortion

        self.constraints = {'type':'ineq',
                            'fun': lambda x: self.maxTime.to(u.d).value - np.sum(x[x*u.d > 0.1*u.s]) - 
                                             np.sum(x*u.d > 0.1*u.s).astype(float)*self.ohTimeTot.to(u.d).value,
                            'jac':lambda x: np.ones(len(x))*-1.}

        self.t0 = None
        if cacheOptTimes:
            #Generate cache Name########################################################################
            cachefname = self.cachefname + 't0'
            
            if os.path.isfile(cachefname):
                self.vprint("Loading cached t0 from %s"%cachefname)
                with open(cachefname, 'rb') as f:
                    self.t0 = pickle.load(f)
                sInds = np.arange(self.TargetList.nStars)
                fZ = np.array([self.ZodiacalLight.fZ0.value]*len(sInds))*self.ZodiacalLight.fZ0.unit
                self.scomp0 = -self.objfun(self.t0.to(u.d).value,sInds,fZ)


        if self.t0 is None:
            #find nominal background counts for all targets in list
            _, Cbs, Csps = self.OpticalSystem.Cp_Cb_Csp(self.TargetList, range(self.TargetList.nStars),  
                    self.ZodiacalLight.fZ0, self.ZodiacalLight.fEZ0, 25.0, self.WAint, self.detmode)

            #find baseline solution with dMagLim-based integration times
            self.vprint('Finding baseline fixed-time optimal target set.')
            t0 = self.OpticalSystem.calc_intTime(self.TargetList, range(self.TargetList.nStars),  
                    self.ZodiacalLight.fZ0, self.ZodiacalLight.fEZ0, self.dMagint, self.WAint, self.detmode)
            comp0 = self.Completeness.comp_per_intTime(t0, self.TargetList, range(self.TargetList.nStars), 
                    self.ZodiacalLight.fZ0, self.ZodiacalLight.fEZ0, self.WAint, self.detmode, C_b=Cbs, C_sp=Csps)

            
            solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
            xs = [ solver.IntVar(0.0,1.0, 'x'+str(j)) for j in range(len(comp0)) ]

            #constraint is x_i*t_i < maxtime
            constraint = solver.Constraint(-solver.infinity(),self.maxTime.to(u.day).value)
            for j,x in enumerate(xs):
                constraint.SetCoefficient(x, t0[j].to(u.day).value + self.ohTimeTot.to(u.day).value)

            #objective is max x_i*comp_i
            objective = solver.Objective()
            for j,x in enumerate(xs):
                objective.SetCoefficient(x, comp0[j])
            objective.SetMaximization()

            cpres = solver.Solve()
            x0 = np.array([x.solution_value() for x in xs])
            self.scomp0 = np.sum(comp0*x0)
            self.t0 = t0

            #now find the optimal eps baseline and use whichever gives you the highest starting completeness
            self.vprint('Finding baseline fixed-eps optimal target set.')
            def totCompfeps(eps):
                compstars,tstars,x = self.inttimesfeps(eps, Cbs.to('1/d').value, Csps.to('1/d').value)
                return -np.sum(compstars*x)
            epsres = minimize_scalar(totCompfeps,method='bounded',bounds = [0,1],options = {'disp':True})
            comp_epsmax,t_epsmax,x_epsmax = self.inttimesfeps(epsres['x'],Cbs.to('1/d').value, Csps.to('1/d').value)
            if np.sum(comp_epsmax*x_epsmax) > self.scomp0:
                x0 = x_epsmax
                self.scomp0 = np.sum(comp_epsmax*x_epsmax) 
                self.t0 = t_epsmax*u.day

            #now optimize the solution
            self.vprint('Optimizing baseline integration times.')
            sInds = np.arange(self.TargetList.nStars)
            fZ = self.valfZmin#np.array([self.ZodiacalLight.fZ0.value]*len(sInds))*self.ZodiacalLight.fZ0.unit
            bounds = [(0,self.maxTime.to(u.d).value) for i in range(len(sInds))]
            initguess = x0*self.t0.to(u.d).value
            ires = minimize(self.objfun, initguess, jac=self.objfun_deriv, args=(sInds,fZ), 
                    constraints=self.constraints, method='SLSQP', bounds=bounds, options={'maxiter':300,'ftol':1e-3})

            assert ires['success'], "Initial time optimization failed."

            self.t0 = ires['x']*u.d
            self.scomp0 = -ires['fun']

            if cacheOptTimes:
                with open(cachefname,'wb') as f:
                    pickle.dump(self.t0, f)
                self.vprint("Saved cached optimized t0 to %s"%cachefname)


    def inttimesfeps(self,eps,Cb,Csp):
        """
        Compute the optimal subset of targets for a given epsilon value
        where epsilon is the maximum completeness gradient.

        Everything is in units of days
        """

        tstars = (-Cb*eps*np.sqrt(np.log(10)) + np.sqrt((Cb*eps)**2.*np.log(10) + 
                   5*Cb*Csp**2.*eps))/(2.0*Csp**2.*eps*np.log(10))
        compstars = self.Completeness.comp_per_intTime(tstars*u.day, self.TargetList, 
                np.arange(self.TargetList.nStars), self.ZodiacalLight.fZ0, 
                self.ZodiacalLight.fEZ0, self.WAint, self.detmode, C_b=Cb/u.d, C_sp=Csp/u.d)

        
        solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        xs = [ solver.IntVar(0.0,1.0, 'x'+str(j)) for j in range(len(compstars)) ]
        constraint = solver.Constraint(-solver.infinity(), self.maxTime.to(u.d).value)

        for j,x in enumerate(xs):
            constraint.SetCoefficient(x, tstars[j] + self.ohTimeTot.to(u.day).value)

        objective = solver.Objective()
        for j,x in enumerate(xs):
            objective.SetCoefficient(x, compstars[j])
        objective.SetMaximization()

        cpres = solver.Solve()

        x = np.array([x.solution_value() for x in xs])

        return compstars,tstars,x


    def objfun(self,t,sInds,fZ):
        """
        Objective Function for SLSQP minimization. Purpose is to maximize summed completeness

        Args:
            t (ndarray):
                Integration times in days. NB: NOT an astropy quantity.
            sInds (ndarray):
                Target star indices (of same size as t)
            fZ (astropy Quantity):
                Surface brightness of local zodiacal light in units of 1/arcsec2
                Same size as t

        """
        good = t*u.d >= 0.1*u.s

        comp = self.Completeness.comp_per_intTime(t[good]*u.d, self.TargetList, sInds[good], fZ[good], 
                self.ZodiacalLight.fEZ0, self.WAint[sInds][good], self.detmode)

        return -comp.sum()


    def objfun_deriv(self,t,sInds,fZ):
        """
        Jacobian of objective Function for SLSQP minimization. 

        Args:
            t (astropy Quantity):
                Integration times in days. NB: NOT an astropy quantity.
            sInds (ndarray):
                Target star indices (of same size as t)
            fZ (astropy Quantity):
                Surface brightness of local zodiacal light in units of 1/arcsec2
                Same size as t

        """
        good = t*u.d >= 0.1*u.s

        tmp = self.Completeness.dcomp_dt(t[good]*u.d, self.TargetList, sInds[good], fZ[good], 
                self.ZodiacalLight.fEZ0, self.WAint[sInds][good], self.detmode).to("1/d").value

        jac = np.zeros(len(t))
        jac[good] = tmp
        return -jac



    def calc_targ_intTime(self, sInds, startTimes, mode):
        """
        Given a subset of targets, calculate their integration times given the
        start of observation time.

        This implementation updates the optimized times based on current conditions and 
        mission time left.

        Note: next_target filter will discard targets with zero integration times.
        
        Args:
            sInds (integer array):
                Indices of available targets
            startTimes (astropy quantity array):
                absolute start times of observations.  
                must be of the same size as sInds 
            mode (dict):
                Selected observing mode for detection

        Returns:
            intTimes (astropy Quantity array):
                Integration times for detection 
                same dimension as sInds
        """
 
        if self.staticOptTimes:
            intTimes = self.t0[sInds]
        else:
            # assumed values for detection
            #fZ = self.ZodiacalLight.fZ(self.Observatory, self.TargetList, sInds, startTimes, mode)
            fZ = self.valfZmin[sInds]


            #### instead of actual time left, try bounding by maxTime - detection time used
            #need to update time used in choose_next_target
            
            timeLeft = (self.TimeKeeping.missionLife - self.TimeKeeping.currentTimeNorm)*self.TimeKeeping.missionPortion
            bounds = [(0,timeLeft.to(u.d).value) for i in range(len(sInds))]

            initguess = self.t0[sInds].to(u.d).value
            ires = minimize(self.objfun, initguess, jac=self.objfun_deriv, args=(sInds,fZ), constraints=self.constraints,
                    method='SLSQP', bounds=bounds, options={'disp':True,'maxiter':300,'ftol':1e-3})
            
            #update default times for these targets
            self.t0[sInds] = ires['x']*u.d

            intTimes = ires['x']*u.d
            
        intTimes[intTimes < 0.1*u.s] = 0.0*u.d
            
        return intTimes

    def choose_next_target(self, old_sInd, sInds, slewTimes, intTimes):
        """
        
        Given a subset of targets (pre-filtered by method next_target or some 
        other means), select the best next one. 

        Args:
            old_sInd (integer):
                Index of the previous target star
            sInds (integer array):
                Indices of available targets
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            intTimes (astropy Quantity array):
                Integration times for detection in units of day
        
        Returns:
            sInd (integer):
                Index of next target star
            waitTime (astropy Quantity):
                the amount of time to wait (this method returns None)
        
        """
                
        # calcualte completeness values for current intTimes
        tmpsInds = sInds
        sInds = sInds[np.where(intTimes.value > 1e-15)]#filter out any intTimes that are essentially 0
        if len(sInds) == 0:#If there are no stars... arbitrarily assign 1 day for observation length...
            sInds = tmpsInds #revert to the saved sInds
            intTimes = (np.zeros(len(sInds)) + 1.)*u.d  
            
        # fZ = self.ZodiacalLight.fZ(self.Observatory, self.TargetList, sInds,  
        #         self.TimeKeeping.currentTimeAbs + slewTimes[sInds], self.detmode)
        comps = self.Completeness.comp_per_intTime(intTimes[np.where(intTimes.value > 1e-15)], self.TargetList, sInds, self.valfZmin[sInds], 
                self.ZodiacalLight.fEZ0, self.WAint[sInds], self.detmode)

        # choose target with maximum completeness
        valfZmax = self.valfZmax[sInds].copy()
        valfZmin = self.valfZmin[sInds].copy()
        #selectInd = np.argmin((fZ - valfZmin)/(valfZmin - valfZmax)*(1/comps))
        #selectInd = np.argmin(1/comps)
        #Select next occuring absTimefZmin
        TK = self.TimeKeeping
        #if max(self.absTimefZmin.value) > self.missionStart + 365.25*u.d
        #sInds2 = np.arange(self.TargetList.nStars)
        #tmp, COPYabsTimefZmin = self.ZodiacalLight.calcfZmin(sInds2, self.Observatory, self.TargetList, self.TimeKeeping, self.mode, self.cachefname) # find fZmin to use in intTimeFilter
        #COPYabsTimefZmin = COPYabsTimefZmin.value
        COPYabsTimefZmin = self.absTimefZmin.value
        #print(max(self.absTimefZmin))
        #self.absTimefZmin #these are the absolute times throughought the first year that fZmin occurs at
        timeWherefZminOccursRelativeToNow = COPYabsTimefZmin - TK.currentTimeAbs.value #of all targetss
        zero = 0#TK.missionStart-TK.missionStart#hack to create a zero astropy time object
        indsLessThan0 = np.where((timeWherefZminOccursRelativeToNow < zero))[0] # find all inds that are less than 0
        cnt = 0.
        while len(indsLessThan0) > 0:#  self.TargetList.nStars:
            cnt += 1.
            # for kk in indsLessThan0:
            #     timeWherefZminOccursRelativeToNow[kk] = (self.absTimefZmin[kk] - TK.currentTimeAbs.copy() + cnt*365.25*u.d).value
            timeWherefZminOccursRelativeToNow[indsLessThan0] = COPYabsTimefZmin[indsLessThan0] - TK.currentTimeAbs.value + cnt*365.25
            indsLessThan0 = np.where((timeWherefZminOccursRelativeToNow < zero))[0]
        timeToStartfZmins = timeWherefZminOccursRelativeToNow#contains all "next occuring" fZmins in absolute time

        absTimefZminAfterNow = [timeToStartfZmins[i] for i in range(len(timeToStartfZmins)) if timeToStartfZmins[i] > 0. and i in sInds]#filter by times in future and times not filtered
        # print len(absTimefZminAfterNow)
        # print len(sInds)
        nextAbsTime = min(np.asarray(absTimefZminAfterNow))#find the minimum time
        sInd = np.where((timeToStartfZmins == nextAbsTime))[0][0]#find the index of the minimum time and return that sInd
        del absTimefZminAfterNow

        #Advance To fZmin of Target
        success = self.TimeKeeping.advanceToAbsTime(Time(nextAbsTime+TK.currentTimeAbs.copy().value, format='mjd', scale='tai'), False)
        waitTime = None
        # print(self.absTimefZmin[sInd])
        # print(absTimefZminAfterNow)
        # print(self.TimeKeeping.currentTimeAbs)

        #Check if exoplanetObsTime would be exceeded
        OS = self.OpticalSystem
        Comp = self.Completeness
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        # allocate settling time + overhead time
        allModes = OS.observingModes
        mode = filter(lambda mode: mode['detectionMode'] == True, allModes)[0]
        maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife = TK.get_ObsDetectionMaxIntTime(Obs, mode)
        maxIntTime = min(maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife)#Maximum intTime allowed
        intTimes2 = self.calc_targ_intTime(sInd, TK.currentTimeAbs.copy(), mode)
        if intTimes2 > maxIntTime:
            sInd = None
            waitTime = 1*u.d

        return sInd, waitTime

    def arbitrary_time_advancement(self,dt):
        """ Handles fully dynamically scheduled case where OBduration is infinite and
        missionPortion is less than 1.

        Input dt is the total amount of time, including all overheads and extras
        used for the previous observation."""

        #Do nothing