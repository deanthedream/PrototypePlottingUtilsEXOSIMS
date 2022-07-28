#simple BIP
import numpy as np

#from ortools.linear_solver import pywraplp
import itertools
#from itertools import product
import random
import time
#import psutil
import os

import numpy as np

from ortools.linear_solver import pywraplp
import itertools

import matplotlib.pyplot as plt

#from mipOptimalScheduler import mipOptimalScheduler
#from mipOptimalScheduler import recurSum
#from mipOptimalScheduler import multiply_along_axis

whichRun = 7
if whichRun == 1:
    time0 = time.time()
    #### Generate Inputs
    numSV = 50 #100 SV
    numT = 60*1000 #7 minutes in 1 ms increments

    numOBSs = 50
    meanOBSduration = 3*1000
    stdOBSduration = 1000

    OBSDurs= np.random.normal(loc=meanOBSduration,scale=stdOBSduration,size=numOBSs)
    OBSDurs = np.round(OBSDurs)
    OBSStarts = list()
    for j in np.arange(numOBSs):
        OBSStarts.append(np.random.randint(low=0,high=(numT - OBSDurs[j])))
    OBSValues = np.ones(numOBSs)


    #### Instantiate MIP Solver
    time1 = time.time()
    solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING) # create solver instance
    print("Instantiate MIP Solver: " + str((time1-time0)))

    #### Free Variables
    ys = dict() #is SV i used at time t
    for (i,t) in itertools.product(np.arange(numSV),np.arange(numT)):
        ys[i,t] = solver.IntVar(0,1,'y' + str(i) + '_' + str(t))
    xs = dict() #is SV i observing OBS j
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        xs[i,j] = solver.IntVar(0,1,'x' + str(i) + '_' + str(j))
    time2 = time.time()
    print("Done Adding Free Variables Total:" + str(numSV*numOBSs + numSV*numT) + " execution time: " + str((time2-time1)))

    #### CONSTRAINTS ################################
    #Each OBS Observed at most once (by one SV)
    observedOnce = list()
    for j in np.arange(numOBSs):
        observedOnce.append(solver.Add(solver.Sum([xs[i,j] for i in np.arange(numSV)]) <= 1))
    #Each OBS set to be observed must must consume the relevant time assignment
    svTimeConstraints = list()
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        svTimeConstraints.append(solver.Add(solver.Sum([ys[i,t] for t in OBSStarts[j]+np.arange(OBSDurs[j])]) == xs[i,j]*OBSDurs[j]))
    time3 = time.time()
    print("Done Adding Constraints:" + str(numSV*numOBSs + numSV*numT) + " execution time: " + str((time3-time2)))

    #### Objective Function
    objective = solver.Objective()
    #Set Total Partial Reward Coefficients
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        objective.SetCoefficient(xs[i,j], OBSValues[i]) #OBSValues could be different for each SV making the observation
    objective.SetMaximization()
    time4 = time.time()
    print('Done Setting Objective Function execution time: ' + " execution time: " + str((time4-time3)))

    #solver.EnableOutput()# this line enables output of the CBC MIXED INTEGER PROGRAM (Was hard to find don't delete)
    solver.SetTimeLimit(60*1000)#time limit for solver in milliseconds
    solver.SetNumThreads(6)
    #might be able to add SetHint() to furnish an IFS
    cpres = solver.Solve() # actually solve MIP
    time5 = time.time()
    print('Done Solving execution time: ' + " execution time: " + str((time5-time4)))
    x0d = dict()
    y0d = dict()
    z0d = dict()
    for (i,j) in xs.keys():
        x0d[(i,j)] = xs[i,j].solution_value()
    for (i,t) in ys.keys():
        y0d[(i,t)] = ys[i,t].solution_value()


    #### Evaluate How Much Reward Was Collected ###############################################################
    #Set Total Partial Reward Coefficients
    TPR = 0
    for (i,j) in xs.keys():
        TPR += xs[i,j].solution_value()*OBSValues[i]

    # Outputs the Solver Objective Function Value
    objFunValue = solver.Objective().Value()

    print("Number of Variables: " + str(solver.NumVariables()))
    print("Number of Constraints: " + str(solver.NumConstraints()))

    #return x0d, y0d, z0d, TPR, TROC, objFunValue





elif whichRun == 2:
    #### SOLVING USING ONLY OBS TIME WINDOWS, CONTINUOUS TIME ##################################################
    print('#### EXECUTION 2 #################################################################')
    time0 = time.time()
    #### Generate Inputs
    numSV = 50 #100 SV
    #numT = 60*1000 #7 minutes in 1 ms increments
    numOBSs = 50 #numOBSs = numT
    meanOBSduration = 30
    stdOBSduration = 5
    tmax = 7*60 #the maximum time range

    OBSDurs= np.random.normal(loc=meanOBSduration,scale=stdOBSduration,size=numOBSs)
    time_windows = list()
    #OBSStarts = list()
    for j in np.arange(numOBSs):
        start = np.random.uniform(low=0.,high=tmax-OBSDurs[j])
        #OBSStarts.append(start)
        time_windows.append(start)
        time_windows.append(start+OBSDurs[j])
    #time_windows = np.zeros(2*numOBSs)
    #for j in np.arange(numOBSs):
    #    time_windows[j] = OBSStarts[j]
    #    time_windows[j+1] = OBSStarts[j]+OBSDurs[j]
    time_windows_argsort = np.argsort(time_windows)
    #time_windows = np.concatenate((OBSStarts,OBSStarts+OBSDurs))
    #time_windows = np.argsort(time_windows)#sort from smallest to largest
    numT = len(time_windows)
    OBSValues = np.ones(numOBSs) #equal values for all


    #### Instantiate MIP Solver
    time1 = time.time()
    solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING) # create solver instance
    print("2 Instantiate MIP Solver: " + str((time1-time0)))

    #### Free Variables
    ys = dict() #is SV i used at time t
    for (i,t) in itertools.product(np.arange(numSV),np.arange(numT)):
        ys[i,t] = solver.IntVar(0,1,'y' + str(i) + '_' + str(t))
    xs = dict() #is SV i observing OBS j
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        xs[i,j] = solver.IntVar(0,1,'x' + str(i) + '_' + str(j))
    time2 = time.time()
    print("2 Done Adding Free Variables Total:" + str(numSV*numOBSs + numSV*numT) + " execution time: " + str((time2-time1)))

    #### CONSTRAINTS ################################
    #Each OBS Observed at most once (by one SV)
    observedOnce = list()
    for j in np.arange(numOBSs):
        observedOnce.append(solver.Add(solver.Sum([xs[i,j] for i in np.arange(numSV)]) <= 1))
    #Each OBS set to be observed must must consume the relevant time assignment
    svTimeConstraints = list()
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        svTimeConstraints.append(solver.Add(solver.Sum([ys[i,t] for t in np.arange(len(time_windows))]) == xs[i,j]*OBSDurs[j]))
    time3 = time.time()
    print("2 Done Adding Constraints:" + str(numSV*numOBSs + numSV*numT) + " execution time: " + str((time3-time2)))

    #### Objective Function
    objective = solver.Objective()
    #Set Total Partial Reward Coefficients
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        objective.SetCoefficient(xs[i,j], OBSValues[i]) #OBSValues could be different for each SV making the observation
    objective.SetMaximization()
    time4 = time.time()
    print('2 Done Setting Objective Function execution time: ' + " execution time: " + str((time4-time3)))

    #solver.EnableOutput()# this line enables output of the CBC MIXED INTEGER PROGRAM (Was hard to find don't delete)
    solver.SetTimeLimit(60*1000)#time limit for solver in milliseconds
    solver.SetNumThreads(6)
    #might be able to add SetHint() to furnish an IFS
    cpres = solver.Solve() # actually solve MIP
    time5 = time.time()
    print('2 Done Solving execution time: ' + " execution time: " + str((time5-time4)))
    x0d = dict()
    y0d = dict()
    z0d = dict()
    for (i,j) in xs.keys():
        x0d[(i,j)] = xs[i,j].solution_value()
    for (i,t) in ys.keys():
        y0d[(i,t)] = ys[i,t].solution_value()


    #### Evaluate How Much Reward Was Collected ###############################################################
    #Set Total Partial Reward Coefficients
    TPR = 0
    for (i,j) in xs.keys():
        TPR += xs[i,j].solution_value()*OBSValues[i]

    # Outputs the Solver Objective Function Value
    objFunValue = solver.Objective().Value()

    print("2 Number of Variables: " + str(solver.NumVariables()))
    print("2 Number of Constraints: " + str(solver.NumConstraints()))





elif whichRun == 3:
    #### SOLVING USING ONLY OBS TIME WINDOWS, CONTINUOUS TIME ##################################################
    print('#### EXECUTION 3 #################################################################')
    time0 = time.time()
    #### Generate Inputs
    numSV = 50 #100 SV
    #numT = 60*1000 #7 minutes in 1 ms increments
    numOBSs = 600 #numOBSs = numT
    meanOBSduration = 30
    stdOBSduration = 5
    tmax = 7*60 #the maximum time range

    OBSDurs= np.random.normal(loc=meanOBSduration,scale=stdOBSduration,size=numOBSs)
    time_windows = list()
    #OBSStarts = list()
    for j in np.arange(numOBSs):
        start = np.random.uniform(low=0.,high=tmax-OBSDurs[j])
        #OBSStarts.append(start)
        time_windows.append(start)
        time_windows.append(start+OBSDurs[j])
    #time_windows = np.zeros(2*numOBSs)
    #for j in np.arange(numOBSs):
    #    time_windows[j] = OBSStarts[j]
    #    time_windows[j+1] = OBSStarts[j]+OBSDurs[j]
    time_windows_argsort = np.argsort(time_windows)
    #time_windows = np.concatenate((OBSStarts,OBSStarts+OBSDurs))
    #time_windows = np.argsort(time_windows)#sort from smallest to largest
    numT = len(time_windows)
    OBSValues = np.ones(numOBSs) #equal values for all


    #### Instantiate MIP Solver
    time1 = time.time()
    solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING) # create solver instance
    print("3 Instantiate MIP Solver: " + str((time1-time0)))

    #### Free Variables
    ys = dict() #is SV i used at time t
    for (i,t) in itertools.product(np.arange(numSV),np.arange(numT)):
        ys[i,t] = solver.IntVar(0,1,'y' + str(i) + '_' + str(t))
    xs = dict() #is SV i observing OBS j
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        xs[i,j] = solver.IntVar(0,1,'x' + str(i) + '_' + str(j))
    time2 = time.time()
    print("3 Done Adding Free Variables Total:" + str(numSV*numOBSs + numSV*numT) + " execution time: " + str((time2-time1)))

    #### CONSTRAINTS ################################
    #Each OBS Observed at most once (by one SV)
    observedOnce = list()
    for j in np.arange(numOBSs):
        observedOnce.append(solver.Add(solver.Sum([xs[i,j] for i in np.arange(numSV)]) <= 1))
    #Each OBS set to be observed must must consume the relevant time assignment
    svTimeConstraints = list()
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        svTimeConstraints.append(solver.Add(solver.Sum([ys[i,t] for t in np.arange(len(time_windows))]) == xs[i,j]*OBSDurs[j]))
    time3 = time.time()
    print("3 Done Adding Constraints:" + str(numSV*numOBSs + numSV*numT) + " execution time: " + str((time3-time2)))

    #### Objective Function
    objective = solver.Objective()
    #Set Total Partial Reward Coefficients
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        objective.SetCoefficient(xs[i,j], OBSValues[i]) #OBSValues could be different for each SV making the observation
    objective.SetMaximization()
    time4 = time.time()
    print('3 Done Setting Objective Function execution time: ' + " execution time: " + str((time4-time3)))

    #solver.EnableOutput()# this line enables output of the CBC MIXED INTEGER PROGRAM (Was hard to find don't delete)
    solver.SetTimeLimit(60*1000)#time limit for solver in milliseconds
    solver.SetNumThreads(6)
    #might be able to add SetHint() to furnish an IFS
    cpres = solver.Solve() # actually solve MIP
    time5 = time.time()
    print('3 Done Solving execution time: ' + " execution time: " + str((time5-time4)))
    x0d = dict()
    y0d = dict()
    z0d = dict()
    for (i,j) in xs.keys():
        x0d[(i,j)] = int(xs[i,j].solution_value())
    for (i,t) in ys.keys():
        y0d[(i,t)] = int(ys[i,t].solution_value())


    #### Evaluate How Much Reward Was Collected ###############################################################
    #Set Total Partial Reward Coefficients
    TPR = 0
    for (i,j) in xs.keys():
        TPR += xs[i,j].solution_value()*OBSValues[i]

    # Outputs the Solver Objective Function Value
    objFunValue = solver.Objective().Value()

    print("3 Number of Variables: " + str(solver.NumVariables()))
    print("3 Number of Constraints: " + str(solver.NumConstraints()))




    time6 = time.time()
    solVars = list()
    solSols = list()
    for (i,j) in xs.keys():
        solVars.append(xs[(i,j)])
        solSols.append(int(xs[(i,j)].solution_value()))
    for (i,t) in ys.keys():
        solVars.append(ys[(i,t)])
        solSols.append(int(ys[(i,t)].solution_value()))
        #y0d[(i,t)] = int(ys[i,t].solution_value())
    time7 = time.time()
    print('3 Declaring IFS execution time: ' + str((time7-time6)))
    #for (i,j) in xs.keys():
    solver.SetHint(solVars,solSols)
    cpres = solver.Solve() # actually solve MIP
    time8 = time.time()
    print('3 Done Solving with IFS execution time: ' + str((time8-time7)))








elif whichRun == 4:
    from earthOccultation import *
    #### SOLVING USING ONLY OBS TIME WINDOWS, CONTINUOUS TIME ##################################################
    print('#### EXECUTION 4 #################################################################')
    time0 = time.time()
    #### Generate Inputs
    numSV = len(satOBJs) #50 #100 SV
    #numT = 60*1000 #7 minutes in 1 ms increments
    numOBSs = 600 #numOBSs = numT
    meanOBSduration = 30
    stdOBSduration = 5
    tmax = 7*60 #the maximum time range

    #Create LOS array between each SV and each time
    for i in np.arange(len(satOBJs)):#iterate over satellite objects
        isVisible = np.zeros((satOBJs[i]['r'].shape[0],r_locs.T.shape[0]))
        for j in np.arange(satOBJs[i]['r'].shape[0]): #iterate over time steps
            satDist = np.linalg.norm(satOBJs[i]['r'][j])
            earthFOVAng = np.arcsin(r_earth/np.linalg.norm(satOBJs[i]['r'][j]))
            for k in np.arange(r_locs.shape[1]):#Iterate over OBSs
                r_obj_OBS = r_locs.T[k] - satOBJs[i]['r'][j]
                OBSDist = np.linalg.norm(r_obj_OBS)
                angBetweenEarthCenterAndLoc = np.arccos(np.dot(r_obj_OBS,-satOBJs[i]['r'][j])/satDist/OBSDist)
                if earthFOVAng <= angBetweenEarthCenterAndLoc:
                    isVisible[j,k] = 1 #it is visible
        satOBJs[i]['isVisible'] = isVisible




    OBSDurs= np.random.normal(loc=meanOBSduration,scale=stdOBSduration,size=numOBSs)
    time_windows = list()
    #OBSStarts = list()
    for j in np.arange(numOBSs):
        start = np.random.uniform(low=0.,high=tmax-OBSDurs[j])
        #OBSStarts.append(start)
        time_windows.append(start)
        time_windows.append(start+OBSDurs[j])
    #time_windows = np.zeros(2*numOBSs)
    #for j in np.arange(numOBSs):
    #    time_windows[j] = OBSStarts[j]
    #    time_windows[j+1] = OBSStarts[j]+OBSDurs[j]
    time_windows_argsort = np.argsort(time_windows)
    #time_windows = np.concatenate((OBSStarts,OBSStarts+OBSDurs))
    #time_windows = np.argsort(time_windows)#sort from smallest to largest
    numT = len(time_windows)
    OBSValues = np.ones(numOBSs) #equal values for all


    #### Instantiate MIP Solver
    time1 = time.time()
    solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING) # create solver instance
    #solver = pywraplp.Solver.CreateSolver('GLOP')
    print("4 Instantiate LP Solver: " + str((time1-time0)))


    #### Count # SV not able to see anything
    numWithNoVisible = 0
    for i in np.arange(len(satOBJs)):
        if np.any(satOBJs[i]['isVisible']):
            numWithNoVisible += 1


    #### Free Variables
    ys = dict() #is SV i used at time t
    for (i,t) in itertools.product(np.arange(numSV),np.arange(numT)):
        #TODO IF SV CANNOT SEE ANY OBSS IN THIS TIMESLOT, DO NOT INCLUDE Y_I,T
        if np.any(satOBJs[i]['isVisible']):
            ys[i,t] = solver.NumVar(0, 1,'y' + str(i) + '_' + str(t))
        #x = solver.NumVar(0, solver.infinity(), 'x')

    xs = dict() #is SV i observing OBS j
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        if np.any(satOBJs[i]['isVisible']):
            xs[i,j] = solver.NumVar(0, 1,'x' + str(i) + '_' + str(j))
    time2 = time.time()
    print("4 Done Adding Free Variables Total:" + str(numSV*numOBSs + numSV*numT) + " execution time: " + str((time2-time1)))

    #### CONSTRAINTS ################################
    #Each OBS Observed at most once (by one SV)
    observedOnce = list()
    for j in np.arange(numOBSs):
        observedOnce.append(solver.Add(solver.Sum([xs[i,j] for i in np.arange(numSV) if np.any(satOBJs[i]['isVisible'])]) <= 1))
    #Each OBS set to be observed must must consume the relevant time assignment
    svTimeConstraints = list()
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        if np.any(satOBJs[i]['isVisible']):
            svTimeConstraints.append(solver.Add(solver.Sum([ys[i,t] for t in np.arange(len(time_windows))]) == xs[i,j]*OBSDurs[j]))
    time3 = time.time()
    print("4 Done Adding Constraints:" + str(numSV*numOBSs + numSV*numT) + " execution time: " + str((time3-time2)))

    #### Objective Function
    objective = solver.Objective()
    #Set Total Partial Reward Coefficients
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        if np.any(satOBJs[i]['isVisible']):
            objective.SetCoefficient(xs[i,j], OBSValues[i]) #OBSValues could be different for each SV making the observation
    objective.SetMaximization()
    time4 = time.time()
    print('4 Done Setting Objective Function execution time: ' + " execution time: " + str((time4-time3)))

    #solver.EnableOutput()# this line enables output of the CBC MIXED INTEGER PROGRAM (Was hard to find don't delete)
    solver.SetTimeLimit(60*1000)#time limit for solver in milliseconds
    solver.SetNumThreads(6)
    #might be able to add SetHint() to furnish an IFS
    cpres = solver.Solve() # actually solve MIP
    time5 = time.time()
    print('4 Done Solving execution time: ' + " execution time: " + str((time5-time4)))
    x0d = dict()
    y0d = dict()
    z0d = dict()
    for (i,j) in xs.keys():
        x0d[(i,j)] = int(xs[i,j].solution_value())
    for (i,t) in ys.keys():
        y0d[(i,t)] = int(ys[i,t].solution_value())


    #### Evaluate How Much Reward Was Collected ###############################################################
    #Set Total Partial Reward Coefficients
    TPR = 0
    for (i,j) in xs.keys():
        TPR += xs[i,j].solution_value()*OBSValues[i]

    # Outputs the Solver Objective Function Value
    objFunValue = solver.Objective().Value()

    print("4 Number of Variables: " + str(solver.NumVariables()))
    print("4 Number of Constraints: " + str(solver.NumConstraints()))

elif whichRun == 5:
    print("SOMETHING IS BROKEN IN 5. RELATING TO ONLY ONE OBS AT A TIME CONSTRAINT")
    from earthOccultation import *
    #### SOLVING USING ONLY OBS TIME WINDOWS, CONTINUOUS TIME ##################################################
    print('#### EXECUTION 5 #################################################################')
    time0 = time.time()
    #### Generate Inputs
    numSV = len(satOBJs) #50 #100 SV
    numOBSs = 600 #numOBSs = numT
    meanOBSduration = 30
    stdOBSduration = 5
    tmax = 7*60 #the maximum time range

    ##### Create LOS array between each SV and each time, 1 if sat can see OBS k location at time j 
    for i in np.arange(len(satOBJs)):#iterate over satellite objects
        isVisible = np.zeros((satOBJs[i]['r'].shape[0],numOBSs))
        for j in np.arange(satOBJs[i]['r'].shape[0]): #iterate over time steps
            satDist = np.linalg.norm(satOBJs[i]['r'][j])
            earthFOVAng = np.arcsin(r_earth/np.linalg.norm(satOBJs[i]['r'][j]))
            for k in np.arange(r_locs.shape[1]):#Iterate over OBSs
                r_obj_OBS = r_locs.T[k] - satOBJs[i]['r'][j]
                OBSDist = np.linalg.norm(r_obj_OBS)
                angBetweenEarthCenterAndLoc = np.arccos(np.dot(r_obj_OBS,-satOBJs[i]['r'][j])/satDist/OBSDist)
                if earthFOVAng <= angBetweenEarthCenterAndLoc:
                    isVisible[j,k] = 1 #it is visible
        satOBJs[i]['isVisible'] = isVisible #shape 7x numpOBSs
        satOBJs[i]['isVisibleAtAll'] = np.any(satOBJs[i]['isVisible'],axis=0).astype('int') #can satellite i see OBS j at all (saved as array of j)
    ####

    ##### Randomly Generate OBS durations
    OBSDurs= np.random.normal(loc=meanOBSduration,scale=stdOBSduration,size=numOBSs) #create a random obs duration for each obs
    time_windows = list() #an array containing the start and end points of each OBS
    time_window_OBSnum = list() #an array containing OBS numbers of this time window
    for j in np.arange(numOBSs):
        start = np.random.uniform(low=0.,high=tmax-OBSDurs[j]) #create a random obs start time within time window
        time_windows.append(start) #save obs start time
        time_windows.append(start+OBSDurs[j]) #save obs end time
        time_window_OBSnum.append(j) #saves OBS index associated with time_window index
        time_window_OBSnum.append(j)
    time_windows = np.asarray(time_windows) #cast to arrays
    time_window_OBSnum = np.asarray(time_window_OBSnum)
    #sort by 
    time_windows_argsort = np.argsort(time_windows) #getting the indicies to sort the timewindows defined by the start and stop of each OBS
    sortedtime_window = time_windows[time_windows_argsort] #sorting the time windows from earliest to lates
    assert np.all(sortedtime_window[:-1] < sortedtime_window[1:]), 'not all time windows sorted'
    sortedtime_window_OBSnum = time_window_OBSnum[time_windows_argsort] #sorted timewindows
    numT = len(time_windows) #number of time windows
    OBSValues = np.ones(numOBSs) #equal values for all
    ####

    #### Find All OBS# intersecting or overlapping given time_windows
    # #first instantiate list storing list of OBSs in each time window
    # otherOBSsInTimeWindow = list()
    # for i in np.arange(numOBSs):
    #     otherOBSsInTimeWindow.append(list())
    # #check all other OBSs also within time window
    # for i in np.arange(numOBSs):#iterate over all OBSs
    #     #find the inds of the time_windows associated with the start and end of OBS i
    #     indsAfflicted = np.where(sortedtime_window_OBSnum == i)[0] #get the two inds associated with the OBS
    #     assert len(indsAfflicted) == 2, 'incorrect number of inds afflicted' #there must be 2, a start and end
    #     #Add all OBSs that start or end within the time_window
    #     for j in range(indsAfflicted[0],indsAfflicted[1]):
    #         #print(sortedtime_window_OBSnum[j])
    #         if not sortedtime_window_OBSnum[j] in otherOBSsInTimeWindow[i]: #only add unique OBSS, no duplicates
    #             #add the jth OBSnum that will be in between
    #             otherOBSsInTimeWindow[i].append(sortedtime_window_OBSnum[j])
    #     #Also check for spanning
    #     for j in np.arange(numOBSs):
    #         indsAfflicted_2 = np.where(sortedtime_window_OBSnum == j)[0] #get the two inds for this other OBS
    #         if indsAfflicted_2[0] < indsAfflicted[0] and indsAfflicted[1] < indsAfflicted_2[1]: #the other OBS starts before and ends after current OBS
    #             if not sortedtime_window_OBSnum[j] in otherOBSsInTimeWindow[i]: #only add unique OBSS, no duplicates
    #                 otherOBSsInTimeWindow[i].append(sortedtime_window_OBSnum[j])

    #NOTE will result in duplicate constraints...
    otherOBSsInTimeWindow = list()
    for j in np.arange(numOBSs):
        otherOBSsInTimeWindow.append(list())
    for (i,j) in itertools.product(np.arange(numOBSs),np.arange(numOBSs)):
        if i==j: #skip this one
            continue
        indsAfflicted = np.where(sortedtime_window_OBSnum == i)[0] #get the two inds associated with the OBS
        indsAfflicted_2 = np.where(sortedtime_window_OBSnum == j)[0] #get the two inds for this other OBS
        """
        This shows the range of sortedtime_window being iterated over is appropriate
        [sortedtime_window[k] for k in range(indsAfflicted[0],indsAfflicted[1])]
        sortedtime_window[indsAfflicted[0]]
        sortedtime_window[indsAfflicted[1]]
        """
        for k in range(indsAfflicted[0],indsAfflicted[1]):
            if not sortedtime_window_OBSnum[k] in otherOBSsInTimeWindow[i]: #only add unique OBSS, no duplicates
                #add the jth OBSnum that will be in between
                otherOBSsInTimeWindow[i].append(sortedtime_window_OBSnum[k])#this is the original OBS number
        #This works as expected according to the plot
        if indsAfflicted_2[0] < indsAfflicted[0] and indsAfflicted[1] < indsAfflicted_2[1]: #the other OBS starts before and ends after current OBS
            if not sortedtime_window_OBSnum[j] in otherOBSsInTimeWindow[i]: #only add unique OBSS, no duplicates
                otherOBSsInTimeWindow[i].append(sortedtime_window_OBSnum[j])
    ####

    #### Instantiate MIP Solver
    time1 = time.time()
    solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CLP_LINEAR_PROGRAMMING) # create solver instance
    #solver = pywraplp.Solver.CreateSolver('GLOP')
    print("5 Instantiate LP Solver: " + str((time1-time0)))


    #### Count # SV not able to see anything
    numWithNoVisible = 0
    for i in np.arange(len(satOBJs)):
        if np.any(satOBJs[i]['isVisible']):
            numWithNoVisible += 1
    print('5 numSV is ' + str(numSV))
    print('5 numSV able to see at least 1: ' + str(numSV-numWithNoVisible))

    #### Free Variables
    xs = dict() #is SV i observing OBS j
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        if satOBJs[i]['isVisibleAtAll'][j] == 1: #the satellite cannot see current OBS at all
            xs[i,j] = solver.BoolVar('x' + str(i) + '_' + str(j))
    time2 = time.time()
    print("5 Done Adding Free Variables Total:" + str(len(xs)) + " execution time: " + str((time2-time1)))

    #### CONSTRAINTS ################################
    #Each SV CAN ONLY OBSERVE ONE OBS AT A TIME
    oneOBSatAtimeConstraints = list()
    for i in np.arange(numSV): #for each SV observing each OBS
        #Should be iterate over OBSs this SV can observe
        for j in np.arange(numOBSs): #iterate over all OBSS
            if satOBJs[i]['isVisibleAtAll'][j] == 1: #if this SV can see this OBS
                #add constraint ensuring only one OBS is observed at a time
                #NOTE should read other OBSs in time window that this sv can see
                oneOBSatAtimeConstraints.append(solver.Add(solver.Sum([xs[i,k] for k in otherOBSsInTimeWindow[j] if (i,k) in xs.keys()]) <= 1))
    

    fig = plt.figure(111133453)
    ax = plt.gca()
    #plt.matshow()
    svObs = list()
    i=0 #the OBS number
    tmpIndTimeWindow = np.where(sortedtime_window_OBSnum == i)[0]
    ax.broken_barh([(sortedtime_window[tmpIndTimeWindow[0]],sortedtime_window[tmpIndTimeWindow[1]]-sortedtime_window[tmpIndTimeWindow[0]])], (-1, 4),facecolors =('green'))
    print((sortedtime_window[tmpIndTimeWindow[0]],sortedtime_window[tmpIndTimeWindow[1]]-sortedtime_window[tmpIndTimeWindow[0]]))
    for k in otherOBSsInTimeWindow[j]:
        tmpIndTimeWindow2 = np.where(sortedtime_window_OBSnum == k)[0]
        ax.broken_barh([(sortedtime_window[tmpIndTimeWindow2[0]],sortedtime_window[tmpIndTimeWindow2[1]]-sortedtime_window[tmpIndTimeWindow2[0]])], (k, 1),facecolors =('black','red'))
    plt.show(block=False)



    #Each OBS is observed only once
    eachOBSonlyOnceConstraints = list()
    for j in np.arange(numOBSs):
        eachOBSonlyOnceConstraints.append(solver.Add(solver.Sum([xs[i,j] for i in np.arange(numSV) if (i,j) in xs.keys()]) <= 1))

    time3 = time.time()
    print("5 Done Adding Constraints:" + str(len(oneOBSatAtimeConstraints)) + " execution time: " + str((time3-time2)))
    ####

    #### Objective Function
    objective = solver.Objective()
    #Set Total Partial Reward Coefficients
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        if (satOBJs[i]['isVisibleAtAll'][j] == 1): #the satellite cannot see current OBS at all
            objective.SetCoefficient(xs[i,j], OBSValues[i]) #OBSValues could be different for each SV making the observation
    objective.SetMaximization()
    time4 = time.time()
    print('5 Done Setting Objective Function execution time: ' + " execution time: " + str((time4-time3)))

    #solver.EnableOutput()# this line enables output of the CBC MIXED INTEGER PROGRAM (Was hard to find don't delete)
    solver.SetTimeLimit(60*1000)#time limit for solver in milliseconds
    solver.SetNumThreads(6)
    #might be able to add SetHint() to furnish an IFS
    cpres = solver.Solve() # actually solve MIP
    time5 = time.time()
    print('5 Done Solving execution time: ' + " execution time: " + str((time5-time4)))
    x0d = dict()
    for (i,j) in xs.keys():
        x0d[(i,j)] = int(xs[i,j].solution_value())
    # for (i,t) in ys.keys():
    #     y0d[(i,t)] = int(ys[i,t].solution_value())


    #### Evaluate How Much Reward Was Collected ###############################################################
    #Set Total Partial Reward Coefficients
    TPR = 0
    for (i,j) in xs.keys():
        TPR += xs[i,j].solution_value()*OBSValues[i]

    # Outputs the Solver Objective Function Value
    objFunValue = solver.Objective().Value()

    print("5 Obj Fun Value: " + str(objFunValue) + " max obj fun val: " + str(numOBSs))
    print("5 Number of Variables: " + str(solver.NumVariables()))
    print("5 Number of Constraints: " + str(solver.NumConstraints()))

    cumSum = 0
    for (i,j) in x0d.keys():
        cumSum += x0d[(i,j)]

    
    fig = plt.figure(6546846464654)
    ax = plt.gca()
    #plt.matshow()
    svObs = list()
    for i in np.arange(numSV):
        svObs.append(list())
    for (i,j) in x0d.keys():
        if x0d[(i,j)] == 1:
            tmpIndTimeWindow = np.where(sortedtime_window_OBSnum == j)[0]
            #tmpIndTimeWindow = np.where(sortedtime_window_OBSnum == i)[0]
            #print(tmpIndTimeWindow)
            assert tmpIndTimeWindow[0] < tmpIndTimeWindow[1], 'time window ordering not right'
            svObs[i].append((sortedtime_window[tmpIndTimeWindow[0]],sortedtime_window[tmpIndTimeWindow[1]]-sortedtime_window[tmpIndTimeWindow[0]])) #(start,duration)

    for i in np.arange(len(svObs)):
        #for j in np.arange(len(svObs[i])):
        ax.broken_barh(svObs[i], (2*i, 2),facecolors =('black','red'))
    plt.show(block=False)



    #Print all keys where an observation would be made
    # for (i,j)in x0d.keys():
    #     if x0d[(i,j)] == 1:
    #         print((i,j))





elif whichRun == 6:
    from earthOccultation import *
    #### SOLVING USING ONLY OBS TIME WINDOWS, CONTINUOUS TIME ##################################################
    print('#### EXECUTION 6 #################################################################')
    time0 = time.time()
    #### Generate Inputs
    numSV = len(satOBJs) #50 #100 SV
    numOBSs = 600 #numOBSs = numT
    meanOBSduration = 30
    stdOBSduration = 5
    tmax = 7*60 #the maximum time range

    ##### Create LOS array between each SV and each time, 1 if sat can see OBS k location at time j 
    for i in np.arange(len(satOBJs)):#iterate over satellite objects
        isVisible = np.zeros((satOBJs[i]['r'].shape[0],numOBSs))
        for j in np.arange(satOBJs[i]['r'].shape[0]): #iterate over time steps
            satDist = np.linalg.norm(satOBJs[i]['r'][j])
            earthFOVAng = np.arcsin(r_earth/np.linalg.norm(satOBJs[i]['r'][j]))
            for k in np.arange(r_locs.shape[1]):#Iterate over OBSs
                r_obj_OBS = r_locs.T[k] - satOBJs[i]['r'][j]
                OBSDist = np.linalg.norm(r_obj_OBS)
                angBetweenEarthCenterAndLoc = np.arccos(np.dot(r_obj_OBS,-satOBJs[i]['r'][j])/satDist/OBSDist)
                if earthFOVAng <= angBetweenEarthCenterAndLoc:
                    isVisible[j,k] = 1 #it is visible
        satOBJs[i]['isVisible'] = isVisible #shape 7x numpOBSs
        satOBJs[i]['isVisibleAtAll'] = np.any(satOBJs[i]['isVisible'],axis=0).astype('int') #can satellite i see OBS j at all (saved as array of j)
    ####

    ##### Randomly Generate OBS durations
    OBSDurs= np.random.normal(loc=meanOBSduration,scale=stdOBSduration,size=numOBSs) #create a random obs duration for each obs
    time_windows = list() #an array containing the start and end points of each OBS
    time_window_OBSnum = list() #an array containing OBS numbers of this time window
    for j in np.arange(numOBSs):
        start = np.random.uniform(low=0.,high=tmax-OBSDurs[j]) #create a random obs start time within time window
        time_windows.append(start) #save obs start time
        time_windows.append(start+OBSDurs[j]) #save obs end time
        time_window_OBSnum.append(j) #saves OBS index associated with time_window index
        time_window_OBSnum.append(j)
    time_windows = np.asarray(time_windows) #cast to arrays
    time_window_OBSnum = np.asarray(time_window_OBSnum)
    #sort by 
    time_windows_argsort = np.argsort(time_windows) #getting the indicies to sort the timewindows defined by the start and stop of each OBS
    sortedTW = time_windows[time_windows_argsort] #sorting the time windows from earliest to lates
    assert np.all(sortedTW[:-1] < sortedTW[1:]), 'not all time windows sorted'
    sortedTW_OBSnum = time_window_OBSnum[time_windows_argsort] #sorted timewindows
    numT = len(time_windows) #number of time windows
    OBSValues = np.linspace(start=1.,stop=1.+1e-4,num=numOBSs)#nearly equal values for all #np.ones(numOBSs) #equal values for all

    #creates a matrix where if i is the OBS being observed, all j that are 1 are OBSs which occur within the same timeframe as this observation
    OBSInterferenceMatrix = np.zeros((numOBSs,numOBSs))
    for (i,j) in itertools.product(np.arange(numOBSs),np.arange(numOBSs)):
        #if i==j: #skip this one
        #    continue
        #If the start of other is within TW, if the end of other is within TW, or if other spans TW
        if (time_windows[i*2] < time_windows[j*2] and time_windows[j*2] < time_windows[i*2+1])\
            or (time_windows[i*2] < time_windows[j*2+1] and time_windows[j*2+1] < time_windows[i*2+1])\
            or (time_windows[j*2] < time_windows[i*2] and time_windows[i*2+1] < time_windows[j*2+1]):
            OBSInterferenceMatrix[i,j] = 1
    ####

    #### Instantiate MIP Solver
    time1 = time.time()
    #solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CLP_LINEAR_PROGRAMMING) # create solver instance
    #solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING) # create solver instance
    #solver = pywraplp.Solver.CreateSolver('SCIP') #DID NOT produce feasible solution MPSOLVER ABNORMAL
    solver = pywraplp.Solver.CreateSolver('CP_SAT')




    #solver = pywraplp.Solver.CreateSolver('GLOP')
    print("6 Instantiate LP Solver: " + str((time1-time0)))


    #### Count # SV not able to see anything
    numWithNoVisible = 0
    for i in np.arange(len(satOBJs)):
        if np.any(satOBJs[i]['isVisible']):
            numWithNoVisible += 1
    print('6 numSV is ' + str(numSV))
    print('6 numSV able to see at least 1: ' + str(numSV-numWithNoVisible))

    #### Free Variables
    xs = dict() #is SV i observing OBS j
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        if satOBJs[i]['isVisibleAtAll'][j] == 1: #the satellite cannot see current OBS at all
            xs[i,j] = solver.IntVar(0,1,'x' + str(i) + '_' + str(j))
            #xs[i,j] = solver.BoolVar('x' + str(i) + '_' + str(j))

    time2 = time.time()
    print("6 Done Adding Free Variables Total:" + str(len(xs)) + " execution time: " + str((time2-time1)))

    #### CONSTRAINTS ################################
    # #Each SV CAN ONLY OBSERVE ONE OBS AT A TIME
    oneOBSatAtimeConstraints = list()
    for i in np.arange(numSV): #for each SV observing each OBS
        #Should be iterate over OBSs this SV can observe
        for j in np.arange(numOBSs): #iterate over all OBSS
            if satOBJs[i]['isVisibleAtAll'][j] == 1: #if this SV can see this OBS
                #add constraint ensuring only one OBS is observed at a time
                oneOBSatAtimeConstraints.append(solver.Add(\
                    solver.Sum([xs[i,k] for k in np.arange(numOBSs) \
                        if (i,k) in xs.keys() and OBSInterferenceMatrix[j,k] == 1 and satOBJs[i]['isVisibleAtAll'][k] == 1]) <= 1))


    #Copied from old code
    # fig = plt.figure(111133453)
    # ax = plt.gca()
    # #plt.matshow()
    # svObs = list()
    # i=0 #the OBS number
    # tmpIndTimeWindow = np.where(sortedTW_OBSnum == i)[0]
    # ax.broken_barh([(sortedtime_window[tmpIndTimeWindow[0]],sortedtime_window[tmpIndTimeWindow[1]]-sortedtime_window[tmpIndTimeWindow[0]])], (-1, 4),facecolors =('green'))
    # print((sortedtime_window[tmpIndTimeWindow[0]],sortedtime_window[tmpIndTimeWindow[1]]-sortedtime_window[tmpIndTimeWindow[0]]))
    # for k in otherOBSsInTimeWindow[j]:
    #     tmpIndTimeWindow2 = np.where(sortedTW_OBSnum == k)[0]
    #     ax.broken_barh([(sortedtime_window[tmpIndTimeWindow2[0]],sortedtime_window[tmpIndTimeWindow2[1]]-sortedtime_window[tmpIndTimeWindow2[0]])], (k, 1),facecolors =('black','red'))
    # plt.show(block=False)



    #MULTIPLE SV CANNOT OBSERVE SAME OBS
    eachOBSonlyOnceConstraints = list()
    for j in np.arange(numOBSs):
        eachOBSonlyOnceConstraints.append(solver.Add(solver.Sum([xs[i,j] for i in np.arange(numSV) if (i,j) in xs.keys() and satOBJs[i]['isVisibleAtAll'][j] == 1]) <= 1))

    time3 = time.time()
    # print("6 Done Adding Constraints:" + str(len(oneOBSatAtimeConstraints)) + " execution time: " + str((time3-time2)))
    ####

    #### Objective Function
    objective = solver.Objective()
    #Set Total Partial Reward Coefficients
    for (i,j) in xs.keys():#itertools.product(np.arange(numSV),np.arange(numOBSs)):
        #if (satOBJs[i]['isVisibleAtAll'][j] == 1): #the satellite cannot see current OBS at all
        objective.SetCoefficient(xs[i,j], 1.)#OBSValues[j]) #OBSValues could be different for each SV making the observation
    objective.SetMaximization()
    time4 = time.time()
    print('6 Done Setting Objective Function execution time: ' + " execution time: " + str((time4-time3)))

    #solver.EnableOutput()# this line enables output of the CBC MIXED INTEGER PROGRAM (Was hard to find don't delete)
    solver.SetTimeLimit(60*1000)#time limit for solver in milliseconds
    solver.SetNumThreads(6)
    #might be able to add SetHint() to furnish an IFS
    cpres = solver.Solve() # actually solve MIP
    time5 = time.time()
    print('6 Done Solving execution time: ' + " execution time: " + str((time5-time4)))
    x0d = dict()
    for (i,j) in xs.keys():
        x0d[(i,j)] = xs[i,j].solution_value()
        print(xs[i,j].solution_value()) 
    # for (i,t) in ys.keys():
    #     y0d[(i,t)] = int(ys[i,t].solution_value())


    #### Evaluate How Much Reward Was Collected ###############################################################
    #Set Total Partial Reward Coefficients
    TPR = 0
    for (i,j) in xs.keys():
        TPR += xs[i,j].solution_value()*OBSValues[j]

    # Outputs the Solver Objective Function Value
    objFunValue = solver.Objective().Value()

    print("6 Obj Fun Value: " + str(objFunValue) + " max obj fun val: " + str(numOBSs))
    print("6 Number of Variables: " + str(solver.NumVariables()))
    print("6 Number of Constraints: " + str(solver.NumConstraints()))


    cumSum = 0
    maxi = 0
    maxj = 0
    for (i,j) in x0d.keys():
        cumSum += x0d[(i,j)]
        if i > maxi:
            maxi = i
        if j > maxj:
            maxj = j
    print("cumSum: " + str(cumSum))
    
    fig = plt.figure(6546846464654)
    ax = plt.gca()
    #plt.matshow()
    svObs = list()
    for i in np.arange(numSV):
        svObs.append(list())
    for (i,j) in x0d.keys():
        if x0d[(i,j)] > 0:
            tmpIndTimeWindow = np.where(sortedTW_OBSnum == j)[0]
            #tmpIndTimeWindow = np.where(sortedTW_OBSnum == i)[0]
            #print(tmpIndTimeWindow)
            assert tmpIndTimeWindow[0] < tmpIndTimeWindow[1], 'time window ordering not right'
            svObs[i].append((sortedTW[tmpIndTimeWindow[0]],sortedTW[tmpIndTimeWindow[1]]-sortedTW[tmpIndTimeWindow[0]])) #(start,duration)

    for i in np.arange(len(svObs)):
        #for j in np.arange(len(svObs[i])):
        ax.broken_barh(svObs[i], (2*i, 2),facecolors =('black','red'))
    plt.show(block=False)



    #Print all keys where an observation would be made
    for (i,j)in x0d.keys():
        if x0d[(i,j)] == 1:
            print((i,j))



    #Plot all potential OBSs
    fig = plt.figure(2234352342222435434)
    ax = plt.gca()
    for i in np.arange(numOBSs):
        ax.broken_barh([(time_windows[i*2],time_windows[i*2+1]-time_windows[i*2])], (2*i, 2),facecolors =('black','red'))
    plt.title('all OBSs')
    plt.show(block=False)
    ####


    #Plot OBS interference matrix
    fig = plt.figure(1111546654654)
    ax = plt.gca()
    i=0
    k = 0 #the index of the line to print at
    for j in np.arange(numOBSs):
        if OBSInterferenceMatrix[i,j]==1:
            ax.scatter([time_windows[j*2],time_windows[j*2+1]],[2*k,2*k],color='red',s=1)
            ax.broken_barh([(time_windows[j*2],time_windows[j*2+1]-time_windows[j*2])], (2*k, 2),facecolors =('black','red'))
            k+=1
    ax.broken_barh([(time_windows[i*2],time_windows[i*2+1]-time_windows[i*2])], (-1, 4),facecolors =('green'))
    plt.title('OBS interferences')
    plt.show(block=False)



elif whichRun == 7:
    from earthOccultation import *
    #### SOLVING USING ONLY OBS TIME WINDOWS, CONTINUOUS TIME ##################################################
    print('#### EXECUTION 7 #################################################################')
    time0 = time.time()
    #### Generate Inputs
    numSV = len(satOBJs) #50 #100 SV
    numOBSs = 600 #numOBSs = numT
    meanOBSduration = 30
    stdOBSduration = 5
    tmax = 7*60 #the maximum time range

    ##### Create LOS array between each SV and each time, 1 if sat can see OBS k location at time j 
    for i in np.arange(len(satOBJs)):#iterate over satellite objects
        isVisible = np.zeros((satOBJs[i]['r'].shape[0],numOBSs))
        for j in np.arange(satOBJs[i]['r'].shape[0]): #iterate over time steps
            satDist = np.linalg.norm(satOBJs[i]['r'][j])
            earthFOVAng = np.arcsin(r_earth/np.linalg.norm(satOBJs[i]['r'][j]))
            for k in np.arange(r_locs.shape[1]):#Iterate over OBSs
                r_obj_OBS = r_locs.T[k] - satOBJs[i]['r'][j]
                OBSDist = np.linalg.norm(r_obj_OBS)
                angBetweenEarthCenterAndLoc = np.arccos(np.dot(r_obj_OBS,-satOBJs[i]['r'][j])/satDist/OBSDist)
                if earthFOVAng <= angBetweenEarthCenterAndLoc:
                    isVisible[j,k] = 1 #it is visible
        satOBJs[i]['isVisible'] = isVisible #shape 7x numpOBSs
        satOBJs[i]['isVisibleAtAll'] = np.any(satOBJs[i]['isVisible'],axis=0).astype('int') #can satellite i see OBS j at all (saved as array of j)
    ####

    ##### Randomly Generate OBS durations
    OBSDurs= np.random.normal(loc=meanOBSduration,scale=stdOBSduration,size=numOBSs) #create a random obs duration for each obs
    time_windows = list() #an array containing the start and end points of each OBS
    time_window_OBSnum = list() #an array containing OBS numbers of this time window
    for j in np.arange(numOBSs):
        start = np.random.uniform(low=0.,high=tmax-OBSDurs[j]) #create a random obs start time within time window
        time_windows.append(start) #save obs start time
        time_windows.append(start+OBSDurs[j]) #save obs end time
        time_window_OBSnum.append(j) #saves OBS index associated with time_window index
        time_window_OBSnum.append(j)
    time_windows = np.asarray(time_windows) #cast to arrays
    time_window_OBSnum = np.asarray(time_window_OBSnum)
    #sort by 
    time_windows_argsort = np.argsort(time_windows) #getting the indicies to sort the timewindows defined by the start and stop of each OBS
    sortedTW = time_windows[time_windows_argsort] #sorting the time windows from earliest to lates
    assert np.all(sortedTW[:-1] < sortedTW[1:]), 'not all time windows sorted'
    sortedTW_OBSnum = time_window_OBSnum[time_windows_argsort] #sorted timewindows
    numT = len(time_windows) #number of time windows
    OBSValues = np.linspace(start=1.,stop=1.+1e-4,num=numOBSs)#nearly equal values for all #np.ones(numOBSs) #equal values for all

    #creates a matrix where if i is the OBS being observed, all j that are 1 are OBSs which occur within the same timeframe as this observation
    OBSInterferenceMatrix = np.zeros((numOBSs,numOBSs))
    for (i,j) in itertools.product(np.arange(numOBSs),np.arange(numOBSs)):
        #if i==j: #skip this one
        #    continue
        #If the start of other is within TW, if the end of other is within TW, or if other spans TW
        if (time_windows[i*2] < time_windows[j*2] and time_windows[j*2] < time_windows[i*2+1])\
            or (time_windows[i*2] < time_windows[j*2+1] and time_windows[j*2+1] < time_windows[i*2+1])\
            or (time_windows[j*2] < time_windows[i*2] and time_windows[i*2+1] < time_windows[j*2+1]):
            OBSInterferenceMatrix[i,j] = 1
    ####

    #### Instantiate MIP Solver
    time1 = time.time()
    solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CLP_LINEAR_PROGRAMMING) # create solver instance
    #solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING) # create solver instance
    #solver = pywraplp.Solver.CreateSolver('SCIP') #DID NOT produce feasible solution MPSOLVER ABNORMAL
    #solver = pywraplp.Solver.CreateSolver('CP_SAT')
    #solver = pywraplp.Solver.CreateSolver('GLOP')
    print("7 Instantiate LP Solver: " + str((time1-time0)))


    #### Count # SV not able to see anything
    numWithNoVisible = 0
    for i in np.arange(len(satOBJs)):
        if np.any(satOBJs[i]['isVisible']):
            numWithNoVisible += 1
    print('7 numSV is ' + str(numSV))
    print('7 numSV able to see at least 1: ' + str(numSV-numWithNoVisible))

    #### Free Variables
    xs = dict() #is SV i observing OBS j
    for (i,j) in itertools.product(np.arange(numSV),np.arange(numOBSs)):
        if satOBJs[i]['isVisibleAtAll'][j] == 1: #the satellite cannot see current OBS at all
            xs[i,j] = solver.IntVar(0,1,'x' + str(i) + '_' + str(j))
            #xs[i,j] = solver.BoolVar('x' + str(i) + '_' + str(j))
    time2 = time.time()
    print("7 Done Adding Free Variables Total:" + str(len(xs)) + " execution time: " + str((time2-time1)))

    #### CONSTRAINTS ################################
    gt0constraint = list()
    for (i,j) in xs.keys():
        gt0constraint.append(solver.Add(xs[(i,j)] >= 0))
    # #Each SV CAN ONLY OBSERVE ONE OBS AT A TIME
    oneOBSatAtimeConstraints = list()
    for i in np.arange(numSV): #for each SV observing each OBS
        #Should be iterate over OBSs this SV can observe
        for j in np.arange(numOBSs): #iterate over all OBSS
            if satOBJs[i]['isVisibleAtAll'][j] == 1: #if this SV can see this OBS
                #add constraint ensuring only one OBS is observed at a time
                oneOBSatAtimeConstraints.append(solver.Add(\
                    solver.Sum([xs[i,k] for k in np.arange(numOBSs) \
                        if (i,k) in xs.keys() and OBSInterferenceMatrix[j,k] == 1 and satOBJs[i]['isVisibleAtAll'][k] == 1]) <= 1))

    #MULTIPLE SV CANNOT OBSERVE SAME OBS
    eachOBSonlyOnceConstraints = list()
    for j in np.arange(numOBSs):
        eachOBSonlyOnceConstraints.append(solver.Add(solver.Sum([xs[i,j] for i in np.arange(numSV) if (i,j) in xs.keys() and satOBJs[i]['isVisibleAtAll'][j] == 1]) <= 1))

    time3 = time.time()
    # print("6 Done Adding Constraints:" + str(len(oneOBSatAtimeConstraints)) + " execution time: " + str((time3-time2)))
    ####

    #### Objective Function
    objective = solver.Objective()
    #Set Total Partial Reward Coefficients
    for (i,j) in xs.keys():#itertools.product(np.arange(numSV),np.arange(numOBSs)):
        #if (satOBJs[i]['isVisibleAtAll'][j] == 1): #the satellite cannot see current OBS at all
        objective.SetCoefficient(xs[i,j],OBSValues[j])# 1.)#OBSValues[j]) #OBSValues could be different for each SV making the observation
    objective.SetMaximization()
    time4 = time.time()
    print('7 Done Setting Objective Function execution time: ' + " execution time: " + str((time4-time3)))

    #solver.EnableOutput()# this line enables output of the CBC MIXED INTEGER PROGRAM (Was hard to find don't delete)
    solver.SetTimeLimit(60*1000)#time limit for solver in milliseconds
    solver.SetNumThreads(6)
    #might be able to add SetHint() to furnish an IFS
    cpres = solver.Solve() # actually solve MIP
    time5 = time.time()
    print('7 Done Solving execution time: ' + " execution time: " + str((time5-time4)))
    x0d = dict()
    for (i,j) in xs.keys():
        x0d[(i,j)] = xs[i,j].solution_value()
        print(xs[i,j].solution_value()) 


    fig = plt.figure(65468464646541111)
    ax = plt.gca()
    #plt.matshow()
    svObs = list()
    for i in np.arange(numSV):
        svObs.append(list())
    for (i,j) in x0d.keys():
        if x0d[(i,j)] > 0:
            tmpIndTimeWindow = np.where(sortedTW_OBSnum == j)[0]
            #tmpIndTimeWindow = np.where(sortedTW_OBSnum == i)[0]
            #print(tmpIndTimeWindow)
            assert tmpIndTimeWindow[0] < tmpIndTimeWindow[1], 'time window ordering not right'
            svObs[i].append((sortedTW[tmpIndTimeWindow[0]],sortedTW[tmpIndTimeWindow[1]]-sortedTW[tmpIndTimeWindow[0]])) #(start,duration)

    for i in np.arange(len(svObs)):
        #for j in np.arange(len(svObs[i])):
        ax.broken_barh(svObs[i], (2*i, 2),facecolors =('black','red'))
    plt.title("Int Schedule")
    plt.show(block=False)

    # #Print all keys where an observation would be made
    # for (i,j)in x0d.keys():
    #     if x0d[(i,j)] == 1:
    #         print((i,j))

    #### REFORMULATION, the LP has been solved. Now consider only the nonzero solutions
    solver2 = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING) # create solver instance

    xs2 = dict() #is SV i observing OBS j
    for (i,j) in xs.keys():
        if xs[(i,j)].solution_value() > 0:
            xs2[i,j] = solver2.IntVar(0,1,'x' + str(i) + '_' + str(j))

    oneOBSatAtimeConstraints2 = list()
    for i in np.arange(numSV): #for each SV observing each OBS
        #Should be iterate over OBSs this SV can observe
        for j in np.arange(numOBSs): #iterate over all OBSS
            if satOBJs[i]['isVisibleAtAll'][j] == 1: #if this SV can see this OBS
                #add constraint ensuring only one OBS is observed at a time
                oneOBSatAtimeConstraints2.append(solver2.Add(\
                    solver2.Sum([xs2[i,k] for k in np.arange(numOBSs) \
                        if (i,k) in xs2.keys() and OBSInterferenceMatrix[j,k] == 1 and satOBJs[i]['isVisibleAtAll'][k] == 1]) <= 1))


    #MULTIPLE SV CANNOT OBSERVE SAME OBS
    eachOBSonlyOnceConstraints2 = list()
    for j in np.arange(numOBSs):
        eachOBSonlyOnceConstraints2.append(solver2.Add(solver2.Sum([xs2[i,j] for i in np.arange(numSV) if (i,j) in xs2.keys() and satOBJs[i]['isVisibleAtAll'][j] == 1]) <= 1))


    objective2 = solver2.Objective()
    #Set Total Partial Reward Coefficients
    for (i,j) in xs2.keys():#itertools.product(np.arange(numSV),np.arange(numOBSs)):
        #if (satOBJs[i]['isVisibleAtAll'][j] == 1): #the satellite cannot see current OBS at all
        objective2.SetCoefficient(xs2[i,j],OBSValues[j])# 1.)#OBSValues[j]) #OBSValues could be different for each SV making the observation
    objective2.SetMaximization()
    time4 = time.time()
    print('7_2 Done Setting Objective Function execution time: ' + " execution time: " + str((time4-time3)))

    #solver.EnableOutput()# this line enables output of the CBC MIXED INTEGER PROGRAM (Was hard to find don't delete)
    solver2.SetTimeLimit(60*1000)#time limit for solver in milliseconds
    solver2.SetNumThreads(6)
    #might be able to add SetHint() to furnish an IFS
    cpres2 = solver2.Solve() # actually solve MIP

    x0d2 = dict()
    for (i,j) in xs2.keys():
        x0d2[(i,j)] = xs2[i,j].solution_value()
        #print(xs2[i,j].solution_value()) 

    #### Evaluate How Much Reward Was Collected ###############################################################
    #Set Total Partial Reward Coefficients
    TPR = 0
    for (i,j) in xs2.keys():
        TPR += xs2[i,j].solution_value()*OBSValues[j]

    # Outputs the Solver Objective Function Value
    objFunValue = solver2.Objective().Value()

    print("7_2 Obj Fun Value: " + str(objFunValue) + " max obj fun val: " + str(numOBSs))
    print("7_2 Number of Variables: " + str(solver2.NumVariables()))
    print("7_2 Number of Constraints: " + str(solver2.NumConstraints()))


    cumSum = 0
    maxi = 0
    maxj = 0
    for (i,j) in x0d2.keys():
        cumSum += x0d2[(i,j)]
        if i > maxi:
            maxi = i
        if j > maxj:
            maxj = j
    print("cumSum: " + str(cumSum))
    
    fig = plt.figure(65468464646542222)
    ax = plt.gca()
    #plt.matshow()
    svObs2 = list()
    for i in np.arange(numSV):
        svObs2.append(list())
    for (i,j) in x0d2.keys():
        if x0d2[(i,j)] > 0:
            tmpIndTimeWindow = np.where(sortedTW_OBSnum == j)[0]
            #tmpIndTimeWindow = np.where(sortedTW_OBSnum == i)[0]
            #print(tmpIndTimeWindow)
            assert tmpIndTimeWindow[0] < tmpIndTimeWindow[1], 'time window ordering not right'
            svObs2[i].append((sortedTW[tmpIndTimeWindow[0]],sortedTW[tmpIndTimeWindow[1]]-sortedTW[tmpIndTimeWindow[0]])) #(start,duration)

    for i in np.arange(len(svObs2)):
        #for j in np.arange(len(svObs[i])):
        ax.broken_barh(svObs2[i], (2*i, 2),facecolors =('black','red'))
    plt.title("Int Schedule")
    plt.show(block=False)



    #Print all keys where an observation would be made
    # for (i,j)in x0d2.keys():
    #     if x0d2[(i,j)] == 1:
    #         print((i,j))



    #Plot all potential OBSs
    fig = plt.figure(22343523422224354342222)
    ax = plt.gca()
    for i in np.arange(numOBSs):
        ax.broken_barh([(time_windows[i*2],time_windows[i*2+1]-time_windows[i*2])], (2*i, 2),facecolors =('black','red'))
    plt.title('all OBSs')
    plt.show(block=False)
    ####


    #Plot OBS interference matrix
    fig = plt.figure(11115466546542222)
    ax = plt.gca()
    i=0
    k = 0 #the index of the line to print at
    for j in np.arange(numOBSs):
        if OBSInterferenceMatrix[i,j]==1:
            ax.scatter([time_windows[j*2],time_windows[j*2+1]],[2*k,2*k],color='red',s=1)
            ax.broken_barh([(time_windows[j*2],time_windows[j*2+1]-time_windows[j*2])], (2*k, 2),facecolors =('black','red'))
            k+=1
    ax.broken_barh([(time_windows[i*2],time_windows[i*2+1]-time_windows[i*2])], (-1, 4),facecolors =('green'))
    plt.title('OBS interferences')
    plt.show(block=False)




