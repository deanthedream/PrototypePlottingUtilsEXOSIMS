import numpy as np

#from ortools.linear_solver import pywraplp
import itertools
#from itertools import product
import random
import time
import psutil
import os

from mipOptimalScheduler import mipOptimalScheduler
from mipOptimalScheduler import recurSum
from mipOptimalScheduler import multiply_along_axis


# #### Inputs #############
# NumNodes = 1000 #number of nodes
# nodes = np.arange(NumNodes) #Set of all Nodes
# nodes_so = range(0,200) #Set of all sources
# nodes_si = range(150,250) #Set of all sinks
# nodes_sv = range(0,100) # set of nodes with fixed total reward collectable
# nodes_one = range(250,750) #set of nodes which can only communicate to one other node at a time
# nodes_two = range(750,1000) #set of nodes which can communicate with up to 2 other nodes at a time
# boldK = range(0,150) #The set of all data types
# nodes_sc = range(250,1000) #The set of all spacecraft nodes
# node_scTypes =  np.concatenate((np.ones(100)*0,np.ones(100)*1,np.ones(100)*2,np.ones(100)*3,np.ones(100)*4,np.ones(50)*5,np.ones(50)*6,np.ones(50)*7,np.ones(50)*8,np.ones(50)*9))#lets say there are 10 sc types indicated by integers 0-9
# #########################

# #### Set of Comm Edges
# #Normally, you would check and see of the spacecraft is capable of communicating with another one by checking the equipment and Rx Tx freq
# #Here, we use edge_scTypes to define the set of spacecraft types that can communicate to one another
# edges_scTypes =   [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),\
#                 (1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),\
#                 (2,0),(2,1),\
#                 (3,0),(3,1),(3,2),(3,4),(3,5),(3,6),\
#                 (4,0),(4,1),(4,2),(4,3),(4,4),\
#                 (5,0),(5,1),(5,2),(5,3),      (5,5),\
#                 (6,0),(6,1),(6,2),(6,3),            (6,6),(6,7),(6,8),\
#                 (7,0),(7,1),                              (7,7),(7,8),(8,9),\
#                 (8,0),(8,1),                                    (8,8),(8,9),\
#                 (9,0),(9,1),                                          (9,9)]
# ####
# #### 
# ####

startTime=time.time()

#### Inputs #############
NumNodes = 10 #number of nodes
nodes = np.arange(NumNodes) #Set of all Nodes
nodes_GS = np.arange(3) #Set of all ground stations
nodes_so = np.zeros(4) #Set of all sources
nodes_si = np.arange(1,3) #Set of all sinks
nodes_one = np.arange(7,10) #set of nodes which can only communicate to one other node at a time
nodes_two = np.arange(3,7) #set of nodes which can communicate with up to 2 other nodes at a time
boldK = np.arange(0,4) #The set of all data types (assumed to be partial reward)
nodes_sv = np.asarray([1 for i in boldK]) # set of nodes with fixed total reward collectable
totalSv = nodes_sv*1000
dtype_partialReward = np.asarray([0,1,2])
dtype_onCompletionReward =  np.asarray([3]) # a set of indicies indicating which of boldK is data on completion
nodes_sc = np.arange(3,10) #The set of all spacecraft nodes
node_scTypes_i =  np.asarray([0,0,0,0,1,1,2])
Tmax = 90 #number of timesteps
##
#normally, tx and rx cap between two satellites is dictated by the link-budget equation, we are doing a poor substitute for that
gsTxCap = 10000
gsRxCap = 1000
scTxRxCap = np.asarray([300,600,1000])
edges_scTypes =   [(0,0),(0,2),(1,1),(1,2),(2,0),(2,1),(2,2)] #The edges of types of spacecraft that can communicate with one another
#########################
#### Generate array of spacecraft data capacity
SCcap_l = np.asarray([200,500,3000])
####
#### Generate array of Source Data Volumes
Sv_byType = np.asarray([10000,10000,100,10])
####
#### Generate Rewards vs time
#Simulate 3 cases
#1 High Volume, High Reward, Low Availability
#2 High Volume, Low Reward, High Availability
#3 Low Volume, High Reward, Low Availability
Rtk = np.asarray([np.concatenate((np.zeros(30-10),100*np.ones(10),np.zeros(Tmax-30))),\
        1*np.ones(Tmax),\
        np.concatenate((np.zeros(35-10),100*np.ones(10),np.zeros(Tmax-35))),\
        1*np.ones(Tmax)]).T
Rzk = np.asarray([np.linspace(start=300,stop=0,num=Tmax)+100])#declining reward for zk

Rpltk = np.zeros((3,Tmax,4))
Rpltk[0] = Rtk
Rctk = np.zeros((Tmax,4))
Rctk[:,3] = Rzk
####

#### Functions for Procedurally Generating Inputs
def genLOS(MaxPeaks,PeakWidths,Tmax):
    """ Generates a vector indicating whether an object can see another object
    """
    numPeaks = random.randint(0,MaxPeaks)
    if numPeaks == 0:
        LOS_t = np.zeros(Tmax)
    else:
        LOS_ts = list()
        for peakInd in np.arange(numPeaks):
            startInd = random.randint(0,Tmax) #where the visibility starts
        
            visible = np.ones(PeakWidths) #the width of the visibility
            LOStmp = np.concatenate((np.zeros(startInd),visible,np.zeros(Tmax)))
            LOS_ts.append(LOStmp[:Tmax])
        LOS_t = np.sum(np.asarray(LOS_ts),axis=0)
        LOS_t = np.clip(LOS_t,a_min=0,a_max=1)
        if (2,Tmax) == LOS_t.shape:
            print(saltyburrito)
    return LOS_t

def genLOSbooleanMatrix(NumNodes,Tmax,nodes_sc,nodes_so,nodes_si,edges_scTypes,MaxPeaks=10,PeakWidths=60):
    """ Generates the LOS matrix using Gen LOS
    """
    MaxPeaks = 10
    PeakWidths = 60
    LOS_ijt = np.zeros((NumNodes,NumNodes,Tmax))
    for (i,j) in itertools.product(np.arange(NumNodes),np.arange(NumNodes)):
        if i == j: # prevents self communication #NOT HANDLED IN FIRST IF STATEMENT
            LOS_ijt[i,j] = np.zeros(Tmax)
        elif i in nodes_sc and j in nodes_sc: #these nodes are spacecraft
            iInd = np.where(nodes_sc == i)[0]
            jInd = np.where(nodes_sc == j)[0]
            if (iInd,jInd) in edges_scTypes: # spacecraft capable of communicating to one another
                LOS_ijt[i,j] = genLOS(MaxPeaks,PeakWidths,Tmax)
            else:
                LOS_ijt[i,j] = np.zeros(Tmax)
        elif i in nodes_so and j in nodes_so: # Sources cannot send to sources
            LOS_ijt[i,j] = np.zeros(Tmax)
        elif i in nodes_si and j in nodes_si: # Sinks cannot send to sinks
            LOS_ijt[i,j] = np.zeros(Tmax)
        elif i in nodes_so and j in nodes_si: # Sources can't sent to sinks
            LOS_ijt[i,j] = np.zeros(Tmax)
        elif i in nodes_si and j in nodes_so: # Sinks can't sent to sources
            LOS_ijt[i,j] = np.zeros(Tmax)
        else: #visibility of spacecraft to either sources or sinks
            LOS_ijt[i,j] = genLOS(MaxPeaks,PeakWidths,Tmax)
    return LOS_ijt

def calcTxRxCap(NumNodes,Tmax,nodes_so,nodes_si,nodes_sc,soTxRxCap,siTxRxCap,node_scTypes_i,scTxRxCap):
    """ Generates the TxRxCap for all i,j,t
    """
    TxRxCap_ijt = np.ones((NumNodes,NumNodes,Tmax))*np.inf
    for (i,j) in itertools.product(np.arange(NumNodes),np.arange(NumNodes)):
        if i in nodes_sc:
            ixcap = scTxRxCap[node_scTypes_i[np.where(i==nodes_sc)[0]]]
        elif i in nodes_so:
            ixcap = soTxRxCap[np.where(i==nodes_so)[0]]
        elif i in nodes_si:
            ixcap = siTxRxCap[np.where(i==nodes_si)[0]]
        else:
            print(error) #just makes an error

        if j in nodes_sc:
            jxcap = scTxRxCap[node_scTypes_i[np.where(j==nodes_sc)[0]]]
        elif j in nodes_so:
            jxcap = soTxRxCap[np.where(j==nodes_so)[0]]
        elif j in nodes_si:
            jxcap = siTxRxCap[np.where(j==nodes_si)[0]]
        else:
            print(error) #just makes an error

        if ixcap < jxcap:
            TxRxCap_ijt[i,j,:] = ixcap*np.linspace(start=0,stop=Tmax-1,num=Tmax).astype(int)
        else: #ixcap > jxcap
            TxRxCap_ijt[i,j,:] = jxcap*np.linspace(start=0,stop=Tmax-1,num=Tmax).astype(int)

    # for i in np.arange(len(nodes_sc)):
    #     jInds = np.where(TxRxCap_ijt[nodex_sc[i],:,0] > scTxRxCap[])
    # for i in np.arange(NumNodes):
    #     if i in nodes_so: #the node is a source node
    #         TxRxCap_ijt[i] = np.ones((NumNodes,Tmax))*soTxRxCap
    #         continue
    #     jInds = np.where(TxRxCap_ijt[i,:,0] > scTxRxCap[node_scTypes_i[i-3]])[0]
    #     if len(jInds) > 0:
    #         TxRxCap_ijt[i] = multiply_along_axis(np.ones((NumNodes,Tmax)), scTxRxCap[node_scTypes_i[np.asarray(jInds)-3]], axis=0)
    return TxRxCap_ijt

def totalNumberOfConstraints(nodes,Tmax,nodes_one,nodes_two,nodes_sc,boldK,dtype_onCompletionReward):
    numCon_binaryCommVariable = len(nodes)*len(nodes)*Tmax
    numCon_onlyOneCommConstraint = len(nodes_one)*Tmax
    numCon_onlyOneCommInOneCommOutConstraint = len(nodes_two)*Tmax
    numCon_OnboardStorageCapacity = len(nodes_sc)*recurSum(Tmax)
    numCon_positiveNetDataAtNode = len(nodes_sc)*recurSum(Tmax)
    numCon_SourceExtractionLimitConstraint = len(boldK)
    numCon_dataOnCompletion = Tmax*len(dtype_onCompletionReward)
    numCon_lessThanOneFinish = len(dtype_onCompletionReward)
    TotalNumberConstraints = numCon_binaryCommVariable+numCon_onlyOneCommConstraint+numCon_onlyOneCommInOneCommOutConstraint+numCon_OnboardStorageCapacity+\
        numCon_positiveNetDataAtNode+numCon_SourceExtractionLimitConstraint+numCon_dataOnCompletion+numCon_lessThanOneFinish
    return TotalNumberConstraints

def genRewardt(Tmax,num=2,widthFrac=0.1):
    """Generates reward boolean (0 or 1) as a function of time
    """
    Rt = np.zeros(Tmax)
    width = int(np.ceil(Tmax*widthFrac))
    for i in np.arange(num):
        startInd = np.random.randint(0,Tmax-width)
        Rt[startInd:(startInd+width)] = np.ones(width)
    return Rt
####


#### Inputs To Procedurally Generate Inputs #####
#DELETEnumGS = 5 # the number of ground stations to generate
numSources = 10 #the number of nodes acting as sources of data
numSinks = 3 # we will assume all GS act as sinks *it might be interesting to look at scenarios where spacecraft are also sinks*
numSC = 30 # the number of spacecraft we will consider
Tmax = 90
numDataTypes = 4

def runMIPSolverInLoop(numSources,numSinks,numSC,Tmax,numDataTypes,OnboardStorageCapacity_boolean):
    ## Construct the sets of nodes
    #DELETEnodes_GS = np.arange(numGS)
    nodes_so = np.arange(numSources)
    nodes_si = np.max(nodes_so) + 1 + np.arange(numSinks)
    nodes_sc = np.max(nodes_si) + 1 + np.arange(numSC)
    nodes = np.concatenate((nodes_so,nodes_si,nodes_sc)) #combine into master set of nodes
    NumNodes = len(nodes) #total number of nodes in the simulation
    numSCTypes = random.randint(1,numSC) #number of different SC Types
    weights = np.ceil(np.exp(np.linspace(start=4,stop=0,num=numSCTypes))).astype(int) #weighting for selecting some spacecraft more than others
    node_scTypes_i = np.sort(np.random.choice(a=list(np.arange(numSCTypes)),size=numSC,p=weights/np.sum(weights))) #randomly assign SC types to SC nodes
    scTxRxCap = np.ceil(np.logspace(start=1.5,stop=3,num=numSCTypes)).astype(int) #generates logarithmically spaced spacecraft max TxRxCap
    SCcap_l = np.ceil(np.logspace(start=2,stop=3.5,num=numSCTypes)).astype(int) #generate data capacities of all spacecraft
    soTxRxCap = np.ones(numSources)*1000
    siTxRxCap = np.ones(numSinks)*1000

    # constructs data types and assigns sources/sinks
    #numDataTypes = random.randint(4,20) #generates the number of data-types to have
    dtype_onCompletionReward =  np.asarray([0,1,2,3]) # a set of indicies indicating which of boldK is data on completion
    boldK = np.arange(0,numDataTypes) #The set of all data types (assumed to be partial reward)
    kFromNode = np.random.choice(a=nodes_so,size=numDataTypes) # indicated which node is the source for a given data-type (does not allow multiple data sources)
    kToNode = np.random.choice(a=nodes_si,size=numDataTypes) # indicates which node is the sink for the given data-type (does not allow multiple sinks)
    Svweights = np.asarray([10,20,50,100])
    Sv_byType = np.random.choice(a=np.asarray([10000,10000,100,10]),size=numDataTypes,p=Svweights/np.sum(Svweights)) #

    allPossibleSCcomms = [(i,j) for (i,j) in itertools.product(np.arange(numSCTypes),np.arange(numSCTypes))]
    indsOfSCcomms = np.random.choice(a=np.arange(len(allPossibleSCcomms)),size=int(np.ceil(numSCTypes*numSCTypes/4))) #A random metric for how many to select
    edges_scTypes = [allPossibleSCcomms[i] for i in indsOfSCcomms]
    ####
    #### Set of Comm Edges
    #Normally, you would check and see of the spacecraft is capable of communicating with another one by checking the equipment and Rx Tx freq
    #Here, we use edge_scTypes to define the set of spacecraft types that can communicate to one another

    ####
    #### Generate LOS Boolean matrix
    LOS_ijt = genLOSbooleanMatrix(NumNodes,Tmax,nodes_sc,nodes_so,nodes_si,edges_scTypes,MaxPeaks=10,PeakWidths=60)
    ####
    #### TxRxCap array of communication capacities
    #Lets assume fixed capacities (they will likely vary)

    TxRxCap_ijt = calcTxRxCap(NumNodes,Tmax,nodes_so,nodes_si,nodes_sc,soTxRxCap,siTxRxCap,node_scTypes_i,scTxRxCap) 
    #### Generate Task Rewards
    Rk = np.random.choice(np.arange(20),size=numDataTypes) #net reward per data-type
    Rptk = np.zeros((Tmax,numDataTypes))
    for k in np.arange(numDataTypes):
        Rptk[:,k] = genRewardt(Tmax,num=2,widthFrac=0.1)*Rk[k]


    # Rtk = np.asarray([np.concatenate((np.zeros(30-10),100*np.ones(10),np.zeros(Tmax-30))),\
    #         1*np.ones(Tmax),\
    #         np.concatenate((np.zeros(35-10),100*np.ones(10),np.zeros(Tmax-35))),\
    #         1*np.ones(Tmax)]).T


    # Rpltk = np.zeros((numSinks,Tmax,numDataTypes))
    # Rpltk[0] = Rtk
    # Rpltk = 

    Rzk = np.asarray([np.linspace(start=300,stop=0,num=Tmax)*3])#declining reward for zk
    Rctk = np.zeros((Tmax,len(dtype_onCompletionReward)))
    for k in dtype_onCompletionReward:
        Rctk[:,k] = Rzk*(k+0.5)

    #### Calculate Number of Constraints #########################
    TotalNumberConstraints = totalNumberOfConstraints(nodes,Tmax,nodes_one,nodes_two,nodes_sc,boldK,dtype_onCompletionReward)
    print('Total Number Of Constraints is: ' + str(TotalNumberConstraints))
    ##############################################################





    ################################################################################################
    print('Starting MIP Solver')
    x0d, y0d, z0d, TPR, TROC, objFunValue = mipOptimalScheduler(nodes, Tmax, boldK, TxRxCap_ijt, \
        nodes_one, nodes_two, SCcap_l, node_scTypes_i, nodes_sc, Sv_byType, \
        nodes_si, Rptk, Rctk, LOS_ijt, nodes_so, nodes_sv, kFromNode, kToNode, \
        dtype_onCompletionReward, OnboardStorageCapacity_boolean=OnboardStorageCapacity_boolean)

    totalExecutionTime = time.time() - startTime
    #print('Total Execution Time: ' + str(totalExecutionTime))

    return TotalNumberConstraints, x0d, y0d, z0d, TPR, TROC, objFunValue, totalExecutionTime

#runMIPSolverInLoop(1,1,10,90,4)

# listnumSources = [1,4,8]
# listnumSinks = [1,4,8]
# listnumSC = [10,20,30]
# listTmax = [45,90,135]
# listnumDataTypes = [4,8,12,16,20,30]
listnumSources = [4,8]
listnumSinks = [4,8]
listnumSC = [10,20,30]
listTmax = [45,90,135]
listnumDataTypes = [12]
#runs resulting in killed
# numSources = 1, numSinks=1, numSC=10, Tmax=180, numDataTypes=16

####Sims Run with OnboardStorageCapacity_boolean
# OnboardStorageCapacity_boolean = True
# simRun = [(1,1,10,90,4),(1,1,10,90,8),(1,1,10,90,12),(1,1,10,90,16),(1,1,10,90,20),(1,1,10,90,30),(1,1,10,45,4),(1,1,10,45,8),\
# (1,1,10,45,12),(1,1,10,45,16),(1,1,10,45,20),(1,1,10,45,30),(1,1,10,135,4),(1,1,10,135,8),(1,1,10,135,12),(1,1,10,135,16),\
# (1,1,10,135,20),(1,1,10,135,30),(1,1,10,180,4),(1,1,10,180,8),(1,1,10,180,12),\
# (1,1,20,45,4),(1,1,20,45,8),(1,1,20,45,12),(1,1,20,45,16),(1,1,20,45,20),(1,1,20,45,30),(1,1,20,90,4),
# (1,1,20,90,8),(1,1,20,90,12),(1,1,20,90,16),(1,1,30,45,20),(1,1,30,45,16),(1,1,30,45,12),(1,1,30,45,8),(1,1,30,45,4),
# (1,4,10,45,8),(1,4,10,45,12),(1,4,10,45,20),(1,4,10,45,30),(1,4,10,45,4),(1,4,10,90,4),(1,4,10,90,8),(1,4,10,90,12),
# (1,4,10,90,16),(1,4,10,90,20),(1,4,10,90,30),(1,4,10,135,4),(1,4,10,135,8),(1,4,10,135,12),(1,4,10,135,16),(1,4,10,45,16),
# (1,4,20,45,4),(1,4,20,45,8),(1,4,20,45,12),(1,4,20,45,16),(1,4,20,45,20),(1,4,20,90,8),(1,4,20,90,4),(1,4,30,45,4),
# (1,4,30,45,8),(1,4,30,45,12),(1,4,30,45,16),(1,4,30,45,20),(1,4,30,90,4),(4,4,10,45,12),(4,4,10,90,12),
# (4,4,10,135,12),(4,4,20,45,12),(4,4,30,45,12),(4,8,10,45,12),(4,8,20,45,12),(4,8,30,45,12),(8,4,10,45,12),(8,4,10,90,12)
# ]
# simFailed = [(1,2,10,90,4),(1,1,10,180,16),(1,1,20,90,20),(1,1,20,90,30),(1,1,20,135,4),(1,1,20,135,8),(1,1,30,45,30),(1,1,30,90,4),
# (1,1,30,90,8),(1,1,30,135,4),(1,1,30,135,8),(1,4,10,135,20),(1,4,10,135,30),(1,4,20,90,12),(1,4,20,90,16),(1,4,20,90,20),(1,4,20,135,8),
# (1,4,30,45,30),(1,4,30,90,8),(1,4,30,90,12),(4,4,20,90,12),(4,4,20,135,12),(4,4,30,90,12),(4,4,30,135,12),(4,8,10,135,12),(4,8,20,90,12),
# (4,8,20,135,12),(4,8,30,90,12),(4,8,30,135,12),(8,4,10,135,12)]
# simProjectedToFail = [(1,1,20,135,12),(1,1,20,135,16),(1,1,20,135,20),(1,1,20,135,30),(1,1,30,90,12),(1,1,30,90,16),(1,1,30,90,20),
# (1,1,30,90,30),(1,1,30,135,12),(1,1,30,135,16),(1,1,30,135,20),(1,1,30,135,30),(1,4,20,90,30),(1,4,20,135,12),(1,4,20,135,16),
# (1,4,20,135,20),(1,4,20,135,30)]
# simNotSolved = [(1,4,20,45,30),(1,4,20,135,4),(4,8,10,90,12)]
####

#### Sims Run with OnboardStorageCapacity_boolean
OnboardStorageCapacity_boolean = False
simRun = [(4,4,10,45,12),(4,4,10,90,12),(4,4,10,135,12),(4,4,20,45,12),(4,4,20,90,12),(4,4,30,45,12),(4,8,10,45,12),
(4,8,10,45,12),(4,8,10,90,12),(4,8,10,135,12),(4,8,20,45,12),(4,8,20,90,12),(8,4,20,45,12),(8,4,10,45,12),(8,4,10,90,12),
(8,4,30,45,12),(8,8,10,45,12),(8,8,20,45,12),(8,8,10,90,12),(8,8,30,45,12)]
simFailed = [(4,4,20,135,12),(4,4,30,90,12),(4,4,30,135,12),(4,8,20,135,12),(4,8,30,90,12),(4,8,30,135,12),(8,4,20,90,12),(8,4,20,135,12),
(8,4,30,90,12),(8,4,30,135,12),(8,8,20,45,12),(8,8,20,90,12),(8,8,20,135,12),(8,8,30,90,12),(8,8,30,135,12)]
simProjectedToFail = []
simNotSolved = [(4,8,30,45,12),(8,4,10,135,12),(8,8,10,135,12)]
####

data = list()
for (numSources,numSinks,numSC,Tmax,numDataTypes) in itertools.product(listnumSources,listnumSinks,listnumSC,listTmax,listnumDataTypes):
    print('Inputs are: numSources ' + str(numSources) + ' numSinks ' + str(numSinks) + ' numSC ' + str(numSC) + ' Tmax ' + str(Tmax) + ' numDataTypes ' + str(numDataTypes))
    if (numSources,numSinks,numSC,Tmax,numDataTypes) in simRun:
        print('skipped already run')
        continue
    if (numSources,numSinks,numSC,Tmax,numDataTypes) in simFailed:
        print('skipping already failed')
        continue
    if (numSources,numSinks,numSC,Tmax,numDataTypes) in simProjectedToFail:
        print('skipping projected to fail')
        continue
    if (numSources,numSinks,numSC,Tmax,numDataTypes) in simNotSolved:
        print('skipping sim did not solve')
        continue
    out = runMIPSolverInLoop(numSources,numSinks,numSC,Tmax,numDataTypes,OnboardStorageCapacity_boolean)
    data.append(out)
    #print(out)
    with open('storageCapBoolean' + str(int(OnboardStorageCapacity_boolean)) + 'numSources' + str(numSources) + 'numSinks' + str(numSinks) + 'numSC' + str(numSC) + 'Tmax' + str(Tmax) + 'numDataTypes' + str(numDataTypes) + '.txt', 'w') as f:
        f.write('Inputs are: numSources ' + str(numSources) + ' numSinks ' + str(numSinks) + ' numSC ' + str(numSC) + ' Tmax ' + str(Tmax) + ' numDataTypes ' + str(numDataTypes) + '\n')
        f.write('TotalNumberConstraints: ' + str(out[0]) + '\n')
        f.write('totalExecutionTime: ' + str(out[7]) + '\n')
        f.write('objFunValue: ' + str(out[6]) + '\n')
        f.write('x0d: ' + str(out[1]) + '\n')
        f.write('y0d: ' + str(out[2]) + '\n')
        f.write('z0d: ' + str(out[3]) + '\n')
        f.write('TPR: ' + str(out[4]) + '\n')
        f.write('TROC: ' + str(out[5]) + '\n')
    print('Inputs were: numSources ' + str(numSources) + ' numSinks ' + str(numSinks) + ' numSC ' + str(numSC) + ' Tmax ' + str(Tmax) + ' numDataTypes ' + str(numDataTypes))
        


#### Plot Stuff
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#### Plot 'run success' vs # SC and time For onboard storage capacity constraint
fig = plt.figure(num=5861313)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
allSims = simRun + simFailed + simProjectedToFail + simNotSolved
maxNumNodes = 0
for i in np.arange(len(allSims)):
    tmpNumNodes = allSims[i][0]+allSims[i][1]+allSims[i][2]
    simTime = allSims[i][3]
    if allSims[i] in simRun:
        plt.scatter(tmpNumNodes,simTime,color='blue',marker='o',s=64)
    elif allSims[i] in simFailed:
        plt.scatter(tmpNumNodes,simTime,color='red',marker='x',s=16)
    elif allSims[i] in simProjectedToFail:
        plt.scatter(tmpNumNodes, simTime, color='red',marker='d',s=16)
    elif allSims[i] in simNotSolved:
        plt.scatter(tmpNumNodes, simTime, color='red', marker='+' ,s=16)
    if tmpNumNodes > maxNumNodes:
        maxNumNodes = tmpNumNodes
plt.ylim([0,1.1*np.max(np.asarray(allSims)[:,3])])
plt.xlim([0,1.1*maxNumNodes])
plt.xlabel('# of Nodes', weight='bold')
plt.ylabel('Simulation Time (units)', weight='bold')
plt.show(block=False)



#### Add Timeline Plot
x0d = out[1]
xkeys = list(x0d.keys())
xp = list()
yp = list()
zp = list()
for i in np.arange(len(xkeys)):
    if x0d[xkeys[i]] == 1:
        xp.append(xkeys[i][0])
        yp.append(xkeys[i][1])
        zp.append(xkeys[i][2])
xp = np.asarray(xp)
yp = np.asarray(yp)
zp = np.asarray(zp)

fig2 = plt.figure(num=1000)
ax = fig2.add_subplot(111,projection='3d')
ax.scatter(xp,yp,zp,marker='o',color='black',s=1)
ax.set_xlabel('Node i', weight='bold')
ax.set_ylabel('Node j', weight='bold')
ax.set_zlabel('Time t', weight='bold')
plt.show(block=False)


