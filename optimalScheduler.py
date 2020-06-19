import numpy as np

from ortools.linear_solver import pywraplp
import itertools
#from itertools import product
import random

# #### Inputs #############
# NumNodes = 1000 #number of nodes
# nodes = np.arange(NumNodes) #Set of all Nodes
# nodes_so = range(0,200) #Set of all sources
# nodes_si = range(150,250) #Set of all sinks
# nodes_sv = range(0,100) # set of nodes with fixed total reward collectable
# nodes_one = range(250,750) #set of nodes which can only communicate to one other node at a time
# nodes_two = range(750,1000) #set of nodes which can communicate with up to 2 other nodes at a time
# nodes_dtypes = range(0,150) #The set of all data types
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

#### Inputs #############
NumNodes = 10 #number of nodes
nodes = np.arange(NumNodes) #Set of all Nodes
nodes_GS = np.arange(2,3) #Set of all ground stations
nodes_so = np.arange(0,1) #Set of all sources
nodes_si = np.arange(2,3) #Set of all sinks
nodes_sv = np.arange(1,1) # set of nodes with fixed total reward collectable
nodes_one = np.arange(7,10) #set of nodes which can only communicate to one other node at a time
nodes_two = np.arange(0,7) #set of nodes which can communicate with up to 2 other nodes at a time
nodes_dtypes = np.arange(0,4) #The set of all data types
nodes_sc = np.arange(3,10) #The set of all spacecraft nodes
node_scTypes =  np.asarray([0,0,0,0,1,1,2])
Tmax = 100 #number of timesteps
##
#normally, tx and rx cap between two satellites is dictated by the link-budget equation, we are doing a poor substitute for that
gsTxCap = 10000
gsRxCap = 1000
scTxRxCap = np.asarray([300,600,1000])
edges_scTypes =   [(0,0),(0,2),(1,1),(1,2),(2,0),(2,1),(2,2)] #The edges of types of spacecraft that can communicate with one another
#########################

#### Set of Comm Edges
#Normally, you would check and see of the spacecraft is capable of communicating with another one by checking the equipment and Rx Tx freq
#Here, we use edge_scTypes to define the set of spacecraft types that can communicate to one another
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
            LOS_ts.append(LOStmp[:100])
        LOS_t = np.sum(np.asarray(LOS_ts),axis=0)
        LOS_t = np.clip(LOS_t,a_min=0,a_max=1)
        if (2,100) == LOS_t.shape:
            print(saltyburrito)
    return LOS_t
####

#### Generate LOS Boolean matrix
MaxPeaks = 2
PeakWidths = 10
LOS_ijt = np.zeros((NumNodes,NumNodes,Tmax))
for (i,j) in itertools.product(np.arange(NumNodes),np.arange(NumNodes)):
    #both nodes are spacecraft
    if i in nodes_sc and j in nodes_sc: #these spacecraft are spacecraft
        iInd = np.where(nodes_sc == i)[0]
        jInd = np.where(nodes_sc == j)[0]
        if (iInd,jInd) in edges_scTypes: # spacecraft capable of communicating to one another
            LOS_ijt[i,j] = genLOS(MaxPeaks,PeakWidths,Tmax)
        else:
            LOS_ijt[i,j] = np.zeros(Tmax)
    elif i in nodes_GS and j in nodes_GS: # Ground stations cannot recieve input from one another
        LOS_ijt[i,j] = np.zeros(Tmax)
    elif i in nodes_so and j in nodes_si: #sources can't sent to sinks
        LOS_ijt[i,j] = np.zeros(Tmax)
    elif i in nodes_si and j in nodes_so: #sinks can't sent to sources
        LOS_ijt[i,j] = np.zeros(Tmax)
    else: #visibility of spacecraft to either sources or sinks
        LOS_ijt[i,j] = genLOS(MaxPeaks,PeakWidths,Tmax)
####

#### Generate array of spacecraft data capacity
SCcap_i = np.asarray([200,500,3000])
####
#### Generate array of Source Data Volumes
Sv_byType = np.asarray([10000,10000,100,10])
####
#### TxRxCap array of communication capacities
#Lets assume fixed capacities (they will likely vary)


# def genTxRxCap(gsInd=None,tx_scTypeInd=None,rx_scTypeInd=None,gsTxCap,gsRxCap,scTxRxCap):
#     if gs

#     return 
# TxRxCap_ijt = np.asarray([])

TxRxCap_ijt = np.ones((NumNodes,NumNodes,Tmax))*np.inf
TxRxCap_ijt[2] = np.ones((NumNodes,Tmax))*gsRxCap
for i in np.arange(NumNodes):
    if i in [0,1,2]:
        continue
    jInds = np.where(TxRxCap_ijt[i,:,0] < scTxRxCap[node_scTypes[i-3]])[0]
    if len(jInds) > 0:
        TxRxCap_ijt[i] = np.ones(NumNodes,Tmax)*scTxRxCap[node_scTypes[np.asarray(jInds)-3]]


# for i in np.arange(NumNodes):
#     if i == 0:
#         jInds = np.where(TxRxCap[i,:,0] < gsTxCap)[0]
#         iInds = np.where(TxRxCap[:,j,0] < gsRxCap)[0]

# tmpTxRxCam_ijt
# for (i,j) in itertools.product(NumNodes,NumNodes):
#     if i == 0:
#         TxRxCap_ijt[i] = np.where(TxRxCap[i,:,0] < gsTxCap)
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
####

################################################################################################


#### 5. Formulating MIP to filter out stars we can't or don't want to reasonably observe
solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING) # create solver instance
#solver = pywraplp.Solver('simple_mip_program',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING) #found online

#### Create set of all possible communication nodes
# allPossibleComms = list(product(N,N))
# tmp = [allPossibleComms.remove((i,i)) for i in N] # remove spacecraft that can only talk to each other
# del tmp
# Edges = set(allPossibleComms)


#xstruct = dict()
#xstruct[i,j,t,k] = solver.IntVar(0,1,'x' + str(i) + str(j) + str(t) + str(k)) for (i,j,t,k) in itertools.product(nodes,nodes,np.arange(Tmax),nodes_dtypes))

#### Free Variables
# Free Variable: Do node i send data to node j in time t?
#xs = np.asarray([solver.IntVar(0,1,'x' + str(i) + str(j) + str(t) + str(k)) for (i,j,t,k) in itertools.product(nodes,nodes,np.arange(Tmax),nodes_dtypes)])
xs = dict()
for (i,j,t,k) in itertools.product(nodes,nodes,np.arange(Tmax),nodes_dtypes):
    xs[i,j,t,k] = solver.IntVar(0,1,'x' + str(i) + str(j) + str(t) + str(k))
# Free Variable: How much data to transmit from node i to node j?
#ys = np.asarray([solver.IntVar(0,TxRxCap_ijt[i,j,t],'y' + str(i) + str(j) + str(t) + str(k)) for (i,j,t,k) in itertools.product(nodes,nodes,np.arange(Tmax),nodes_dtypes)])
ys = dict()
for (i,j,t) in itertools.product(nodes,nodes,np.arange(Tmax)):
    ys[i,j,t] = solver.IntVar(0,TxRxCap_ijt[i,j,t],'y' + str(i) + str(j) + str(t))


# Create zs descirbing when data transmitted must be done being transmitted
##################

#### CONSTRAINTS ################################
#All ys leq xs
# for (i,j,t) in itertools.product(nodes,nodes,np.arange(Tmax)):
#     print((i,j,t))

binaryCommConstraint = [solver.Add(ys[i,j,t] <= np.sum([xs[i,j,t,k] for k in nodes_dtypes])) for (i,j,t) in itertools.product(nodes,nodes,np.arange(Tmax))]
# Only One Comm Constraint
onlyOneCommConstraint = [solver.Add(solver.Sum([xs[i,j,t,k]+xs[j,i,t,k] for (j,k) in itertools.product(nodes,nodes_dtypes)]) <= 1) for (i,t) in itertools.product(nodes_one,np.arange(Tmax))]
# Only Two Comm Constraint
onlyTwoCommConstraint = [solver.Add([solver.Sum([xs[i,j,t,k] for (j,k) in itertools.product(nodes,nodes_dtypes)]) <= 1][0]) for (i,t) in itertools.product(nodes_two,np.arange(Tmax))]
#################################################

#### Create does my SC communicate constraint

####

#xs = [ solver.IntVar(0.0,1.0, 'x'+str(j)) for j in N] # define x_i variables for each star either 0 or 1
#print('Finding baseline fixed-time optimal target set.')

#constraint is x_i*t_i < maxtime
# constraint = solver.Constraint(-solver.infinity(),maxTime.to(u.day).value) #hmmm I wonder if we could set this to 0,maxTime
# for j,x in enumerate(xs):
#     constraint.SetCoefficient(x, t0[j].to('day').value + ohTimeTot.to(u.day).value) # this forms x_i*(t_0i+OH) for all i

#objective is max x_i*comp_i
objective = solver.Objective()
for (i,j,t,k) in xs.keys():
    objective.SetCoefficient(ys[i,j,t], Rtk[t,k])
objective.SetMaximization()

#solver.EnableOutput()# this line enables output of the CBC MIXED INTEGER PROGRAM (Was hard to find don't delete)
solver.SetTimeLimit(5*60*1000)#time limit for solver in milliseconds
cpres = solver.Solve() # actually solve MIP
x0 = np.array([xs[i,j,t,k].solution_value() for (i,j,t,k) in xs.keys()]) # convert output solutions
y0 = np.array([ys[i,j,t].solution_value() for (i,j,t) in ys.keys()]) # convert output solutions

print('Yay it solved!')
