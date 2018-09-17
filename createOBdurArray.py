import numpy as np
import math
from pylab import *

close('all')

#### Inputs ##########################################################
exoplanetObsTime = 365.25#Maximum amount of observing time
maxNumYears = 6.#maximum number of years we will allow
maxNumDays = maxNumYears*365.25

OBdur = 1.
#####################################################################

def maxNumRepInTime(OBdur,Time):
    #Time in days
    #OBdur in days
    maxNumRepInTime = np.asarray([math.ceil(Time/OBduri) for OBduri in OBdur])#.astype('int')
    maxRepsPerYear = np.asarray([math.ceil(365.25/OBduri) for OBduri in OBdur])#.astype('int')
    return maxNumRepInTime, maxRepsPerYear


missionPortion = exoplanetObsTime/(maxNumYears*365.25) # This is constrained


#Create List of OB durations
OBdur2 = list(set(np.logspace(np.log10(0.1),np.log10(365.),num=50,base=10.).astype(int)))
tmp = list(np.asarray(range(10))+1.5)
OBdur2.remove(0)
OBdur2 = sort(np.asarray(OBdur2 + tmp))


#Calculate Maximum number of repetitions within exoplanetObsTime
maxNumRepTot2, maxRepsPerYear2 = maxNumRepInTime(OBdur2,exoplanetObsTime)
fig = figure(1)
loglog(OBdur2,maxNumRepTot2,marker='o')
xlabel('Num Days')
ylabel('Max Num Reps')
show(block=False)


##### Distribute Observing Blocks throughout years ##################################
#Even Distribution
def evenDist(numOB,OBdur,maxNumDays):
    """
    maxNumDays - missionLife in days
    OBdur - duration of an OB in days
    numOB - number of Observing blocks to create
    """
    OBstartTimes = np.linspace(0,maxNumDays-OBdur,num=numOB, endpoint=True)
    #Check maxNumDays - OBstartTimes[-1] < OBdur
    return OBstartTimes

#### GeomDist OB Spacing ############################################
def geomDist(numOB,OBdur,maxNumDays):
    """
    maxNumDays - missionLife in days
    OBdur - duration of an OB in days
    numOB - number of Observing blocks to create
    """
    OBstartTimes = np.geomspace(1e-10,maxNumDays-OBdur,num=numOB, endpoint=True)
    #Check maxNumDays - OBstartTimes[-1] < OBdur
    return OBstartTimes

GeomDistOBstartTimes = list()
for i in np.arange(len(OBdur2)):
    GeomDistOBstartTimes.append(geomDist(maxNumRepTot2[i], OBdur2[i], maxNumDays))
####################################################################

#### HarmonicDist ##################################################
def harmonicDist(numOB, OBdur, exoplanetObsTime):#, missionPortion):
    """
    """
    daysInYear = 365.25 #Days in a year
    numYears = exoplanetObsTime/daysInYear#Number of years

    numOBperYear = np.ceil(numOB/numYears)#Number of Observing blocks that can fit into 1 year

    OneYrOBstartTimes = np.asarray([])
    OneYrOBendTimes = np.asarray([])
    for i in range(int(np.ceil(numYears))):#Note it should be fine if we fully fill out the remaining year
        OneYrOBstartTimes = np.append(OneYrOBstartTimes,np.linspace(0.,365.25,num=numOBperYear, endpoint=False)+i*daysInYear)
        OneYrOBendTimes = np.append(OneYrOBendTimes,np.linspace(0.,365.25,num=numOBperYear, endpoint=False)+float(i*daysInYear+OBdur))
    return OneYrOBstartTimes, OneYrOBendTimes
HarmonicDistOB = list()
for i in np.arange(len(OBdur2)):
    tmpStart, tmpEnd = harmonicDist(maxNumRepTot2[i], OBdur2[i], maxNumDays)
    HarmonicDistOB.append([tmpStart, tmpEnd])


#Write to output files
path = '/home/dean/Documents/exosims/Scripts/'
tmp = ''
for i in np.arange(len(OBdur2)):
    myList = list()
    for j in range(len(HarmonicDistOB[i][0])):
        myList.append(str(HarmonicDistOB[i][0][j]) + ',' + str(HarmonicDistOB[i][1][j]) + '\n')
    outString = ''.join(myList)
    #print outString
    fname = path + 'harmonicOB' + str(i) + '.csv'
    f = open(fname, "w")
    f.write(outString)
    print '"' + fname.split('/')[-1] + '",'
#####################################################################





#######################################################################################
#maxNumReps = maxNumYears*maxRepsPerYear2#number of Reps/ number of years #The minimum number of repetitions to go into 1 year in order to finish before 6 years

figure(2)
num = np.linspace(0,50,num=50)
tmp = np.geomspace(0.0001,maxNumDays,num=50)
frac = 0.6
tmp2 = frac*np.geomspace(0.0001,maxNumDays,num=50) + (1-frac)*np.linspace(0,maxNumDays,num=50)
tmp3 = frac*np.geomspace(0.0001,maxNumDays,num=50) + (1-frac)*np.linspace(0,maxNumDays,num=50)*np.geomspace(0.0001,maxNumDays,num=50)/maxNumDays
tmp4 = np.linspace(0,maxNumDays,num=50)
def func(x,valMax):
    m=200.
    frac = 0.8
    val1 = m*x
    val2 = x**3#12*x**2#np.exp(x)
    val = frac*val1 + (1-frac)*val2
    val = val*valMax/max(val)
    return val
def dfunc(x):
    dval = 0.8*200 + 3*(1-0.8)*x**2.
    return dval
tmp5 = func(num,maxNumDays)

plot(tmp,num,marker='o',color='blue')
plot(tmp2,num,marker='o',color='black')
plot(tmp3,num,marker='o',color='red')
plot(tmp4,num,marker='o',color='green')
plot(tmp5,num,marker='o',color='orange')
ylabel('Points Number')
xlabel('Start Times')
show(block=False)






import scipy.integrate as integrate
tmp5L = integrate.quad(dfunc,0,max(num))#This is the total length of the path from (0,0) to (num,maxNumDays)



minNumReps = 0




numRep = 10#number of repetitions in 1 year
assert(numRep <= 365.25/OBdur - 365.25%OBdur, 'numRep too large')
missionPortion = numRep*OBdur/365.25
missionLife = exoplanetObsTime/missionPortion

def isoMissionDuration(mL,mP,mdur):
    #missionLife,missionPortion,mission duration
    #mdur = mL*mP #total amount of time to elapse during the mission
    if mL is None:
        mL = mdur/mP
    elif mP is None:
        mP = mdur/mL
    return mL, mP

# def OBharmonics(num,mP):
#     #num of repetitions to occur within one year
#     #mP missionPortion



tmp = np.asarray(range(30))*12.
tmp1 = np.asarray(range(30))
tmp2 = np.asarray(range(12))+0.5
denom = np.asarray(range(30),)+1.
tmp3 = 365.25/denom

OBdurs = list()
[OBdurs.append(x) for x in tmp.tolist()]
[OBdurs.append(x) for x in tmp1.tolist()]
[OBdurs.append(x) for x in tmp2.tolist()]
[OBdurs.append(x) for x in tmp3.tolist()]