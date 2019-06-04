"""
This script demonstrates the longevity probability of a population using metapopulation dynamics
We specifically compare the probabilistic postponement of population extinction in a single patch and multiple patch system
(a patch can be considered analagous to a multipatch system)
Written By: Dean Keithly
"""
import numpy as np
import numpy.random as random


pe = 0.15 #this is the probability of a patch going extinct (becoming unoccupied)
pr = 0.3 #this is the probability of an occupied patch inhabiting an unoccupied patch

#calculate state change probabilities
def stateChangeProb(pe,pr):
    """
    Args:
        pe (float) - probability of patch going extinct
        pr (float) - probability of patch inhabiting unoccupied patch
    """
    #### Starting at state 11 (both occupied)
    P11_P11 = (1.-pe)**2. # probability both planets remain occupied
    P11_P10 = 2.*(1.-pe)*pe #probability one planet goes to unoccupied state
    P11_P00 = pe**2. #probability both planets got to unoccupied
    print(P11_P11 + P11_P10 + P11_P00)

    #### Starting at state 10 (only 1 occupied)
    P10_P11 = pr*(1.-pe) #probability the one occupied planet re-inhabits the other occupied planet and the occupied planet does not go extinct
    P10_P10 = (1.-pe)*(1.-pr) + pe*pr 
        # prob the occupied planet stays occupied and the other is not colonized + # prob the occupied planets goes extinct and the other is colonized
    P10_P00 = pe*(1.-pr) #probability the occupied planet goes exinct and the other planet was not colonized
    print(P10_P11 + P10_P10 + P10_P00)
    return P11_P11, P11_P10, P11_P00, P10_P11, P10_P10, P10_P00
P11_P11, P11_P10, P11_P00, P10_P11, P10_P10, P10_P00 = stateChangeProb(pe,pr)

#### Probability we are in state 11 or 10 at epoch N
N=10.

#### Assuming single patch with no recolonization
print('Single Planet Life Probability: ' + str((1.-pe)**N))

#Assumes starting state of 11 and assuming re-colonization
p11 = 1.
p10 = 0.
p00 = 0.
for i in np.arange(N):
    tp11 = p11#save initial values
    tp10 = p10
    tp00 = p00
    p11 = tp11*P11_P11 + tp10*P10_P11#the probability of ending in state P11
    p10 = tp11*P11_P10 + tp10*P10_P10#the probability of staying in state P10
    p00 = tp00 + tp11*P11_P00 + tp10*P10_P00#the probability of going to state P00 from either state plus previous p00

print('p11: ' + str(p11))
print('p10: ' + str(p10))
print('p00: ' + str(p00))

#Assumes starting state of 11 and assuming NO re-colonization
p11 = 1.
p10 = 0.
p00 = 0.
pr=0.
P11_P11, P11_P10, P11_P00, P10_P11, P10_P10, P10_P00 = stateChangeProb(pe,pr)
for i in np.arange(N):
    tp11 = p11#save initial values
    tp10 = p10
    tp00 = p00
    p11 = tp11*P11_P11 + tp10*P10_P11#the probability of ending in state P11
    p10 = tp11*P11_P10 + tp10*P10_P10#the probability of staying in state P10
    p00 = tp00 + tp11*P11_P00 + tp10*P10_P00#the probability of going to state P00 from either state plus previous p00

print('p11: ' + str(p11))
print('p10: ' + str(p10))
print('p00: ' + str(p00))






#######################################
#Extinction Modeling with Poisson Distributed Random Events
def colonizeLam():
    # rate at which colonization events occur
    return 1./10**4.
def extinctionLam():
    # rate at which extinction level events occur
    return 5./570.*10**9.

def extinctionEventDict(toIndex, time_c):
    #DELETE TEx = random.poisson(lam=extinctionLam()) # the time from now for the event to occur
    #Inverse Transform Sampling of a poisson distributed random event
    TEx = -np.log(1.-np.random.uniform(low=0.,high=1.))/extinctionLam() # the time from now for the event to occur
    tmpEvent1 = 'extinct' + '_' + str(toIndex) + '_' + str(toIndex) # event name
    eventDict = {'eventName':tmpEvent1}
    eventDict['toIndex'] = toIndex
    eventDict['fromIndex'] = [] # should be empty
    eventDict['time'] = TEx + time_c # time in the future when event would occur
    return eventDict

#DONE
def extinctEvent(time_c, index, planetStates, eventStack, fromIndex, planetIndices):
    """
    1. Updates planet State to EXTINCTION
    2. Adds all Colonization events with destination to this planet
    3. Removes any Colonization events eminating from this planet
    *Future update, split colonization into launch date and arrival date
        to account for extinction events occuring after launch
    Args:
        time_c (float) - current time
        index (ind) - planet index of extinction event
        planetStates (list) - state of the planet
        eventStack (list) - list of dicts of events
        fromIndex (ind) - place holder variable
        planetIndices (list) - list of planet indices
    Returns:
        eventStack (list) - updated list of dicts of events
        planetStates (list) - updated planet states
    """
    #1. Updates planet State to EXTINCTION
    planetStates[index] = 0 # planet is now extinct

    #2. Calculate next colonize event to index from ALIVE planets
    for pInd in planetIndices:
        if planetStates[pInd] == 1:
            #Planet is inhabited! generate colonization event time
            eventStack.append(colonizationEventDict(planetStates[pInd], index, time_c)) # Add event to event stack

    #3. Remove any colonize events eminating from this planet
    tmpEventStack = eventStack#.copy()
    for i in np.arange(len(tmpEventStack)):
        if tmpEventStack[i]['eventName'].split('_')[0] == 'colonize':
            if tmpEventStack[i]['fromIndex'] == index:
                #DELETE eventNames = [eventStack[i]['eventName'] for i in np.arange(len(eventStack))]
                #DELETE tInd = np.where(np.asarray(eventNames) == tmpEventStack[i]['eventName'])[0][0] # find where in actual stack event occurs
                #DELETE del eventStack[tInd] #delete the event occurrence
                eventStack = deleteEventDict(eventStack, tmpEventStack[i])#DELETE['eventName'])

    return eventStack, planetStates

def colonizationEventDict(fromIndex, toIndex, time_c):
    #DELETE TEx = random.poisson(lam=colonizeLam()) # the time from now for the event to occur
    #Inverse Transform Sampling of a poisson distributed random event
    TEx = -np.log(1.-np.random.uniform(low=0.,high=1.))/extinctionLam() # the time from now for the event to occur
    tmpEvent2 = 'colonize' + '_' + str(fromIndex) + '_' + str(toIndex) # event name
    eventDict = {'eventName':tmpEvent2}
    eventDict['fromIndex'] = fromIndex
    eventDict['toIndex'] = toIndex
    eventDict['time'] = TEx + time_c # time in the future when event would occur
    return eventDict

#DONE
def colonizeEvent(time_c, toIndex, planetStates, eventStack, fromIndex, planetIndices):
    """
    1. Updates planet state to ALIVE
    2. Add next Extunction event for this planet
    Args:
        time_c (float) - current time
        toIndex (int) - planet index being colonized
        planetStates (list) - state of the planet
        eventStack (list) - list of dicts of events
        fromIndex (int) - indicates planet being colonized from
        planetIndices (list) - indices of planets
    Returns:
        eventStack (list) - updated list of dicts of events
        planetStates (list) - updated planet states
    """
    #1. Updates planet state to ALIVE
    planetStates[toIndex] = 1 # planet is now Alive

    #2. Calculate next extinction event for colonized planet
    eventStack.append(extinctionEventDict(toIndex, time_c)) # Add event to event stack
    # Calculate colonize event time for any extinct planets
    
    #3. Add colonization events eminating from this planet
    for i in np.arange(len(planetStates)):
        if planetStates[i] == 0:
            eventStack.append(colonizationEventDict(toIndex, planetIndices[i], time_c))
            # toIndex was just colonized and is the "fromIndex" for the colonization events added

    return eventStack, planetStates

#DONE
def eventSwitch(time_c, eventDict, eventStack, planetStates, planetIndices):
    """ *Future update: figure out a way to avoid recreating event switches... shouldn't be hard
    Args:
        time_c (time) - current time
        eventDict (dict) - dict of event info includes:
            event index, event name, time of event 
        eventStack (list) - list of dicts containing eventDict
        planetStates (list) - list of planet states
        planetIndices (list) - list of planet indicies

    Returns:
        eventStack (list) - list of dicts of events with enacted event removed from it
        planetStates (list) - 
    """
    #1. Enumerate possible events for all planets
    switcher = dict() # set of all possible events for each planet
    for index in planetIndices: #iterate over planets
        tmpEvent1 = 'extinct' + '_' + str(index) + '_' + str(index)
        switcher[tmpEvent1] = extinctEvent #(time_c, index, planetStates, eventStack, index2, planetIndices) #extinction event
        for index2 in planetIndices: #from originating star (index2) to this star (index)
            tmpEvent2 = 'colonize' + '_' + str(index2) + '_' + str(index)
            switcher[tmpEvent2] = colonizeEvent #(time_c, index, planetStates, eventStack, index2, planetIndices) #colonize event
    return switcher

    #2. Get Event Name
    event = eventDict['eventName'] # this is the event Name indicator
    #eventName takes for eventType_originatingPlanet_destinationPlanet

    #3. Update Event Stack and Planet States
    eventStack, planetStates = switcher.get(event, 'Invalid Event') # update planet states
    #DELETE eventNames = [eventStack[i]['eventName'] for i in np.arange(len(eventStack))]
    #DELETE tInd = np.where(np.asarray(eventNames) == eventDict['eventName'])[0][0] # find where in actual stack event occurs
    #DELETE del eventStack[tInd]    # remove event enacted from eventDict
    eventStack = deleteEventDict(eventStack, eventDict)

    return eventStack, planetStates

def deleteEventDict(eventStack, eventDict):
    eventNames = [eventStack[i]['eventName'] for i in np.arange(len(eventStack))]
    tInd = np.where(np.asarray(eventNames) == eventDict['eventName'])[0][0] # find where in actual stack event occurs
    del eventStack[tInd]    # remove event enacted from eventDict
    return eventStack

#DONE
def getNextEvent(time_c, eventStack):
    """
    Args:
        time_c (float) - the current time
        eventStack (list) - a list of dicts containing all events 
    Returns:
        nextEventTime (float) - the time of the next event
        eventDict (dict) - the dict of the next event
    """
    #Generate list of all event times
    eventTimes = [eventStack[i]['time'] for i in np.arange(len(eventStack))]
    ind = np.argmin(eventTimes)
    return eventTimes[ind], eventStack[ind]

#### Initialize Simulation
time_c = 0. # start at generalized time = 0.
planetIndices = [0,1]
planetStates = [1,0]
eventStack = [extinctionEventDict(0, time_c), colonizationEventDict(0, 1, time_c)]
eventRecord = list()
eventIndex = 0

# Run Simulation
while not np.all(planetStates == 0) and time_c < 10**6.:
    #1 Advance time_c to next event
    nextEventTime, eventDict = getNextEvent(time_c, eventStack)
    time_c = nextEventTime # update current time to next event time

    #2. Append eventDict to eventRecord
    eventRecord.append(eventDict)

    #3. Get Event Cases
    switcher = eventSwitch(time_c, eventDict, eventStack, planetStates, planetIndices)

    #4. Run eventDict
    eventStack, planetStates = switcher[eventDict['eventName']](time_c, eventDict['toIndex'], planetStates, eventStack, eventDict['fromIndex'], planetIndices)

    # #3 Do event Cases
    # eventStack, planetStates = eventSwitch(time_c, eventDict, eventStack, planetStates, planetIndices)

    eventIndex = eventIndex + 1
    print('ind: ' + str(eventIndex) + ' time: ' + str(np.round(time_c)))
 