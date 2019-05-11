"""
This script demonstrates the longevity probability of a population using metapopulation dynamics
We specifically compare the probabilistic postponement of population extinction in a single patch and multiple patch system
(a patch can be considered analagous to a multipatch system)
"""
import numpy as np


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