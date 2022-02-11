import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

Amin = 0.24
Amax = 3.4
A = 3.4
C_d = 2.2
rho = 5*10**-13
m = 6
a = (6378+400)*1000
mu = 398600*10**9
T0 = 2.*np.pi*np.sqrt(a**3./mu)

#dndt = 1.5*A*C_d*a**(-2.0)*mu*rho/m


#The time we have until the two spacecraft need to be at the same separation
timeToGetToSeparations = 3*30*24*60*60 #3 months in seconds
numOrbits = int(np.floor(timeToGetToSeparations/T0))

th0 = 0.
th1 = 20./90*2.*np.pi #initial 20 minutes of separation between lead and following spacecraft
thd0 = np.sqrt(mu/a**3)
thd1 = np.sqrt(mu/a**3)

desiredSeparations = 2./90*2.*np.pi


def posVelDiff(x):
    #Extract A(t)
    A0 = x[:numOrbits]
    A1 = x[numOrbits+1:] #double check these lengths are correct

    #Need to run through orbits iteratively
    a0 = np.zeros(numOrbits)
    a0[0] = a
    a1 = np.zeros(numOrbits)
    a1[0] = a
    T0 = 2.*np.pi*np.sqrt(a0**3./mu)
    T1 = 2.*np.pi*np.sqrt(a1**3./mu)
    th0 = np.zeros(numOrbits)
    th1 = np.zeros(numOrbits)
    n0 = np.zeros(numOrbits)
    n0[0] = np.sqrt(mu/a0[0]**3)
    n1 = np.zeros(numOrbits)
    n1[0] = np.sqrt(mu/a1[0]**3)
    dndt0 = np.zeros(numOrbits)
    dndt1 = np.zeros(numOrbits)
    for i in np.arange(numOrbits-1):
        dadt0 = -1.0*A0[i]*C_d*np.sqrt(a0[i])*np.sqrt(mu)*rho/m
        a0[i+1] = a0[i] + dadt0
        dadt1 = -1.0*A1[i]*C_d*np.sqrt(a1[i])*np.sqrt(mu)*rho/m
        a1[i+1] = a1[i] + dadt1
        T0[i+1] = 2.*np.pi*np.sqrt(a0[i+1]**3./mu)
        T1[i+1] = 2.*np.pi*np.sqrt(a1[i+1]**3./mu)
        dndt0[i+1] = 1.5*A0[i]*C_d*a0[i+1]**(-2.0)*mu*rho/m
        n0[i+1] = n0[i] + dndt0[i+1]*T0[i+1]
        dndt1[i+1] = 1.5*A1[i]*C_d*a1[i+1]**(-2.0)*mu*rho/m
        n1[i+1] = n1[i] + dndt1[i+1]*T1[i+1]
        th0[i+1] = th0[i] + n0[i+1]*T0[i+1]
        th1[i+1] = th1[i] + n1[i+1]*T1[i+1]
    print("th0: " + str(th0[-1]) + " th1: " + str(th1[-1]) + " n0: " + str(n0[-1]) + " n1: " + str(n1[-1]))
    #Evaluation function
    erf = np.abs(th0[-1] - th1[-1]) + np.abs(n0[-1] - n1[-1])
    #print(erf)
    return erf

#initial starting array of times
x0 = np.concatenate(((Amin)*np.ones(numOrbits),Amin*np.ones(numOrbits)),axis=0)
x0[1150:1200] = Amax*np.ones(50)
x0[2750:] = Amax*np.ones(50)
bounds = [(Amin,Amax) for i in np.arange(len(x0))]

out = minimize(posVelDiff, x0, bounds=bounds)#constraints=[]) #constraints=const)
A_SV0 = out['x'][:1400]
A_SV1 = out['x'][1400:]


import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(np.arange(len(A_SV0)),A_SV0,color='blue')
plt.plot(np.arange(len(A_SV1)),A_SV1,color='black')
plt.show(block=False)



#### NEED TO REFORMULATE FREE VARIABLES INTO: FIRST MANEUVER START, FIRST MANEUVER DURATION, TIME BETWEEN MANEUVERS, SECOND MANEUVER DURATION


