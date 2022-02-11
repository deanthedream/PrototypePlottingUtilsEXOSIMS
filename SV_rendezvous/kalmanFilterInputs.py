import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import matplotlib.pyplot as plt

Re = 6371.
alt = 600.
a = Re+alt
e = 0.003 #as large as 0.01
i = np.pi/2.*80./90. #idk using 80 deg but it wont be that high
w = 0. #Argument of periapsis
W = 0. #longitude of the ascending node
nu = 0. #true anomaly
r_Earth_sun = np.asarray([1.,0.,0.])#Assumed Sun to Earth vector

def XYZ_from_KOE(a,e,i,w,W,nu):
    r = a*(1-e**2.)/(1.+e*np.cos(nu))
    return r*np.asarray([np.cos(W)*cos(w+nu) - np.sin(W)*np.sin(w+nu)*np.cos(I),\
        np.sin(W)*np.cos(w+nu) + np.cos(W)*np.sin(w+nu)*np.cos(I),\
        np.sin(w+nu)*np.sin(I)])

#Using simple circle as the Earth, we can calculate when eclipses occur
#solve where Re^2 = Y^2 + Z^2



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
thd0 = np.sqrt(mu/a**3.)
thd1 = np.sqrt(mu/a**3.)

desiredSeparations = 2./90*2.*np.pi

def simpleInputs_to_At(firstStart, firstDur, timeBetween, secondDur, numOrbits):
    """ Converts simple inputs into spacecraft SA vs time inputs
    """
    #Error checking so we cannot start the maneuver too close to the end
    # print(firstStart)
    # print(firstDur)
    # print(timeBetween)
    # print(secondDur)
    # print(numOrbits)
    if firstStart + firstDur + timeBetween + secondDur > numOrbits:
        firstStart = numOrbits - firstDur - timeBetween - secondDur
        print("constraint violated")
    A0 = Amin*np.ones(numOrbits)
    A0[int(np.floor(firstStart)):int(np.floor(firstStart+firstDur))] = Amax*np.ones(int(np.floor(firstStart+firstDur))-int(np.floor(firstStart)))
    
    A1 = Amin*np.ones(numOrbits)
    secondStart = int(np.floor(firstStart + firstDur + timeBetween))
    secondEnd = int(np.floor(firstStart + firstDur + timeBetween + secondDur))
    A1[secondStart:secondEnd] = Amax*np.ones(secondEnd-secondStart)
    # plt.figure()
    # plt.plot(np.arange(len(A0)),A0,color='blue')
    # plt.plot(np.arange(len(A1)),A1,color='black')
    # plt.show(block=False)
    return A0, A1

def posVelDiff(x,th0_0,th1_0,C_d,rho,m,mu):
    # print(x)
    firstStart = x[0]
    firstDur = x[1]
    timeBetween = x[2]
    secondDur = x[3]
    A0, A1 = simpleInputs_to_At(firstStart, firstDur, timeBetween, secondDur, numOrbits)
    #Extract A(t)
    # A0 = x[:numOrbits]
    # A1 = x[numOrbits+1:] #double check these lengths are correct

    #Need to run through orbits iteratively
    a0 = np.zeros(numOrbits)
    a0[0] = a
    a1 = np.zeros(numOrbits)
    a1[0] = a
    T0 = 2.*np.pi*np.sqrt(a0**3./mu)
    T1 = 2.*np.pi*np.sqrt(a1**3./mu)
    th0 = np.zeros(numOrbits)
    th0[0] = th0_0
    th1 = np.zeros(numOrbits)
    th1[0] = th1_0
    n0 = np.zeros(numOrbits)
    n0[0] = np.sqrt(mu/a0[0]**3.)
    n1 = np.zeros(numOrbits)
    n1[0] = np.sqrt(mu/a1[0]**3.)
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
    #Evaluation function
    erf = np.abs(th0[-1] - th1[-1]) + 1000*np.abs(n0[-1] - n1[-1]) + 100000*n0[-1]
    print("th0: " + str(th0[-1]) + " th1: " + str(th1[-1]) + " n0: " + str(n0[-1]) + " n1: " + str(n1[-1]) + " erf: " + str(erf))
    #todo update to have constant separation
    #print(erf)
    return erf

#initial starting array of times
x0 = [int(np.floor(numOrbits/10)),int(np.floor(numOrbits/10)),int(np.floor(numOrbits/10)),int(np.floor(numOrbits/10))]
bounds = [(0,numOrbits),(0,numOrbits/3),(0,numOrbits),(0,numOrbits/3)]
#bounds = [(Amin,Amax) for i in np.arange(len(x0))]

# erf0 = posVelDiff(x0,th0,th1,C_d,rho,m,mu)
# x0[1] = 100
# erf1 = posVelDiff(x0,th0,th1,C_d,rho,m,mu)
# print(saltyburrito)

out = minimize(posVelDiff, x0, bounds=bounds, method='SLSQP', options={'eps':20}, args=(th0, th1,C_d,rho,m,mu))#constraints=[]) #constraints=const)
A_SV0, A_SV1 = simpleInputs_to_At(out['x'][0], out['x'][1], out['x'][2], out['x'][3], numOrbits)

plt.figure(1)
plt.plot(np.arange(len(A_SV0)),A_SV0,color='blue')
plt.plot(np.arange(len(A_SV1)),A_SV1,color='black')
plt.show(block=False)



#### NEED TO REFORMULATE FREE VARIABLES INTO: FIRST MANEUVER START, FIRST MANEUVER DURATION, TIME BETWEEN MANEUVERS, SECOND MANEUVER DURATION


