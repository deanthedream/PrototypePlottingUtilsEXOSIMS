import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#3 Body Problem
m1 = 1.98847*10**30 #kg mass of sun
m2 = 5.972*10**24 #kg mass of earth
G = 6.67408*10**-11 #m3 kg-1 s-2

AU = 1.496*10**8. #km
mu2 = AU*m2/(m1+m2)
mu1 = AU*m1/(m1+m2)

####### Initial Condutions
#https://engineering.purdue.edu/people/kathleen.howell.1/Publications/Masters/2006_Grebow.pdf
x0 = 0.8234
y0d = 0.1263
T = 2.7430
v = 1180.5771

import numpy as np

#### DAE Attempt
# R1 = np.asarray([0.,0.,0.])
# R2 = np.asarray([1.,0.,0.])
# ls = np.linalg.norm(R1)+np.linalg.norm(R2)
# ms = m1+m2
# ts = np.sqrt(ls**3/(G*ms))
# mu = m2/ms

# def rdd(t,x,y,z,xd,yd,zd,r13,r23,mu):
#     #turn stuff into KWARGS
#     ydd = -2*xd + y + (1.-mu)*y/np.linalg.norm(r13)**3. - mu*y/np.linalg.norm(r23)**3.
#     xdd = x + 2*ydd - (1.-mu)*(x+mu)/np.linalg.norm(r13)**3. - mu*(x-(1.-mu))/np.linalg.norm(r23)**3.
#     zdd = -(1.-mu)*z/np.linalg.norm(r13)**3.-mu*z/np.linalg.norm(r23)**3.
#     return (xdd, ydd, zdd)

# def f(rrr,t):
#     #there is probably an issue with xdd,ydd,zdd being from the divided by ls numbers...
#     x = rrr[0]
#     y = rrr[1]
#     z = rrr[2]
#     xd = rrr[3]
#     yd = rrr[4]
#     zd = rrr[5]
#     R1 = np.asarray([0.,0.,0.])
#     R2 = np.asarray([1.,0.,0.])
#     ls = np.linalg.norm(R1)+np.linalg.norm(R2)
#     R3 = np.asarray([x,y,z])
#     r13 = (R3-R1)/ls
#     r23 = (R3-R2)/ls
#     m1 = 1.98847*10**30 #kg mass of sun
#     m2 = 5.972*10**24 #kg mass of earth
#     G = 6.67408*10**-11 #m3 kg-1 s-2
#     ms = m1+m2
#     mu = m2/ms

#     (xdd, ydd, zdd) = rdd(t,x,y,z,xd,yd,zd,r13,r23,mu)
#     return (xd, yd, zd, xdd, ydd, zdd)
# #asdf x               y          z            xd          yd          zd           period?   
# #2501 355594.29313093 0.00000001 924.49213028 0.48315827 -0.85609027 -0.02635009 3.53584262
# x0 = 355594.29313093
# y0 = 0.00000001
# z0 = 924.49213028
# x0d = 0.48315827
# y0d = -0.85609027
# z0d = -0.02635009
# # x0 = 0.8234
# # y0 = 0.
# # z0 = 0.
# # x0d = 0.
# # y0d = 0.1263
# # z0d = 0.
# # t0 = 0.
# b0 = np.asarray([x0,y0,z0,x0d,y0d,z0d])/(1.49610*10**8)
# import scipy
# import scipy.integrate
# t = np.linspace(0,30,num=100)
# out = scipy.integrate.odeint(f,b0,t)
# # rr = scipy.integrate.ode(f, jac=None)
# # rr.set_initial_value(b0, t0)#.set_f_params(2.0).set_jac_params(2.0)

#### Lagrangian Attempt

def f(rrr,t):
    m1 = 1.98847*10**30 #kg mass of sun
    m2 = 5.972*10**24 #kg mass of earth
    G = 6.67408*10**-11 #m3 kg-1 s-2
    AU = 1.496*10**8. #km
    mu2 = m2/(m1+m2)
    mu1 = m1/(m1+m2)

    x = rrr[0]
    y = rrr[1]
    z = rrr[2]
    xd = rrr[3]
    yd = rrr[4]
    zd = rrr[5]

    #Fronm Lagrange Method in chapter 2 koon, lo and written in my Notes
    C1 = ((x+mu2)**2. + y**2. + z**2.)**(-3./2.)
    C2 = ((x-mu1)**2. + y**2. + z**2.)**(-3./2.)
    xdd = yd + xd - y -3.*mu1*(x+mu2)*C1 + 3*mu2*(x-mu1)*C2
    ydd = -y -3.*mu1*y*C1 + 3.*mu2*z*C2
    zdd = -3.*mu1*z*C1 + 3.*mu2*z*C2


    #DELETE (xdd, ydd, zdd) = rdd(t,x,y,z,xd,yd,zd,r13,r23,mu)
    return (xd, yd, zd, xdd, ydd, zdd)

#asdf x               y          z            xd          yd          zd           period?   
#2501 355594.29313093 0.00000001 924.49213028 0.48315827 -0.85609027 -0.02635009 3.53584262
AU = 1.496*10**(8.) #km
# x0 = 355594.29313093
# y0 = 0.00000001
# z0 = 924.49213028
# x0d = 0.48315827
# y0d = -0.85609027
# z0d = -0.02635009
# x0 = 0.8234
# y0 = 0.
# z0 = 0.
# x0d = 0.
# y0d = 0.1263
# z0d = 0.
# t0 = 0.
x0 = 1.1*AU
y0 = 0.
z0 = 0.
x0d = 0.
y0d = 0.
z0d = 0.

b0 = np.asarray([x0,y0,z0,x0d,y0d,z0d])/AU
import scipy
import scipy.integrate
t = np.linspace(0,1,num=100)
out = scipy.integrate.odeint(f,b0,t)


# t1 = 10
# dt = 1
# while rr.successful() and rr.t < t1:
#     rr.integrate(rr.t+dt)
#from pylab import *
plt.close('all')
#figure()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(out[:,0],out[:,1],out[:,2])
plt.show(block=False)