#smax_smin_dsdnuExperimental

#from sympy import * 
import sympy as sp
#Symbol, symbols
#from sympy import sp.sin, sp.cos, asp.sin, asp.cos, tan, atan2, exp
#from sympy import power
#from sympy import log
#from mpmath import *
#DELTEfrom mpmath import log10
import numpy as np
import eqnsEXOSIMS2020

#### Solving s eqn
#Solves in Wolfram alpha but not Here
# a, b, c, d, f, xxx = sp.symbols('a b c d f xxx', real=True, positive=True)
# eqnSLimit = a+b*xxx+c*xxx**2.+d*xxx*sp.sqrt(1.-xxx**2.)
# out = sp.solve(eqnSLimit,xxx)

# print('Solving for v')
# print(0.45-eqnS.subs(W,0.).subs(a,1).subs(e,0.1).subs(w,0.2).subs(inc,0.3))
# out = sp.solve(0.45-eqnS.subs(W,0.).subs(a,1).subs(e,0.1).subs(w,0.2).subs(inc,0.3),v) #Uses all the ram :(
omega, sma, eccen, sep, xxx, inc, nu = sp.symbols('omega, sma, eccen, sep, xxx, inc, nu', real=True)
tmpEqn = sma*(1.*eccen**2.)/(1.+eccen*sp.cos(nu))*(sp.cos(omega+nu)**2. + sp.sin(omega+nu)**2.*sp.cos(inc)**2.)**(0.5) - sep
tmpSep = sma*(1.+0.5*eccen).subs(eccen,0.1).subs(sma,1.)
tmpEqn2 = tmpEqn.subs(sep,tmpSep).subs(eccen,0.1).subs(sma,1.).subs(omega,0.25).subs(inc,0.1)
#out = sp.solve(tmpEqn2,nu)#Takes too long/doesn't finish
print('Done out')

#Calculate smin and smax using differential
difftmpEqn = sp.diff(tmpEqn,nu)
diffdifftmpEqn = sp.diff(difftmpEqn,nu)
#out2 = sp.solve(difftmpEqn.subs(sma,1.).subs(eccen,0.1).subs(omega,0.25).subs(inc,0.1),nu)#takes too long/doesn't finish
print('Done out2')

#trying minimize_scalar to find 0's
from scipy.optimize import minimize_scalar
import time
def diffError(x,a,e,w,i):
    error = np.abs(difftmpEqn.subs(sma,a).subs(eccen,e).subs(omega,w).subs(inc,i).subs(nu,x))
    return float(error)
start = time.time()
tmpa=1.
tmpe=0.1
tmpw=0.25
tmpi=0.1
out3 = minimize_scalar(diffError,bounds=[-1,1],args=(tmpa,tmpe,tmpw,tmpi)) #0.037 seconds. 
stop=time.time()
print(stop-start)
print('Done out3')

#trying minimize
from scipy.optimize import minimize
out4 = minimize(diffError,x0=0.,args=(tmpa,tmpe,tmpw,tmpi))
print('Done out4')

#trying root
from scipy.optimize import root
def diffError2(x,a,e,w,i):
    error = difftmpEqn.subs(sma,a).subs(eccen,e).subs(omega,w).subs(inc,i).subs(nu,x)
    return float(error)
diffdifftmpEqn = sp.diff(difftmpEqn,nu)
def diffdiffError2(x,a,e,w,i):
    error = diffdifftmpEqn.subs(sma,a).subs(eccen,e).subs(omega,w).subs(inc,i).subs(nu,x)
    return [error]
start = time.time()
outs = list()
x0s = np.linspace(start=0,stop=2.*np.pi)
for i in np.arange(len(x0s)):
    #NEED TO FIND ROOT SOLVER WITH BOUNDS BETWEEN 0 AND 2PI
    out5 = root(diffError2,x0=x0s[i],args=(tmpa,tmpe,tmpw,tmpi), jac=diffdiffError2)
    #resolve using one of these algorithms.... newton??? bisect???
    outs.append(out5['x'])
stop=time.time()
print(stop-start)
print(str((stop-start)))
print('total time (hrs): ' + str((stop-start)*10**8./60/60.))
print('Done out5')

#see if I can take out omega from eqn (subs with 0.) with this, I can add omega back in after solving eqn for nu.



#plotting
import matplotlib.pyplot as plt
x = np.linspace(start=0.,stop=2.*np.pi)
plt.close(1)
plt.figure(num=1)
plt.plot(x,[difftmpEqn.subs(sma,1.).subs(eccen,0.1).subs(omega,0.25).subs(inc,0.1).subs(nu,x[i]) for i in np.arange(len(x))])
plt.show(block=False)

plt.close(2)
plt.figure(num=2)
plt.plot(x,[tmpEqn.subs(sma,1.).subs(eccen,0.1).subs(omega,0.25).subs(inc,0.1).subs(sep,0.).subs(nu,x[i]) for i in np.arange(len(x))])
plt.show(block=False)

plt.close(3)
plt.figure(num=3)
incs = np.linspace(start=0.,stop=0.9*np.pi/2.,num=10)
for j in np.arange(len(incs)):
    R = incs[j]/np.max(incs)
    G = 0.
    B = 1.-incs[j]/np.max(incs)
    plt.plot(x,[tmpEqn.subs(sma,1.).subs(eccen,0.3).subs(omega,0.).subs(inc,incs[j]).subs(sep,0.).subs(nu,x[i]) for i in np.arange(len(x))],color=(R,G,B,1.))
plt.xlabel('True Anomaly (rad)')
plt.ylabel('Planet-Star Separation')
plt.title('Red=90 deg inc')
plt.show(block=False)

#plotting derivative vs inclination
plt.close(33)
plt.figure(num=33)
fig, ax = plt.subplots(nrows=4, num=33, sharex=True)
e = 0.3
i_crit = np.arccos(((1.-e**2.)**0.5/(1.+e)))
incs = np.linspace(start=0.,stop=0.9*np.pi/2.,num=10)
incs = np.append(incs,[i_crit])
for j in np.arange(len(incs)):
    R = incs[j]/np.max(incs)
    G = 0.
    B = 1.-incs[j]/np.max(incs)
    ax[0].plot(x,[difftmpEqn.subs(sma,1.).subs(eccen,0.3).subs(omega,0.).subs(inc,incs[j]).subs(sep,0.).subs(nu,x[i]) for i in np.arange(len(x))],color=(R,G,B,1.))
ax[0].plot(x,np.zeros(len(x)),color='black')
ax[0].set_xlabel('True Anomaly (rad)')
ax[0].set_ylabel('ds/dnu, w=0')
#ax[0].title('Red=90 deg inc')
plt.show(block=False)

plt.close(4)
plt.figure(num=4)
#incs = np.linspace(start=0.,stop=0.9*np.pi/2.,num=10)
for j in np.arange(len(incs)):
    R = incs[j]/np.max(incs)
    G = 0.
    B = 1.-incs[j]/np.max(incs)
    plt.plot(x,[tmpEqn.subs(sma,1.).subs(eccen,0.3).subs(omega,np.pi/6.).subs(inc,incs[j]).subs(sep,0.).subs(nu,x[i]) for i in np.arange(len(x))],color=(R,G,B,1.))
plt.xlabel('True Anomaly (rad)')
plt.ylabel('Planet-Star Separation')
plt.title('Red=90 deg inc')
plt.show(block=False)

#plt.close(44)
#plt.figure(num=44)
#incs = np.linspace(start=0.,stop=0.9*np.pi/2.,num=10)
for j in np.arange(len(incs)):
    R = incs[j]/np.max(incs)
    G = 0.
    B = 1.-incs[j]/np.max(incs)
    ax[1].plot(x,[difftmpEqn.subs(sma,1.).subs(eccen,0.3).subs(omega,np.pi/6.).subs(inc,incs[j]).subs(sep,0.).subs(nu,x[i]) for i in np.arange(len(x))],color=(R,G,B,1.))
ax[1].plot(x,np.zeros(len(x)),color='black')
ax[1].set_xlabel('True Anomaly (rad)')
ax[1].set_ylabel('ds/dnu, w=pi/6')
#plt.title('Red=90 deg inc')
plt.show(block=False)

#It appears there is 1 true anomaly of complete independence of i (when theta=v+w=0).


plt.close(5)
plt.figure(num=5)
#incs = np.linspace(start=0.,stop=0.9*np.pi/2.,num=10)
for j in np.arange(len(incs)):
    R = incs[j]/np.max(incs)
    G = 0.
    B = 1.-incs[j]/np.max(incs)
    plt.plot(x,[tmpEqn.subs(sma,1.).subs(eccen,0.3).subs(omega,np.pi/3.).subs(inc,incs[j]).subs(sep,0.).subs(nu,x[i]) for i in np.arange(len(x))],color=(R,G,B,1.))
plt.xlabel('True Anomaly (rad)')
plt.ylabel('Planet-Star Separation')
plt.title('Red=90 deg inc')
plt.show(block=False)

#plt.close(55)
#plt.figure(num=55)
#incs = np.linspace(start=0.,stop=0.9*np.pi/2.,num=10)
for j in np.arange(len(incs)):
    R = incs[j]/np.max(incs)
    G = 0.
    B = 1.-incs[j]/np.max(incs)
    ax[2].plot(x,[difftmpEqn.subs(sma,1.).subs(eccen,0.3).subs(omega,np.pi/3.).subs(inc,incs[j]).subs(sep,0.).subs(nu,x[i]) for i in np.arange(len(x))],color=(R,G,B,1.))
ax[2].plot(x,np.zeros(len(x)),color='black')
ax[2].set_xlabel('True Anomaly (rad)')
ax[2].set_ylabel('ds/dnu, w=pi/3')
#plt.title('Red=90 deg inc')
plt.show(block=False)

#plt.close(66)
#plt.figure(num=66)
#incs = np.linspace(start=0.,stop=0.9*np.pi/2.,num=10)
for j in np.arange(len(incs)):
    R = incs[j]/np.max(incs)
    G = 0.
    B = 1.-incs[j]/np.max(incs)
    ax[3].plot(x,[difftmpEqn.subs(sma,1.).subs(eccen,0.3).subs(omega,np.pi/2.).subs(inc,incs[j]).subs(sep,0.).subs(nu,x[i]) for i in np.arange(len(x))],color=(R,G,B,1.))
ax[3].plot(x,np.zeros(len(x)),color='black')
ax[3].set_xlabel('True Anomaly (rad)')
ax[3].set_ylabel('ds/dnu, w=pi/2')
#plt.title('Red=90 deg inc')
fig.subplots_adjust(hspace=0.)#, wspace=)
plt.show(block=False)

#The questions I could ask. Left of theta=v+w coincidence point, what i causes excursion before v=pi?


from scipy.optimize import fsolve



def error_s_nu(x,a,e,w,i,s):
    error = (a*(1.-e**2.))/(1.+e*x)*( np.cos(w)**2. * x**2
    -2*np.cos(w)*x*np.sin(w)*(1. - x**2.)**0.5
    +np.sin(w)**2.
    -np.sin(w)**2.* x**2.
    + np.sin(w)**2.* x**2.*np.cos(i)**2.
    + 2.*np.cos(w)*(1. - x**2.)*np.sin(w)*x*np.cos(i)**2.
    + np.cos(w)**2. *np.cos(i)**2.
    - np.cos(w)**2.* x**2. *np.cos(i)**2.)**0.5 - s
    return error

tmpArgs = (1.,0.1,0.25,0.1,0.4)#a,e,w,i,s
out = fsolve(error_s_nu,x0=0.1,args=tmpArgs)


def tmpSep(x,a,e,w,i,s):
    s = (a*(1.-e**2.))/(1.+e*x)*( np.cos(w)**2. * x**2
    -2*np.cos(w)*x*np.sin(w)*(1. - x**2.)**0.5
    +np.sin(w)**2.
    -np.sin(w)**2.* x**2.
    + np.sin(w)**2.* x**2.*np.cos(i)**2.
    + 2.*np.cos(w)*(1. - x**2.)*np.sin(w)*x*np.cos(i)**2.
    + np.cos(w)**2. *np.cos(i)**2.
    - np.cos(w)**2.* x**2. *np.cos(i)**2.)**0.5
    return s

x = np.linspace(start=0.,stop=1.)
sout = tmpSep(x,1.,0.1,0.25,0.1,0.4)

plt.figure(num=99)
plt.plot(x,sout)
plt.show(block=False)



#### ds/dnu
omega, xxx, inc, nu = sp.symbols('omega, xxx, inc, nu', real=True)
sma, eccen, sep = sp.symbols('sma, eccen, sep', real=True, positive=True)
#The LHS here is 0
dsbydnuzeros = ((eccen**2. *sma* sp.cos(inc)**2. - 1.)*sp.sin(nu+omega)*sp.cos(nu+omega))/ ((eccen*sp.cos(nu) + 1.)*sp.sqrt(sp.sin(nu+omega)**2.*sp.cos(inc)**2. + sp.cos(nu+omega)**2.)  )\
    + eccen**3.* sma* sp.sin(nu)* sp.sqrt(sp.sin(nu+omega)**2.*sp.cos(inc)**2. + sp.cos(nu+omega)**2.)/(eccen*sp.cos(nu)**2. + 1)
#Cross multiply to remove denominators    
dsbydnuzeros_1 = (eccen*sp.cos(nu)**2. + 1)*(eccen**2.*sma*sp.cos(inc)**2. - 1.)*sp.sin(nu+omega)*sp.cos(nu+omega)\
    + eccen**3.*sma*sp.sin(nu)*(sp.sin(nu+omega)**2.*sp.cos(inc)**2. + sp.cos(nu+omega)**2.)*(eccen*sp.cos(nu) + 1.)
#Expand numerator and denominator
dsbydnuzeros_2 = eccen*sp.cos(nu)**2.*eccen**2.*sma*sp.cos(inc)**2.*sp.sin(nu+omega)*sp.cos(nu+omega)\
- eccen*sp.cos(nu)**2.*sp.sin(nu+omega)*sp.cos(nu+omega)\
+ eccen**2.*sma*sp.cos(inc)**2.*sp.sin(nu+omega)*sp.cos(nu+omega)\
- 1.*sp.sin(nu+omega)*sp.cos(nu+omega)\
+ sp.sin(nu+omega)**2.*sp.cos(inc)**2.*eccen*sp.cos(nu)*eccen**3.*sma*sp.sin(nu)\
+ sp.sin(nu+omega)**2.*sp.cos(inc)**2.*eccen**3.*sma*sp.sin(nu)\
+ eccen*sp.cos(nu)*sp.cos(nu+omega)**2.*eccen**3.*sma*sp.sin(nu)\
+ sp.cos(nu+omega)**2.*eccen**3.*sma*sp.sin(nu)

#Replace sin(nu+omega)**2.
sin2nuomega = (1.-sp.cos(nu+omega)**2.)
dsbydnuzeros_3 = dsbydnuzeros_2.subs(sp.sin(nu+omega)**2.,1.-sp.cos(nu+omega)**2.)

#Angle Addition Expanders
sinnupomega = sp.sin(nu)*sp.cos(omega) + sp.cos(nu)*sp.sin(omega)
cosnupomega = sp.cos(nu)*sp.cos(omega) - sp.sin(nu)*sp.sin(omega)
sinnumomega = sp.sin(nu)*sp.cos(omega) - sp.cos(nu)*sp.sin(omega)
cosnumomega = sp.cos(nu)*sp.cos(omega) + sp.sin(nu)*sp.sin(omega)

#Squared Angle Addition Expanders
sinnupomega2 = sp.sin(nu)**2.*sp.cos(omega)**2. + 2.*sp.cos(nu)*sp.sin(omega) + sp.cos(nu)**2.*sp.sin(omega)**2.
cosnupomega2 = sp.cos(nu)**2.*sp.cos(omega)**2. - 2.*sp.cos(nu)*sp.cos(omega)*sp.sin(nu)*sp.sin(omega) + sp.sin(nu)**2.*sp.sin(omega)**2.
sinnumomega2 = sp.sin(nu)**2.*sp.cos(omega)**2. - 2.*sp.sin(nu)*sp.cos(omega)*sp.cos(nu)*sp.sin(omega) + sp.cos(nu)**2.*sp.sin(omega)**2.
cosnumomega2 = sp.cos(nu)**2.*sp.cos(omega)**2. + 2.*sp.cos(nu)*sp.cos(omega)*sp.sin(nu)*sp.sin(omega) + sp.sin(nu)**2.*sp.sin(omega)**2.

#Do (nu+omega) replacements
dsbydnuzeros_4 = dsbydnuzeros_3.subs(sp.sin(nu+omega)**2.,sinnupomega2).subs(sp.cos(nu+omega)**2.,cosnupomega2).subs(sp.sin(nu+omega),sinnupomega).subs(sp.cos(nu+omega),cosnupomega)

#Replace sin(nu) with sp.sqrt(1.-sp.cos(nu)**2.)
sinnu = sp.sqrt(1.-sp.cos(nu)**2.)
dsbydnuzeros_5 = dsbydnuzeros_4.subs(sp.sin(nu),sinnu)


