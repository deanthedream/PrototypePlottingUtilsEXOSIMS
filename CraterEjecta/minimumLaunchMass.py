

from scipy.optimize import fsolve
import numpy as np

def funEM():
    return (6.67*np.exp(2)-3.8)*np.exp(14.66)

def funEL1():
    return (6.67*np.exp(2*0.841**2.)-3.8)*np.exp(12.7)

def funL1M(m0):
    return (0.159)**2.-np.log(np.sqrt((m0/np.exp(12.7)+3.8)/6.67)) * np.log(np.sqrt((m0/np.exp(14.66)+3.8)/6.67))

#Where we assume the antenna at L1 for L1-Moon is the same size as Earth-L1
GEL1 = np.log(np.sqrt((funEL1()/np.exp(12.7)+3.8)/6.67))
def funL1M_2(GEL1):
    return (6.67*np.exp(2*(0.159**2./GEL1))-3.8)*np.exp(14.66)
m0L1M_2 = funL1M_2(GEL1)


x = fsolve(funL1M,0)

x = fsolve(fun,0)

#Earth-L1
m0EL1 = (6.67*np.exp(2*0.841**2.)-3.8)*np.exp(12.7)
7749579.871780219

#L1-moon
m0L1M = 542551.08953663

#Earth-Moon
m0EM = (6.67*np.exp(2)-3.8)*np.exp(14.66)
105834032.42202497




######### REFORMULATION
M_0m = np.exp(14.66)*(6.67*np.exp((0.159/0.841)**2.)-3.8) #From L1 to Moon




