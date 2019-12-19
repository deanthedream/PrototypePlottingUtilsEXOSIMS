#Calculate Average Orbital Radius over true, Eccentric, and Mean Anomaly

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
import matplotlib.gridspec as gridspec
from EXOSIMS.util.eccanom import *
from astropy import constants as const
from matplotlib import ticker

def r_nuae(nu,a,e):
    """ planet distance from central body as a function of true anomaly
    """
    r = a*(1.-e**2.)/(1.+e*np.cos(nu))
    return r

def nu_Ee(E,e):
    """ true anomaly as a function of Eccentric anomaly
    Args:
        E (float) - Eccentric anomaly
        e (float) - eccentricity
    Returns:
        nu (float) - true anomaly
    """
    nu = np.arccos((np.cos(E)-e)/(1.-e*np.cos(E)))
    return nu

#Next 2 functions are unnecessary???
# def nu_Ee_fsolve(E,nu,e):
#     """ A helper function for E_nue
#     Args:
#         E (float) - Eccentric Anomaly
#         nu (float) - ture anomaly
#         e (float) - eccentricity
#     Returns:
#         error (float) - 
#     """
#     return (np.cos(E)-e)/(1.-e*np.cos(E)) - np.cos(nu)

# def E_nue(nu,e):
#     """ Calculate Eccentric Anomaly Given true anomaly and eccentricity
#     Args:
#         nu (float) - true anomaly
#         e (float) - eccentricity
#     Returns:
#         E (float) - Eccentric Anomaly
#     """
#     out = fsolve(nu_Ee_fsolve,nu,args=(nu,e))
#     E = out[0]
#     return E

def T_period(a,mu=const.G*const.M_sun):
    """ Calculate orbital period in seconds??
    Args:
        a (float) - semi-major axis in units of AU
    """
    T = np.sqrt(mu.to('AU3/s2').value/a**3.)
    return T

def E_t(a,e,t):
    """ Calculate Eccentric Anomaly From t
    Args:
        a (float) - semi-major axis
        e (float) - eccentricity
        t (float) - time
    Returns:
        E (float) - eccentric anomaly
    """
    #Assume 1 solar mass star
    #mu = const.G*const.M_sun
    n = 2.*np.pi/T_period(a) #in rad per second #np.sqrt(mu/a**3.)
    M = n*t
    E = eccanom(M,e)
    return E

def r_avgt(a,e):
    """Temporal average of planet-star distance
    From Mendex et al. 2017 
    """
    r = a*(1+e**2./2.)
    return r

#Average Orbital Radius Test
a = 1.
e = 0.3
T = T_period(a)
ts = np.linspace(start=0.,stop=T,num=100.)
Es = E_t(a,e,ts)
nus = nu_Ee(Es,e)
rs_1 = r_nuae(nus,a,e)
r_avgt_1 = np.average(rs_1)
r_avgt_2 = r_avgt(a,e)

#### Example for A single a e combo
# plt.close(1)
# plt.plot([0.,2.*np.pi],[r_avgt_1,r_avgt_1],color='blue',label='numerical')
# plt.plot([0.,2.*np.pi],[r_avgt_2,r_avgt_2],color='red',label='Mendez et al. 2017')
# plt.plot(Es,rs_1,color='green',linestyle='--')
# plt.show(block=False)

####
a_s=np.logspace(start=-2,stop=2,num=100)
e_s=np.linspace(start=0.,stop=0.95,num=100)
r_avgt_1 = np.zeros((len(a_s),len(e_s)))
r_avgt_2 = np.zeros((len(a_s),len(e_s)))
r_avgt_diff = np.zeros((len(a_s),len(e_s)))
for ii,jj in itertools.product(np.arange(len(a_s)),np.arange(len(e_s))):
    T = T_period(a_s[ii])
    ts = np.linspace(start=0.,stop=T,num=100.)
    Es = E_t(a_s[ii],e_s[jj],ts)
    nus = nu_Ee(Es,e_s[jj])
    rs_1 = r_nuae(nus,a_s[ii],e_s[jj])
    r_avgt_1[ii,jj] = np.average(rs_1)
    r_avgt_2[ii,jj] = r_avgt(a_s[ii],e_s[jj])
    r_avgt_diff[ii,jj] = np.abs(r_avgt_1[ii,jj] - r_avgt_2[ii,jj])
plt.close(2)
levels=np.logspace(start=np.log(np.min(r_avgt_1)),stop=np.log(np.max(r_avgt_2)),num=300)
fig2, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, num=2)
ax1.contourf(a_s,e_s,r_avgt_1,cmap='bwr',levels=levels, locator=ticker.LogLocator())
ax2.contourf(a_s,e_s,r_avgt_2,cmap='bwr',levels=levels, locator=ticker.LogLocator())
ax3.contourf(a_s,e_s,r_avgt_diff,cmap='bwr', locator=ticker.LogLocator())
ax1.set_ylabel('eccentricity')
ax2.set_ylabel('eccentricity')
ax3.set_ylabel('eccentricity')
ax3.set_xlabel('SMA')
plt.show(block=False)

print(saltyburrito)

nu = np.linspace(start=0.,stop=2.*np.pi,endpoint=True)
plt.figure(1)
plt.plot(nu,r_nuae(nu,1.,0.3))
plt.show(block=False)

#Test Integration
out = quad(r_nuae,0.,2.*np.pi,args=(1.0,0.))
r_avg = out[0]/(2.*np.pi)
print(saltyburrito)

#Test E_nue
nu = np.pi/2.
e = 0.3
E = E_nue(nu,e)
nu2 = nu_Ee(E,e)

# #Integrate over true anomaly
a_s=np.logspace(start=-2,stop=2,num=100)
e_s=np.linspace(start=0.,stop=1.,num=100)
r_avg = np.zeros((len(a_s),len(e_s)))
for ii,j in itertools.product(np.arange(len(a_s)),np.arange(len(e_s))):
    out = quad(r_nuae,0.,2.*np.pi,args=(a_s[ii],e_s[j]))
    r_avg[ii,j] = out[0]/(2.*np.pi)
    #1/(b-a)*int_a^b f(x)dx

plt.close(1)
fig = plt.figure(num=1)
gs = gridspec.GridSpec(1,2, width_ratios=[7.5,0.5], height_ratios=[8])
gs.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
levels = np.logspace(start=-3,stop=3,num=1000)
cax = ax0.contourf(a_s,e_s,r_avg,cmap='jet', levels=levels, norm=LogNorm())
ax0.set_xscale('symlog')
cbar = fig.colorbar(cax, cax=ax1,orientation='vertical')
cscaleMin = np.floor(np.nanmin(np.log10(r_avg))) # 10**min, min order of magnitude
cscaleMax = np.ceil(np.nanmax(np.log10(r_avg))) # 10**max, max order of magnitude
#levels = 10.**np.arange(cscaleMin,cscaleMax+1)
levels = [5.,10.,20.,30.,40.,50.]
CS4 = ax0.contour(cax, colors=('k',), linewidths=(1,), origin='lower', levels=levels, norm=LogNorm())
ax1.set_ylabel('Average Orbital Radius',weight='bold')
ax0.set_ylabel('Eccentricity', weight='bold')
ax0.set_xlabel('Semi-major axis (unitless)',weight='bold')
plt.show(block=False)


r_avg2 = np.zeros((len(a_s),len(e_s)))
for ii,j in itertools.product(np.arange(len(a_s)),np.arange(len(e_s))):
    r_avg2[ii,j] = r_avg[ii,j]/a_s[ii]

plt.close(2)
fig2 = plt.figure(num=2)
gs2 = gridspec.GridSpec(1,2, width_ratios=[7.5,0.5], height_ratios=[8])
gs2.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 
ax0 = plt.subplot(gs2[0])
ax1 = plt.subplot(gs2[1])
levels = np.logspace(start=-3,stop=3,num=1000)
cax = ax0.contourf(a_s,e_s,r_avg2,cmap='jet', levels=levels, norm=LogNorm())
ax0.set_xscale('symlog')
cbar = fig2.colorbar(cax, cax=ax1,orientation='vertical')
cscaleMin = np.floor(np.nanmin(np.log10(r_avg))) # 10**min, min order of magnitude
cscaleMax = np.ceil(np.nanmax(np.log10(r_avg))) # 10**max, max order of magnitude
levels = 10.**np.arange(cscaleMin,cscaleMax+1)
CS4 = ax1.contour(cax, colors=('k',), linewidths=(1,), origin='lower', levels=levels, norm=LogNorm())
ax1.set_ylabel('Average Orbital Radius',weight='bold')
ax0.set_ylabel('Eccentricity', weight='bold')
ax0.set_xlabel('Semi-major axis (unitless)',weight='bold')
plt.show(block=False)

plt.close(3)
plt.figure(num=3)
plt.plot(a_s,r_avg2[0,:],color='black')
plt.plot(a_s,r_avg2[50,:],color='red')
plt.show(block=False)




#Verifying something against the analytical version
def overNu(nu,e):
    out = np.arctan(np.tan(nu/2.)*np.sqrt((1.-e)/(1.+e)))
    return out

e=0.001
out = quad(overNu,0.,np.pi,args=(e))


plt.close(99)
plt.figure(num=99)
tmpe_s = np.linspace(start=0.,stop=1.,num=300)
plt.plot(tmpe_s,np.sqrt((1.-tmpe_s)/(1.+tmpe_s)),color='k')
plt.xlabel('e')
plt.show(block=False)



