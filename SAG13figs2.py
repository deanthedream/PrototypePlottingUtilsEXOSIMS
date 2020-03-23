# -*- coding: utf-8 -*-
#A rehash of SAG13figs2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import astropy.units as u
import os
from EXOSIMS.PlanetPopulation.SAG13 import SAG13
import itertools
import scipy.integrate as integrate




def SAG13jcdf_RP(Rmin, Rmax, Pmin, Pmax):
    '''Return the SAG13 joint pdf of Period and Radius
    Pmin and Pmax in days
    Rmin and Rmaxin R_earth
    '''
    Rlim = 3.4 #R_earth
    Pmin = Pmin*u.day.to('year')
    Pmax = Pmax*u.day.to('year')
    if Rmax < Rlim: #use i=0 when Rmax<Rlim=3.4
        out = integrate.nquad(SAG13jpdf_RP,ranges=[(Rmin,Rmax),(Pmin,Pmax)],args=(0,)) #marginalized
        eta_SAG13 = out[0]
    elif Rmin >= Rlim: #use i=1 when Rmin>=Rlim=3.4
        out = integrate.nquad(SAG13jpdf_RP,ranges=[(Rmin,Rmax),(Pmin,Pmax)],args=(1,)) #marginalized
        eta_SAG13 = out[0]
    else: #Rmin<Rlim and Rmax > Rlim
        #Independently marginalize over each
        out0 = integrate.nquad(SAG13jpdf_RP,ranges=[(Rlim,Rmax),(Pmin,Pmax)],args=(1,)) #marginalized
        out1 = integrate.nquad(SAG13jpdf_RP,ranges=[(Rmin,Rlim),(Pmin,Pmax)],args=(0,)) #marginalized
        eta_SAG13 = out0[0] + out1[0]
    #eta_SAG13, tmp = integrate.nquad(SAG13jpdf_RP,ranges=[(Rmin,Rmax),(Pmin,Pmax)]) #marginalized
    return eta_SAG13

def SAG13jpdf_RP(R,P,i):
    ''' Returns value of JPDF at single P and R combination
    P in units of years
    '''
    gamma = np.array([0.38, 0.73]) #gamma
    alpha = np.array([-0.19, -1.18]) #alpha
    beta = np.array([0.26, 0.59]) #beta
    # if R > 3.4:
    #     i=0
    # else: #R<=3.4
    #     i=1
    f = gamma[i]* R**(alpha[i]-1.) * P**(beta[i]-1.) #what it should be
    return f

#From Belikov 2018 SAG13 G Star Occurrence Rate Fig
P = np.array([10.,20.,40.,80.,160.,320.,640.]) #in units of years
R = np.array([0.67,1.,1.5,2.2,3.4,5.1,7.6,11.,17.]) #R_earth

#Total SAG13 Occurrance Rate
Rmin=np.min(R)
Rmax=np.max(R)
Pmin=np.min(P)
Pmax=np.max(P)
#eta_SAG13, tmp = integrate.nquad(SAG13jpdf_RP,ranges=[(Rmin,Rmax),(Pmin,Pmax)]) #marginalized
eta_SAG13 = SAG13jcdf_RP(Rmin, Rmax, Pmin, Pmax)

eta_s = np.zeros((len(P)-1,len(R)-1))
for ii,jj in itertools.product(np.arange(len(P)-1),np.arange(len(R)-1)):
    eta_s[ii,jj] = SAG13jcdf_RP(R[jj], R[jj+1], P[ii], P[ii+1])

plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
fontsize1=None#32
fontsize2=None#36
#plt.rc('font',weight='bold',family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath',r'\renewcommand{\seriesdefault}{\bfdefault}']

cmap = plt.cm.bwr
cmap.set_over('w')
# plot SAG13 original cdf
fig1, ax1 = plt.subplots(figsize=(6,4.5),num=1)
a1 = ax1.pcolormesh(P,R,100*eta_s.T/eta_SAG13,norm=colors.LogNorm(vmin=1e-1,vmax=11.603844662288289),rasterized=True,edgecolor='none',cmap=cmap)
c1 = fig1.colorbar(a1,ticks=[0.1,1,10])
c1.ax.set_yticklabels(['{}'.format(10.**(vv)) for vv in range(-1,3)],fontsize=fontsize1)
c1.set_label('% Stars with Planets in Bin',fontsize=fontsize1, weight='bold')
xlabel = ['{:.3g}'.format(_x) for _x in P]
ylabel = ['{:.3g}'.format(_y) for _y in R]
# scale for axes
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xticks(P)
ax1.set_xticklabels(xlabel)
ax1.set_yticks(R)
ax1.set_yticklabels(ylabel)
for ii,jj in itertools.product(np.arange(len(P)-1),np.arange(len(R)-1)):
    ax1.text(P[ii],R[jj]*1.35,"{:2.2f}".format(100.*eta_s[ii,jj]/eta_SAG13,weight='bold')+'%')
ax1.tick_params(axis='both', bottom='on', top='off', right='off', left='on', which='major', labelsize=fontsize1)
ax1.tick_params(axis='both', bottom='off', top='off', right='off', left='off', which='minor')
ax1.set_title(r'Parametric Fit Integrated Across Bins',fontsize=fontsize2,y=1.05, weight='bold')
ax1.set_xlabel('Orbital Period in d',fontsize=fontsize2, weight='bold')
ax1.set_ylabel('Planet Radius in ' + r'$R_\oplus$',fontsize=fontsize2, weight='bold')
plt.show(block=False)



# Rmin = 2./3. #R_earth
# Rmax = 1. #R_earth
# Pmin = P[0] #in units of years #d
# Pmax = P[1] #in units of years #d
# #mf = integrate.nquad(SAG13jpdf_RP,ranges=[(Rmin,Rmax),(Pmin,Pmax)]) #marginalized
# eta = SAG13jcdf_RP(Rmin, Rmax, Pmin, Pmax)


# print(eta/eta_SAG13*100.)

