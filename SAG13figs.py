# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import astropy.units as u
import os
from EXOSIMS.PlanetPopulation.SAG13 import SAG13
import itertools


def SAG13jcdf(P, R):
    '''Return the SAG13 joint pdf of Period and Radius'''
    g = np.array([0.38, 0.73])
    r = np.array([-0.19, -1.18])
    b = np.array([0.26, 0.59])
    Ri = 3.4
    f = np.zeros((P.shape[0] - 1, P.shape[1] - 1))
    # section below Ri
    Rb = R[R <= Ri]
    Rb = Rb.reshape((int(len(Rb) / 7), 7))

    Pb = P[R <= Ri]
    Pb = Pb.reshape((int(len(Pb) / 7), 7))
    Rl = Rb[:-1, :-1]
    Ru = Rb[1:, :-1]
    Pl = Pb[:-1, :-1]
    Pu = Pb[:-1, 1:]
    f1 = g[0] / r[0] / b[0] * (Ru ** r[0] - Rl ** r[0]) * (Pu ** b[0] - Pl ** b[0])
    # section above Ri
    Rb = R[R >= Ri]
    Rb = Rb.reshape((int(len(Rb) / 7), 7))
    Pb = P[R >= Ri]
    Pb = Pb.reshape((int(len(Pb) / 7), 7))
    Rl = Rb[:-1, :-1]
    Ru = Rb[1:, :-1]
    Pl = Pb[:-1, :-1]
    Pu = Pb[:-1, 1:]
    f2 = g[1] / r[1] / b[1] * (Ru ** r[1] - Rl ** r[1]) * (Pu ** b[1] - Pl ** b[1])
    f = np.vstack((f1, f2))

    return f

def SAG13jpdf(P, R):
    '''Returns SAG13 joint pdf'''
    f = np.zeros(P.shape)
    f[R <= 3.4], nd = down(P[R <= 3.4], R[R <= 3.4])
    f[R >= 3.4], nu = up(P[R >= 3.4], R[R >= 3.4])
    #    print (nd+nu)
    f /= (nd + nu)
    return f

def up(P, R):
    '''Returns upper R portion of SAG13 joint pdf'''
    g, r, b = 0.73, -1.18, 0.59
    f = g * R ** (r - 1.) * P ** (b - 1.)
    nu = g / r / b * (R.max() ** r - R.min() ** r) * (P.max() ** b - P.min() ** b)
    return f, nu

def down(P, R):
    '''Returns lower R portion of SAG13 joint pdf'''
    g, r, b = 0.38, -0.19, 0.26
    f = g * R ** (r - 1.) * P ** (b - 1.)
    nd = g / r / b * (R.max() ** r - R.min() ** r) * (P.max() ** b - P.min() ** b)
    return f, nd

P = np.array([10.,20.,40.,80.,160.,320.,640.]) #days
R = np.array([0.67,1.,1.5,2.2,3.4,5.1,7.6,11.,17.]) #R_earth

PP, RR = np.meshgrid(P,R)

vals = SAG13jcdf(PP*u.day.to('year'),RR)

# plot setup
# use TeX fonts
#plt.rc('text',usetex=True)
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
fig1, ax1 = plt.subplots(figsize=(6,4.5))
a1 = ax1.pcolormesh(P,R,100*vals,norm=colors.LogNorm(vmin=1e-1,vmax=11.603844662288289),rasterized=True,edgecolor='none',cmap=cmap)
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
    ax1.text(P[ii],R[jj]*1.35,"{:2.2f}".format(100.*vals[jj,ii],weight='bold')+'%')
ax1.tick_params(axis='both', bottom='on', top='off', right='off', left='on', which='major', labelsize=fontsize1)
ax1.tick_params(axis='both', bottom='off', top='off', right='off', left='off', which='minor')
ax1.set_title(r'Parametric Fit Integrated Across Bins',fontsize=fontsize2,y=1.05, weight='bold')
ax1.set_xlabel('Orbital Period in d',fontsize=fontsize2, weight='bold')
ax1.set_ylabel('Planet Radius in ' + r'$R_\oplus$',fontsize=fontsize2, weight='bold')
# fig1.show()

# plot SAG13 original pdf
P1 = np.logspace(np.log10(10.),np.log10(640.),500)
R1 = np.logspace(np.log10(0.67),np.log10(17.),500)
PP1,RR1 = np.meshgrid(P1,R1)
pdf_val = SAG13jpdf(PP1*u.day.to('year'),RR1)

fig2, ax2 = plt.subplots(figsize=(6,4.5))
a2 = ax2.pcolormesh(P1,R1,pdf_val,rasterized=True,edgecolor='none',cmap=cmap,norm=colors.LogNorm(vmin=0.00061858170506433525,vmax=4.5019016669992835))
c2 = fig2.colorbar(a2,ticks=[0.0001,0.001,0.01,0.1,1,10])
c2.ax.set_yticklabels(['{}'.format(10.**(vv)) for vv in range(-4,2)],fontsize=fontsize1)
c2.set_label('# Planets per Star/Radius/Period',fontsize=fontsize1, weight='bold')
xlabel = ['{:.3g}'.format(_x) for _x in P]
ylabel = ['{:.3g}'.format(_y) for _y in R]
# scale for axes
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xticks(P)
ax2.set_xticklabels(xlabel)
ax2.set_yticks(R)
ax2.set_yticklabels(ylabel)
ax2.tick_params(axis='both', bottom='on', top='off', right='off', left='on', which='major', labelsize=fontsize1)
ax2.tick_params(axis='both', bottom='off', top='off', right='off', left='off', which='minor')
#ax2.set_title(r'Parametric Fit Joint PDF',fontsize=fontsize2,y=1.05, weight='bold')
ax2.set_xlabel('Orbital Period in d',fontsize=fontsize2, weight='bold')
ax2.set_ylabel('Planet Radius in ' + r'$R_\oplus$',fontsize=fontsize2, weight='bold')
# fig2.show()

# plot updated SAG13 pdf
#spec = {'arange': [0.1,100.0], 'Rprange':[0.67,17.0]}
spec = {'arange': [0.1,30.0], 'Rprange':[2./3.,17.0]}
modules = {'PlanetPopulation':'SAG13', 'PlanetPhysicalModel':'PlanetPhysicalModel'}
spec['modules'] = modules

pop = SAG13(**spec)
a = np.logspace(np.log10(pop.arange[0].to('AU').value), np.log10(pop.arange[1].to('AU').value), 500)
R2 = np.logspace(np.log10(pop.Rprange[0].to('earthRad').value), np.log10(pop.Rprange[1].to('earthRad').value), 500)
aa, RR2 = np.meshgrid(a, R2)
faR = pop.dist_sma_radius(aa,RR2)

xa = np.array([0.1,1.,10.,100.])
fig3, ax3 = plt.subplots(figsize=(6,4.5))
a3 = ax3.pcolormesh(a,R2,faR,rasterized=True,edgecolor='none',cmap=cmap,norm=colors.LogNorm(vmin=1e-5,vmax=0.34904989340963599))
c3 = fig3.colorbar(a3,ticks=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1])
fmt = ticker.LogFormatterMathtext()
fmt.create_dummy_axis()
c3.solids.set_rasterized(True)
c3.solids.set_edgecolor('face')
c3.ax.tick_params(labelsize=fontsize1, which='major')
c3.set_label('#' + ' Planets per Star/AU/' + r'$R_\oplus$',fontsize=fontsize1, weight='bold')
xlabel = ['{:.3g}'.format(_x) for _x in xa]
ylabel = ['{:.3g}'.format(_y) for _y in R]
# scale for axes
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xticks(xa)
ax3.set_xticklabels(xlabel)
ax3.set_yticks(R)
ax3.set_yticklabels(ylabel)
ax3.tick_params(axis='both', bottom='on', top='off', right='off', left='on', which='major', labelsize=fontsize1)
ax3.tick_params(axis='both', bottom='off', top='off', right='off', left='off', which='minor')
#ax3.set_title(r'SAG 13 Extrapolated Joint PDF',fontsize=fontsize2,y=1.05, weight='bold')
ax3.set_xlabel('Semi-major Axis in AU',fontsize=fontsize2, weight='bold')
ax3.set_ylabel('Planet Radius in ' + r'$R_\oplus$',fontsize=fontsize2, weight='bold')
plt.show(block=False)

#### Add SAG13 Grid
#Calculate SMA bin edges
Msun = 1.989 * 10.**30.*u.kg #kg
G = 6.67408 * 10.**-11.*u.m**3.*u.kg**-1.*u.s**-2. #m3 kg-1 s-2
mu = G*Msun
#period = 2.*np.pi*np.sqrt(a**3/mu)
a_SAG13_occurrance_bins = ((mu*( ((P*u.day).to('s')/(2.*np.pi))**2. ).decompose())**(1./3.)).to('AU')
period = 2.*np.pi*np.sqrt((30.*u.AU)**3/mu).decompose().to('day')
parP= (3.*np.pi)*np.sqrt(a/mu)

bigP = (2.*np.pi)**(beta_0)*(a**(3./2.)/np.sqrt(mu))**(beta_0-1.)*(3./2.*np.sqrt(a)/np.sqrt(mu))
#HERE SEE IF THE FULL EXPANSION WITH EXPONENT ENABLES ME TO CONVERT FROM P SPACE TO A SPACE AND MAKE GRIDS ON THE SMA VS R PLOT

# plot marginalized pdfs
fa = pop.dist_sma(a)
fR = pop.dist_radius(R2)
fig4, (ax5,ax4) = plt.subplots(2,figsize=(5,4))

ax4.loglog(a,fa)#,linewidth=7)
ax4.set_xlim(left=0.1,right=100.)
ax4.tick_params(axis='both', which='major', labelsize=fontsize1)
ax4.tick_params(axis='both', bottom='off', top='off', right='off', left='off', which='minor')
ax4.set_xlabel('Semi-major axis in ' + 'AU', fontsize=fontsize2, weight='bold')
ax4.set_ylabel(r'$ f_{\bar{a}}\left(a\right) $', fontsize=fontsize2, weight='bold')
ax4.set_xticks(xa)
ax4.set_xticklabels(xlabel)

ax5.loglog(R2,fR,'r-')#,linewidth=7)
ax5.set_xlim(left=0.67, right=17.)
ax5.tick_params(axis='both', which='major', labelsize=fontsize1)
ax5.tick_params(axis='both', bottom='off', top='off', right='off', left='off', which='minor')
ax5.set_xlabel('Planetary Radius in ' + r'$R_{\oplus}$', fontsize=fontsize2, weight='bold')
ax5.set_ylabel(r'$ f_{\bar{R}}\left(R\right) $', fontsize=fontsize2, weight='bold')
ax5.set_xticks(R)
ax5.set_xticklabels(ylabel)
#ax5.set_title(r'Marginalized PDFs',fontsize=fontsize2,y=1.05)
# fig4.show()
fig4.tight_layout(h_pad=1)#5)
plt.show(block=False)

