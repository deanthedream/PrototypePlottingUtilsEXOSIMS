# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import astropy.units as u
import os
from EXOSIMS.PlanetPopulation.SAG13 import SAG13
import itertools
import astropy.constants as const
import scipy.integrate as integrate
import datetime
import re

PPoutpath = './SAG13figs/'
folder = './'
date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date


def SAG13jcdf(P, R):
    '''Return the SAG13 joint pdf of Period and Radius'''
    gamma = np.array([0.38, 0.73]) #gamma
    alpha = np.array([-0.19, -1.18]) #alpha
    beta = np.array([0.26, 0.59]) #beta
    Ri = 3.4 #R_earth
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
    f1 = gamma[0] / alpha[0] / beta[0] * (Ru ** alpha[0] - Rl ** alpha[0]) * (Pu ** beta[0] - Pl ** beta[0])
    # section above Ri
    Rb = R[R >= Ri]
    Rb = Rb.reshape((int(len(Rb) / 7), 7))
    Pb = P[R >= Ri]
    Pb = Pb.reshape((int(len(Pb) / 7), 7))
    Rl = Rb[:-1, :-1]
    Ru = Rb[1:, :-1]
    Pl = Pb[:-1, :-1]
    Pu = Pb[:-1, 1:]
    f2 = gamma[1] / alpha[1] / beta[1] * (Ru ** alpha[1] - Rl ** alpha[1]) * (Pu ** beta[1] - Pl ** beta[1])
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
    gamma, alpha, beta = 0.73, -1.18, 0.59
    f = gamma * R ** (alpha - 1.) * P ** (beta - 1.)
    nu = gamma / alpha / beta * (R.max() ** alpha - R.min() ** alpha) * (P.max() ** beta - P.min() ** beta) #Integrated to get total number
    return f, nu

def down(P, R):
    '''Returns lower R portion of SAG13 joint pdf'''
    gamma, alpha, beta = 0.38, -0.19, 0.26
    f = gamma * R ** (alpha - 1.) * P ** (beta - 1.) #occurrance rate distribution value
    nd = gamma / alpha / beta * (R.max() ** alpha - R.min() ** alpha) * (P.max() ** beta - P.min() ** beta) #Integrated to get total number
    return f, nd

P = np.array([10.,20.,40.,80.,160.,320.,640.]) #days
R = np.array([0.67,1.,1.5,2.2,3.4,5.1,7.6,11.,17.]) #R_earth

PP, RR = np.meshgrid(P,R)

vals = SAG13jcdf(PP*u.day.to('year'),RR) #the converstion from days to years here appears to be crucial
#vals = SAG13jcdf(PP,RR)
#for ii,jj in itertools.product(np.arange(len(P)-1),np.arange(len(R)-1)):

# Daniel's SAG13 Paremetric Fit Plot ##############################################################
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

fig1, ax1 = plt.subplots(figsize=(6,4.5),num=1)
a1 = ax1.pcolormesh(P,R,100*vals/np.sum(vals),norm=colors.LogNorm(vmin=1e-1,vmax=11.603844662288289),rasterized=True,edgecolor='none',cmap=cmap)
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
    ax1.text(P[ii],R[jj]*1.32,"{:2.2f}".format(100.*vals[jj,ii]/np.sum(vals),weight='bold')+'%',fontsize='10', color='black')
    ax1.text(P[ii],R[jj]*1.17,"{:2.2f}".format(100.*vals[jj,ii],weight='bold'),fontsize='10', color='purple')
ax1.tick_params(axis='both', bottom='on', top='off', right='off', left='on', which='major', labelsize=fontsize1)
ax1.tick_params(axis='both', bottom='off', top='off', right='off', left='off', which='minor')
#ax1.set_title(r'Parametric Fit Integrated Across Bins',fontsize=fontsize2,y=1.05, weight='bold')
ax1.set_xlabel('Orbital Period in d',fontsize=fontsize2, weight='bold')
ax1.set_ylabel('Planet Radius in ' + r'$R_\oplus$',fontsize=fontsize2, weight='bold')
plt.show(block=False)
plt.figure(fig1.number)
fname = 'DanielSAG13ParemetricFitPercente_RP' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=300)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=300)
#########################################################################################################




# plot SAG13 original pdf ################################################################################
P1 = np.logspace(np.log10(10.),np.log10(640.),500)
R1 = np.logspace(np.log10(0.67),np.log10(17.),500)
PP1,RR1 = np.meshgrid(P1,R1)
pdf_val = SAG13jpdf(PP1*u.day.to('year'),RR1)

fig2, ax2 = plt.subplots(figsize=(6,4.5),num=2)
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
plt.figure(fig2.number)
fname = 'DanielSAG13originalPDFplotcount_RP' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=300)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=300)
##############################################################################################################

# plot updated SAG13 pdf implemented in EXOSIMS ##############################################################
#spec = {'arange': [0.1,100.0], 'Rprange':[0.67,17.0]}
spec = {'arange': [0.1,30.0], 'Rprange':[2./3.,17.0]}
modules = {'PlanetPopulation':'SAG13', 'PlanetPhysicalModel':'PlanetPhysicalModel'}
spec['modules'] = modules

pop = SAG13(**spec)
a = np.logspace(np.log10(pop.arange[0].to('AU').value), np.log10(pop.arange[1].to('AU').value), 500)
R2 = np.logspace(np.log10(pop.Rprange[0].to('earthRad').value), np.log10(pop.Rprange[1].to('earthRad').value), 500)
aa, RR2 = np.meshgrid(a, R2)
faR = pop.dist_sma_radius(aa,RR2)

xa = np.array([0.1,1.,10.,30.])#100.])
fig3, ax3 = plt.subplots(figsize=(6,4.5),num=3)
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
plt.figure(fig3.number)
fname = 'EXOSIMSSAG13count_perRa' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=300)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=300)
################################################################################################################

# plot marginalized pdfs from EXOSIMS ##########################################################################
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
plt.figure(fig4.number)
fname = 'EXOSIMSSAG13marginalized_RA' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=300)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=300)
#################################################################################################################



# plot EXOSIMS SAG13 pdf OVER Fitted Range ######################################################################
#DOUBLE CHECK THIS PLOT... THERE MAY BE SOMETHING WRONG
plt.close(10)
#### Add SAG13 Grid P edges
#Calculate SMA bin edges
#DELETE Msun = 1.989 * 10.**30.*u.kg #kg
#DELETE G = 6.67408 * 10.**-11.*u.m**3.*u.kg**-1.*u.s**-2. #m3 kg-1 s-2
#DELETE mu = G*Msun
#DELETE mu = mu.to('AU3/year2').value
SAG13starMass = 1.*u.solMass
mu = (const.G*SAG13starMass).to('AU3/year2').value
#period = 2.*np.pi*np.sqrt(a**3/mu)
a_SAG13_occurrance_bins = (((P*u.day).to('year').value/2./np.pi)**(2./3.)*mu**(1./3.)) #In units of AU
#DELETE identical a_SAG13_occurrance_bins2 = P**(2./3.)*mu**(1./3.)/(2.*np.pi)**(2./3.)
#a_SAG13_occurrance_bins2 = (((mu*(P*u.day.to('s')/2./np.pi)**2. ).decompose())**(1./3.)).to('AU')
# period = 2.*np.pi*np.sqrt((30.*u.AU)**3/mu).decompose().to('day')
# parP= (3.*np.pi)*np.sqrt(a/mu)
# bigP = (2.*np.pi)**(beta_0)*(a**(3./2.)/np.sqrt(mu))**(beta_0-1.)*(3./2.*np.sqrt(a)/np.sqrt(mu))
#HERE SEE IF THE FULL EXPANSION WITH EXPONENT ENABLES ME TO CONVERT FROM P SPACE TO A SPACE AND MAKE GRIDS ON THE SMA VS R PLOT

#spec2 = {'arange': [0.1,100.0], 'Rprange':[0.67,17.0]}
spec2 = {'arange': [0.1,30.0], 'Rprange':[0.67,17.0]}
#spec2 = {'arange': [a_SAG13_occurrance_bins[0],a_SAG13_occurrance_bins[-1]], 'Rprange':[2./3.,17.0]}
modules = {'PlanetPopulation':'SAG13', 'PlanetPhysicalModel':'PlanetPhysicalModel'}
spec2['modules'] = modules

pop2 = SAG13(**spec2)
a = np.logspace(np.log10(pop2.arange[0].to('AU').value), np.log10(pop2.arange[1].to('AU').value), 500)
R2 = np.logspace(np.log10(pop2.Rprange[0].to('earthRad').value), np.log10(pop2.Rprange[1].to('earthRad').value), 500)
aa, RR2 = np.meshgrid(a, R2)
faR2 = pop.dist_sma_radius(aa,RR2)

xa = np.array([0.1,1.,10.,30.])#100.])
#xa = a_SAG13_occurrance_bins
fig10, ax10 = plt.subplots(figsize=(6,4.5),num=10)
a10 = ax10.pcolormesh(a,R2,faR2,rasterized=True,edgecolor='none',cmap=cmap,norm=colors.LogNorm(vmin=1e-5,vmax=0.34904989340963599))
c10 = fig10.colorbar(a10,ticks=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1])
fmt = ticker.LogFormatterMathtext()
fmt.create_dummy_axis()
c10.solids.set_rasterized(True)
c10.solids.set_edgecolor('face')
c10.ax.tick_params(labelsize=fontsize1, which='major')
c10.set_label('#' + ' Planets per Star/AU/' + r'$R_\oplus$',fontsize=fontsize1, weight='bold')
xlabel = ['{:.3g}'.format(_x) for _x in xa]
ylabel = ['{:.3g}'.format(_y) for _y in R]
# scale for axes
ax10.set_xscale('log')
ax10.set_yscale('log')
ax10.set_xticks(xa)
ax10.set_xticklabels(xlabel)
ax10.set_yticks(R)
ax10.set_yticklabels(ylabel)
ax10.tick_params(axis='both', bottom='on', top='off', right='off', left='on', which='major', labelsize=fontsize1)
ax10.tick_params(axis='both', bottom='off', top='off', right='off', left='off', which='minor')
#ax10.set_title(r'SAG 13 Extrapolated Joint PDF',fontsize=fontsize2,y=1.05, weight='bold')
ax10.set_xlabel('Semi-major Axis in AU',fontsize=fontsize2, weight='bold')
ax10.set_ylabel('Planet Radius in ' + r'$R_\oplus$',fontsize=fontsize2, weight='bold')
plt.show(block=False)
plt.figure(fig10.number)
fname = 'EXOSIMSSAG13pdf_RA' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=300)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=300)

#DELETE
# a2 = a_SAG13_occurrance_bins #np.logspace(np.log10(pop.arange[0].to('AU').value), np.log10(pop.arange[1].to('AU').value), 500)
# aa2, RR3 = np.meshgrid(a2, R)
# faR3 = pop2.dist_sma_radius(aa2,RR3)
# faR3 = faR3/np.sum(faR3)
# for ii,jj in itertools.product(np.arange(len(a2)-1),np.arange(len(R)-1)):
#     ax10.text(a2[ii],R[jj]*1.33,"{:2.2f}".format(100.*faR3[jj,ii],weight='bold')+'%')
plt.show(block=False)
###########################################################################################################################


# CONFIRM SAG13 PAREMTRIC FIT PLOT ########################################################
# -*- coding: utf-8 -*-
#A rehash of SAG13figs2

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
fig20, ax20 = plt.subplots(figsize=(6,4.5),num=20)
a20 = ax20.pcolormesh(P,R,100*eta_s.T/eta_SAG13,norm=colors.LogNorm(vmin=1e-1,vmax=11.603844662288289),rasterized=True,edgecolor='none',cmap=cmap)
c20 = fig20.colorbar(a20,ticks=[0.1,1,10])
c20.ax.set_yticklabels(['{}'.format(10.**(vv)) for vv in range(-1,3)],fontsize=fontsize1)
c20.set_label('% Stars with Planets in Bin',fontsize=fontsize1, weight='bold')
xlabel = ['{:.3g}'.format(_x) for _x in P]
ylabel = ['{:.3g}'.format(_y) for _y in R]
# scale for axes
ax20.set_xscale('log')
ax20.set_yscale('log')
ax20.set_xticks(P)
ax20.set_xticklabels(xlabel)
ax20.set_yticks(R)
ax20.set_yticklabels(ylabel)
for ii,jj in itertools.product(np.arange(len(P)-1),np.arange(len(R)-1)):
    ax20.text(P[ii],R[jj]*1.35,"{:2.2f}".format(100.*eta_s[ii,jj]/eta_SAG13,weight='bold')+'%')
    ax20.text(P[ii],R[jj]*1.17,"{:2.2f}".format(100.*eta_s[ii,jj],weight='bold'),fontsize='10', color='purple')
ax20.tick_params(axis='both', bottom='on', top='off', right='off', left='on', which='major', labelsize=fontsize1)
ax20.tick_params(axis='both', bottom='off', top='off', right='off', left='off', which='minor')
#ax20.set_title(r'Parametric Fit Integrated Across Bins',fontsize=fontsize2,y=1.05, weight='bold')
ax20.set_xlabel('Orbital Period in d',fontsize=fontsize2, weight='bold')
ax20.set_ylabel('Planet Radius in ' + r'$R_\oplus$',fontsize=fontsize2, weight='bold')
plt.show(block=False)
plt.figure(fig20.number)
fname = 'ANALYTICSAG13Percent_RP' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=300)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=300)

# Rmin = 2./3. #R_earth
# Rmax = 1. #R_earth
# Pmin = P[0] #in units of years #d
# Pmax = P[1] #in units of years #d
# #mf = integrate.nquad(SAG13jpdf_RP,ranges=[(Rmin,Rmax),(Pmin,Pmax)]) #marginalized
# eta = SAG13jcdf_RP(Rmin, Rmax, Pmin, Pmax)
# print(eta/eta_SAG13*100.)
#############################################################################################

# Plot Percent Grids for Analytic Implementation ############################################
def SAG13jcdf_Ra(Rmin, Rmax, amin, amax):
    '''Return the SAG13 joint pdf of Radius and semi-major axis
    amin and amax in AU??
    Rmin and Rmaxin R_earth
    '''
    Rlim = 3.4 #R_earth
    amin = amin#*u.AU
    amax = amax#*u.AU
    if Rmax < Rlim: #use i=0 when Rmax<Rlim=3.4
        out = integrate.nquad(SAG13jpdf_Ra,ranges=[(Rmin,Rmax),(amin,amax)],args=(0,)) #marginalized
        eta_SAG13 = out[0]
    elif Rmin >= Rlim: #use i=1 when Rmin>=Rlim=3.4
        out = integrate.nquad(SAG13jpdf_Ra,ranges=[(Rmin,Rmax),(amin,amax)],args=(1,)) #marginalized
        eta_SAG13 = out[0]
    else: #Rmin<Rlim and Rmax > Rlim
        #Independently marginalize over each
        out0 = integrate.nquad(SAG13jpdf_Ra,ranges=[(Rlim,Rmax),(amin,amax)],args=(1,)) #marginalized
        out1 = integrate.nquad(SAG13jpdf_Ra,ranges=[(Rmin,Rlim),(amin,amax)],args=(0,)) #marginalized
        eta_SAG13 = out0[0] + out1[0]
    #eta_SAG13, tmp = integrate.nquad(SAG13jpdf_RP,ranges=[(Rmin,Rmax),(Pmin,Pmax)]) #marginalized
    return eta_SAG13

def SAG13jpdf_Ra(R,a,i):
    ''' Returns value of JPDF at single P and R combination
    a in units of --
    '''
    gamma = np.array([0.38, 0.73]) #gamma
    alpha = np.array([-0.19, -1.18]) #alpha
    beta = np.array([0.26, 0.59]) #beta
    SAG13starMass = 1.*u.solMass
    mu = (const.G*SAG13starMass).to('AU3/year2').value
    f = gamma[i]* R**(alpha[i]-1.) * (2.*np.pi*np.sqrt(a**3./mu))**(beta[i]-1.)*(3.*np.pi*np.sqrt(a/mu)) #what it should be
    return f

#Total SAG13 occurrance Rate
Rmin=np.min(R)
Rmax=np.max(R)
amin=np.min(a_SAG13_occurrance_bins)
amax=np.max(a_SAG13_occurrance_bins)
#eta_SAG13, tmp = integrate.nquad(SAG13jpdf_RP,ranges=[(Rmin,Rmax),(Pmin,Pmax)]) #marginalized
eta_SAG132 = SAG13jcdf_Ra(Rmin, Rmax, amin, amax)

eta_s2 = np.zeros((len(a_SAG13_occurrance_bins)-1,len(R)-1))
for ii,jj in itertools.product(np.arange(len(a_SAG13_occurrance_bins)-1),np.arange(len(R)-1)):
    eta_s2[ii,jj] = SAG13jcdf_Ra(R[jj], R[jj+1], a_SAG13_occurrance_bins[ii], a_SAG13_occurrance_bins[ii+1])

cmap = plt.cm.bwr
cmap.set_over('w')
# plot SAG13 R vs a pdf
fig21, ax21 = plt.subplots(figsize=(6,4.5),num=21)
a21 = ax21.pcolormesh(a_SAG13_occurrance_bins,R,100*eta_s2.T/eta_SAG132,norm=colors.LogNorm(vmin=1e-1,vmax=11.603844662288289),rasterized=True,edgecolor='none',cmap=cmap)
c21 = fig21.colorbar(a21,ticks=[0.1,1,10])
c21.ax.set_yticklabels(['{}'.format(10.**(vv)) for vv in range(-1,3)],fontsize=fontsize1)
c21.set_label('% Stars with Planets in Bin',fontsize=fontsize1, weight='bold')
xlabel = ['{:.3g}'.format(_x) for _x in a_SAG13_occurrance_bins]
ylabel = ['{:.3g}'.format(_y) for _y in R]
# scale for axes
ax21.set_xscale('log')
ax21.set_yscale('log')
ax21.set_xticks(a_SAG13_occurrance_bins)
ax21.set_xticklabels(xlabel)
ax21.set_yticks(R)
ax21.set_yticklabels(ylabel)
for ii,jj in itertools.product(np.arange(len(a_SAG13_occurrance_bins)-1),np.arange(len(R)-1)):
    ax21.text(a_SAG13_occurrance_bins[ii],R[jj]*1.35,"{:2.2f}".format(100.*eta_s2[ii,jj]/eta_SAG132,weight='bold')+'%')
    ax21.text(a_SAG13_occurrance_bins[ii],R[jj]*1.17,"{:2.2f}".format(100.*eta_s2[ii,jj],weight='bold'),fontsize='10', color='purple')
ax21.tick_params(axis='both', bottom='on', top='off', right='off', left='on', which='major', labelsize=fontsize1)
ax21.tick_params(axis='both', bottom='off', top='off', right='off', left='off', which='minor')
#ax21.set_title(r'Parametric Fit Integrated Across Bins',fontsize=fontsize2,y=1.05, weight='bold')
ax21.set_xlabel('Semi-major axis in AU',fontsize=fontsize2, weight='bold')
ax21.set_ylabel('Planet Radius in ' + r'$R_\oplus$',fontsize=fontsize2, weight='bold')
plt.show(block=False)
plt.figure(fig21.number)
fname = 'ANALYTICSAG13Percent_RA' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=300)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=300)
############################################################################################################




#Manual Grid Errors ###########################################################
valsEXOSIMS = np.asarray([[0.15,0.23,0.34,0.51,0.77,1.16],
                            [0.20,0.31,0.46,0.70,1.05,1.58],
                            [0.35,0.52,0.79,1.19,1.79,2.69],
                            [0.57,0.86,1.29,1.94,2.92,4.40],
                            [4.07,4.88,5.84,6.99,8.38,10.03],
                            [3.87,4.64,5.55,6.65,7.96,9.54],
                            [4.42,5.29,6.34,7.59,9.09,10.88],
                            [4.71,5.64,6.76,8.09,9.69,11.60]])
valsBelikov = np.asarray([[0.134,0.202,0.304,0.458,0.691,1.04],
                            [0.216,0.326,0.492,0.741,1.12,1.68],
                            [0.35,0.527,0.795,1.2,1.81,2.72],
                            [0.565,0.852,1.28,1.94,2.92,4.4],
                            [3.75,4.5,5.39,6.46,7.75,9.29],
                            [4.05,4.85,5.82,6.98,8.36,10.0],
                            [4.37,5.24,6.28,7.53,9.03,10.8],
                            [4.72,5.65,6.78,8.13,9.74,11.7]])
errorss = np.abs(valsEXOSIMS-valsBelikov)
percentError = errorss/valsBelikov
bins = np.linspace(start=0.,stop=np.max(percentError),num=10)
plt.figure(num=9999)
plt.hist(percentError,range=[0.,np.max(percentError)])
plt.show(block=False)

###############################################################################

