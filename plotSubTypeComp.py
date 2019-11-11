#Plot Completeness Subtypes

import sys, os.path, EXOSIMS, EXOSIMS.MissionSim

folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts/'))
filename = 'compSubtype2.json'

scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile,nopar=True)


import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import itertools
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

comp = sim.Completeness

#### Plot Population JPDF
plt.close(983098098203845)
plt.figure(num=983098098203845)
plt.contourf(comp.xnew,comp.ynew,comp.Cpdf_pop,cmap='jet',intepolation='nearest', locator=ticker.LogLocator())
plt.show(block=False)

#### Plot EarthLike JPDF
plt.close(88853333333111)
plt.figure(num=88853333333111)
plt.contourf(comp.xnew,comp.ynew,comp.Cpdf_earthLike,cmap='jet',intepolation='nearest', locator=ticker.LogLocator())
plt.show(block=False)


#### Plot Sub-Type JPDF
for ii,j in itertools.product(np.arange(len(comp.Rp_hi)),np.arange(len(comp.L_lo[:,0]))):
    if np.max(comp.Cpdf_hs[ii,j]) == 0.:
        print('Skipping: ' + str(ii) + ',' + str(j))
        continue
    plt.close(88853333333111+int(ii)*len(comp.Rp_hi)+int(j))
    plt.figure(num=88853333333111+int(ii)*len(comp.Rp_hi)+int(j))
    plt.title(str(ii) + ',' + str(j))
    plt.contourf(comp.xnew,comp.ynew,comp.Cpdf_hs[ii,j],cmap='jet',intepolation='nearest', locator=ticker.LogLocator())
    plt.show(block=False)




#### Plot Gridspec
plt.close(1)
fig = plt.figure(num=1, figsize=(len(comp.L_lo[0,:])*3,len(comp.Rp_bins)*3+0.75))
numRows = len(comp.Rp_bins)-1+1 #Rp bins + 1 colorbar
numCols = len(comp.L_lo[0,:]) #Luminosity bins
#height_ratios = ([0.75] + [3,0.5]*(len(comp.Rp_bins)-1))[:-1]
#width_ratios = ([3,0.5]*(len(comp.L_lo)))[:-1]
height_ratios = ([0.75] + [3]*(len(comp.Rp_bins)-1))
width_ratios = ([3]*(len(comp.L_lo[0,:])))
gs = gridspec.GridSpec(numRows,numCols, width_ratios=width_ratios, height_ratios=height_ratios)
gs.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')

#Find Levels for Contour Colors
tmpH = comp.Cpdf_pop.copy()
tmpH[tmpH==0.] = np.nan
cscaleMin = np.floor(np.nanmin(np.log10(tmpH))) # 10**min, min order of magnitude
cscaleMax = np.ceil(np.nanmax(np.log10(tmpH))) # 10**max, max order of magnitude
levels = 10.**np.arange(cscaleMin,cscaleMax+1)

#Find xmin, xmax, ymin, ymax
xmin=0.
xmax=25.
ymin=15.
ymax=50.

#What the plot layout looks like
###---------------------------------------------------------
# | gs[0]  gs[1]  gs[2]  gs[3]  gs[4]  |
# | gs[5]  gs[6]  gs[7]  gs[8]  gs[9]  |
# | gs[10] gs[11] gs[12] gs[13] gs[14] |
# | gs[15] gs[16] gs[17] gs[18] gs[19] |
# | gs[20] gs[21] gs[22] gs[23] gs[24] |
# | gs[25] gs[26] gs[27] gs[28] gs[29] |
# | gs[30] gs[31] gs[32] gs[33] gs[34] |
# | gs[35] gs[36] gs[37] gs[38] gs[39] |
###---------------------------------------------------------
axCBAR = plt.subplot(gs[0:5])
axij = dict()
for ii,j in itertools.product(np.arange(len(comp.Rp_hi)),np.arange(len(comp.L_lo[0,:]))):
    axij[ii,j] = plt.subplot(gs[5+j+ii*len(comp.L_lo[0,:])])
    axij[ii,j].contourf(comp.xnew,comp.ynew,comp.Cpdf_hs[ii,j],cmap='jet',intepolation='nearest', levels=levels, norm = LogNorm())
    axij[ii,j].set_xlim([xmin,xmax])
    axij[ii,j].set_ylim([ymin,ymax])
    axij[ii,j].text(15,45,'(ii:' + str(ii) + ',j:' + str(j) + ')')
    #Add total Count Per Grid

#Temporary For coloring purposes
plt.close(9000)
plt.figure(num=9000)
ax1= plt.gca()
cax = ax1.contourf(comp.xnew, comp.ynew, comp.Cpdf_pop, extent=[xmin, xmax, ymin, ymax], cmap='jet', levels=levels, norm = LogNorm())
CS4 = ax1.contour(cax, colors=('k',), linewidths=(1,), origin='lower', levels=levels, norm = LogNorm())

#Add Colorbar
cbar = fig.colorbar(cax, cax=axCBAR, orientation='horizontal')#pad=0.05,
plt.rcParams['axes.titlepad']=-10
axCBAR.set_xlabel('Joint Probability Density: Planet Sub-types', weight='bold', labelpad=-35)
axCBAR.tick_params(axis='x',direction='in',labeltop=True,labelbottom=False)#'off'
cbar.add_lines(CS4)

plt.show(block=False)



#DELETE
# ax1 = plt.subplot(gs[5+5])#2D histogram of planet pop
# ax2 = plt.subplot(gs[0+5])#1D histogram of a
# ax3 = plt.subplot(gs[6+5])#1D histogram of Rp
# ax4 = plt.subplot(gs[8+5])#2D histogram of detected Planet Population
# ax5 = plt.subplot(gs[3+5])#1D histogram of detected planet a
# ax6 = plt.subplot(gs[9+5])#1D histogram of detected planet Rp
# TXT1 = plt.subplot(gs[1+5])
# TXT4 = plt.subplot(gs[4+5])


#### Plot Kopparapu Grid
plt.close(88888)
plt.figure(num=88888)

#Plot Nodes
for i in np.arange(len(comp.Rp_bins)-1):
    plt.scatter(comp.L_bins[i],np.zeros(len(comp.L_bins[i]))+comp.Rp_bins[i],color='blue',alpha=0.5)
#Plot Horizontal Lines
for i in np.arange(len(comp.Rp_bins)):
    for j in np.arange(len(comp.L_bins[i])-1):
        plt.plot([comp.L_bins[i,j],comp.L_bins[i,j+1]],[comp.Rp_bins[i],comp.Rp_bins[i]],color='red')
#Plot Vertical Lines
for i in np.arange(len(comp.Rp_bins)-1):
    for j in np.arange(len(comp.L_bins[i])):
        plt.plot([comp.L_bins[i,j],comp.L_bins[i+1,j]],[comp.Rp_bins[i],comp.Rp_bins[i+1]],color='blue')

ax = plt.gca()
ax.set_ylim(0.4, 20.)
#ax.set_xlim(500., 0.001)
ax.set_xlim(0.001, 500.)
ax.set_yscale('log')
ax.set_xscale('log')
ax.invert_xaxis()
ax.set_ylabel('Planet Radius in ' + r'$R_{\oplus}$', weight='bold')
ax.set_xlabel('Stellar Flux in Earth Units', weight='bold')
plt.show(block=False)

