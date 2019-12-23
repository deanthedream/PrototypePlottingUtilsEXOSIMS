#Plot Completeness Subtypes

import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import datetime
import re
import astropy.units as u

folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts/'))
filename = 'compSubtype2.json'

scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile,nopar=True)


PPoutpath = './'
folder = './'
date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date

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
plt.contourf(comp.xnew,comp.ynew,comp.Cpdf_pop,cmap='jet', locator=ticker.LogLocator(), zorder=0)
for ii,j in itertools.product(np.arange(len(comp.Rp_hi)),np.arange(len(comp.L_lo[0,:]))):
    for k in np.arange(len(comp.jpdf_props['lower_limits'][ii,j])):
        start = comp.jpdf_props['lower_limits'][ii,j][k]
        stop = comp.jpdf_props['upper_limits'][ii,j][k]
        sranges = np.linspace(start=start,stop=stop,num=200)
        plt.plot(sranges,comp.jpdf_props['limit_funcs'][ii,j][k](sranges),color='black', zorder=10)
plt.show(block=False)

#### Plot EarthLike JPDF
plt.close(88853333333111)
plt.figure(num=88853333333111)
plt.contourf(comp.xnew,comp.ynew,comp.Cpdf_earthLike,cmap='jet', locator=ticker.LogLocator())
plt.show(block=False)


#### Plot Sub-Type JPDF ##########################################################
def plotIndividualSubTypeJPDFs(comp):
    for ii,j in itertools.product(np.arange(len(comp.Rp_hi)),np.arange(len(comp.L_lo[:,0]))):
        if np.max(comp.Cpdf_hs[ii,j]) == 0.:
            print('Skipping: ' + str(ii) + ',' + str(j))
            continue
        plt.close(88853333333111+int(ii)*len(comp.Rp_hi)+int(j))
        plt.figure(num=88853333333111+int(ii)*len(comp.Rp_hi)+int(j))
        plt.title(str(ii) + ',' + str(j))
        plt.contourf(comp.xnew,comp.ynew,comp.Cpdf_hs[ii,j],cmap='jet', locator=ticker.LogLocator())
        plt.show(block=False)
#plotIndividualSubTypeJPDFs(comp)
##################################################


#### Plot Gridspec of Kopparapu Bins 1111111111111111111111111111111111111
plt.close(9876)
fig9876 = plt.figure(num=9876, figsize=(len(comp.L_lo[0,:]-2)*4,len(comp.Rp_bins-1)*4+0.75))
numRows = len(comp.Rp_bins)-1-1+1 #Rp bins + 1 colorbar
numCols = len(comp.L_lo[0,:])-2 #Luminosity bins
#height_ratios = ([0.75] + [3,0.5]*(len(comp.Rp_bins)-1))[:-1]
#width_ratios = ([3,0.5]*(len(comp.L_lo)))[:-1]
height_ratios = ([0.75] + [4]*(len(comp.Rp_bins)-1-1))
width_ratios = ([4]*(len(comp.L_lo[0,:])-2))
gs = gridspec.GridSpec(numRows,numCols, width_ratios=width_ratios, height_ratios=height_ratios)
gs.update(wspace=0.03, hspace=0.05) # set the spacing between axes. 
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
# | gs[0]  gs[1]  gs[2]  |
# | gs[3]  gs[4]  gs[5]  |
# | gs[6]  gs[7]  gs[8]  |
# | gs[9]  gs[10] gs[11] |
# | gs[12] gs[13] gs[14] |
# | gs[15] gs[16] gs[17] |
# | gs[18] gs[19] gs[20] |
# | gs[21] gs[22] gs[23] |
###---------------------------------------------------------
axCBAR = plt.subplot(gs[0:2])
axij = dict()
for ii,j in itertools.product(np.arange(len(comp.Rp_hi)-1)+1,np.arange(len(comp.L_lo[0,:])-2)+1):
    #DELETE axij[ii,j] = plt.subplot(gs[5+j+ii*len(comp.L_lo[0,:])]) #old mapping
    axij[ii,j] = plt.subplot(gs[2+j+(len(comp.Rp_hi)-1-ii)*(len(comp.L_lo[0,:])-2)])#3 from cbar, j iterates over row rest defines starting spot
    axij[ii,j].contourf(comp.xnew,comp.ynew,comp.Cpdf_hs[ii,j],cmap='jet', levels=levels, norm = LogNorm())
    axij[ii,j].set_xlim([xmin,xmax])
    axij[ii,j].set_ylim([ymin,ymax])
    axij[ii,j].text(18.5,45,'(i:' + str(ii) + ',j:' + str(j) + ')', weight='bold')
    if j != 1:
        axij[ii,j].get_yaxis().set_visible(False)
    if ii != 1:#len(comp.Rp_hi)-1:
        axij[ii,j].get_xaxis().set_visible(False)
    axij[ii,j].text(0.5,45, comp.type_names[ii,j], weight='bold')
    #Add total Count per grid
    axij[ii,j].text(16,41, "{:.2e}".format(comp.count_hs[ii,j]),weight='bold')
    #Add bounding edges
    for k in np.arange(len(comp.jpdf_props['lower_limits'][ii,j])):
        start = comp.jpdf_props['lower_limits'][ii,j][k]
        stop = comp.jpdf_props['upper_limits'][ii,j][k]
        sranges = np.linspace(start=start,stop=stop,num=200)
        axij[ii,j].plot(sranges,comp.jpdf_props['limit_funcs'][ii,j][k](sranges),color='black')
    #Add bounding edges For Other in Column
    for ii2 in (np.arange(len(comp.Rp_hi)-ii-1-1)+ii+1+1): #All plots above this one in the column
        for k in np.arange(len(comp.jpdf_props['lower_limits'][ii2,j])-1):
            start = comp.jpdf_props['lower_limits'][ii2,j][k]
            stop = comp.jpdf_props['upper_limits'][ii2,j][k]
            sranges = np.linspace(start=start,stop=stop,num=200)
            axij[ii,j].plot(sranges,comp.jpdf_props['limit_funcs'][ii2,j][k](sranges),color='red',linewidth=1.0)
    #Add dmag and s, (dmag=23, s=0.7) - Earth from dmag vs s plot of solar system
    earth_separation = 0.7 #AU
    earth_dmag = 23. #planet-star difference in magnitude
    axij[ii,j].scatter(earth_separation,earth_dmag,c='white',s=2,edgecolor='black',linewidth=0.5)
    #Add ddmag and ds Error Bars
    uncertainty_dmag = 0.01 #HabEx requirement is 1%
    uncertainty_s = 5.*u.mas.to('rad')*10.*u.pc.to('AU')
    axij[ii,j].errorbar(earth_separation, earth_dmag, yerr=[uncertainty_dmag*earth_dmag,], xerr=[uncertainty_s], fmt='--',color='black',linewidth=1)
fig9876.tight_layout(pad=0)

#Temporary For coloring purposes, pop hist
plt.close(9000)
figpop = plt.figure(num=9000)
ax1= plt.gca()
cax = ax1.contourf(comp.xnew, comp.ynew, comp.Cpdf_pop, extent=[xmin, xmax, ymin, ymax], cmap='jet', levels=levels, norm = LogNorm())
CS4 = ax1.contour(cax, colors=('k',), linewidths=(1,), origin='lower', levels=levels, norm = LogNorm())
for k in np.arange(len(comp.lower_limits)):
    start = comp.lower_limits[k]
    stop = comp.upper_limits[k]
    sranges = np.linspace(start=start,stop=stop,num=200)
    ax1.plot(sranges, comp.dmag_limit_functions[k](sranges),color='black')
for ii,j in itertools.product(np.arange(len(comp.Rp_hi)-1)+1,np.arange(len(comp.L_lo[0,:])-2)+1):
    for k in np.arange(len(comp.jpdf_props['lower_limits'][ii,j])):
        start = comp.jpdf_props['lower_limits'][ii,j][k]
        stop = comp.jpdf_props['upper_limits'][ii,j][k]
        sranges = np.linspace(start=start,stop=stop,num=200)
        ax1.plot(sranges,comp.jpdf_props['limit_funcs'][ii,j][k](sranges),color='black', zorder=10)
ax1.set_ylim([ymin,ymax])
ax1.set_xlim([xmin,xmax])

#Add Colorbar, and pop dist
cbar = fig9876.colorbar(cax, cax=axCBAR, orientation='horizontal')#pad=0.05,
plt.rcParams['axes.titlepad']=-10
axCBAR.set_xlabel('Joint Probability Density: Planet Sub-types', weight='bold', labelpad=-55)
axCBAR.tick_params(axis='x',direction='in',labeltop=True,labelbottom=False)#'off'
cbar.add_lines(CS4)
plt.title('SAG13 Population JPDF', weight='bold')
plt.xlabel('Luminosity Scaled Planet-star Separation ' + r'$(s/\sqrt{L})$' + ' in AU', weight='bold')
plt.ylabel('Planet-star Difference in Magnitude', weight='bold')
plt.text(20,48,"{:.2e}".format(comp.count_pop),weight='bold')

#Add Labels
fig9876.text(0.5, 0.075, 'Luminosity Scaled Planet-star Separation ' + r'$(s/\sqrt{L})$' + ' in AU', ha='center', weight='bold')
fig9876.text(0.09, 0.5, 'Planet-star Difference in Magnitude', va='center', rotation='vertical', weight='bold')

#### add Earth-Like dist
plt.close(2)
figearth = plt.figure(num=2)
ax2= plt.gca()
cax2 = ax2.contourf(comp.xnew, comp.ynew, comp.Cpdf_earthLike, extent=[xmin, xmax, ymin, ymax], cmap='jet', levels=levels, norm = LogNorm())
CS42 = ax2.contour(cax2, colors=('k',), linewidths=(1,), origin='lower', levels=levels, norm = LogNorm())
ax2.set_ylim([ymin,ymax])
ax2.set_xlim([xmin,xmax])
plt.text(20,48,"{:.2e}".format(comp.count_earthLike),weight='bold')
plt.title('SAG13 Earth-Like Sub-Population JPDF', weight='bold')
plt.xlabel('Luminosity Scaled Planet-star Separation ' + r'$(s/\sqrt{L})$' + ' in AU', weight='bold')
plt.ylabel('Planet-star Difference in Magnitude', weight='bold')

plt.show(block=False)
print(saltyburrito)
###########################################################################



#### Plot Gridspec of Kopparapu Bins 1111111111111111111111111111111111111
plt.close(1)
fig = plt.figure(num=1, figsize=(len(comp.L_lo[0,:])*3,len(comp.Rp_bins)*3+0.75))
numRows = len(comp.Rp_bins)-1+1 #Rp bins + 1 colorbar
numCols = len(comp.L_lo[0,:]) #Luminosity bins
#height_ratios = ([0.75] + [3,0.5]*(len(comp.Rp_bins)-1))[:-1]
#width_ratios = ([3,0.5]*(len(comp.L_lo)))[:-1]
height_ratios = ([0.75] + [3]*(len(comp.Rp_bins)-1))
width_ratios = ([3]*(len(comp.L_lo[0,:])))
gs = gridspec.GridSpec(numRows,numCols, width_ratios=width_ratios, height_ratios=height_ratios)
gs.update(wspace=0.03, hspace=0.05) # set the spacing between axes. 
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
    #DELETE axij[ii,j] = plt.subplot(gs[5+j+ii*len(comp.L_lo[0,:])]) #old mapping
    axij[ii,j] = plt.subplot(gs[5+j+(len(comp.Rp_hi)-1-ii)*len(comp.L_lo[0,:])])#5 from cbar, j is 
    axij[ii,j].contourf(comp.xnew,comp.ynew,comp.Cpdf_hs[ii,j],cmap='jet', levels=levels, norm = LogNorm())
    axij[ii,j].set_xlim([xmin,xmax])
    axij[ii,j].set_ylim([ymin,ymax])
    axij[ii,j].text(18.5,45,'(i:' + str(ii) + ',j:' + str(j) + ')', weight='bold')
    if j != 0:
        axij[ii,j].get_yaxis().set_visible(False)
    if ii != 0:#len(comp.Rp_hi)-1:
        axij[ii,j].get_xaxis().set_visible(False)
    axij[ii,j].text(0.5,45, comp.type_names[ii,j], weight='bold')
    #Add total Count per grid
    axij[ii,j].text(16,41, "{:.2e}".format(comp.count_hs[ii,j]),weight='bold')
    #Add bounding edges
    for k in np.arange(len(comp.jpdf_props['lower_limits'][ii,j])):
        start = comp.jpdf_props['lower_limits'][ii,j][k]
        stop = comp.jpdf_props['upper_limits'][ii,j][k]
        sranges = np.linspace(start=start,stop=stop,num=200)
        axij[ii,j].plot(sranges,comp.jpdf_props['limit_funcs'][ii,j][k](sranges),color='black')
    #Add bounding edges For Other in Column
    for ii2 in (np.arange(len(comp.Rp_hi)-ii-1)+ii+1): #All plots above this one in the column
        for k in np.arange(len(comp.jpdf_props['lower_limits'][ii2,j])-1):
            start = comp.jpdf_props['lower_limits'][ii2,j][k]
            stop = comp.jpdf_props['upper_limits'][ii2,j][k]
            sranges = np.linspace(start=start,stop=stop,num=200)
            axij[ii,j].plot(sranges,comp.jpdf_props['limit_funcs'][ii2,j][k](sranges),color='red',linewidth=1.0)
    #Add dmag and s, (dmag=23, s=0.7) - Earth from dmag vs s plot of solar system
    earth_separation = 0.7 #AU
    earth_dmag = 23. #planet-star difference in magnitude
    axij[ii,j].scatter(earth_separation,earth_dmag,c='white',s=2,edgecolor='black',linewidth=0.5)
    #Add ddmag and ds Error Bars
    uncertainty_dmag = 0.01 #HabEx requirement is 1%
    uncertainty_s = 5.*u.mas.to('rad')*10.*u.pc.to('AU')
    axij[ii,j].errorbar(earth_separation, earth_dmag, yerr=[uncertainty_dmag*earth_dmag,], xerr=[uncertainty_s], fmt='--',color='black',linewidth=1)
fig.tight_layout(pad=0)

#Temporary For coloring purposes, pop hist
plt.close(9000)
figpop = plt.figure(num=9000)
ax1= plt.gca()
cax = ax1.contourf(comp.xnew, comp.ynew, comp.Cpdf_pop, extent=[xmin, xmax, ymin, ymax], cmap='jet', levels=levels, norm = LogNorm())
CS4 = ax1.contour(cax, colors=('k',), linewidths=(1,), origin='lower', levels=levels, norm = LogNorm())
for k in np.arange(len(comp.lower_limits)):
    start = comp.lower_limits[k]
    stop = comp.upper_limits[k]
    sranges = np.linspace(start=start,stop=stop,num=200)
    ax1.plot(sranges, comp.dmag_limit_functions[k](sranges),color='black')
for ii,j in itertools.product(np.arange(len(comp.Rp_hi)),np.arange(len(comp.L_lo[0,:]))):
    for k in np.arange(len(comp.jpdf_props['lower_limits'][ii,j])):
        start = comp.jpdf_props['lower_limits'][ii,j][k]
        stop = comp.jpdf_props['upper_limits'][ii,j][k]
        sranges = np.linspace(start=start,stop=stop,num=200)
        ax1.plot(sranges,comp.jpdf_props['limit_funcs'][ii,j][k](sranges),color='black', zorder=10)
ax1.set_ylim([ymin,ymax])
ax1.set_xlim([xmin,xmax])

#Add Colorbar, and pop dist
cbar = fig.colorbar(cax, cax=axCBAR, orientation='horizontal')#pad=0.05,
plt.rcParams['axes.titlepad']=-10
axCBAR.set_xlabel('Joint Probability Density: Planet Sub-types', weight='bold', labelpad=-55)
axCBAR.tick_params(axis='x',direction='in',labeltop=True,labelbottom=False)#'off'
cbar.add_lines(CS4)
plt.title('SAG13 Population JPDF', weight='bold')
plt.xlabel('Luminosity Scaled Planet-star Separation ' + r'$(s/\sqrt{L})$' + ' in AU', weight='bold')
plt.ylabel('Planet-star Difference in Magnitude', weight='bold')
plt.text(20,48,"{:.2e}".format(comp.count_pop),weight='bold')

#Add Labels
fig.text(0.5, 0.075, 'Luminosity Scaled Planet-star Separation ' + r'$(s/\sqrt{L})$' + ' in AU', ha='center', weight='bold')
fig.text(0.09, 0.5, 'Planet-star Difference in Magnitude', va='center', rotation='vertical', weight='bold')

#### add Earth-Like dist
plt.close(2)
figearth = plt.figure(num=2)
ax2= plt.gca()
cax2 = ax2.contourf(comp.xnew, comp.ynew, comp.Cpdf_earthLike, extent=[xmin, xmax, ymin, ymax], cmap='jet', levels=levels, norm = LogNorm())
CS42 = ax2.contour(cax2, colors=('k',), linewidths=(1,), origin='lower', levels=levels, norm = LogNorm())
ax2.set_ylim([ymin,ymax])
ax2.set_xlim([xmin,xmax])
plt.text(20,48,"{:.2e}".format(comp.count_earthLike),weight='bold')
plt.title('SAG13 Earth-Like Sub-Population JPDF', weight='bold')
plt.xlabel('Luminosity Scaled Planet-star Separation ' + r'$(s/\sqrt{L})$' + ' in AU', weight='bold')
plt.ylabel('Planet-star Difference in Magnitude', weight='bold')

plt.show(block=False)
print(saltyburrito)


plt.figure(fig.number)
fname = 'JPDFsubtype_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=300)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=300)
plt.figure(figpop.number)
fname = 'JPDFpop_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=300)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=300)
plt.figure(figearth.number)
fname = 'JPDFearth_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=300)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=300)

###########################################################################





#### Plot Gridspec of Kopparapu Bins 22222222222222222222222222222222222222222222
plt.close(11)
fig2 = plt.figure(num=11, figsize=(len(comp.L_lo[0,:])*3,len(comp.Rp_bins)*3+0.75))
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
maxSubPop = 0.
maxSubPopii = 0
maxSubPopj = 0
for ii,j in itertools.product(np.arange(len(comp.Rp_hi)),np.arange(len(comp.L_lo[0,:]))):
    if np.nanmax(comp.Cpdf_hs[ii,j]) > maxSubPop:
        maxSubPop = np.nanmax(comp.Cpdf_hs[ii,j])
        maxSubPopii = ii
        maxSubPopj = j
tmpH = comp.Cpdf_hs[maxSubPopii,maxSubPopj].copy()
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
    #DELETE axij[ii,j] = plt.subplot(gs[5+j+ii*len(comp.L_lo[0,:])]) #old mapping
    axij[ii,j] = plt.subplot(gs[5+j+(len(comp.Rp_hi)-1-ii)*len(comp.L_lo[0,:])])#5 from cbar, j is 
    axij[ii,j].contourf(comp.xnew,comp.ynew,comp.Cpdf_hs[ii,j],cmap='jet', levels=levels, norm = LogNorm())
    axij[ii,j].set_xlim([xmin,xmax])
    axij[ii,j].set_ylim([ymin,ymax])
    axij[ii,j].text(18.5,45,'(i:' + str(ii) + ',j:' + str(j) + ')', weight='bold')
    if j != 0:
        axij[ii,j].get_yaxis().set_visible(False)
    if ii != 0:#len(comp.Rp_hi)-1:
        axij[ii,j].get_xaxis().set_visible(False)
    axij[ii,j].text(0.5,45, comp.type_names[ii,j], weight='bold')
    #Add total Count per grid
    axij[ii,j].text(16,41, "{:.2e}".format(comp.count_hs[ii,j]),weight='bold')

#Temporary For coloring purposes, pop hist
plt.close(9000)
figpop = plt.figure(num=9000)
ax1= plt.gca()
cax = ax1.contourf(comp.xnew, comp.ynew, comp.Cpdf_hs[maxSubPopii,maxSubPopj], extent=[xmin, xmax, ymin, ymax], cmap='jet', levels=levels, norm = LogNorm())
CS4 = ax1.contour(cax, colors=('k',), linewidths=(1,), origin='lower', levels=levels, norm = LogNorm())
ax1.set_ylim([ymin,ymax])
ax1.set_xlim([xmin,xmax])

#Add Colorbar, and pop dist
cbar = fig2.colorbar(cax, cax=axCBAR, orientation='horizontal')#pad=0.05,
plt.rcParams['axes.titlepad']=-10
axCBAR.set_xlabel('Joint Probability Density: Planet Sub-types', weight='bold', labelpad=-55)
axCBAR.tick_params(axis='x',direction='in',labeltop=True,labelbottom=False)#'off'
cbar.add_lines(CS4)
plt.title('SAG13 Population JPDF', weight='bold')
plt.xlabel('Luminosity Scaled Planet-star Separation ' + r'$(s/\sqrt{L})$' + ' in AU', weight='bold')
plt.ylabel('Planet-star Difference in Magnitude', weight='bold')
plt.text(20,48,"{:.2e}".format(comp.count_pop),weight='bold')

#Add Labels
fig2.text(0.5, 0.075, 'Luminosity Scaled Planet-star Separation ' + r'$(s/\sqrt{L})$' + ' in AU', ha='center', weight='bold')
fig2.text(0.09, 0.5, 'Planet-star Difference in Magnitude', va='center', rotation='vertical', weight='bold')

plt.show(block=False)

plt.figure(fig2.number)
fname = 'JPDFsubtype2_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=300)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=300)
#########################################################################################






#### Plot Kopparapu Grid###################################################
plt.close(88888)
figGrid = plt.figure(num=88888)

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
plt.figure(figGrid.number)
fname = 'KopGrid_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=300)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=300)
#############################################################################


#### Checking Classification Method
import astropy.units as u
Rp = np.asarray([0.6])*u.earthRad
a = np.asarray([0.75])*u.AU
e = np.asarray([0.])
starInds = np.asarray([0])
TL = sim.TargetList
L_plan = TL.L[starInds]/a.value**2.
out = comp.classifyPlanets(Rp,TL,starInds,a,e)
out2 = comp.classifyPlanet(Rp[0].value,TL,starInds[0],a[0].value,e[0])

bini = np.zeros(len(e),dtype=int) + len(comp.Rp_hi)
print(bini)
for ind in np.arange(len(comp.Rp_hi)):
    bini -= np.asarray(Rp.value<comp.Rp_hi[ind],dtype=int)*1
    print("bini: " + str(bini) + " Rp_hi: " + str(comp.Rp_hi[ind]) + " :" + str(Rp.value<comp.Rp_hi[ind]) + " :" + str(np.asarray(Rp.value<comp.Rp_hi[ind])*1))
comp.Rp_hi[bini]


#Find Luminosity Ranges for the Given Rp
L_lo1 = comp.L_lo[bini] # lower bin range of luminosity
L_lo2 = comp.L_lo[bini+1] # lower bin range of luminosity
L_hi1 = comp.L_hi[bini] # upper bin range of luminosity
L_hi2 = comp.L_hi[bini+1] # upper bin range of luminosity        
k1 = (L_lo2 - L_lo1)
k2 = (comp.Rp_hi[bini] - comp.Rp_lo[bini])
k3 = (Rp.value - comp.Rp_lo[bini])
k4 = k1/k2[:,np.newaxis]
L_lo = k4*k3[:,np.newaxis] + L_lo1
#Find Planet Stellar Flux range
binj = np.zeros(len(e),dtype=int)-1
for ind in np.arange(len(L_lo[0,:])):
    binj += np.asarray(L_plan<L_lo[:,ind])*1
    print("binj: " + str(binj) + " L_lo[:,ind]: " + str(L_lo[:,ind]) + "")


print(out)
print(out2)




#### Plot Distribution of Stellar Luminosity
plt.close(6666666655555)
figHist = plt.figure(num=6666666655555)
plt.hist(TL.L[TL.L<100],bins=20,color='black')
plt.ylabel('Frequency',weight='bold')
plt.xlabel('Stellar Luminosities',weight='bold')#Double Check
plt.show(block=False)
plt.figure(figHist.number)
fname = 'LumHist_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=300)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=300)


