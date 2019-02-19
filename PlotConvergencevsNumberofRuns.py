""" Plot Convergence vs Number of Runs

Written by: Dean Keithly on 5/29/2018
"""

try:
    import cPickle as pickle
except:
    import pickle
import os
import numpy as np
from pylab import *
from numpy import nan
import matplotlib.pyplot as plt
import argparse
import json


runDir = '/home/dean/Documents/SIOSlab/Dean22May18RS09CXXfZ01OB01PP01SU01/'
saveFolder = '/home/dean/Documents/SIOSlab/SPIE2018Journal/'


#Given Filepath for pklfile, Plot a pkl from each testrun in subdir
pklPaths = list()
pklfname = list()


#Look for all directories in specified path with structured folder name
dirs = runDir

pklFiles = [myFileName for myFileName in os.listdir(dirs) if 'run' in myFileName and '.pkl' in myFileName]  # Get names of all pkl files in path
for i in np.arange(len(pklFiles)):
    pklPaths.append(dirs + pklFiles[i])  # append a random pkl file to path



#Iterate over all pkl files
meanNumDets = list()
for cnt in np.arange(len(pklPaths)):
    try:
        with open(pklPaths[cnt], 'rb') as f:#load from cache
            DRM = pickle.load(f)
    except:
        print('Failed to open pklfile %s'%pklPaths[cnt])
        pass

    #Calculate meanNumDets #raw detections, not unique detections
    AllDetsInPklFile = [(DRM['DRM'][i]['det_status'] == 1).tolist().count(True) for i in np.arange(len(DRM['DRM']))]
    meanNumDetsTMP = sum(AllDetsInPklFile)

    
    #Append to list
    if cnt == 0:
        meanNumDets.append(float(meanNumDetsTMP))
    else:
        meanNumDets.append((meanNumDets[cnt-1]*float(cnt-1+1) + meanNumDetsTMP)/float(cnt+1))

    print "%d/%d  %d %f"%(cnt,len(pklPaths), meanNumDetsTMP, meanNumDets[cnt])

plt.close('all')
fig = plt.figure(8000)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold') 
plt.#rcParams['axes.titlepad']=-50
plt.plot(abs(np.asarray(meanNumDets) - meanNumDets[:-1]), color='purple')

plt.plot([100,100],[abs(np.asarray(meanNumDets[99]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))], linewidth=1, color='k')
gca().text(90,5,r"$\mu_{det_{100}}=$" + ' %2.1f'%(meanNumDets[99]/meanNumDets[-1]*100.) + '%', rotation=45)
plt.plot([1000,1000],[abs(np.asarray(meanNumDets[999]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))], linewidth=1, color='k')
gca().text(900,5,r"$\mu_{det_{1000}}=$" + ' %2.1f'%(meanNumDets[999]/meanNumDets[-1]*100.) + '%', rotation=45)
#plot([10000,10000],[abs(np.asarray(meanNumDets[-1]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))], linewidth=1, color='k')
#ADD LABEL
plt.xscale('log')
#yscale('log')

plt.plot([82,82],[abs(np.asarray(meanNumDets[82]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))],linestyle='--', linewidth=1, color='k')
gca().text(60,4.85,r"$\mu_{det_{82}}=$" + ' %2.0f'%(meanNumDets[81]/meanNumDets[-1]*100.) + '%', rotation=45)
plt.plot([132,132],[abs(np.asarray(meanNumDets[132]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))],linestyle='--', linewidth=1, color='k')
gca().text(130,4.85,r"$\mu_{det_{132}}=$" + ' %2.0f'%(meanNumDets[131]/meanNumDets[-1]*100.) + '%', rotation=45)
plt.plot([363,363],[abs(np.asarray(meanNumDets[363]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))],linestyle='--', linewidth=1, color='k')
gca().text(310,4.85,r"$\mu_{det_{363}}=$" + ' %2.0f'%(meanNumDets[362]/meanNumDets[-1]*100.) + '%', rotation=45)
plt.plot([1550,1550],[abs(np.asarray(meanNumDets[1550]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))],linestyle='--', linewidth=1, color='k')
gca().text(1400,5,r"$\mu_{det_{1550}}=$" + ' %2.1f'%(meanNumDets[1555]/meanNumDets[-1]*100.) + '%', rotation=45)
#gca().text(9000,4,r"$\mu_{det_{1000}}=$" + ' %2.1f'%(meanNumDets[999]/meanNumDets[-1]*100.))

plt.xlim([1,lwn(meanNumDets)])
plt.ylim([0,max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))])
plt.ylabel("Mean # of Detections Error\n$|\mu_{det_i}-\mu_{det_{10000}}|$", weight='bold')
plt.xlabel("# of Simulations, i", weight='bold')
#tight_layout()
#margins(1)
gcf().subplots_adjust(top=0.75)
plt.show(block=False)

plt.savefig(saveFolder + 'meanNumDetectionDiffConvergence' + '.png')
plt.savefig(saveFolder + 'meanNumDetectionDiffConvergence' + '.svg')
plt.savefig(saveFolder + 'meanNumDetectionDiffConvergence' + '.eps')


fig = plt.figure(8001)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold') 
plt.plot(meanNumDets, color='purple')
plt.xscale('log')
plt.xlim([1,len(meanNumDets)])
plt.ylim([0,meanNumDets[-1]*1.05])
plt.ylabel("Mean # of Detections", weight='bold')
plt.xlabel("# of Simulations, i", weight='bold')
plt.show(block=False)
plt.savefig(saveFolder + 'meanNumDetectionConvergence' + '.png')
plt.savefig(saveFolder + 'meanNumDetectionConvergence' + '.svg')
plt.savefig(saveFolder + 'meanNumDetectionConvergence' + '.eps')


fig = plt.figure(8002)
#ax = fig.add_subplot(111)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold') 
plt.plot(np.asarray(meanNumDets)/meanNumDets[-1]*100., color='purple')
plt.xscale('log')
plt.xlim([1,1e4])
plt.ylim([0,100*1.05])
plt.ylabel(r"Percentage of $\mu_{det_{10000}}$, $\frac{\mu_{det_i}}{\mu_{det_{10000}}} \times 100$", weight='bold')
plt.xlabel("# of Simulations, i", weight='bold')
plt.show(block=False)
plt.savefig(saveFolder + 'percentErrorFromMeanConvergence' + '.png')
plt.savefig(saveFolder + 'percentErrorFromMeanConvergence' + '.svg')
plt.savefig(saveFolder + 'percentErrorFromMeanConvergence' + '.eps')



# $\mu_1000=9.82700$ which is 99.27\% of the $\mu_10000$.
# $\mu_10000=9.898799$.
# $\mu_100=9.100$ which is 91.9\% of the $\mu_10000$. 
# 90\% of the $\mu_10000$ is achieved at 82sims.
# 95\% of the $\mu_10000$ is achieved at 132sims.
# 99\% of the $\mu_10000$ is achieved at 363sims.
# 99.5\% of the $\mu_10000$ is achieved at 1550sims.