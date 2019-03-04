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
import scipy.stats as st


runDir = '/home/dean/Documents/SIOSlab/EXOSIMSres/Dean22May18RS09CXXfZ01OB01PP01SU01/'
saveFolder = './'#'/home/dean/Documents/SIOSlab/SPIE2018Journal/'


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
numDetsInSim = list()
incVariance = list()
ci90u = list()
ci95u = list()
ci99u = list()
ci99p5u = list()
ci90l = list()
ci95l = list()
ci99l = list()
ci99p5l = list()
#CIpmlZ = list() # Incremental plus/minus to add to mean for confidence interval, CIpmlZ*Z = CI
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
    numDetsInSim.append(meanNumDetsTMP)

    if len(numDetsInSim) > 10:
        u90, l90 = st.t.interval(0.90, len(numDetsInSim)-1, loc=np.mean(numDetsInSim), scale=st.sem(numDetsInSim))
        u95, l95 = st.t.interval(0.95, len(numDetsInSim)-1, loc=np.mean(numDetsInSim), scale=st.sem(numDetsInSim))
        u99, l99 = st.t.interval(0.99, len(numDetsInSim)-1, loc=np.mean(numDetsInSim), scale=st.sem(numDetsInSim))
        u995, l995 = st.t.interval(0.995, len(numDetsInSim)-1, loc=np.mean(numDetsInSim), scale=st.sem(numDetsInSim))
    else:
        u90, l90, u95, l95, u99, l99, u995, l995 = (0., 0., 0., 0., 0., 0., 0., 0.)
    ci90u.append(u90)
    ci95u.append(u95)
    ci99u.append(u99)
    ci99p5u.append(u995)
    ci90l.append(l90)
    ci95l.append(l95)
    ci99l.append(l99)
    ci99p5l.append(l995)
    
    #Append to list and incrementally update the new mean
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
#rcParams['axes.titlepad']=-50
plt.plot(abs(np.asarray(meanNumDets) - meanNumDets[-1]), color='purple', zorder=1)

DetsDeltas = abs(np.asarray(meanNumDets) - meanNumDets[-1])
#http://circuit.ucsd.edu/~yhk/ece250-win17/pdfs/lect07.pdf
#Example 7.7
# varMU = np.var(numDetsInSim)
# N90 = varMU/(1.-0.90)
# N95 = varMU/(1.-0.95)
# N99 = varMU/(1.-0.99)
# N99p5 = varMU/(1.-0.995)

# inds90 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x]/meanNumDets[-1] < (1.-0.90) and i > 50][0]
# inds95 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x]/meanNumDets[-1] < (1.-0.95) and i > 50][0]
# inds99 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x]/meanNumDets[-1] < (1.-0.99) and i > 50][0]
# inds99p5 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x]/meanNumDets[-1] < (1.-0.995) and i > 50][0]

inds90 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x] < ci90u[-1] and i > 50][0]
inds95 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x] < ci95u[-1] and i > 50][0]
inds99 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x] < ci99u[-1] and i > 50][0]
inds99p5 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x] < ci99p5u[-1] and i > 50][0]


maxDetsDelta = np.max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))
plt.plot([100,100],[abs(np.asarray(meanNumDets[99]) - meanNumDets[-1]),maxDetsDelta], linewidth=1, color='k')
plt.text(90,1.3*maxDetsDelta,r"$\mu_{det_{100\ }}=$" + ' %2.1f'%(meanNumDets[99]/meanNumDets[-1]*100.) + '%', rotation=45)
plt.plot([1000,1000],[abs(np.asarray(meanNumDets[999]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))], linewidth=1, color='k')
plt.text(900,1.3*maxDetsDelta,r"$\mu_{det_{1000}}=$" + ' %2.1f'%(meanNumDets[999]/meanNumDets[-1]*100.) + '%', rotation=45)
#plot([10000,10000],[abs(np.asarray(meanNumDets[-1]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))], linewidth=1, color='k')
#ADD LABEL
plt.xscale('log')
#yscale('log')

plt.plot([inds90,inds90],[abs(np.asarray(meanNumDets[inds90]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))],linestyle='--', linewidth=1, color='k')
plt.text(inds90-12,1.3*maxDetsDelta,r"$\mu_{det_{82\ \ }}=$" + ' %2.1f'%(meanNumDets[inds90]/meanNumDets[-1]*100.) + '%', rotation=45)
plt.plot([inds95,inds95],[abs(np.asarray(meanNumDets[inds95]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))],linestyle='--', linewidth=1, color='k')
plt.text(inds95-12,1.3*maxDetsDelta,r"$\mu_{det_{132\ }}=$" + ' %2.1f'%(meanNumDets[inds95]/meanNumDets[-1]*100.) + '%', rotation=45)
plt.plot([inds99,inds99],[abs(np.asarray(meanNumDets[inds99]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))],linestyle='--', linewidth=1, color='k')
plt.text(inds99-12,1.3*maxDetsDelta,r"$\mu_{det_{363\ }}=$" + ' %2.1f'%(meanNumDets[inds99]/meanNumDets[-1]*100.) + '%', rotation=45)
plt.plot([inds99p5,inds99p5],[abs(np.asarray(meanNumDets[inds99p5]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))],linestyle='--', linewidth=1, color='k')
plt.text(inds99p5-12,1.3*maxDetsDelta,r"$\mu_{det_{1550}}=$" + ' %2.1f'%(meanNumDets[1555]/meanNumDets[-1]*100.) + '%', rotation=45)
#gca().text(9000,4,r"$\mu_{det_{1000}}=$" + ' %2.1f'%(meanNumDets[999]/meanNumDets[-1]*100.))

plt.xlim([1,len(meanNumDets)])
plt.ylim([0,max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))])
plt.ylabel("Mean # of Detections Error\n$|\mu_{det_i}-\mu_{det_{10000}}|$", weight='bold')
plt.xlabel("# of Simulations, i", weight='bold')
#tight_layout()
#margins(1)
plt.gcf().subplots_adjust(top=0.75, left=0.15)
plt.show(block=False)

plt.savefig(saveFolder + 'meanNumDetectionDiffConvergence' + '.png')
plt.savefig(saveFolder + 'meanNumDetectionDiffConvergence' + '.svg')
plt.savefig(saveFolder + 'meanNumDetectionDiffConvergence' + '.eps')



fig = plt.figure(8001)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
plt.plot(meanNumDets, color='purple', zorder=10)
plt.plot(np.arange(len(ci90u)), ci90u, linestyle={0,(1,5)}, color='black', label='90% CI')
plt.plot(np.arange(len(ci90u)), ci95u, linestyle={0,(5,10)}, color='black', label='95% CI')
plt.plot(np.arange(len(ci90u)), ci99u, linestyle={0,(5,5)}, color='black', label='99% CI')
plt.plot(np.arange(len(ci90u)), ci99p5u, linestyle={0,(5,1)}, color='black', label='99.5% CI')
plt.plot(np.arange(len(ci90u)), ci90l, linestyle={0,(1,5)}, color='black')
plt.plot(np.arange(len(ci90u)), ci95l, linestyle={0,(5,10)}, color='black')
plt.plot(np.arange(len(ci90u)), ci99l, linestyle={0,(5,5)}, color='black')
plt.plot(np.arange(len(ci90u)), ci99p5l, linestyle={0,(5,1)}, color='black')
plt.plot([0.,len(np.asarray(meanNumDets))],[meanNumDets[-1],meanNumDets[-1]],linestyle='--',color='black', zorder=1)
plt.xscale('log')
plt.xlim([1,len(meanNumDets)])
plt.ylim([0,meanNumDets[-1]*1.05])
plt.ylabel("Mean # of Unique Detections", weight='bold')
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
plt.plot(np.asarray(meanNumDets)/meanNumDets[-1]*100., color='purple', zorder=10)
plt.plot(ci90u/meanNumDets[-1]*100., linestyle=(0,(1,5)),color='black', label='90% CI')
plt.plot(ci95u/meanNumDets[-1]*100., linestyle=(0,(5,10)),color='black', label='95% CI')
plt.plot(ci99u/meanNumDets[-1]*100., linestyle=(0,(5,5)),color='black', label='99% CI')
plt.plot(ci99p5u/meanNumDets[-1]*100., linestyle=(0,(5,1)),color='black', label='99.5% CI')
plt.plot(ci90l/meanNumDets[-1]*100., linestyle=(0,(1,5)),color='black')
plt.plot(ci95l/meanNumDets[-1]*100., linestyle=(0,(5,10)),color='black')
plt.plot(ci99l/meanNumDets[-1]*100., linestyle=(0,(5,5)),color='black')
plt.plot(ci99p5l/meanNumDets[-1]*100., linestyle=(0,(5,1)),color='black')
plt.plot([0.,len(np.asarray(meanNumDets))],[100.,100.],linestyle='--',color='black', zorder=1)
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