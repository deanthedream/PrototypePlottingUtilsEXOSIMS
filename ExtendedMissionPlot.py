#This represents depricated work from AAS 2018


import scipy
#from scipy.optimize import fmin
import timeit
import csv
import os.path
#import datetime
import hashlib
import inspect
from astropy.coordinates import SkyCoord
try:
    import cPickle as pickle
except:
    import pickle
import csv
from pylab import *
from numpy import nan
from scipy.optimize import curve_fit
from scipy.special import factorial

#data = np.loadtxt("/home/dean/Downloads/SumCompandTargListvsCIGTime.csv")
#data = np.loadtxt("/home/dean/Downloads/rawData1.csv")
tmp = list()
with open("/home/dean/Downloads/rawData1.csv", 'rb') as f:#
    for row in csv.reader(f):
        tmp.append(row)
tmpArray = np.asarray(tmp)
MLmo = np.asfarray(tmpArray[:,0],float)
MLmoRounded = np.asfarray(tmpArray[:,1],float)
sumC = np.asfarray(tmpArray[:,2],float)
TLraw = np.asfarray(tmpArray[:,3],float)
TLRounded = np.asfarray(tmpArray[:,4],float)
#NA = np.asfarray(tmpArray[:,5],float)
MLdays = np.asfarray(tmpArray[:,6],float)
sumTint = np.asfarray(tmpArray[:,7],float)

MLfig = plt.figure(10)
fig, ax1 = plt.subplots()
ax1.plot(MLmoRounded,sumC,color='k',marker='o',label='Overhead = 1days')
ax1.set_ylabel(r'$\sum{Completeness}$', color='k',weight='bold',fontsize=14)
ax1.tick_params('y', colors='k')
ax1.set_xlabel('CGI Time (months)',weight='bold')
new_xtick_locations = np.array([4,6,8,10,12,14,16,18,20,22,24,26,28])
ax1.set_xticks(new_xtick_locations)
ax2 = ax1.twinx()
ax2.plot(MLmoRounded,TLRounded,color='purple',marker='o',label='Targets in List')
ax2.set_ylabel('# Targets in Observation List', color='purple',weight='bold',fontsize=14)
ax2.tick_params('y', colors='purple')

#ax2.set_xlim(ax1.get_xlim())
#ax2.set_xticks(new_tick_locations)

#MLmoRounded - TLRounded*1
sumTint15 = (sumTint + TLRounded*1.5)/30
sumTint05 = (sumTint + TLRounded*0.5)/30
ax1.plot(sumTint15,sumC,color='k',marker='s',linestyle='--',alpha=0.25,label='Overhead = 1.5days')
ax1.plot(sumTint05,sumC,color='k',marker='v',linestyle='--',alpha=0.25,label='Overhead = 0.5days')
#ax2.set_xticklabels([])
#ax2.set_xlabel(r"Modified x-axis: $1/(1+X)$")

fig.tight_layout()
#out = plt.plot([mean(maxmagfZ2),mean(maxmagfZ2)],[0,25],color='r',label=r'$mean(magfZ_{max})$',linestyle='--')
rcParams['axes.linewidth']=2
rc('font',weight='bold') 
#plt.title('Histogram of '+r'$magfZ_{min}$'+' and '+r'$magfZ_{max}$',weight='bold',fontsize=12)
#plt.xlabel('magfZ',weight='bold',fontsize=12)
#plt.ylabel('# of Targets',weight='bold',fontsize=12)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
#plt.rc('axes',prop_cycle=(cycler('color',['red','blue','black','purple'])))
ax1.plot(0,0,color='purple',marker='o',label='#Targets in List')
ax1.legend(loc=4)
ax1.set_xlim([0.975*min(sumTint05),1.025*max(sumTint15)])
ax1.set_ylim([0.975*min(sumC),1.025*max(sumC)])
plt.show(block=False)