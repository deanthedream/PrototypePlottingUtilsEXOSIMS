#Plot fZ for 1 year for 1 star, fZ for all stars over 1 year,
import numpy as np
import scipy
from scipy.optimize import fmin
import timeit
import csv
import os.path
import datetime
import hashlib
import inspect
try:
    import cPickle as pickle
except:
    import pickle
from pylab import *

cachefname = '/home/dean/Documents/exosims/EXOSIMS/SurveySimulation/KeplerLike2ForecasterBrownCompletenessTargetListNemati8aad7ceb4e61a4d387941f4d6802a699.starkfZ'

with open(cachefname, 'rb') as f:#load from cache
    tmpfZ = pickle.load(f)

dt = 365.25/len(np.arange(1000))
time = [j*dt for j in range(1000)]
#fZ = np.zeros([sInds.shape[0], len(resolution)])
#dt = 365.25/len(resolution)*u.d
#time = 365.25/resolution

magfZ = 2.5*np.log10(tmpfZ)
i = 0
fig = plt.figure()
for i in np.arange(651):
	plt.plot(time,2.5*np.log10(tmpfZ[i][:]))
#plt.plot(time,2.5*np.log10(tmpfZ[1][:]),color='r',label='fZmin')
#plt.plot(time,2.5*np.log10(tmpfZ[2][:]),color='g',label='fZmin')
#plt.plot(time,2.5*np.log10(tmpfZ[3][:]),color='b',label='fZmin')
#plt.plot(time,2.5*np.log10(tmpfZ[4][:]),color='m',label='fZmin')

#plt.xscale('log')
#plt.show(block=False)
plt.ylabel('fZ in magfZ')
plt.xlabel('Time (days)')
plt.title('sInd=' + str(i))
#plt.legend(loc='lower right')
plt.show(block=False)

#Count number of stars where magfZ > -18
magfZ18 = 0
for i in np.arange(651):
	if max(magfZ[i]) > -18:
		magfZ18 += 1
print('magfZ18')
print(magfZ18)

#Find fZmax-fZmin for each star
maxDeltamagfZ = np.arange(651,dtype=np.float)
for i in np.arange(651):
	maxDeltamagfZ[i] = max(magfZ[i])-min(magfZ[i])
print('MaxDeltamagfZ')
print(maxDeltamagfZ)




#foldername = os.path.normpath(os.path.expandvars('$HOME/Pictures/Compfitfigs'))
#fname = 'CvsTforfZ' + str(i) + '.png'
#figPathName = os.path.join(foldername,fname)
#fig.savefig(figPathName)
#plt.close()	


print(saltyBurrito)