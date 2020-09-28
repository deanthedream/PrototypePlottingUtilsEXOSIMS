# Historical Nation Longevity

import csv
import matplotlib.pyplot as plt
import numpy as np
import os

PPoutpath = './'

nations = list()
with open('NationLongevity.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        tmpList = ' '.join(row).split(',')    
        nations.append(tmpList)

lenNations = np.zeros(len(nations))
LongevityNations = np.zeros(len(nations))
for i in np.arange(len(nations)):
    lenNations[i] = len(nations[i])
    LongevityNations[i] = nations[i][5]

plt.figure(num=1)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')

plt.hist(LongevityNations,color='black',bins=20)
plt.yscale('log')
plt.ylabel('Frequency',weight='bold')
plt.xlabel('Nation Longevity (years)', weight='bold')
plt.xlim([0,1.1*np.max(LongevityNations)])
plt.show(block=False)
plt.gcf().canvas.draw()
fname = 'NationLongevityHistogram'
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=300)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=300)

