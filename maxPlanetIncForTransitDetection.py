# Determining maximum planet inclination for transit as a function of AU

import numpy as np
import matplotlib.pyplot as plt
import datetime
import re
import os

folder='./'
PPoutpath = './'

R_sun_AU = 0.00465047 #AU radius of the sun in AU

a = (np.arange(100)+1.)/105.*39.48
i_max = np.arctan2(a,R_sun_AU)*180./np.pi



### Planets Inclination
#https://en.wikipedia.org/wiki/Orbital_inclination
# Mercury  6.34°
# Venus    2.19°
# Earth    1.57°
# Mars     1.67°
# Jupiter  0.32°
# Saturn   0.93°
# Uranus   1.02°
# Neptune  0.72°
# Pluto    15.55°
pIncs = [6.34, 2.19, 1.57, 1.67, 0.32, 0.93, 1.02, 0.72, 15.55]
solarSystemAU = [0.39, 0.72, 1.00, 1.52, 5.20, 9.54, 19.2, 30.1, 39.4 ]

plt.figure(12980124)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
plt.semilogy(a,90.-i_max, color='k')
plt.ylabel('Maximum Inclination (deg)', weight='bold')
plt.xlabel('Distance form Sun (AU)', weight='bold')
plt.xlim([0., 1.05*np.max(solarSystemAU)])
plt.ylim([0.01, 1.05*np.max(i_max)])
plt.show(block=False)
date = datetime.datetime.now().strftime("%d_%b_%Y_%H_%M_%S_%f")
#date = str(datetime.datetime.now(),'utf-8')#Python3.5 unicode(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'maxIncForTransit_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))


plt.scatter(solarSystemAU, pIncs, color='blue')
plt.ylim([0.005, 1.05*np.max(pIncs)])
plt.show(block=False)
fname = 'maxIncForTransitwPlanets_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'))
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))
