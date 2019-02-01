"""
Kormagorov-Smirnov 2 sample Testing
Determines whether two simulation's results are statistically unique

If the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis
that the distributions of the two samples are the same

Written By: Dean Keithly
Written on: 1/22/2019
"""

from EXOSIMS.util.read_ipcluster_ensemble import gen_summary
import numpy as np
import math
from scipy.stats import ks_2samp
from scipy.stats import anderson


PPoutpath1 = '/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3momaxC/WFIRSTcycle6core_CKL2_PPKL2'
folder1 = '/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3momaxC/WFIRSTcycle6core_CKL2_PPKL2'
PPoutpath2 = '/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3momaxC/WFIRSTcycle6core_CSAG13_PPKL2'
folder2 = '/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3momaxC/WFIRSTcycle6core_CSAG13_PPKL2'

#Generate Summaries
res1 = gen_summary(folder1)
res2 = gen_summary(folder2)


rcounts = []
res = [res1['detected'], res2['detected']]
for el in res:
    rcounts.append(np.array([np.unique(r).size for r in el]).astype(float))#unique detections


out = ks_2samp(rcounts[0], rcounts[1])
#out2 = anderson()


## Quick proof distributions are different
# X0 = np.random.normal(loc=1.,scale=1.,size=(1000))
# Y0 = np.random.normal(loc=2.,scale=1.,size=(1000))
# Y1 = np.random.normal(loc=20.,scale=1.,size=(1000))
# #compare X0 to Y0 Should be very close
# out0 = ks_2samp(X0,Y0)
# #compare X0 to Y1 Should be totally separated
# out1 = ks_2samp(X0,Y1)
# print('because p from out0 and p from out1 are so small, we know smaller p means the distributions are more unique')
