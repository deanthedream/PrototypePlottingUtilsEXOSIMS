"""
Written By: Dean Keithly
"""

from EXOSIMS.util.read_ipcluster_ensemble import gen_summary
from pylab import *
import numpy as np
from cycler import cycler
import math

#5/14/2018 SLSQP Scheduler static Yield vs Mission Length
# pathRuns = '/home/dean/Documents/SIOSlab/'
# t = ['Dean6May18RS09CXXfZ01OB08PP01SU01',\
# 'Dean6May18RS09CXXfZ01OB09PP01SU01',\
# 'Dean6May18RS09CXXfZ01OB10PP01SU01',\
# 'Dean6May18RS09CXXfZ01OB11PP01SU01',\
# 'Dean6May18RS09CXXfZ01OB12PP01SU01',\
# 'Dean6May18RS09CXXfZ01OB13PP01SU01',\
# 'Dean6May18RS09CXXfZ01OB14PP01SU01',\
# 'Dean6May18RS09CXXfZ01OB15PP01SU01',\
# 'Dean6May18RS09CXXfZ01OB16PP01SU01',\
# 'Dean6May18RS09CXXfZ01OB17PP01SU01',\
# 'Dean6May18RS09CXXfZ01OB18PP01SU01',\
# 'Dean2May18RS09CXXfZ01OB01PP01SU01',\
# 'Dean6May18RS09CXXfZ01OB19PP01SU01',\
# 'Dean6May18RS09CXXfZ01OB20PP01SU01',\
# 'Dean6May18RS09CXXfZ01OB21PP01SU01']
# res1 = gen_summary(pathRuns + t[0])
# res2 = gen_summary(pathRuns + t[1])
# res3 = gen_summary(pathRuns + t[2])
# res4 = gen_summary(pathRuns + t[3])
# res5 = gen_summary(pathRuns + t[4])
# res6 = gen_summary(pathRuns + t[5])
# res7 = gen_summary(pathRuns + t[6])
# res8 = gen_summary(pathRuns + t[7])
# res9 = gen_summary(pathRuns + t[8])
# res10 = gen_summary(pathRuns + t[9])
# res11 = gen_summary(pathRuns + t[10])
# res12 = gen_summary(pathRuns + t[11])
# res13 = gen_summary(pathRuns + t[12])
# res14 = gen_summary(pathRuns + t[13])
# res15 = gen_summary(pathRuns + t[14])

def calcStatistics(el):
    """
    Args:
        el (gen_summary output)
    Returns:

    """
    tmp = np.array([np.unique(r).size for r in el]).astype(float)
    if len(tmp) == 0:
        return  np.asarray(0)
    return tmp

pathRuns = '/home/dean/Documents/SIOSlab/EXOSIMSres/HabExTimeSweep_HabEx_CSAG13_PPSAG13/'
t = ['auto_2019_05_18_13_50__HabExTimeSweep_HabEx_CSAG13_PPSAG13_0',\
'auto_2019_05_18_13_50__HabExTimeSweep_HabEx_CSAG13_PPSAG13_1',\
'auto_2019_05_18_13_50__HabExTimeSweep_HabEx_CSAG13_PPSAG13_2',\
'auto_2019_05_18_13_50__HabExTimeSweep_HabEx_CSAG13_PPSAG13_3',\
'auto_2019_05_18_13_50__HabExTimeSweep_HabEx_CSAG13_PPSAG13_4',\
'auto_2019_05_18_13_50__HabExTimeSweep_HabEx_CSAG13_PPSAG13_5',\
'auto_2019_05_18_13_50__HabExTimeSweep_HabEx_CSAG13_PPSAG13_6',\
'auto_2019_05_18_13_50__HabExTimeSweep_HabEx_CSAG13_PPSAG13_7',\
'auto_2019_05_18_13_50__HabExTimeSweep_HabEx_CSAG13_PPSAG13_8',\
'auto_2019_05_18_13_50__HabExTimeSweep_HabEx_CSAG13_PPSAG13_9',\
'auto_2019_05_18_13_50__HabExTimeSweep_HabEx_CSAG13_PPSAG13_10']
# res1 = gen_summary(pathRuns + t[0])
# res2 = gen_summary(pathRuns + t[1])
# res3 = gen_summary(pathRuns + t[2])
# res4 = gen_summary(pathRuns + t[3])
# res5 = gen_summary(pathRuns + t[4])
# res6 = gen_summary(pathRuns + t[5])
# res7 = gen_summary(pathRuns + t[6])
# res8 = gen_summary(pathRuns + t[7])
# res9 = gen_summary(pathRuns + t[8])
# res10 = gen_summary(pathRuns + t[9])

rcounts = list()
rcounts.append(calcStatistics(gen_summary(pathRuns + t[0])['detected']))
rcounts.append(calcStatistics(gen_summary(pathRuns + t[1])['detected']))
rcounts.append(calcStatistics(gen_summary(pathRuns + t[2])['detected']))
rcounts.append(calcStatistics(gen_summary(pathRuns + t[3])['detected']))
rcounts.append(calcStatistics(gen_summary(pathRuns + t[4])['detected']))
rcounts.append(calcStatistics(gen_summary(pathRuns + t[5])['detected']))
rcounts.append(calcStatistics(gen_summary(pathRuns + t[6])['detected']))
rcounts.append(calcStatistics(gen_summary(pathRuns + t[7])['detected']))
rcounts.append(calcStatistics(gen_summary(pathRuns + t[8])['detected']))
rcounts.append(calcStatistics(gen_summary(pathRuns + t[9])['detected']))
rcounts.append(calcStatistics(gen_summary(pathRuns + t[10])['detected']))

# 1mo, 0.08333yr=08
# 2mo, 0.1666yr=09
# 3mo, 0.25yr=10
# 4mo, 0.33333yr=11
# 5mo, 0.416666yr=12
# 6mo, 0.5yr=13
# 7mo, 0.583333yr=14
# 8mo, 0.6666666yr=15
# 9mo, 0.75yr=16
# 10mo, 0.83333yr=17
# 11mo, 0.91666yr=18
# 13mo, 1.08333yr=19
# 14mo, 1.166666yr=20
# 15mo, 1.25yr=21
months = np.arange(15)+1


# #Clump
# rcounts = []
# #el = res1['detected']
# res = [res1['detected'], res2['detected'], res3['detected'], res4['detected'], res5['detected'],\
#     res6['detected'], res7['detected'], res8['detected'], res9['detected'], res10['detected']]#,\
#     #res11['detected'], res12['detected'], res13['detected'], res14['detected'], res15['detected']]
# for el in res:
#     rcounts.append(np.array([np.unique(r).size for r in el]).astype(float))#unique detections
#     #rcounts.append(np.array([len(r) for r in el]))

#calculate mean detections, standard deviations, and percentiles
meanUniqueDetections = list()
fifthPercentile = list()
twentyfifthPercentile = list()
fiftiethPercentile = list()
seventyfifthPercentile = list()
ninetiethPercentile = list()
nintyfifthPercentile = list()
minNumDetected = list()
percentAtMinimum = list()
maxNumDetected = list()
stdUniqueDetections = list()
for el in rcounts:
    meanUniqueDetections.append(np.mean(el))
    stdUniqueDetections.append(np.std(el))
    fifthPercentile.append(np.percentile(el,5))
    twentyfifthPercentile.append(np.percentile(el,25))
    fiftiethPercentile.append(np.percentile(el,50))
    seventyfifthPercentile.append(np.percentile(el,75))
    ninetiethPercentile.append(np.percentile(el,90))
    nintyfifthPercentile.append(np.percentile(el,95))
    minNumDetected.append(np.min(el))
    percentAtMinimum.append(float(el.tolist().count(np.min(el)))/len(el))
    maxNumDetected.append(np.max(el))
print meanUniqueDetections

fig = figure(1, figsize=(8.5,4.5))
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rc('axes',prop_cycle=(cycler('color',['purple'])))#,'blue','black','purple'])))
rcParams['axes.linewidth']=2
rc('font',weight='bold') 

B = boxplot(np.transpose(asarray(rcounts)), widths=0.15, positions=list(np.round(10.*np.asarray([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.195, 0.225, 0.25, 0.3]),decimals=2)), sym='', whis= 1000.)
xlabel('Total Mission Time (years)', weight='bold')
ylabel('Unique Detections', weight='bold')
axes = gca()
#axes.set_xlim([xmin,xmax])
maxCount = np.max([np.max(rcounts[i]) for i in np.arange(len(rcounts))])
axes.set_ylim([0,1.1*maxCount])#np.amax(rcounts)])
show(block=False)

filename = 'UniqDetvsMissionLength'
#runPath = '/home/dean/Documents/SIOSlab/SPIE2018Journal/DELETE'
runPath = '/home/dean/Documents/SIOSlab/EXOSIMSres/HabExTimeSweep_HabEx_CSAG13_PPSAG13/'
savefig(runPath + filename + '.png', format='png', dpi=500)
savefig(runPath + filename + '.svg')
savefig(runPath + filename + '.eps', format='eps', dpi=500)
#scatter(months, meanUniqueDetections)

#How to get box plot quartiles means and such
[item.get_ydata() for item in B['boxes']]


#Calculate Percent of Mission Time Wasted
# key = 'tottime'
# res_detTime = [res1[key], res2[key], res3[key], res4[key], res5[key],\
#     res6[key], res7[key], res8[key], res9[key], res10[key]]#,\
    #res11[key], res12[key], res13[key], res14[key], res15[key]]

# key = 'tottime'
# res_detTime = [res1[key], res2[key], res3[key], res4[key], res5[key],\
#     res6[key], res7[key], res8[key], res9[key], res10[key],\
#     res11[key], res12[key], res13[key], res14[key], res15[key]]


#pklfiles = glob.glob(os.path.join(run_dir,'*.pkl'))


### Make Dmitry's violin plots
fig2 = figure(2, figsize=(8.5,4.5))
parts = violinplot(np.transpose(asarray(rcounts)), widths=0.15, positions=list(10.*np.asarray([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.195, 0.225, 0.25, 0.3])), showmeans=False, showmedians=False, showextrema=False)#, widths=0.75)
for pc in parts['bodies']:
    #pc.set_facecolor('#D43F3A')
    pc.set_facecolor('purple')
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rc('axes',prop_cycle=(cycler('color',['purple'])))#,'blue','black','purple'])))
rcParams['axes.linewidth']=2
rc('font',weight='bold') 
xlabel('Total Mission Time (years)', weight='bold')
ylabel('Unique Detections', weight='bold')
axes = gca()
#axes.set_xlim([xmin,xmax])
maxCount = np.max([np.max(rcounts[i]) for i in np.arange(len(rcounts))])
axes.set_ylim([0,1.1*maxCount])#np.amax(rcounts)])




inds = np.arange(len(rcounts))+1
scatter(list(10.*np.asarray([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.195, 0.225, 0.25, 0.3])), meanUniqueDetections, marker='o', color='k', s=30, zorder=3)
vlines(list(10.*np.asarray([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.195, 0.225, 0.25, 0.3])), minNumDetected, maxNumDetected, color='k', linestyle='-', lw=2)
vlines(list(10.*np.asarray([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.195, 0.225, 0.25, 0.3])), twentyfifthPercentile, seventyfifthPercentile, color='silver', linestyle='-', lw=5)

show(block=False)

filename = 'UniqDetvsMissionLengthVIOLIN'
#runPath = '/home/dean/Documents/SIOSlab/SPIE2018Journal/DELETE'
runPath = '/home/dean/Documents/SIOSlab/EXOSIMSres/HabExTimeSweep_HabEx_CSAG13_PPSAG13/'
savefig(runPath + filename + '.png', format='png', dpi=500)
savefig(runPath + filename + '.svg')
savefig(runPath + filename + '.eps', format='eps', dpi=500)
show(block=False)

