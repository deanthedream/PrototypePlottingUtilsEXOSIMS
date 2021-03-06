# #Run this script from EXOSIMS/util (otherwise read_ipcluster_ensemble wont load)
# #from read_ipcluster_ensemble import gen_summary
# from EXOSIMS.util.read_ipcluster_ensemble import gen_summary
# from pylab import *
# import numpy as np

# #import matplotlib.mlab as mlab
# #import matplotlib.pyplot as plt


# #dir = "/data2/extmount/EXOSIMSres/drk94_starkAYOoct9_2017"
# #dir1 = "/home/dean/starkAYOoct9_2017"
# #dir1 = "/home/dean/drk94_starkAYOstaticSchedulefZmin"
# dir1 = "/home/dean/drk94_starkAYOoct20_2017"
# out = gen_summary(dir1)
# print('Done starkAYO')
# dir2 = "/home/dean/wfirst_radDos0"
# out2 = gen_summary(dir2)
# print('Done radDos0')


# rc('axes', linewidth=2)

# #Histogram Planets Detected############################
# bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
# print(len(out['detected']))
# numPlanDetSTARK = np.zeros(len(out['detected']))
# for i in range(len(out['detected'])):
#     numPlanDetSTARK[i] = len(out['detected'][i])
# numPlanDetRADDOS = np.zeros(len(out2['detected']))
# for i in range(len(out2['detected'])):
# 	numPlanDetRADDOS[i] = len(out2['detected'][i])

# hist(numPlanDetSTARK, bins, alpha=0.5, label='stark AYO', color = "purple")
# hist(numPlanDetRADDOS, bins, alpha=0.5, label='radDos0', color = "green")
# meanNumPlanDetSTARK = mean(numPlanDetSTARK)
# meanNumPlanDetRADDOS = mean(numPlanDetRADDOS)
# plot([meanNumPlanDetSTARK, meanNumPlanDetSTARK], [0, 200], color = "purple")
# plot([meanNumPlanDetRADDOS, meanNumPlanDetRADDOS], [0, 200], color = "green")
# xlabel('# Planets Detected',fontsize=16,fontweight='bold')
# #pyplot.hist(x, bins, alpha=0.5, label='x')
# #pyplot.hist(y, bins, alpha=0.5, label='y')
# legend(loc='upper right')
# show(block=False)

# print('Mean num Plan Dets Stark ' + str(meanNumPlanDetSTARK))
# print('Std num Plan Dets Stark ' + str(std(numPlanDetSTARK)))
# print('Mean num Plan Dets RADDOS ' + str(meanNumPlanDetRADDOS))
# print('Std num Plan Dets RADDOS ' + str(std(numPlanDetRADDOS)))


# #Histogram sInds####################################
# bins = np.arange(700)
# sIndsSTARK = np.array([])#np.zeros(len(out['starinds']))
# for i in range(len(out['starinds'])):
#     sIndsSTARK = np.concatenate([sIndsSTARK, out['starinds'][i]])
# sIndsRADDOS = np.array([])#np.zeros(len(out2['starinds']))
# for i in range(len(out2['starinds'])):
# 	sIndsRADDOS = np.concatenate([sIndsRADDOS, out2['starinds'][i]])

# figure(figsize=(13,4))
# hist(sIndsSTARK, bins, alpha=0.5, label='stark AYO', color = "purple")
# hist(sIndsRADDOS, bins, alpha=0.5, label='radDos0', color = "green")

# axis([-1,652,0,300])
# xlabel('sInds Where Planets Detected',fontsize=14,fontweight='bold')
# #pyplot.hist(x, bins, alpha=0.5, label='x')
# #pyplot.hist(y, bins, alpha=0.5, label='y')
# legend(loc='upper right')
# tight_layout()
# show(block=False)


# #Histogram of fZ############################
# bins = np.arange(50)*10e-11
# fZsSTARK = np.array([])#np.zeros(len(out['starinds']))
# for i in range(len(out['fZs'])):
#     fZsSTARK = np.concatenate([fZsSTARK, out['fZs'][i]])
# fZsRADDOS = np.array([])#np.zeros(len(out2['starinds']))
# for i in range(len(out2['fZs'])):
# 	fZsRADDOS = np.concatenate([fZsRADDOS, out2['fZs'][i]])

# figure()
# hist(fZsSTARK, bins, alpha=0.5, label='stark AYO', color = "purple")
# hist(fZsRADDOS, bins, alpha=0.5, label='radDos0', color = "green")
# meanfZsSTARK = mean(fZsSTARK)
# stdfZsSTARK = std(fZsSTARK)
# meanfZsRADDOS = mean(fZsRADDOS)
# stdfZsRADDOS = std(fZsRADDOS)
# plot([meanfZsSTARK, meanfZsSTARK],[0,1400], color = 'purple')
# plot([meanfZsRADDOS, meanfZsRADDOS],[0,1400], color = 'green')
# legend(loc='upper right')
# xlabel('fZs Where Planets Detected',fontsize=14,fontweight='bold')
# tight_layout()
# show(block=False)

#Histogram of dMags#####################################





#Plot Unique Detections################################################################
from EXOSIMS.util.read_ipcluster_ensemble import gen_summary
from pylab import *
import numpy as np
from cycler import cycler
import math

#res1 = gen_summary('/home/dean/drk94_starkAYOoct21_2017')
#res2 = gen_summary('/home/dean/drk94_starkAYOstaticSchedulefZmin')#starkAYOoct9_2017')


#11/9/2017 c-KeplerLike pp-KeplerLike
#res1 = gen_summary('/home/dean/wfirst_radDos0')
#res2 = gen_summary('/home/dean/wfirst_radDos0_SLSQP')
#res3 = gen_summary('/home/dean/wfirst_radDos0_SLSQP_static')
#res4 = gen_summary('/home/dean/drk94_starkAYOfZhelioDecCompnov8_2017')#drk94_starkayOfZhelioDecCompnov8_2017')

#eq Dec vs Helio
#res1 = gen_summary('/home/dean/drk94_starkAYO_fZeqDecCompnov8_2017')
#res2 = gen_summary('/home/dean/drk94_starkAYOfZhelioDecCompnov8_2017')
#res3 = gen_summary('/home/dean/drk94_starkAYO_fZnoDecCompnov13_2017')
#res4 = gen_summary('/home/dean/drk94_starkAYO_fZnoDecnoCompnov14_2017')

#11/10/2017 c-JTWIN pp-JTWIN
#res1 = gen_summary('/home/dean/drk94_raddos0_JTWINJTWINnov9_2017')
#res2 = gen_summary('/home/dean/drk94_starkAYO_JTWINJTWINnov10_2017')



#11/9/2017 c-JTWIN pp-KeplerLike
#res1 = gen_summary('/home/dean/drk94_radDos0JTWINKeplernov6_2017')
#res2 = gen_summary('/home/dean/drk94_SLSQPJTWINKeplernov4_2017')
#res3 = gen_summary('/home/dean/drk94_SLSQPstaticJTWINKeplernov3_2017')
#res4 = gen_summary('/home/dean/drk94_starkAYOJTWINKeplernov9_2017')

#2/14/2018 c-KeplerLike New implementation
#res1 = gen_summary('/home/dean/drk94_starkAYO_fZnoDecnoCompnov14_2017')
#res2 = gen_summary('/home/dean/Feb13_2018starkAYOstaticFunctionalTest')

#2/15/2018 c-KeplerLike New implementation compared to Last master commit
res1 = gen_summary('/home/dean/Feb15_2018starkAYOstaticmasterFunctionalTest')
res2 = gen_summary('/home/dean/Feb13_2018starkAYOstaticFunctionalTest')
res3 = gen_summary('/home/dean/Feb16_2018SLSQPmasterFunctionalTest')


#res1 = gen_summary('/home/dean/wfirst_radDos0_SLSQP')
#res2 = gen_summary('/home/dean/drk94_SLSQP_CbyTnov7_2017')

#dir1 = "/home/dean/drk94_starkAYOoct20_2017"
#out = gen_summary(dir1)
#print('Done starkAYO')
#dir2 = "/home/dean/wfirst_radDos0"
#out2 = gen_summary(dir2)



def dist_plot(res,uniq = True,fig=None,lstyle='--',plotmeans=True,legtext=None):
    rcounts = []
    for el in res:
        if uniq:
            rcounts.append(np.array([np.unique(r).size for r in el]))
        else:
            rcounts.append(np.array([len(r) for r in el]))

    bins = range(np.min(np.hstack(rcounts).astype(int)),np.max(np.hstack(rcounts).astype(int))+2)
    bcents = np.diff(bins)/2. + bins[:-1]

    pdfs = []
    for j in range(len(res)):
        pdfs.append(np.histogram(rcounts[j],bins=bins,density=True)[0].astype(float))

    mx = math.ceil(np.max(pdfs)*10)/10#np.round(np.max(pdfs),decimals=1)
    print(mx)

    syms = 'osp^v<>h'
    if fig is None:
        plt.figure()
    else:
        plt.figure(fig)
        plt.gca().set_prop_cycle(None)

    if legtext is None:
        legtext = [None]*len(res)

    #Set linewidth and color cycle

    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rc('axes',prop_cycle=(cycler('color',['red','purple'])))#,'blue','black','purple'])))
    rcParams['axes.linewidth']=2
    rc('font',weight='bold') 

    for j in range(len(res)):
        leg = legtext[j]
        c = plt.gca()._get_lines.prop_cycler.next()['color']
        if plotmeans:
            mn = np.mean(rcounts[j])
            plot([mn]*2,[0,mx],'--',color=c)
            if leg is not None:
                leg += ' ($\\mu = %2.2f$)'%mn
        plot(bcents, pdfs[j], syms[np.mod(j,len(syms))]+lstyle,color=c,label=leg)

    plt.ylim([0,mx])
    if legtext[0] is not None:
        plt.legend()
    plt.xlabel('Unique Detections',weight='bold')
    plt.ylabel('Normalized Yield Frequency (NYF)',weight='bold')

#11/9/2017 c-JTWIN pp-Kepler Like
#dist_plot([res1['detected'],res2['detected'],res3['detected'],res4['detected']],legtext=['max(C)','SLSQP','SLSQPstatic','StarkAYOstatic'])

#11/9/2017 c-Kepler pp-Kepler
#dist_plot([res1['detected'],res2['detected'],res3['detected'],res4['detected']],legtext=['max(C)','SLSQP','SLSQPstatic','StarkAYOstatic'])#legtext=[r'$max(C)$',r'$SLSQP$',r'$SLSQP_{static}$',r'$StarkAYO_{static}$'])

#11/10/2017 c-JTWIN pp-JTWIN
#dist_plot([res1['detected'],res2['detected']],legtext=['max(C)','StarkAYOstatic'])

#11/13/2017 starkAYO dec, helio dec, no dec
#dist_plot([res1['detected'],res2['detected'],res3['detected'],res4['detected']],legtext=['Equatorial Dec','Ecliptic Dec','no Dec','no Dec no Comp'])

#2/14/2018 c-keplerlike new implementation
#dist_plot([res1['detected'],res2['detected']],legtext=['StarkAYOOLD','StarkAYOstaticNEW'])

#2/15/2018 c-keplerlike new implementation
dist_plot([res1['detected'],res2['detected'],res3['detected']],legtext=['StarkAYOinMaster','StarkAYOstaticNEW','SLSQPdynamicMaster'])

#dist_plot([res1['detected'],res2['detected']],legtext=['SLSQP maxComp','SLSQP maxCbyT'])
#dist_plot([res1['detected'],res2['detected']],legtext=['Proto:C-JTWIN PP-JTWIN','Proto:C-JTWIN PP-Kepler'])
#'$t \\sim \\Delta$mag=22.5',

plt.show(block=False)
###########################################################################