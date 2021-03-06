"""Purpose: Plot Observation Timeline for mission

Written by Dean Keithly on 23 Apr, 2018
"""
"""Example 1
I have 1000 pkl files in /home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/run146279583107.pkl and
1qty outspec file in /home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/outspec.json

To generate timelines for these run the following code from an ipython session
from ipython
%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/run146279583107.pkl' \
'/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/outspec.json'
"""
"""Example 2
I have several folders with foldernames /home/dean/Documents/SIOSlab/*fZ*OB*PP*SU*/
each containing ~1000 pkl files and 1 outspec.json file

To plot a random Timeline from each folder, from ipython
%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/' None
"""
#%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/run136726516274.pkl' '/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/outspec.json'
#%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/Dean21May18RS09CXXfZ01OB56PP03SU03/run5991056964408.pkl' '/home/dean/Documents/SIOSlab/Dean21May18RS09CXXfZ01OB56PP03SU03/outspec.json'
#%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/Dean19May18RS09CXXfZ01OB57PP01SU01/run5152585081560.pkl' '/home/dean/Documents/SIOSlab/Dean19May18RS09CXXfZ01OB57PP01SU01/outspec.json'
#%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/Dean2June18RS26CXXfZ01OB65PP01SU01/run9239688957.pkl' '/home/dean/Documents/SIOSlab/Dean2June18RS26CXXfZ01OB65PP01SU01/outspec.json'
#%run DRMtoTimelinePlot.py '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/run/run1324783950.pkl' '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/run/outspec.json'
#%run DRMtoTimelinePlotSPIEposter.py '/home/dean/Documents/SIOSlab/Dean19May18RS09CXXfZ01OB58PP01SU01/run5210579638085.pkl' '/home/dean/Documents/SIOSlab/Dean19May18RS09CXXfZ01OB58PP01SU01/outspec.json'
#%run DRMtoTimelinePlotSPIEposter.py '/home/dean/Documents/SIOSlab/Dean19May18RS09CXXfZ01OB59PP01SU01/run5273430013323.pkl' '/home/dean/Documents/SIOSlab/Dean19May18RS09CXXfZ01OB58PP01SU01/outspec.json'
#%run DRMtoTimelinePlotSPIEposter.py '/home/dean/Documents/SIOSlab/Dean2June18RS09CXXfZ01OB64PP01SU01/run166028223287.pkl' '/home/dean/Documents/SIOSlab/Dean2June18RS09CXXfZ01OB64PP01SU01/outspec.json'
#%run DRMtoTimelinePlotSPIEposter.py '/home/dean/Documents/SIOSlab/Dean19May18RS09CXXfZ01OB56PP01SU01/run5081951104978.pkl' '/home/dean/Documents/SIOSlab/Dean19May18RS09CXXfZ01OB56PP01SU01/outspec.json'
#%run DRMtoTimelinePlotSPIEposter.py '/home/dean/Documents/SIOSlab/Dean19May18RS09CXXfZ01OB57PP01SU01/run5147420512699.pkl' '/home/dean/Documents/SIOSlab/Dean19May18RS09CXXfZ01OB57PP01SU01/outspec.json'

try:
    import cPickle as pickle
except:
    import pickle
import os
import numpy as np
if not 'DISPLAY' in os.environ.keys(): #Check environment for keys
    from matplotlib import *
    use('Agg')
from pylab import *
from numpy import nan
import matplotlib.pyplot as plt
import argparse
import json
import pdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Mission Timeline Figures")
    parser.add_argument('--pklfile', nargs=1, type=str, help='Full path to pkl file (string).')
    parser.add_argument('--outspecfile', nargs=1, type=str, help='Full path to outspec file (string).')

    args = parser.parse_args()
    pklfile = args.pklfile[0]
    outspecfile = args.outspecfile[0]

    if not os.path.exists(pklfile):
        raise ValueError('%s not found'%pklfile)

    #Given Filepath for pklfile, Plot a pkl from each testrun in subdir
    pklPaths = list()
    pklfname = list()
    outspecPaths = list()
    if(os.path.isdir(pklfile)):
        #Look for all directories in specified path with structured folder name
        fp1 = pklfile
        dirs = [myString for myString in next(os.walk(fp1))[1] if 'SU' in myString \
            and 'PP' in myString \
            and 'OB' in myString \
            and 'fZ' in myString \
            and 'RS' in myString]  # Folders containing Monte Carlo Runs

        for i in np.arange(len(dirs)):
            pklFiles = [myFileName for myFileName in os.listdir(fp1+dirs[i]) if 'run' in myFileName and '.pkl' in myFileName]  # Get names of all pkl files in path
            pklfname.append(np.random.choice(pklFiles))
            pklPaths.append(fp1 + dirs[i] + '/' + pklfname[i])  # append a random pkl file to path
            outspecPaths.append(fp1 + dirs[i] + '/' + 'outspec.json')
    elif(os.path.isfile(pklfile)):
        dirs = [pklfile.split('/')[-2]]
        pklfname.append(pklfile.split('/')[-1].split('.')[0])
        pklPaths.append(pklfile)#append filepath provided in args
        outspecPaths.append(outspecfile)#append filepath provided in args

    #Iterate over all pkl files
    for cnt in np.arange(len(pklPaths)):
        


        try:
            with open(pklPaths[cnt], 'rb') as f:#load from cache
                DRM = pickle.load(f)
        except:
            print('Failed to open pklfile %s'%pklPaths[cnt])
            pass
        try:
            with open(outspecPaths[cnt], 'rb') as g:
                outspec = json.load(g)
        except:
            print('Failed to open outspecfile %s'%outspecPaths[cnt])
            pass

        allModes = outspec['observingModes']
        mode1 = [mode for mode in allModes if 'detectionMode' in mode.keys()]
        assert len(mode1) == 1, 'This needs to be enhanced'
        mode = mode1[0]
        if not 'timeMultiplier' in mode.keys():
            mode['timeMultiplier'] = 1.

        #det_mode = filter(lambda mode: mode['detectionMode'] == True, allModes)[0]
        #mode = det_mode
        LD = np.arange(len(DRM['DRM']))
        arrival_times = [DRM['DRM'][i]['arrival_time'].value for i in LD]
        sumOHTIME = outspec['settlingTime'] + outspec['starlightSuppressionSystems'][0]['ohTime']
        det_times = [DRM['DRM'][i]['det_time'].value*(mode['timeMultiplier'])+sumOHTIME for i in LD]
        det_timesROUNDED = [round(DRM['DRM'][i]['det_time'].value*(mode['timeMultiplier'])+sumOHTIME,1) for i in LD]
        ObsNums = [DRM['DRM'][i]['ObsNum'] for i in LD]
        y_vals = np.zeros(len(det_times)).tolist()
        char_times = [DRM['DRM'][i]['char_time'].value*(1.+outspec['charMargin'])+sumOHTIME*(DRM['DRM'][i]['char_time'].value > 0.) for i in LD]
        OBdurations = np.asarray(outspec['OBendTimes'])-np.asarray(outspec['OBstartTimes'])
        #sumOHTIME = [1 for i in np.arange(len(DRM['DRM']))]
        print(sum(det_times))
        print(sum(char_times))

        from pylab import *
        arr = [DRM['DRM'][i]['arrival_time'].value for i in np.arange(len(DRM['DRM']))]
        dt = [DRM['DRM'][i]['det_time'].value + 1. for i in np.arange(len(DRM['DRM']))]

        fig = figure()
        ax = fig.add_subplot(111)
        colors=['red','blue']
        for i in np.arange(len(arr)):
            ax.barh(1., dt[i], align='center', left=arr[i],color=colors[int(i) % len(colors)])
        plt.show(block=False)


        #Check if plotting font #########################################################
        tmpfig = plt.figure(figsize=(20,5),num=0)
        ax = tmpfig.add_subplot(111)
        t = ax.text(0, 0, "Obs#   ,  d", ha='center',va='center',rotation='vertical', fontsize=8)
        r = tmpfig.canvas.get_renderer()
        bb = t.get_window_extent(renderer=r)
        Obstxtwidth = bb.width#Width of text
        Obstxtheight = bb.height#height of text
        FIGwidth, FIGheight = tmpfig.get_size_inches()*tmpfig.dpi
        #plt.show(block=False)
        plt.close()
        daysperpixelapprox = max(arrival_times)/FIGwidth#approximate #days per pixel
        if mean(det_times)*0.8/daysperpixelapprox > Obstxtwidth:
            ObstextBool = True
        else:
            ObstextBool = False

        tmpfig = plt.figure(figsize=(25,5),num=0)
        ax = tmpfig.add_subplot(111)
        t = ax.text(0, 0, "OB#  , dur.=    d", ha='center',va='center',rotation='horizontal', fontsize=12)
        r = tmpfig.canvas.get_renderer()
        bb = t.get_window_extent(renderer=r)
        OBtxtwidth = bb.width#Width of text
        OBtxtheight = bb.height#height of text
        FIGwidth, FIGheight = tmpfig.get_size_inches()*tmpfig.dpi
        #plt.show(block=False)
        plt.close()
        if mean(OBdurations)*0.8/daysperpixelapprox > OBtxtwidth:
            OBtextBool = True
        else:
            OBtextBool = False
        #################################################################################


        ######################################################################
        #Finds arrival times that occur within that year
        ObsNumsL = list()
        det_timesL = list()
        char_timesL = list()
        arrival_timesL = list()
        truthArr = list()
        for i in np.arange(int(np.ceil(max(arrival_times)/365.25))):
            truthArr = (np.asarray(arrival_times) >= 365.25*np.float(i))*(np.asarray(arrival_times) < 365.25*np.float(i+1.))
            arrival_timesL.append([arrival_times[ii] for ii in np.where(truthArr)[0]])
            det_timesL.append([det_times[ii] for ii in np.where(truthArr)[0]])
            char_timesL.append([char_times[ii] for ii in np.where(truthArr)[0]])
            ObsNumsL.append([ObsNums[ii] for ii in np.where(truthArr)[0]])
        #######################################################################

        #######################################################################
        # Plotting 
        colors = 'rb'#'rgbwmc'
        patch_handles = []
        fig = plt.figure(figsize=(20,3+int(np.ceil(max(arrival_times)/365.25))/2.),num=cnt)
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rcParams['axes.linewidth']=2
        plt.rc('font',weight='bold') 
        ax = fig.add_subplot(111)

        char_color=(0./255.,128/255.,0/255.)

        #Plot individual blocks
        # Plot All Detection Observations for Year
        for iyr in np.arange(int(np.ceil(max(arrival_times)/365.25))):
            ind = 0
            obs = 0
            for (det_time, l, char_time, arrival_times_yr) in zip(det_timesL[iyr], ObsNumsL[iyr], char_timesL[iyr], arrival_timesL[iyr]):
                #print det_time, l
                patch_handles.append(ax.barh(int(np.ceil(max(arrival_times)/365.25))-iyr, det_time, align='center', left=arrival_times_yr-365.25*iyr,
                    color=colors[int(obs) % len(colors)]))
                if not char_time == 0.:
                    ax.barh(int(np.ceil(max(arrival_times)/365.25))-iyr, char_time, align='center', left=arrival_times_yr+det_time-365.25*iyr,color=char_color)
                ind += 1
                obs += 1
                patch = patch_handles[-1][0]
                bl = patch.get_xy()
                x = 0.5*patch.get_width() + bl[0]
                y = 0.5*patch.get_height() + bl[1]
                if ObstextBool: 
                    ax.text(x, y, "Obs#%d, %dd" % (l,det_time), ha='center',va='center',rotation='vertical', fontsize=8)

        #Set Plot Xlimit so the end of the timeline is at the end of the figure box
        ax.set_xlim([None, 365.25])


        # Plot Asthetics
        y_pos = np.arange(int(np.ceil(max(arrival_times)/365.25)))+1#Number of xticks to have

        yticklabels = list()
        for i in np.arange(int(np.ceil(max(arrival_times)/365.25))):
            yticklabels.append(str(int(np.ceil(max(arrival_times)/365.25)) - i) + 'yr')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(yticklabels,fontsize=30)
        #ax.set_yticklabels(('3yr','2yr','1yr'),fontsize=30)
        ax.xaxis.set_tick_params(labelsize=30)
        ax.set_xlabel('Time Since Start of Mission Year (days)', weight='bold',fontsize=30)
        plt.title('Mission Timeline for runName: ' + dirs[cnt] + '\nand pkl file: ' + pklfname[cnt], weight='bold',fontsize=12)
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'TimelineStacked' + '.png')
        plt.savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'TimelineStacked' + '.svg')
        plt.savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'TimelineStacked' + '.eps')

        #plt.close()



        # #Simple Plot of Temporal Coverage of Sky
        # tmpfig2 = plt.figure(figsize=(8,8),num=1)
        # rc('axes',linewidth=2)
        # rc('lines',linewidth=2)
        # rcParams['axes.linewidth']=2
        # rc('font',weight='bold')
        # sb1 = plt.subplot(111,projection="hammer")
        # plt.grid(True)
        # #sb1.scatter(np.pi/2,np.pi/4,color='r')#plot([0,np.pi/2,0],color='r')
        # #sb1.plot([np.pi/2,0,0],color='b')
        # OBdur = np.asarray(outspec['OBendTimes'])-np.asarray(outspec['OBstartTimes'])
        # angular_widths = np.asarray(OBdur)/365.25*360*np.pi/180.
        # #FOR OBSERVATIONS#start_angles = np.asarray(outspec['arrival_times'])%365.25 /365.25*360*np.pi/180.
        # start_angles = np.asarray(outspec['OBstartTimes'])%365.25 /365.25*360*np.pi/180. - np.pi
        # for i in range(len(start_angles)):
        #     rect = Rectangle((start_angles[i],-np.pi/2),angular_widths[i],np.pi,angle=0.0, alpha=0.2)
        #     sb1.add_patch(rect)
        # # rect = Rectangle((0,-np.pi/2),np.pi/4.,np.pi,angle=0.0, alpha=0.2)
        # # sb1.add_patch(rect)
        # title('Mission Length: ' + str(outspec['missionLife'])  +'yr OBduration: ' + str(outspec['OBduration']) + 'd MissionPortion: ' + str(outspec['missionPortion']),weight='bold',fontsize=12)
        # plt.show(block=False)
        # savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'SkyCoverage' + '.png')
        # savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'SkyCoverage' + '.svg')
        # savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'SkyCoverage' + '.eps')
