# pick a sim
#from EXOSIMS.util.read_ipcluster_ensemble import gen_summary

#ares = gen_summary('/data/extmount/EXOSIMSres/wfirst_radDos0_SLSQP_static2')
#cnt = np.array([len(np.unique(r)) for r in ares['detected']])
#np.argsort(cnt)
#ind = 546

#--------------------------
import matplotlib
try:
    __IPYTHON__
    matplotlib.interactive(True)
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os.path
import glob
import EXOSIMS, EXOSIMS.MissionSim
import cPickle as pickle
import astropy.units as u
import astropy.constants as const
import numpy as np
import EXOSIMS.MissionSim
from astropy.time import Time
from astropy.coordinates import SkyCoord
import time
from matplotlib import animation
import copy

#ind = 546
#run_dir = '/data/extmount/EXOSIMSres/wfirst_radDos0_SLSQP_static2'
#pklfiles = glob.glob(os.path.join(run_dir,'*.pkl'))  
#with open(pklfiles[ind], 'rb') as g:
#    res = pickle.load(g)   

#sim = EXOSIMS.MissionSim.MissionSim(os.path.join(run_dir,'outspec.json'),nopar=True,seed=res['seed'])



#-------------------------------------------
#with open('run176529615289.pkl', 'rb') as g:
#    res = pickle.load(g)   

#sim = EXOSIMS.MissionSim.MissionSim('wfirst_radDos0_SLSQP_static2.json',nopar=True,seed=res['seed'])

#-------------------------------------------

sim = EXOSIMS.MissionSim.MissionSim('mission_anim1_script.json')
mode = filter(lambda mode: mode['detectionMode'] == True, sim.OpticalSystem.observingModes)[0]

if False:
    #sim.reset_sim()
    out = []
    starsobserved = []
    while not sim.TimeKeeping.mission_is_over():
        print sim.TimeKeeping.currentTimeNorm
        out.append({})
        out[-1]['t'] = sim.TimeKeeping.currentTimeNorm.copy()
        sInds = np.arange(sim.TargetList.nStars)
        out[-1]['fZ'] = sim.ZodiacalLight.fZ(sim.Observatory, sim.TargetList, sInds, sim.TimeKeeping.currentTimeAbs, mode)

        kogoodStart = sim.Observatory.keepout(sim.TargetList, sInds, sim.TimeKeeping.currentTimeAbs, mode)
        inds = sInds[np.where(kogoodStart)[0]]
        inds = np.array(list(set(inds) - set(starsobserved)))
        
        intTimes = sim.SurveySimulation.calc_targ_intTime(inds,sim.TimeKeeping.currentTimeAbs + np.array([1.]*len(inds))*u.d,mode)
        
        inds = inds[intTimes != 0*u.d]
        intTimes = intTimes[intTimes != 0*u.d]
        out[-1]['star_inds'] = inds
        out[-1]['intTimes'] = intTimes

        sInd = sim.SurveySimulation.choose_next_target(None,inds,np.array([1.]*sim.TargetList.nStars)*u.d,intTimes)

        out[-1]['sInd'] = sInd
        out[-1]['det_time'] = 1*u.d + intTimes[inds == sInd][0]

        starsobserved.append(sInd)

        sim.TimeKeeping.allocate_time((1*u.d+intTimes[inds == sInd][0])/sim.TimeKeeping.missionPortion)
else:
    with open('mission_anim2.pkl', 'rb') as g:
        out = pickle.load(g)


#want 30 fps for 25 seconds
fps = 30.
nframes = int(fps*25.0)
dt = sim.TimeKeeping.missionLife/nframes
ts = Time(np.linspace(sim.TimeKeeping.missionStart.value, sim.TimeKeeping.missionStart.value + sim.TimeKeeping.missionLife.to(u.d).value, int(nframes)), format='mjd', scale='tai')

nomsize = matplotlib.rcParams['lines.markersize']
xsz = 10; ysz = 6; ssz = 0.25
fontsize = 16

Obs = sim.Observatory
OS = sim.OpticalSystem
TL = sim.TargetList
mode = OS.observingModes[0]
koMin = Obs.koAngleMin.to('deg')
sInds = np.arange(TL.nStars)
th = np.linspace(0,2*np.pi,100)
bcols =  ['orange','g', 'b']


tnorm =  matplotlib.colors.Normalize(vmin=np.log10(0.1/86400.),vmax=np.log10(10),clip=True)
vnorm =  matplotlib.colors.Normalize(vmin=20,vmax=25)


def setupmp():
    fig,ax = plt.subplots(figsize=(xsz,ysz))
    mp = Basemap(projection='hammer',lat_0=0,lon_0=0,ax=ax)
    mp.drawmapboundary(fill_color='white')
    parallels = np.arange(-75,76,15)
    mp.drawparallels(parallels, labels=[1,0,0,0],fontsize=fontsize)
    meridians = np.arange(-150,151,30)
    mp.drawmeridians(meridians)
    
    txt1 = ax.text(-2e6,-1.5e6,'CGI Time Used: %0.1f days'%0,fontsize=fontsize, horizontalalignment='left',verticalalignment='bottom')
    txt2 = ax.text(ax.get_xlim()[1],-1.5e6,'Mission Time: %0.1f MJD'%ts[0].value,fontsize=fontsize,horizontalalignment='right',verticalalignment='bottom')
    txt3 = ax.text(ax.get_xlim()[1],ax.get_ylim()[1]+1.5e6,'Other Science Observations', fontsize=fontsize,horizontalalignment='right',verticalalignment='top')

    kogood, r_body, r_targ, _ = Obs.keepout(TL, sInds, ts[0], mode, returnExtra=True)

    targ = SkyCoord(SkyCoord(r_targ[:,0], r_targ[:,1], r_targ[:,2],
            representation='cartesian').represent_as('spherical'), 
            frame='icrs', unit='deg,deg,pc')

    xs,ys = mp(targ.ra,targ.dec)
    p = mp.scatter(xs,ys, marker='o', facecolor='none', edgecolor='k',s=1)

    tbxs = [txt1,txt2,txt3]
    return fig,ax,mp,tbxs


def drawStuff(j,ax,mp,DRM,tbxs,observed,CGItime):
    print j
    t = ts[j]
    tbxs[1].set_text(u"Mission Time: %0.1f MJD" %t.value)

    #look for starting to integrate on a target
    #if (t >= DRM[0]['arrival_time'] + DRM[0]['det_time'] + DRM[0]['char_time'] + sim.TimeKeeping.missionStart):
    if DRM and (t >= DRM[0]['t']+DRM[0]['det_time'] + sim.TimeKeeping.missionStart):
        DRM.pop(0)
        tbxs[2].set_text(u"Other Science Observations")

    #if (t >= DRM[0]['arrival_time'] + sim.TimeKeeping.missionStart - 1*u.d):
    if DRM and (t >= DRM[0]['t'] + sim.TimeKeeping.missionStart - dt):
        #tbxs[2].set_text(u"Integrating on %s"%DRM[0]['star_name'])
        tbxs[2].set_text(u"Integrating on %s"%TL.Name[DRM[0]['sInd']])
        #sInd = DRM[0]['star_ind']
        sInd = DRM[0]['sInd']
        if sInd not in observed:
            observed.append(sInd)

        CGItime += DRM[0]['det_time']/(np.ceil((DRM[0]['det_time']/dt).decompose().value)+1)
        tbxs[0].set_text(u"CGI Time Used: %0.1f days" %CGItime.value)
    else:
        sInd = None

    #remove old instances
    tmp = copy.copy(ax.collections)
    for a in tmp:
        if a.get_label() == 'stars':
            a.remove()
    
    tmp = copy.copy(ax.patches)
    for a in tmp:
        if a.get_label() == 'ko':
            a.remove()

    
    kogood, r_body, r_targ, _ = Obs.keepout(TL, sInds, t, mode, returnExtra=True)

    targ = SkyCoord(SkyCoord(r_targ[:,0], r_targ[:,1], r_targ[:,2],
            representation='cartesian').represent_as('spherical'), 
            frame='icrs', unit='deg,deg,pc')
    body = SkyCoord(SkyCoord(r_body[:,0,0], r_body[:,0,1], r_body[:,0,2],
            representation='cartesian').represent_as('spherical'), 
            frame='icrs', unit='deg,deg,AU')

    ko = []
    for j in range(len(bcols)):
        ras = koMin*np.cos(th)+body[j].ra
        decs = koMin*np.sin(th)+body[j].dec

        if np.any(decs < -90*u.deg):
            ras = ras[decs >= -90*u.deg]
            decs = decs[decs >= -90*u.deg]

        #account for wrap
        if np.abs(body[j].ra - 180*u.deg) < koMin:
            ras2 = copy.copy(ras)
            b180 = ras < 180*u.deg
            ras2[b180] = 181*u.deg
            xy = np.vstack(mp(ras2,decs)).T
            feh = ax.add_patch(matplotlib.patches.Polygon(xy,closed=True,alpha=0.2,color=bcols[j],label='ko'))
            ras[~b180] = 180*u.deg
        xy = np.vstack(mp(ras,decs)).T
        ko.append(ax.add_patch(matplotlib.patches.Polygon(xy,closed=True,alpha=0.2,color=bcols[j],label='ko')))
 

    if DRM:
        xs,ys = mp(targ.ra,targ.dec)
        
        p = mp.scatter(xs,ys, marker='o',label = 'stars', c=-2.5*np.log10(DRM[0]['fZ'].value),  linewidths=1, norm=vnorm, cmap=matplotlib.cm.viridis_r)
        
        if observed:
            p4 = mp.scatter(xs[np.array(observed)],ys[np.array(observed)],color='blue')
            ko.append(p4)
        if sInd:
            p2 = mp.scatter(xs[sInd],ys[sInd], marker='o',label = 'stars',color='red',s=(nomsize*2.0)**2.)
            ko.append(p2)
      
    return ko


if __name__ == '__main__':
    observed = []
    CGItime = 0*u.d
    #DRM = copy.copy(res['DRM'])
    DRM = copy.copy(out)
    fig,ax,mp,tbxs = setupmp()
    anim = animation.FuncAnimation(fig, drawStuff, fargs=(ax,mp,DRM,tbxs,observed,CGItime), frames=nframes, interval=1./fps*1000, blit=False)
    anim.save('mission_anim2.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])




