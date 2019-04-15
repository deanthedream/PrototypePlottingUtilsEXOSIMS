#Plotting Fits Files

try:
    import cPickle as pickle
except:
    import pickle
import os
if not 'DISPLAY' in os.environ.keys(): #Check environment for keys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
else:
    import matplotlib.pyplot as plt 
import numpy as np
from numpy import nan
import argparse
import json
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import astropy.units as u
import copy
import random
import datetime
import re
from EXOSIMS.util.vprint import vprint
from copy import deepcopy

from astropy.io import fits
import scipy.interpolate
from EXOSIMS.MissionSim import MissionSim
import numbers
from matplotlib import ticker, cm



# fullPathPKL = '/home/dean/Documents/SIOSlab/EXOSIMSres/HabExCompSpecPriors_HabEx_4m_TSDD_pop100DD_revisit_20180424/HabEx_CSAG13_PPSAG13/run94611245591.pkl'
# inputScript = '/home/dean/Documents/SIOSlab/EXOSIMSres/HabExCompSpecPriors_HabEx_4m_TSDD_pop100DD_revisit_20180424/HabEx_CSAG13_PPSAG13/HabEx_CSAG13_PPSAG13.json'
# folder = '/home/dean/Documents/SIOSlab/EXOSIMSres/HabExCompSpecPriors_HabEx_4m_TSDD_pop100DD_revisit_20180424/HabEx_CSAG13_PPSAG13'
#WFIRST pkl
fullPathPKL = '/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_405_19/WFIRSTcycle6core_CKL2_PPKL2/run16348297762.pkl'
folder = '/home/dean/Documents/SIOSlab/EXOSIMSres/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_405_19/WFIRSTcycle6core_CKL2_PPKL2'



#Load pkl and outspec files
try:
    with open(fullPathPKL, 'rb') as f:#load from cache
        DRM = pickle.load(f)
except:
    vprint('Failed to open fullPathPKL %s'%fullPathPKL)
    pass
outspecPath = os.path.join(folder,'outspec.json')
try:
    with open(outspecPath, 'rb') as g:
        outspec = json.load(g)
except:
    vprint('Failed to open outspecfile %s'%outspecPath)
    pass
#Load Input Script
# try:
#     with open(inputScript, 'rb') as g:
#         inputJSON = json.load(g)
# except:
#     vprint('Failed to open outspecfile %s'%inputScript)
#     pass


# outspec = {
#   "scienceInstruments": [
#     { "name": "imager",
#       "QE": "$HOME/Documents/exosims/fitFilesFolder/WFIRST_cycle6/QE/Basic_Multi2_std_Si.fits",
#       "optics": 1.,
#       "FoV": 1.,
#       "pixelNumber": 1,
#       "pixelSize": 1e-3,
#       "sread": 0,
#       "idark": 0.,
#       "CIC": 0,
#       "texp": 100,
#       "ENF": 1
#     }],
#   "starlightSuppressionSystems":[
#     { "name": "HLC-565",
#     "optics": 1,
#     "lam": 565,
#     "BW": 0.10,
#     "IWA": 0.15,
#     "ohTime": 0.5,
#     "occ_trans": "$HOME/Documents/exosims/fitFilesFolder/WFIRST_cycle6/G22_FIT_565/G22_FIT_565_occ_trans.fits",
#     "core_thruput": "$HOME/Documents/exosims/fitFilesFolder/WFIRST_cycle6/G22_FIT_565/G22_FIT_565_thruput.fits",
#     "core_mean_intensity": "$HOME/Documents/exosims/fitFilesFolder/WFIRST_cycle6/G22_FIT_565/G22_FIT_565_mean_intensity.fits",
#     "core_area": "$HOME/Documents/exosims/fitFilesFolder/WFIRST_cycle6/G22_FIT_565/G22_FIT_565_area.fits",
#     "core_platescale": 0.30
#   }],
# "modules": {
#     "PlanetPopulation": " ",
#     "StarCatalog": " ",
#     "OpticalSystem": " ",
#     "ZodiacalLight": " ",
#     "BackgroundSources": " ",
#     "PlanetPhysicalModel": " ",
#     "Observatory": " ",
#     "TimeKeeping": " ",
#     "PostProcessing": " ",
#     "Completeness": " ",
#     "TargetList": " ",
#     "SimulatedUniverse": " ",
#     "SurveySimulation": " ",
#     "SurveyEnsemble": " "
#   }}

#sim = MissionSim(scriptfile=inputScript,nopar=True, verbose=True)#,**deepcopy(outspec))
sim = MissionSim(scriptfile=None,nopar=True, verbose=True,**deepcopy(outspec))

#DELETE SSS = [{ "name": "HLC-565",
#       "optics": 0.983647,
#       "lam": 565,
#       "BW": 0.10,
#       "IWA": 0.15,
#       "ohTime": 0.5,
#       "occ_trans": "$HOME/Documents/exosims/fitFilesFolder/WFIRST_cycle6/G22_FIT_565/G22_FIT_565_occ_trans.fits",
#       "core_thruput": "$HOME/Documents/exosims/fitFilesFolder/WFIRST_cycle6/G22_FIT_565/G22_FIT_565_thruput.fits",
#       "core_mean_intensity": "$HOME/Documents/exosims/fitFilesFolder/WFIRST_cycle6/G22_FIT_565/G22_FIT_565_mean_intensity.fits",
#       "core_area": "$HOME/Documents/exosims/fitFilesFolder/WFIRST_cycle6/G22_FIT_565/G22_FIT_565_area.fits",
#       "core_platescale": 0.30
#     }]

#Some Initializations
occ_trans = 0.
core_thruput = 0.
core_contrast = 0.
PSF = 0.
lam = 0.
BW = 0.

def stripUnits(syst):
  for key in syst.keys():
    try:
      syst[key] = syst[key].value
    except:
      continue
  return syst

#OS = sim.OpticalSystem.__init__(specs=outspec)#starlightSuppressionSystems=SSS)
syst = outspec['starlightSuppressionSystems'][0]
nsyst = 0 # this is the index of above
########syst = stripUnits(syst)
# for nsyst,syst in enumerate(self.starlightSuppressionSystems):
assert isinstance(syst,dict),\
        "Starlight suppression systems must be defined as dicts."
assert syst.has_key('name') and isinstance(syst['name'],basestring),\
        "All starlight suppression systems must have key name."
# populate with values that may be filenames (interpolants)
syst['occ_trans'] = syst.get('occ_trans', occ_trans)
syst['core_thruput'] = syst.get('core_thruput', core_thruput)
syst['core_contrast'] = syst.get('core_contrast', core_contrast)
syst['core_mean_intensity'] = syst.get('core_mean_intensity') # no default
syst['core_area'] = syst.get('core_area', 0.) # if zero, will get from lam/D
syst['PSF'] = syst.get('PSF', PSF)
#self._outspec['starlightSuppressionSystems'].append(syst.copy())

# attenuation due to optics specific to the coronagraph (defaults to 1)
# e.g. polarizer, Lyot stop, extra flat mirror
syst['optics'] = float(syst.get('optics', 1.))

# set an occulter, for an external or hybrid system
syst['occulter'] = syst.get('occulter', False)
# if syst['occulter'] == True:
#     self.haveOcculter = True

# handle inf OWA
if syst.get('OWA') == 0:
    syst['OWA'] = np.Inf

# when provided, always use deltaLam instead of BW (bandwidth fraction)
syst['lam'] = float(syst.get('lam', lam))*u.nm      # central wavelength (nm)
syst['deltaLam'] = float(syst.get('deltaLam', syst['lam'].to('nm').value*
        syst.get('BW', BW)))*u.nm                   # bandwidth (nm)
syst['BW'] = float(syst['deltaLam']/syst['lam'])    # bandwidth fraction
# default lam and BW updated with values from first instrument
if nsyst == 0:
    lam, BW = syst.get('lam').value, syst.get('BW')
    #lam, BW = syst.get('lam'), syst.get('BW')

# get coronagraph input parameters
# syst['lam'] = syst['lam']*u.nm
syst['IWA'] = syst['IWA']#*u.arcsec
syst['OWA'] = syst['OWA']#*u.arcsec
syst['lam'] = syst['lam'].to('nm').value
#syst['core_contrast'] = '/home/dean/Documents/exosims/fitFilesFolder/WFIRST_cycle6/B22_FIT_565/B22_FIT_565_contrast.fits'

def get_coro_param(syst, param_name, fill=0.):
    """For a given starlightSuppressionSystem, this method loads an input 
    parameter from a table (fits file) or a scalar value. It then creates a
    callable lambda function, which depends on the wavelength of the system
    and the angular separation of the observed planet.
    
    Args:
        syst (dict):
            Dictionary containing the parameters of one starlight suppression system
        param_name (string):
            Name of the parameter that must be loaded
        fill (float):
            Fill value for working angles outside of the input array definition
    
    Returns:
        syst (dict):
            Updated dictionary of parameters
    
    Note 1: The created lambda function handles the specified wavelength by 
        rescaling the specified working angle by a factor syst['lam']/mode['lam'].
    Note 2: If the input parameter is taken from a table, the IWA and OWA of that 
        system are constrained by the limits of the allowed WA on that table.
    
    """
    
    assert isinstance(param_name, basestring), "param_name must be a string."
    if isinstance(syst[param_name], basestring):
        pth = os.path.normpath(os.path.expandvars(syst[param_name]))
        assert os.path.isfile(pth), "%s is not a valid file."%pth
        dat = fits.open(pth)[0].data
        assert len(dat.shape) == 2 and 2 in dat.shape, \
                param_name + " wrong data shape."
        WA, D = (dat[0], dat[1]) if dat.shape[0] == 2 else (dat[:,0], dat[:,1])
        # if not self.haveOcculter:
        #     assert np.all(D >= 0) and np.all(D <= 1), \
        #         param_name + " must be positive and smaller than 1."
        # table interpolate function
        Dinterp = scipy.interpolate.interp1d(WA.astype(float), D.astype(float),
                kind='cubic', fill_value=fill, bounds_error=False)
        # create a callable lambda function
        syst[param_name] = lambda l, s: np.array(Dinterp((s \
                *syst['lam']/l)), ndmin=1)#.to('arcsec').value), ndmin=1)
        # IWA and OWA are constrained by the limits of the allowed WA on that table
        syst['IWA'] = max(np.min(WA), syst.get('IWA', np.min(WA)))
        syst['OWA'] = min(np.max(WA), syst.get('OWA', np.max(WA)))
        
    elif isinstance(syst[param_name], numbers.Number):
        # if not self.haveOcculter:
        #     assert syst[param_name] >= 0 and syst[param_name] <= 1, \
        #         param_name + " must be positive and smaller than 1."
        syst[param_name] = lambda l, s, D=float(syst[param_name]): \
                ((s*syst['lam']/l >= syst['IWA']) & \
                (s*syst['lam']/l <= syst['OWA']))*(D - fill) + fill
        
    else:
        syst[param_name] = None
    
    return syst

syst = get_coro_param(syst, 'occ_trans')
syst = get_coro_param(syst, 'core_thruput')
syst = get_coro_param(syst, 'core_contrast', fill=1.)
syst = get_coro_param(syst, 'core_mean_intensity')
syst = get_coro_param(syst, 'core_area')
# syst = sim.OpticalSystem.get_coro_param(syst, 'occ_trans')
# syst = sim.OpticalSystem.get_coro_param(syst, 'core_thruput')
# syst = sim.OpticalSystem.get_coro_param(syst, 'core_contrast', fill=1.)
# syst = sim.OpticalSystem.get_coro_param(syst, 'core_mean_intensity')
# syst = sim.OpticalSystem.get_coro_param(syst, 'core_area')
# syst['lam'] = syst['lam'].value
# syst['IWA'] = syst['IWA'].value
# syst['OWA'] = syst['OWA'].value


#### Plot core_contrast
def plotParam(param, syst, sLambda, sWA, fignum=1, PPoutpath='./', folder='./'):
  """
  Args:
    param (string) - 'core_contrast', 'core_thruput', 'core_mean_intensity',
                      'core_area', 'occ_trans'

  """
  if param == 'core_thruput':
    return plotCoreThruput(syst, sLambda, sWA)
  if param == 'core_mean_intensity':
    return plotCoreMeanIntensity(syst, sLambda, sWA)
  if param == 'core_area':
    return plotCoreArea(syst, sLambda, sWA)
  if param == 'occ_trans':
    return plotOCCTRANS(syst, sLambda, sWA)

  plt.close(fignum)
  lamMin = 400.0
  lamMax = 1200.0
  lams = np.linspace(lamMin,lamMax,num=6)#*u.nm#syst['lam']
  ### TODO ADD LAMS
  WA = np.linspace(syst['IWA'],syst['OWA'],num=100, endpoint=True)#*u.arcsec
  core_contrast = []
  for l2 in lams:
    lcon = []
    for wa in WA:
      tmp = syst[param](l2,wa)
      if tmp == 1.:
        tmp = np.nan
      lcon.append(tmp)
    core_contrast.append(lcon)

  fig = plt.figure(num=fignum)
  ax = plt.subplot(1,1,1)

  plt.subplots_adjust(left=0.2)
  plt.rc('axes',linewidth=2)
  plt.rc('lines',linewidth=2)
  plt.rcParams['axes.linewidth']=2
  plt.rc('font',weight='bold')
  for i in np.arange(len(lams)):
    plt.semilogy(WA,core_contrast[i],label=r'$\lambda$: ' + str(np.round(lams[i])))
  plt.plot([WA[0],WA[0]],[0.5*np.min(core_contrast),1.5*np.max(core_contrast)],color='black',label='IWA')
  plt.plot([WA[-1],WA[-1]],[0.5*np.min(core_contrast),1.5*np.max(core_contrast)],color='black',label='OWA')
  #plt.contour(lams,WA,core_contrast)
  if param == 'core_contrast':
    param = r'Contrast $\zeta$'
  # if param == 'core_thruput':
  #   param = 'Throughput T'
  # if param == 'core_mean_intensity':
  #   param = 'Core Mean Intensity'
  # if param == 'core_area':
  #   param = 'Core Area'
  # if param == 'occ_trans':
  #   param = 'occ trans'
  plt.ylabel(param, weight='bold')
  #plt.xlabel(r'$\lambda$ (nm)')
  plt.xlabel('Working Angle (WA) in arcsec', weight='bold')
  plt.legend()
  plt.show(block=False)
  return core_contrast

def plotCoreThruput(syst, sLambda, sWA, fignum=65496832, PPoutpath='./', folder='./'):
  plt.close(fignum)
  lamMin = 400.0
  lamMax = 1200.0
  lams = np.linspace(lamMin, lamMax, num=500, endpoint=True)#*u.nm#syst['lam']
  WA = np.linspace(syst['IWA'], syst['OWA'], num=500, endpoint=True)#*u.arcsec
  core_thruput = []
  for l2 in lams:
    lcon = []
    for wa in WA:
      tmp = syst['core_thruput'](l2,wa)[0]
      if tmp == 1.:
        tmp = np.nan
      lcon.append(tmp)
    core_thruput.append(lcon)
  core_thruput = np.asarray(core_thruput)

  fig = plt.figure(num=fignum)
  plt.rc('axes',linewidth=2)
  plt.rc('lines',linewidth=2)
  plt.rcParams['axes.linewidth']=2
  plt.rc('font',weight='bold')

  CS = plt.contourf(lams, WA, core_thruput, cmap='bwr')
  plt.plot([np.min(lams), sLambda],[sWA, sWA], color='black')
  plt.plot([sLambda, sLambda],[np.min(WA), sWA], color='black')
  plt.scatter(sLambda,sWA, marker='o',facecolors='white', edgecolors='black',zorder=3)
  plt.xlabel(r'Wavelength, $\lambda$ (nm)', weight='bold')
  #plt.xlabel(r'$\lambda$ (nm)')
  plt.ylabel(r'Working Angle, $WA$ in arcsec', weight='bold')
  cbar = plt.colorbar(CS)
  cbar.set_label(r'Throughput, $T$', weight='bold')
  plt.show(block=False)

  date = unicode(datetime.datetime.now())
  date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
  fname = 'Throughput_' + folder.split('/')[-1] + '_' + date
  plt.savefig(os.path.join(PPoutpath,fname+'.png'))
  plt.savefig(os.path.join(PPoutpath,fname+'.svg'))
  plt.savefig(os.path.join(PPoutpath,fname+'.eps'))
  plt.savefig(os.path.join(PPoutpath,fname+'.pdf'))

  return core_thruput

def plotCoreMeanIntensity(syst, sLambda, sWA, fignum=458845, PPoutpath='./', folder='./'):
  plt.close(fignum)
  lamMin = 400.0
  lamMax = 1200.0
  lams = np.linspace(lamMin, lamMax, num=500, endpoint=True)#*u.nm#syst['lam']
  WA = np.linspace(syst['IWA'], syst['OWA'], num=500, endpoint=True)#*u.arcsec
  core_mean_intensity = []
  for l2 in lams:
    lcon = []
    for wa in WA:
      tmp = syst['core_mean_intensity'](l2,wa)[0]
      if tmp == 1.:
        tmp = np.nan
      lcon.append(tmp)
    core_mean_intensity.append(lcon)
  core_mean_intensity = np.asarray(core_mean_intensity)

  fig = plt.figure(num=fignum)
  plt.rc('axes',linewidth=2)
  plt.rc('lines',linewidth=2)
  plt.rcParams['axes.linewidth']=2
  plt.rc('font',weight='bold')

  CS = plt.contourf(lams, WA, core_mean_intensity, cmap='bwr')
  plt.plot([np.min(lams), sLambda],[sWA, sWA], color='black')
  plt.plot([sLambda, sLambda],[np.min(WA), sWA], color='black')
  plt.scatter(sLambda,sWA, marker='o',facecolors='white', edgecolors='black',zorder=3)
  plt.xlabel(r'Wavelength, $\lambda$ (nm)', weight='bold')
  #plt.xlabel(r'$\lambda$ (nm)')
  plt.ylabel(r'Working Angle, $WA$ in arcsec', weight='bold')
  cbar = plt.colorbar(CS)
  cbar.set_label(r'Core Mean Intensity, $\Psi$', weight='bold')
  plt.show(block=False)

  date = unicode(datetime.datetime.now())
  date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
  fname = 'MeanIntensity_' + folder.split('/')[-1] + '_' + date
  plt.savefig(os.path.join(PPoutpath,fname+'.png'))
  plt.savefig(os.path.join(PPoutpath,fname+'.svg'))
  plt.savefig(os.path.join(PPoutpath,fname+'.eps'))
  plt.savefig(os.path.join(PPoutpath,fname+'.pdf'))

  return core_mean_intensity

def plotCoreArea(syst, sLambda, sWA, fignum=89129, PPoutpath='./', folder='./'):
  plt.close(fignum)
  lamMin = 400.0
  lamMax = 1200.0
  lams = np.linspace(lamMin, lamMax, num=500, endpoint=True)#*u.nm#syst['lam']
  WA = np.linspace(syst['IWA'], syst['OWA'], num=500, endpoint=True)#*u.arcsec
  core_area = []
  for l2 in lams:
    lcon = []
    for wa in WA:
      tmp = syst['core_area'](l2,wa)[0]
      if tmp == 1.:
        tmp = np.nan
      lcon.append(tmp)
    core_area.append(lcon)
  core_area = np.asarray(core_area)

  fig = plt.figure(num=fignum)
  plt.rc('axes',linewidth=2)
  plt.rc('lines',linewidth=2)
  plt.rcParams['axes.linewidth']=2
  plt.rc('font',weight='bold')

  CS = plt.contourf(lams, WA, core_area, cmap='bwr')
  plt.plot([np.min(lams), sLambda],[sWA, sWA], color='black')
  plt.plot([sLambda, sLambda],[np.min(WA), sWA], color='black')
  plt.scatter(sLambda,sWA, marker='o',facecolors='white', edgecolors='black',zorder=3)
  plt.xlabel(r'Wavelength, $\lambda$ (nm)', weight='bold')
  #plt.xlabel(r'$\lambda$ (nm)')
  plt.ylabel(r'Working Angle, $WA$ in arcsec', weight='bold')
  cbar = plt.colorbar(CS)
  cbar.set_label(r'Core Area, $\Gamma$', weight='bold')
  plt.show(block=False)

  date = unicode(datetime.datetime.now())
  date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
  fname = 'CoreArea_' + folder.split('/')[-1] + '_' + date
  plt.savefig(os.path.join(PPoutpath,fname+'.png'))
  plt.savefig(os.path.join(PPoutpath,fname+'.svg'))
  plt.savefig(os.path.join(PPoutpath,fname+'.eps'))
  plt.savefig(os.path.join(PPoutpath,fname+'.pdf'))

  return core_area

def plotOCCTRANS(syst, sLambda, sWA, fignum=12380, PPoutpath='./', folder='./'):
  """
  Plot Intensity transmission of extended background sources such as zodiacal light.
  Includes pupil mask, occulter, Lyot stop, and polarizer
  """
  plt.close(fignum)
  lamMin = 400.0
  lamMax = 1200.0
  lams = np.linspace(lamMin, lamMax, num=500, endpoint=True)#*u.nm#syst['lam']
  WA = np.linspace(syst['IWA'], syst['OWA'], num=500, endpoint=True)#*u.arcsec
  occ_trans = []
  for l2 in lams:
    lcon = []
    for wa in WA:
      tmp = syst['occ_trans'](l2,wa)[0]
      if tmp == 1.:
        tmp = np.nan
      lcon.append(tmp)
    occ_trans.append(lcon)
  occ_trans = np.asarray(occ_trans)

  fig = plt.figure(num=fignum)
  plt.rc('axes',linewidth=2)
  plt.rc('lines',linewidth=2)
  plt.rcParams['axes.linewidth']=2
  plt.rc('font',weight='bold')

  CS = plt.contourf(lams, WA, occ_trans, cmap='bwr')
  plt.plot([np.min(lams), sLambda],[sWA, sWA], color='black')
  plt.plot([sLambda, sLambda],[np.min(WA), sWA], color='black')
  plt.scatter(sLambda,sWA, marker='o',facecolors='white', edgecolors='black',zorder=3)
  plt.xlabel(r'Wavelength, $\lambda$ (nm)', weight='bold')
  #plt.xlabel(r'$\lambda$ (nm)')
  plt.ylabel(r'Working Angle, $WA$ in arcsec', weight='bold')
  cbar = plt.colorbar(CS)
  cbar.set_label(r'Intensity Transmission of Extended\nBackground Sources, $\Upsilon$', weight='bold')
  plt.show(block=False)

  date = unicode(datetime.datetime.now())
  date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
  fname = 'occtrans_' + folder.split('/')[-1] + '_' + date
  plt.savefig(os.path.join(PPoutpath,fname+'.png'))
  plt.savefig(os.path.join(PPoutpath,fname+'.svg'))
  plt.savefig(os.path.join(PPoutpath,fname+'.eps'))
  plt.savefig(os.path.join(PPoutpath,fname+'.pdf'))

  return occ_trans

CC = plotParam(fignum=1,param='core_contrast',syst=syst, sWA=sim.SurveySimulation.WAint[0].value, sLambda=sim.SurveySimulation.detmode['lam'].value)
CT = plotParam(fignum=2,param='core_thruput',syst=syst, sWA=sim.SurveySimulation.WAint[0].value, sLambda=sim.SurveySimulation.detmode['lam'].value)
CMI = plotParam(fignum=3,param='core_mean_intensity',syst=syst, sWA=sim.SurveySimulation.WAint[0].value, sLambda=sim.SurveySimulation.detmode['lam'].value)
CA = plotParam(fignum=4,param='core_area',syst=syst, sWA=sim.SurveySimulation.WAint[0].value, sLambda=sim.SurveySimulation.detmode['lam'].value)
OT = plotParam(fignum=5,param='occ_trans',syst=syst, sWA=sim.SurveySimulation.WAint[0].value, sLambda=sim.SurveySimulation.detmode['lam'].value)

minCC = min([cc for cc in CC[1] if not np.isnan(cc)])
minCMI = min([cc for cc in CMI[1] if not np.isnan(cc) and not (cc == np.asarray([0.]))[0]])


def plotSpectralFluxDensity(sim, PPoutpath='./', folder='./'):
  #### Plot Spectral Flux Density vs Lambda #########################
  plt.figure(6548631)
  lams = np.linspace(start=400.0,stop=1000.0,num=1000-400+1)
  plt.plot(lams,sim.OpticalSystem.F0(lams*u.nm).value, color='blue', label='')
  plt.plot([np.min(lams), np.min(lams)],[0.9*np.min(sim.OpticalSystem.F0(lams*u.nm).value), 1.1*np.max(sim.OpticalSystem.F0(lams*u.nm).value)],color='black', label=r'$\lambda$ Bounds')
  plt.plot([np.max(lams), np.max(lams)],[0.9*np.min(sim.OpticalSystem.F0(lams*u.nm).value), 1.1*np.max(sim.OpticalSystem.F0(lams*u.nm).value)],color='black')
  plt.xlim(left=0.9*np.min(lams),right=1.1*np.max(lams))
  plt.ylim(bottom=0.9*np.min(sim.OpticalSystem.F0(lams*u.nm).value),top=1.1*np.max(sim.OpticalSystem.F0(lams*u.nm).value))
  plt.xlabel(r'$\lambda$ in (nm)', weight='bold')
  plt.ylabel(r'Spectral Flux Density      in $(ph/s/m^2/nm)$', weight='bold', usetex=False)
  alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
  plt.text((1.-0.5-0.1)*(np.max(lams)+np.min(lams))/2., (1.+0.15)*(np.max(sim.OpticalSystem.F0(lams*u.nm).value)+np.min(sim.OpticalSystem.F0(lams*u.nm).value))/2., r'$\mathcal{F}_0$', family='normal', rotation=90, usetex=True, **alignment)
  plt.show(block=False)
  date = unicode(datetime.datetime.now())
  date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
  fname = 'PowerSpectralFluxDensity_' + folder.split('/')[-1] + '_' + date
  plt.savefig(os.path.join(PPoutpath,fname+'.png'))
  plt.savefig(os.path.join(PPoutpath,fname+'.svg'))
  #plt.savefig(os.path.join(PPoutpath,fname),format='eps')
  plt.savefig(os.path.join(PPoutpath,fname+'.pdf'))
  ###################################################################
plotSpectralFluxDensity(sim, PPoutpath='./', folder='./')




print saltyburrito
#sim.OpticalSystem.get_coro_param(syst,'core_thruput')
            # syst = self.get_coro_param(syst, 'occ_trans')
            # syst = self.get_coro_param(syst, 'core_thruput')
            # syst = self.get_coro_param(syst, 'core_contrast', fill=1.)
            # syst = self.get_coro_param(syst, 'core_mean_intensity')
            # syst = self.get_coro_param(syst, 'core_area')

pth = '/home/dean/Documents/exosims/fitFilesFolder/WFIRST_cycle6/B22_FIT_565/B22_FIT_565_contrast.fits'
dat = fits.open(pth)[0].data

print saltyburrito


#### occ_trans
pth = '/home/dean/Documents/exosims/fitFilesFolder/HabExMay3/G_VC6_occ_trans_asec.fits'
dat = fits.open(pth)[0].data
WA, D = (dat[0], dat[1]) if dat.shape[0] == 2 else (dat[:,0], dat[:,1])
Dinterp = scipy.interpolate.interp1d(WA.astype(float), D.astype(float),
                    kind='cubic', fill_value=fill, bounds_error=False)
WAint = np.asarray([0.045     , 0.045     , 0.19756445, 0.18430155, 0.13013536,
           0.045     , 0.045     , 1.41807054, 0.045     , 0.76020878,
           0.06810318, 0.0556777 , 0.045     , 0.045     , 0.045     ,
           0.045     , 0.045     , 0.045     , 0.045     , 0.045     ,
           0.045     , 0.045     , 0.045     , 0.045     , 0.045     ,
           0.045     , 0.63738941, 0.10301575, 0.09035079, 0.07160575,
           0.045     , 0.13248153, 0.045     , 0.045     , 0.045     ,
           0.06097561])*u.arcsec

lam = 500.*u.nm
syst_occ_trans = lambda l, s: np.array(Dinterp((s*lam/l).to('arcsec').value), ndmin=1)
#NO

#### core_thruput
pth = "/home/dean/Documents/exosims/fitFilesFolder/HabExMay3/G_VC6_core_thruput_asec.fits"
dat = fits.open(pth)[0].data
WA, D = (dat[0], dat[1]) if dat.shape[0] == 2 else (dat[:,0], dat[:,1])
Dinterp = scipy.interpolate.interp1d(WA.astype(float), D.astype(float),
                    kind='cubic', fill_value=fill, bounds_error=False)
syst_core_thruput = lambda l, s: np.array(Dinterp((s*lam/l).to('arcsec').value), ndmin=1)

#### mean_intensity
pth = "/home/dean/Documents/exosims/fitFilesFolder/HabExMay3/G_VC6_core_mean_intensity_asec.fits"
dat = fits.open(pth)[0].data
WA, D = (dat[0], dat[1]) if dat.shape[0] == 2 else (dat[:,0], dat[:,1])
Dinterp = scipy.interpolate.interp1d(WA.astype(float), D.astype(float),
                    kind='cubic', fill_value=fill, bounds_error=False)
syst_core_mean_intensity = lambda l, s: np.array(Dinterp((s*lam/l).to('arcsec').value), ndmin=1)


# #### core_area
# pth = 
# dat = fits.open(pth)[0].data
# WA, D = (dat[0], dat[1]) if dat.shape[0] == 2 else (dat[:,0], dat[:,1])
# Dinterp = scipy.interpolate.interp1d(WA.astype(float), D.astype(float),
#                     kind='cubic', fill_value=fill, bounds_error=False)
# syst_core_area = lambda l, s: np.array(Dinterp((s*lam/l).to('arcsec').value), ndmin=1)

