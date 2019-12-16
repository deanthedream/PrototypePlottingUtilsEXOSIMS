# Calculate Indentical dMag of planets

import matplotlib.pyplot as plt
import numpy as np
#DELETE import matplotlib.dates as mdates
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import numpy as np
from EXOSIMS.util.deltaMag import *
#DELETimport EXOSIMS.PlanetPhysicalModel as PPM#calc_Phi
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import astropy.units as u
from scipy.interpolate import interp1d, PchipInterpolator


folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_40519/'))#/WFIRSTCompSpecPriors_WFIRSTcycle6core'))#HabExCompSpecPriors_HabEx_4m_TSDD_pop100DD_revisit_20180424'))#prefDistOBdursweep_WFIRSTcycle6core'))
filename = 'WFIRSTcycle6core_CKL2_PPKL2.json'#'HabEx_CKL2_PPSAG13.json'#'auto_2018_11_03_15_09__prefDistOBdursweep_WFIRSTcycle6core_9.json'#'./TestScripts/02_KnownRV_FAP=1_WFIRSTObs_staticEphem.json'#'Dean17Apr18RS05C01fZ01OB01PP01SU01.json'#'sS_SLSQP.json'#'sS_AYO4.json'#'sS_differentPopJTwin.json'#AYO4.json'
#filename = 'sS_intTime6_KeplerLike2.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile,nopar=True)
#sim.run_sim()

def phiD(dMag, Rp_max, p, s):
    """ Calculates a set of Phi and D combinations to achieve the selected dMag
    Args:
        dMag (float) - planet-star difference in magnitudes
        Rp_max (float) - maximum planet size considered
        p (float) - 
        s (float) - observed planet-star separation
    Returns:
        phi (numpy array) - planet phase function
        d (numpy array) - planet-star distance
    """
    #         o   y
    #         |
    #         |
    # *-------o   x == s
    #Assume circular orbits
    d_min = s
    #d_max = inf
    phi_d2 = 10.**(-dMag/2.5)/p/Rp**2. #equals phi/d**2
    #solve this function for beta
    out = fsolve(phi_d2*s**2.- calc_Phi(beta)/(1.+np.tan(beta)**2.),np.pi/4.)
    print(saltyburrito)
    return phi, d

def calc_Phi(beta):
    """Calculate the phase function. Prototype method uses the Lambert phase 
    function from Sobolev 1975.
    From sim.PlanetPhysicalModel.calc_Phi
    
    Args:
        beta (astropy Quantity array):
            Planet phase angles at which the phase function is to be calculated,
            in units of rad
            
    Returns:
        Phi (ndarray):
            Planet phase function
    
    """
    
    beta = beta.to('rad').value
    Phi = (np.sin(beta) + (np.pi - beta)*np.cos(beta))/np.pi
    
    return Phi

#### Plot Phase function value
plt.close(11112233)
plt.figure(num=11112233)
betas = np.linspace(start=0.,stop=np.pi,num=1000,endpoint=True)*u.rad
Phis = calc_Phi(betas)
plt.plot(betas,Phis,color='blue')
plt.show(block=False)

betaFunction2 = interp1d(Phis,betas,kind='quadratic') # NOT USED
betaFunction = PchipInterpolator(-Phis,betas) #the -Phis ensure the function monotonically increases
def calc_beta(Phi):
    """ Calculates the Phase angle based on the assumed planet phase function
    Args:
        Phi (float) - Phase angle function value ranging from 0 to 1
    Returns:
        beta (float) - Phase angle from 0 rad to pi rad
    """
    beta = betaFunction(-Phi)
    #Note: the - is because betaFunction uses -Phi when calculating the Phase Function
    #This is because PchipInterpolator used requires monotonically increasing function
    return beta

#### Plot Phase Function Inverse Function
plt.close(22221113655)
plt.figure(num=22221113655)
plt.plot(betaFunction2(Phis),Phis, color='blue')
plt.plot(calc_beta(Phis),Phis, color='black',linestyle='--')
plt.show(block=False)

#### Planet Properties #####################################
planProp = dict() #all in units of meters
planProp['mercury'] = {'R':2439.7*1000.,'a':57.91*10.**9.,'p':0.142}
planProp['venus'] = {'R':6051.8*1000.,'a':108.21*10.**9.,'p':0.689}
planProp['earth'] = {'R':6371.0*1000.,'a':149.60*10.**9.,'p':0.434}
planProp['mars'] = {'R':3389.92*1000.,'a':227.92*10.**9.,'p':0.150}
planProp['jupiter'] = {'R':69911.*1000.,'a':778.57*10.**9.,'p':0.538}
planProp['saturn'] = {'R':58232.*1000.,'a':1433.53*10.**9.,'p':0.499}
planProp['uranus'] = {'R':25362.*1000.,'a':2872.46*10.**9.,'p':0.488}
planProp['neptune'] = {'R':24622.*1000.,'a':4495.*10.**9.,'p':0.442}

R_venus = 6051.8*1000. #m
a_venus = 108.21*10.**9. #m
p_venus = 0.689
R_earth = 6371.0*1000. #m
a_earth = 149.60*10.**9. #m
p_earth = 0.434
R_mars = 3389.92*1000. #m
a_mars = 227.92*10.**9. #m
p_mars = 0.150 
R_jupiter = 69911.*1000. #m
a_jupiter = 778.57*10.**9. #m
p_jupiter = 0.538
R_saturn = 58232.*1000. #m
a_saturn = 1433.53*10.**9. #m
p_saturn = 0.499
R_uranus = 25362.*1000. #m
a_uranus = 2872.46*10.**9. #m
p_uranus = 0.488
R_neptune = 24622.*1000. #m
a_neptune = 4495.*10.**9. #m
p_neptune = 0.442


R_s = np.asarray([R_venus,R_earth,R_mars,R_jupiter,R_saturn,R_uranus,R_neptune])*u.m
a_s = np.asarray([a_venus,a_earth,a_mars,a_jupiter,a_saturn,a_uranus,a_neptune])*u.m
p_s = np.asarray([p_venus,p_earth,p_mars,p_jupiter,p_saturn,p_uranus,p_neptune])
color_s = ['coral','blue','red','purple','black','cyan','darkblue']
label_s = ['Venus','Earth','Mars','Jupiter','Saturn','uranus','neptune']

#### Maximum Solar System Planet dMags
dMag_max = [deltaMag(p_s[i],R_s[i],a_s[i],1.) for i in np.arange(len(p_s))]
print(dMag_max)
planetFluxes = 10.**(np.asarray(dMag_max)/-2.5)
print(planetFluxes)

#### How much dimmer to achieve dMag_Earth
FluxRatios = planetFluxes/planetFluxes[1]
print('FluxRatio to Earth: ' + str(FluxRatios))
# PhiToMakeEarthLike = 1./np.asarray(FluxRatios)

# ddMag_needed = np.asarray(dMag_max)-np.asarray(dMag_max[1])
# print(ddMag_needed)
# dFlux_needed = 10.**(ddMag_needed/-2.5)
# print(dFlux_needed)


#dMag = -2.5*np.log10(p*(Rp/d).decompose()**2*Phi) .value

#### Create list of Solar System planet properties
#deltaMag.phiD(25.,)

#### Given s, this is the Phi Range possible for a given SMA assuming circular orbits
def beta_range_s_circular(s,sma):
    """ Calculates the minimum and maximum Phi the planet can gave assuming a circular orbit,
    the given s, and the given SMA
    0 deg phase is when the planet is behind the host star
    180 deg phase is when planet is in front of the host star
    Args:
        s (float) - observed planet-star sepatation
        sma (float) - actual planet semi-major axis
    Returns:
        beta_min (float) - in radians, angle from planet behind star
        beta_max (float) - in radians, angle from planet occulting star
    """
    #sma**2. = s**2. + y**2.
    if s > sma: #failure mode/nonphysical
        s = sma
    y = np.sqrt(sma**2.-s**2.)
    beta_min = np.pi/2. - np.arctan2(y,s)
    beta_max = np.pi/2. + np.arctan2(y,s)
    return beta_min, beta_max

#### Calculate beta_min and beta_max for Each Planet at each sepatation
s_ss = np.zeros((100,len(a_s)))
for j in np.arange(len(a_s)):
    s_ss[:,j] = np.linspace(start=0.,stop=a_s[j].value,num=100,endpoint=True)
beta_min = np.zeros((s_ss.shape[0],len(a_s)))
beta_max = np.zeros((s_ss.shape[0],len(a_s)))
for i in np.arange(s_ss.shape[0]):
    for j in np.arange(len(a_s)):
        beta_min[i,j], beta_max[i,j] = beta_range_s_circular(s_ss[i,j],a_s[j].value)


plt.close(6666999878787)
plt.figure(num=6666999878787)
for j in np.arange(len(a_s)):
    plt.plot((s_ss[:,j]*u.m).to('AU'),beta_min[:,j],color=color_s[j],marker='2',label=label_s[j])
    plt.plot((s_ss[:,j]*u.m).to('AU'),beta_max[:,j],color=color_s[j],marker='1')
plt.xlabel('Planet-Star Separation in AU', weight='bold')
plt.ylabel('Phase Angle Value (beta) in deg')
plt.xlim([0.,1.05*np.max(a_s.to('AU').value)])
plt.show(block=False)

#### Max and Min beta for for Earth-like separations
betaE_min = np.zeros((s_ss.shape[0],len(a_s)))
betaE_max = np.zeros((s_ss.shape[0],len(a_s)))
for i in np.arange(s_ss.shape[0]):
    for j in np.arange(len(a_s)):
        betaE_min[i,j], betaE_max[i,j] = beta_range_s_circular(s_ss[i,1],a_s[j].value)

plt.close(55555888883333)
plt.figure(num=55555888883333)
for j in np.arange(len(a_s)):
    plt.plot((s_ss[:,1]*u.m).to('AU'),betaE_min[:,j],color=color_s[j],marker='2',label=label_s[j])
    plt.plot((s_ss[:,1]*u.m).to('AU'),betaE_max[:,j],color=color_s[j],marker='1')
plt.xlabel('Planet-Star Separation in AU', weight='bold')
plt.ylabel('Phase Angle Value (beta) in deg')
plt.xlim([0.,1.05*a_s[1].to('AU').value])
plt.show(block=False)


#Calculate Phi_max and Phi_min
PhiE_min = np.zeros((s_ss.shape[0],len(a_s)))
PhiE_max = np.zeros((s_ss.shape[0],len(a_s)))
for i in np.arange(s_ss.shape[0]):
    for j in np.arange(len(a_s)):
        PhiE_min[i,j] = calc_Phi(betaE_min[i,j]*u.rad)
        PhiE_max[i,j] = calc_Phi(betaE_max[i,j]*u.rad)

plt.close(4444888883333)
plt.figure(num=4444888883333)
for j in np.arange(len(a_s)):
    plt.plot((s_ss[:,1]*u.m).to('AU'),PhiE_min[:,j],color=color_s[j],marker='2',label=label_s[j])
    plt.plot((s_ss[:,1]*u.m).to('AU'),PhiE_max[:,j],color=color_s[j],marker='1')
plt.xlabel('Planet-Star Separation in AU', weight='bold')
plt.ylabel('Phase Function Value')
plt.xlim([0.,1.05*a_s[1].to('AU').value])
plt.show(block=False)


#### Where Venus has same Visual Magnitude as Earth
s_venus = np.linspace(start=0,stop=a_s[0].value,num=100)
beta_venus_min = np.zeros(100)
beta_venus_max = np.zeros(100)
beta_earth_min = np.zeros(100)
beta_earth_max = np.zeros(100)
Phi_Emin = np.zeros(100)
Phi_Emax = np.zeros(100)
Phi_Vmin = np.zeros(100)
Phi_Vmax = np.zeros(100)
for i in np.arange(len(s_venus)):
    beta_venus_min[i], beta_venus_max[i] = beta_range_s_circular(s_venus[i],a_s[0].value)
    beta_earth_min[i], beta_earth_max[i] = beta_range_s_circular(s_venus[i],a_s[1].value)
    Phi_Emin[i] = calc_Phi(beta_earth_min[i]*u.rad)
    Phi_Emax[i] = calc_Phi(beta_earth_max[i]*u.rad)
    Phi_Vmin[i] = calc_Phi(beta_venus_min[i]*u.rad)
    Phi_Vmax[i] = calc_Phi(beta_venus_max[i]*u.rad)


plt.close(5)
plt.figure(num=5)
plt.plot((s_venus*u.m).to('AU').value,Phi_Vmax,color='red')#,marker='D')
plt.plot((s_venus*u.m).to('AU').value,Phi_Vmin,color='blue')#,marker='o')
plt.plot((s_venus*u.m).to('AU').value,Phi_Emax,color='purple')#,marker='s')
plt.plot((s_venus*u.m).to('AU').value,Phi_Emin,color='black')#,marker='+')
plt.show(block=False)

plt.close(33333222233333)
plt.figure(num=33333222233333)
plt.plot((s_venus*u.m).to('AU').value,Phi_Emax/Phi_Vmax, color='red')#color_s[0])
plt.plot((s_venus*u.m).to('AU').value,Phi_Emin/Phi_Vmax, color='blue')#color_s[0])
plt.plot((s_venus*u.m).to('AU').value,Phi_Emin/Phi_Vmin, color='purple')#color_s[0])
plt.plot((s_venus*u.m).to('AU').value,Phi_Emax/Phi_Vmin, color='black')#color_s[0])
plt.xlabel('Planet-Star Separation in AU', weight='bold')
plt.ylabel('Phi E/Phi V')
plt.xlim([0.,1.05*a_s[1].to('AU').value])
plt.ylim([0.,10.])
plt.show(block=False)

plt.close(11111133333222233333)
plt.figure(num=11111133333222233333)
plt.plot((s_venus*u.m).to('AU').value,Phi_Vmax/Phi_Emax, color='red')#color_s[0])
plt.plot((s_venus*u.m).to('AU').value,Phi_Vmin/Phi_Emax, color='blue')#color_s[0])
plt.plot((s_venus*u.m).to('AU').value,Phi_Vmin/Phi_Emin, color='purple')#color_s[0])
plt.plot((s_venus*u.m).to('AU').value,Phi_Vmax/Phi_Emin, color='black')#color_s[0])
plt.xlabel('Planet-Star Separation in AU', weight='bold')
plt.ylabel('Phi V/Phi E')
plt.xlim([0.,1.05*a_s[1].to('AU').value])
plt.ylim([0.,10.])
plt.title('Venus')
plt.show(block=False)

#FluxRatios = VenusFlux/EarthFlux
#EarthFlux/VenusFlux

def venusEarthPhi(x):
    #beta_min, beta_max = beta_range_s_circular(x,a_s[0].value)
    tbeta_venus_min, tbeta_venus_max = beta_range_s_circular(x,a_s[0].value)
    tbeta_earth_min, tbeta_earth_max = beta_range_s_circular(x,a_s[1].value)
    error1 = np.abs(1./FluxRatios[0] - calc_Phi(tbeta_venus_min*u.rad)/calc_Phi(tbeta_earth_max*u.rad))
    error2 = np.abs(1./FluxRatios[0] - calc_Phi(tbeta_venus_min*u.rad)/calc_Phi(tbeta_earth_min*u.rad))
    error3 = np.abs(1./FluxRatios[0] - calc_Phi(tbeta_venus_max*u.rad)/calc_Phi(tbeta_earth_max*u.rad))
    error4 = np.abs(1./FluxRatios[0] - calc_Phi(tbeta_venus_max*u.rad)/calc_Phi(tbeta_earth_min*u.rad))
    return np.min([error1,error2,error3,error4])

#Neither of these worked
#res = minimize(venusEarthPhi,0.72*u.AU.to('m'),bounds=[(0.,0.99999*a_s[0].value)],method='SLSQP',options={'maxiter':300,'eps':0.001*u.AU.to('m'),'ftol':1e-7,'disp':True})
#res = minimize_scalar(venusEarthPhi,bounds=(0.5*a_s[0].value, a_s[0].value))#0.7*u.AU.to('m'))
# print(res.x*u.m.to('AU'))
# print(a_s[0].to('AU'))
# beta_venus_min, beta_venus_max = beta_range_s_circular(res.x,a_s[0].value)
# beta_earth_min, beta_earth_max = beta_range_s_circular(res.x,a_s[1].value)
# print(calc_Phi(beta_venus_max*u.rad)/calc_Phi(beta_earth_min*u.rad))
# print(1./FluxRatios[0]-calc_Phi(beta_venus_max*u.rad)/calc_Phi(beta_earth_min*u.rad))

tx = np.linspace(start=0.71*u.AU.to('m'),stop=a_s[0].value,num=1000)
out = np.asarray([venusEarthPhi(tx[i]) for i in np.arange(len(tx))])
tx_min = tx[np.argmin(out)]
ind = np.argmin(out)
minError = np.min(out)
beta_venus_min, beta_venus_max = beta_range_s_circular(tx_min,a_s[0].value)
beta_earth_min, beta_earth_max = beta_range_s_circular(tx_min,a_s[1].value)
Ratio = calc_Phi(beta_venus_max*u.rad)/calc_Phi(beta_earth_min*u.rad)
FluxRatios[0]*Ratio

#DELETE
# print("The planet-star separation resulting in Earth and Venus having the same visual magnitude is")
# print(tx_min*u.m.to('AU'))

print('Venus has the same visual magnitude as Earth at (AU): ' + str(tx_min*u.m.to('AU')))
print('Venus average Orbital Radius (AU): ' + str(a_s[0].to('AU').value))


#### Jupiter 
def jupiterEarthPhi(x):
    #beta_min, beta_max = beta_range_s_circular(x,a_s[0].value)
    tbeta_jupiter_min, tbeta_jupiter_max = beta_range_s_circular(x,a_s[3].value)
    tbeta_earth_min, tbeta_earth_max = beta_range_s_circular(x,a_s[1].value)
    error1 = np.abs(1./FluxRatios[3] - calc_Phi(tbeta_jupiter_min*u.rad)/calc_Phi(tbeta_earth_max*u.rad))
    error2 = np.abs(1./FluxRatios[3] - calc_Phi(tbeta_jupiter_min*u.rad)/calc_Phi(tbeta_earth_min*u.rad))
    error3 = np.abs(1./FluxRatios[3] - calc_Phi(tbeta_jupiter_max*u.rad)/calc_Phi(tbeta_earth_max*u.rad))
    error4 = np.abs(1./FluxRatios[3] - calc_Phi(tbeta_jupiter_max*u.rad)/calc_Phi(tbeta_earth_min*u.rad))
    return np.min([error1,error2,error3,error4])

tx2 = np.linspace(start=0.,stop=a_s[1].value,num=1000)
out2 = np.asarray([jupiterEarthPhi(tx2[i]) for i in np.arange(len(tx2))])
tx2_min = tx2[np.argmin(out2)]
ind2 = np.argmin(out2)
minError2 = np.min(out2)
beta_jupiter_min, beta_jupiter_max = beta_range_s_circular(tx2_min,a_s[3].value)
beta_earth_min, beta_earth_max = beta_range_s_circular(tx2_min,a_s[1].value)
Ratio = calc_Phi(beta_jupiter_min*u.rad)/calc_Phi(beta_earth_max*u.rad)
FluxRatios[3]*Ratio

ratio1 = np.zeros(len(tx2))
ratio2 = np.zeros(len(tx2))
ratio3 = np.zeros(len(tx2))
ratio4 = np.zeros(len(tx2))
for i in np.arange(len(tx2)):
    tbeta_jupiter_min, tbeta_jupiter_max = beta_range_s_circular(tx2[i],a_s[3].value)
    tbeta_earth_min, tbeta_earth_max = beta_range_s_circular(tx2[i],a_s[1].value)
    ratio1[i] = calc_Phi(tbeta_jupiter_min*u.rad)/calc_Phi(tbeta_earth_min*u.rad)
    ratio2[i] = calc_Phi(tbeta_jupiter_min*u.rad)/calc_Phi(tbeta_earth_max*u.rad)
    ratio3[i] = calc_Phi(tbeta_jupiter_max*u.rad)/calc_Phi(tbeta_earth_max*u.rad)
    ratio4[i] = calc_Phi(tbeta_jupiter_max*u.rad)/calc_Phi(tbeta_earth_min*u.rad)

plt.close(4444444222222)
plt.figure(num=4444444222222)
plt.plot(tx2*u.m.to('AU'),ratio1,color='red')
plt.plot(tx2*u.m.to('AU'),ratio2,color='blue')
plt.plot(tx2*u.m.to('AU'),ratio3,color='purple')
plt.plot(tx2*u.m.to('AU'),ratio4,color='black')
plt.ylim([0.,10.])
plt.ylabel('Jupiter Phi/Earth Phi')
plt.title('Jupiter')
plt.show(block=False)

#For Jupiter, the appropriate Flux Ratio between Earth and Jupiter to make the same
#visual magnitude occurs near the 0 separation

#### Saturn
def saturnEarthPhi(x):
    #beta_min, beta_max = beta_range_s_circular(x,a_s[0].value)
    tbeta_saturn_min, tbeta_saturn_max = beta_range_s_circular(x,a_s[4].value)
    tbeta_earth_min, tbeta_earth_max = beta_range_s_circular(x,a_s[1].value)
    error1 = np.abs(1./FluxRatios[4] - calc_Phi(tbeta_saturn_min*u.rad)/calc_Phi(tbeta_earth_max*u.rad))
    error2 = np.abs(1./FluxRatios[4] - calc_Phi(tbeta_saturn_min*u.rad)/calc_Phi(tbeta_earth_min*u.rad))
    error3 = np.abs(1./FluxRatios[4] - calc_Phi(tbeta_saturn_max*u.rad)/calc_Phi(tbeta_earth_max*u.rad))
    error4 = np.abs(1./FluxRatios[4] - calc_Phi(tbeta_saturn_max*u.rad)/calc_Phi(tbeta_earth_min*u.rad))
    return np.min([error1,error2,error3,error4])

tx3 = np.linspace(start=0.,stop=a_s[1].value,num=1000)
out3 = np.asarray([saturnEarthPhi(tx3[i]) for i in np.arange(len(tx3))])
tx3_min = tx3[np.argmin(out3)]
ind3 = np.argmin(out3)
minError3 = np.min(out3)
beta_saturn_min, beta_saturn_max = beta_range_s_circular(tx3_min,a_s[4].value)
beta_earth_min, beta_earth_max = beta_range_s_circular(tx3_min,a_s[1].value)
Ratio = calc_Phi(beta_saturn_min*u.rad)/calc_Phi(beta_earth_max*u.rad)
FluxRatios[4]*Ratio

ratio1 = np.zeros(len(tx3))
ratio2 = np.zeros(len(tx3))
ratio3 = np.zeros(len(tx3))
ratio4 = np.zeros(len(tx3))
for i in np.arange(len(tx3)):
    tbeta_saturn_min, tbeta_saturn_max = beta_range_s_circular(tx3[i],a_s[4].value)
    tbeta_earth_min, tbeta_earth_max = beta_range_s_circular(tx3[i],a_s[1].value)
    ratio1[i] = calc_Phi(tbeta_saturn_min*u.rad)/calc_Phi(tbeta_earth_min*u.rad)
    ratio2[i] = calc_Phi(tbeta_saturn_min*u.rad)/calc_Phi(tbeta_earth_max*u.rad)
    ratio3[i] = calc_Phi(tbeta_saturn_max*u.rad)/calc_Phi(tbeta_earth_max*u.rad)
    ratio4[i] = calc_Phi(tbeta_saturn_max*u.rad)/calc_Phi(tbeta_earth_min*u.rad)

plt.close(555554444444222222)
plt.figure(num=555554444444222222)
plt.plot(tx3*u.m.to('AU'),ratio1,color='red')
plt.plot(tx3*u.m.to('AU'),ratio2,color='blue')
plt.plot(tx3*u.m.to('AU'),ratio3,color='purple')
plt.plot(tx3*u.m.to('AU'),ratio4,color='black')
plt.ylim([0.,10.])
plt.ylabel('Saturn Phi/Earth Phi')
plt.title('Saturn')
plt.show(block=False)

#### Uranus
def uranusEarthPhi(x):
    #beta_min, beta_max = beta_range_s_circular(x,a_s[0].value)
    tbeta_uranus_min, tbeta_uranus_max = beta_range_s_circular(x,a_s[5].value)
    tbeta_earth_min, tbeta_earth_max = beta_range_s_circular(x,a_s[1].value)
    error1 = np.abs(1./FluxRatios[5] - calc_Phi(tbeta_uranus_min*u.rad)/calc_Phi(tbeta_earth_max*u.rad))
    error2 = np.abs(1./FluxRatios[5] - calc_Phi(tbeta_uranus_min*u.rad)/calc_Phi(tbeta_earth_min*u.rad))
    error3 = np.abs(1./FluxRatios[5] - calc_Phi(tbeta_uranus_max*u.rad)/calc_Phi(tbeta_earth_max*u.rad))
    error4 = np.abs(1./FluxRatios[5] - calc_Phi(tbeta_uranus_max*u.rad)/calc_Phi(tbeta_earth_min*u.rad))
    return np.min([error1,error2,error3,error4])

tx4 = np.linspace(start=0.,stop=a_s[1].value,num=10000)
out4 = np.asarray([uranusEarthPhi(tx4[i]) for i in np.arange(len(tx4))])
tx4_min = tx4[np.argmin(out4)]
ind4 = np.argmin(out4)
minError4 = np.min(out4)
beta_uranus_min, beta_uranus_max = beta_range_s_circular(tx4_min,a_s[5].value)
beta_earth_min, beta_earth_max = beta_range_s_circular(tx4_min,a_s[1].value)
Ratio = calc_Phi(beta_uranus_min*u.rad)/calc_Phi(beta_earth_max*u.rad)
FluxRatios[5]*Ratio

ratio1 = np.zeros(len(tx4))
ratio2 = np.zeros(len(tx4))
ratio3 = np.zeros(len(tx4))
ratio4 = np.zeros(len(tx4))
for i in np.arange(len(tx4)):
    tbeta_uranus_min, tbeta_uranus_max = beta_range_s_circular(tx4[i],a_s[5].value)
    tbeta_earth_min, tbeta_earth_max = beta_range_s_circular(tx4[i],a_s[1].value)
    ratio1[i] = calc_Phi(tbeta_uranus_min*u.rad)/calc_Phi(tbeta_earth_min*u.rad)
    ratio2[i] = calc_Phi(tbeta_uranus_min*u.rad)/calc_Phi(tbeta_earth_max*u.rad)
    ratio3[i] = calc_Phi(tbeta_uranus_max*u.rad)/calc_Phi(tbeta_earth_max*u.rad)
    ratio4[i] = calc_Phi(tbeta_uranus_max*u.rad)/calc_Phi(tbeta_earth_min*u.rad)

plt.close(666555554444444222222)
plt.figure(num=666555554444444222222)
plt.plot(tx4*u.m.to('AU'),ratio1,color='red')
plt.plot(tx4*u.m.to('AU'),ratio2,color='blue')
plt.plot(tx4*u.m.to('AU'),ratio3,color='purple')
plt.plot(tx4*u.m.to('AU'),ratio4,color='black')
plt.ylim([0.,25.])
plt.ylabel('uranus Phi/Earth Phi')
plt.title('uranus')
plt.show(block=False)

print('Uranus has the same visual magnitude as Earth at (AU): ' + str(tx4_min*u.m.to('AU')))
print('Uranus average Orbital Radius (AU): ' + str(a_s[5].to('AU').value))
print(Ratio)



#### Neptune
def neptuneEarthPhi(x):
    #beta_min, beta_max = beta_range_s_circular(x,a_s[0].value)
    tbeta_neptune_min, tbeta_neptune_max = beta_range_s_circular(x,a_s[6].value)
    tbeta_earth_min, tbeta_earth_max = beta_range_s_circular(x,a_s[1].value)
    error1 = np.abs(1./FluxRatios[6] - calc_Phi(tbeta_neptune_min*u.rad)/calc_Phi(tbeta_earth_max*u.rad))
    error2 = np.abs(1./FluxRatios[6] - calc_Phi(tbeta_neptune_min*u.rad)/calc_Phi(tbeta_earth_min*u.rad))
    error3 = np.abs(1./FluxRatios[6] - calc_Phi(tbeta_neptune_max*u.rad)/calc_Phi(tbeta_earth_max*u.rad))
    error4 = np.abs(1./FluxRatios[6] - calc_Phi(tbeta_neptune_max*u.rad)/calc_Phi(tbeta_earth_min*u.rad))
    return np.min([error1,error2,error3,error4])

tx5 = np.linspace(start=0.,stop=a_s[1].value,num=10000)
out5 = np.asarray([neptuneEarthPhi(tx5[i]) for i in np.arange(len(tx5))])
tx5_min = tx5[np.argmin(out5)]
ind5 = np.argmin(out5)
minError5 = np.min(out5)
beta_neptune_min, beta_neptune_max = beta_range_s_circular(tx5_min,a_s[6].value)
beta_earth_min, beta_earth_max = beta_range_s_circular(tx5_min,a_s[1].value)
Ratio = calc_Phi(beta_neptune_min*u.rad)/calc_Phi(beta_earth_max*u.rad)
FluxRatios[6]*Ratio

ratio1 = np.zeros(len(tx5))
ratio2 = np.zeros(len(tx5))
ratio3 = np.zeros(len(tx5))
ratio4 = np.zeros(len(tx5))
for i in np.arange(len(tx5)):
    tbeta_neptune_min, tbeta_neptune_max = beta_range_s_circular(tx5[i],a_s[6].value)
    tbeta_earth_min, tbeta_earth_max = beta_range_s_circular(tx5[i],a_s[1].value)
    ratio1[i] = calc_Phi(tbeta_neptune_min*u.rad)/calc_Phi(tbeta_earth_min*u.rad)
    ratio2[i] = calc_Phi(tbeta_neptune_min*u.rad)/calc_Phi(tbeta_earth_max*u.rad)
    ratio3[i] = calc_Phi(tbeta_neptune_max*u.rad)/calc_Phi(tbeta_earth_max*u.rad)
    ratio4[i] = calc_Phi(tbeta_neptune_max*u.rad)/calc_Phi(tbeta_earth_min*u.rad)

plt.close(666555554444444222222)
plt.figure(num=666555554444444222222)
plt.plot(tx5*u.m.to('AU'),ratio1,color='red')
plt.plot(tx5*u.m.to('AU'),ratio2,color='blue')
plt.plot(tx5*u.m.to('AU'),ratio3,color='purple')
plt.plot(tx5*u.m.to('AU'),ratio4,color='black')
plt.ylim([0.,70.])
plt.ylabel('neptune Phi/Earth Phi')
plt.title('neptune')
plt.show(block=False)

print('Neptune has the same visual magnitude as Earth at (AU): ' + str(tx5_min*u.m.to('AU')))
print('Neptune average Orbital Radius (AU): ' + str(a_s[6].to('AU').value))
print(Ratio)



#### Planet Orbital Radius Plot
plt.close(999922222244444)
plt.figure(num=999922222244444)
ax = plt.gca()
ax.set_yscale('symlog')
#ax.set_xscale('log')
#plt.yscale('symlog')
ax.set_xlim([4.*1e-1,a_s[1].to('AU').value])
ax.set_ylim([-1.1*1e2,1.1*1e0])
#plt.scatter(np.zeros(len(a_s)),a_s,color='blue')
x_earth = np.linspace(start=0.,stop=a_s[1].to('AU').value,num=300,endpoint=True)
x_venus1 = np.linspace(start=0.,stop=a_s[0].to('AU').value,num=300,endpoint=True)
y_venus1 = np.sqrt(a_s[0].to('AU').value**2. - x_venus1**2.)
plt.plot(x_venus1,y_venus1,color='orange')
plt.plot(x_venus1,-y_venus1,color='orange')
plt.scatter(tx_min*u.m.to('AU'), -np.sqrt(a_s[0].to('AU').value**2. - (tx_min*u.m.to('AU'))**2.), color='orange', marker='o')
plt.plot(np.zeros(2)+tx_min*u.m.to('AU'),np.linspace(start=-1e2,stop=np.sqrt(a_s[1].to('AU').value**2. - (tx_min*u.m.to('AU'))**2.),num=2,endpoint=True),linestyle='--',color='k')
y_earth1 = np.sqrt(a_s[1].to('AU').value**2. - x_earth**2.)
plt.plot(x_earth,y_earth1,color='green')
plt.plot(x_earth,-y_earth1,color='green')
plt.scatter(tx5_min*u.m.to('AU'), -np.sqrt(a_s[1].to('AU').value**2. - (tx5_min*u.m.to('AU'))**2.), color='green', marker='o')
plt.scatter(tx4_min*u.m.to('AU'), -np.sqrt(a_s[1].to('AU').value**2. - (tx4_min*u.m.to('AU'))**2.), color='green', marker='o')
plt.scatter(tx_min*u.m.to('AU'), np.sqrt(a_s[1].to('AU').value**2. - (tx_min*u.m.to('AU'))**2.), color='green', marker='o')
y_neptune1 = np.sqrt(a_s[6].to('AU').value**2. - x_earth**2.)
plt.plot(x_earth,-y_neptune1,color='blue')
plt.scatter(tx5_min*u.m.to('AU'), -np.sqrt(a_s[6].to('AU').value**2. - (tx5_min*u.m.to('AU'))**2.), color='blue',marker='o')
plt.plot(np.zeros(2)+tx5_min*u.m.to('AU'),np.linspace(start=-np.sqrt(a_s[1].to('AU').value**2. - (tx5_min*u.m.to('AU'))**2.),stop=-1e2,num=2,endpoint=True),linestyle='--',color='k')
y_uranus1 = np.sqrt(a_s[5].to('AU').value**2. - x_earth**2.)
plt.plot(x_earth,-y_uranus1,color='cyan')
plt.scatter(tx4_min*u.m.to('AU'),-np.sqrt(a_s[5].to('AU').value**2. - (tx4_min*u.m.to('AU'))**2.), color='cyan', marker='o')
plt.plot(np.zeros(2)+tx4_min*u.m.to('AU'),np.linspace(start=-np.sqrt(a_s[1].to('AU').value**2. - (tx4_min*u.m.to('AU'))**2.) ,stop=-1e2,num=2,endpoint=True),linestyle='--',color='k')

plt.xlabel('Semi-Major Axis (AU)')
plt.show(block=False)

#### Planet Orbital Radius Plot 2
plt.close(111999922222244444)
plt.figure(num=111999922222244444)
plt.xlim([4.*1e-1,a_s[1].to('AU').value])
plt.ylim([-np.log10(1.1*1e2),np.log10(1.1*1e0)])
x_earth = np.linspace(start=0.,stop=a_s[1].to('AU').value,num=300,endpoint=True)
x_venus1 = np.linspace(start=0.,stop=a_s[0].to('AU').value,num=300,endpoint=True)
y_venus1 = np.sqrt(a_s[0].to('AU').value**2. - x_venus1**2.)
plt.plot(x_venus1,np.log10(y_venus1),color='orange')
plt.plot(x_venus1,-np.log10(y_venus1),color='orange')
plt.scatter(tx_min*u.m.to('AU'), -np.log10(np.sqrt(a_s[0].to('AU').value**2. - (tx_min*u.m.to('AU'))**2.)), color='orange', marker='o')
plt.plot(np.zeros(2)+tx_min*u.m.to('AU'),np.linspace(start=-np.log10(1e2),stop=np.log10(np.sqrt(a_s[1].to('AU').value**2. - (tx_min*u.m.to('AU'))**2.)),num=2,endpoint=True),linestyle='--',color='k')
y_earth1 = np.sqrt(a_s[1].to('AU').value**2. - x_earth**2.)
plt.plot(x_earth,np.log10(y_earth1),color='green')
plt.plot(x_earth,-np.log10(y_earth1),color='green')
plt.scatter(tx5_min*u.m.to('AU'), -np.log10(np.sqrt(a_s[1].to('AU').value**2. - (tx5_min*u.m.to('AU'))**2.)), color='green', marker='o')
plt.scatter(tx4_min*u.m.to('AU'), -np.log10(np.sqrt(a_s[1].to('AU').value**2. - (tx4_min*u.m.to('AU'))**2.)), color='green', marker='o')
plt.scatter(tx_min*u.m.to('AU'), np.log10(np.sqrt(a_s[1].to('AU').value**2. - (tx_min*u.m.to('AU'))**2.)), color='green', marker='o')
y_neptune1 = np.sqrt(a_s[6].to('AU').value**2. - x_earth**2.)
plt.plot(x_earth,-np.log10(y_neptune1),color='blue')
plt.scatter(tx5_min*u.m.to('AU'), -np.log10(np.sqrt(a_s[6].to('AU').value**2. - (tx5_min*u.m.to('AU'))**2.)), color='blue',marker='o')
plt.plot(np.zeros(2)+tx5_min*u.m.to('AU'),np.linspace(start=-np.log10(np.sqrt(a_s[1].to('AU').value**2. - (tx5_min*u.m.to('AU'))**2.)),stop=-np.log10(1e2),num=2,endpoint=True),linestyle='--',color='k')
y_uranus1 = np.sqrt(a_s[5].to('AU').value**2. - x_earth**2.)
plt.plot(x_earth,-np.log10(y_uranus1),color='cyan')
plt.scatter(tx4_min*u.m.to('AU'),-np.log10(np.sqrt(a_s[5].to('AU').value**2. - (tx4_min*u.m.to('AU'))**2.)), color='cyan', marker='o')
plt.plot(np.zeros(2)+tx4_min*u.m.to('AU'),np.linspace(start=-np.log10(np.sqrt(a_s[1].to('AU').value**2. - (tx4_min*u.m.to('AU'))**2.)) ,stop=-np.log10(1e2),num=2,endpoint=True),linestyle='--',color='k')


plt.xlabel('Semi-Major Axis (AU)')
plt.show(block=False)






#### Plot dMag vs S for Solar System Planets ####################################################################################
plt.close(1)
plt.figure(num=1)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
betas = np.linspace(start=0.,stop=np.pi,num=1000,endpoint=True)*u.rad
Phis = calc_Phi(betas)
dMag_s = np.zeros((len(R_s),len(Phis)))
s_s = np.zeros((len(R_s),len(Phis)))
for i in np.arange(len(R_s)):
    for j in np.arange(len(Phis)):
        dMag_s[i,j] = deltaMag(p_s[i],R_s[i],a_s[i],Phis[j])
        s_s[i,j] = a_s[i].to('AU').value*np.sin(betas[j])
    plt.plot(s_s[i,:],dMag_s[i,:],color=color_s[i],label=label_s[i])
plt.xlabel('Planet-star Sepatation ' + r'$(s)$' + ' in AU',weight='bold')
plt.ylabel('Planet-star ' + r'$\Delta\mathrm{mag}$',weight='bold')
plt.xlim([0.,1.1*np.max(s_s)])
plt.ylim([20.,50.])
plt.legend()
plt.show(block=False)
#################################################################################################################################
plt.close('all')


def alpha_ap_D(D,a_p): #OK
    """ Given some Angle between r_earth/sun and r_planet/sun
    Args:
        D (float) - angle between r_earth/sun and r_planet/sun from 0 to 180 deg
        a_p (float) - planet star semi-major axis
    Return:
        alpha (float) - phase angle
    """
    a_earth = 1. #AU
    x_p = a_p*np.cos(D)
    y_p = a_p*np.sin(D)
    alpha = np.zeros(len(D))

    #ang2 extends beyone a_earth
    x_p_GT_a_earth_Inds = np.where(x_p > a_earth)[0]
    #if x_p > a_earth:
    ang1 = np.arctan2(x_p[x_p_GT_a_earth_Inds], y_p[x_p_GT_a_earth_Inds])
    ang2 = np.arctan2(np.abs(x_p[x_p_GT_a_earth_Inds]-a_earth),y_p[x_p_GT_a_earth_Inds]) #from a_earth and a_p aligned up to p directly above.
    alpha[x_p_GT_a_earth_Inds] = ang1 - ang2
    #elif x_p < a_earth and x_p > 0.:
    x_p_GTa_earth_GT0_Inds = np.where((x_p < a_earth)*(x_p > 0.))[0]
    ang1 = np.arctan2(x_p[x_p_GTa_earth_GT0_Inds], y_p[x_p_GTa_earth_GT0_Inds])
    ang2 = np.arctan2(a_earth - x_p[x_p_GTa_earth_GT0_Inds],y_p[x_p_GTa_earth_GT0_Inds])
    alpha[x_p_GTa_earth_GT0_Inds] = ang1 + ang2
    #else: #x_p < 0
    x_p_LT0_Inds = np.where(x_p < 0.)[0]
    ang1 = np.arctan2(np.abs(x_p[x_p_LT0_Inds]) + a_earth, y_p[x_p_LT0_Inds])
    ang2 = np.arctan2(np.abs(x_p[x_p_LT0_Inds]), y_p[x_p_LT0_Inds])
    alpha[x_p_LT0_Inds] = ang1 - ang2
    return alpha

def d_ap_D(D,a_p): #OK
    """ Calculate planet-Earth distance d given D
    D is the angle between r_k/star and r_earth/star
    """
    a_earth = 1. #AU
    d = np.sqrt(a_p**2. + a_earth**2. - 2.*a_p*a_earth*np.cos(D))
    return d

def alpha_crit_fromEarth(a_p): #OK
    """ Calculate the maximum phase angle for an object 
    Args:
        a_p (float) - planet semi-major axis in AU
    Results:
        alpha_max (float) - maximum alpha in radians
    """
    a_earth = 1. #in AU
    if a_p > a_earth:
        alpha_max = np.arcsin(a_earth/a_p) #occurs at quadrature
    else: #if a_p < a_earth:
        alpha_max = np.pi #DELETEnp.arctan2(a_earth,a_p)
    return alpha_max

def D_alphacrit(a_p): #OK
    """ This assumes alpha_crit occurs when planet is at quadrature (same x location but different y)
    Returns:
        Dcrit (float) - the D angle where theta crit occurs in radians
    """
    Dcrit = np.pi - np.pi/2. - alpha_crit_fromEarth(a_p)
    return Dcrit

def d_crit_ap(a_p): #OK
    """ We know this because at quadrature, alpha is at its maximum
    Additionally, we assume this maximum occurs when the planets for a right triangle
    """
    a_earth = 1 #AU
    dcrit = np.sqrt(a_p**2. - a_earth**2.)
    return dcrit

Ds = np.linspace(start=0.,stop=np.pi,num=100)
alphas1 = alpha_ap_D(Ds,0.7)
alphas2 = alpha_ap_D(Ds,1.5)
ds1 = d_ap_D(Ds,0.7)
ds2 = d_ap_D(Ds,1.5)
plt.close(99)
plt.figure(num=99)
plt.plot(1.5*np.cos(Ds),1.5*np.sin(Ds),color='blue')
plt.show(block=False)
plt.close(199)
plt.figure(num=199)
plt.plot(Ds,alphas1,color='red')
plt.plot(Ds,alphas2,color='green')
plt.xlabel('Ds',weight='bold')
plt.ylabel('alphas',weight='bold')
plt.show(block=False)
plt.close(299)
plt.figure(num=299)
plt.plot(Ds,ds1,color='red')
plt.plot(Ds,ds2,color='green')
plt.xlabel('Ds',weight='bold')
plt.ylabel('ds',weight='bold')
plt.show(block=False)


def calc_Vmag(solarFlux, bulk_albedo, r_sun_sc, nhat, A_aperture, r_gs_sc, vegaFlux):
    """ The fully parameterized equation for visual magnitude
    Args:
        solarFlux (float) - solar flux in W/m^2
        bulk_albedo (float) - Bulk Albedo of spacecraft surface
        r_sun_sc (numpy array) - 3 component vector of sun from spacecraft
        nhat (numpy array) - 3 component unit vector describing spacecraft surface normal
        A_aperture (float) - Area of telescope aperture in m^2
        r_gs_sc (numpy array) - 3 component vector of ground station w.r.t SC
        vegaFlux (float) - Flux From Vega in W/m^2
    """
    numerator = solarFlux*(1.-bulk_albedo)*\
        (no.dot(r_sun_sc,nhat)/np.linalg.norm(r_sun_sc))*\
        (A_aperture/np.linalg.norm(r_gs_sc)**3.)*\
        (np.dot(r_gs_sc,nhat))
    denominator = 2.*np.pi*vegaFlux
    Vmag = -2.5*np.log10(numerator/denominator)
    return Vmag

# def calc_elongation_alpha_ap(alpha,a_p):
#     """ Calculates elongation given phase angle and planet semi-major axis
#     https://en.wikipedia.org/wiki/Elongation_(astronomy)
#     """
#     return elongation


def d_planet_earth_D(D,a_p):
    """ Assuming circular orbits for the Earth and a general planet p, the earth-planet distances 
        is directly calculable from phase angle
    Args:
        D (float) - angular position of planet around sun relative to Earth position ranging from 0 to pi in deg
        a_p (float) - planet semi-major axis in AU
    Returns:
        d (float) - Earth to planet distance at given phase angle in AU
    """
    a_earth = 1. #in AU
    d = np.sqrt(a_p**2. + a_earth**2. - 2.*a_p*a_earth*np.cos(D))
    return d

# def d_planet_earth_alpha(alpha,a_p):
#     """ Assuming circular orbits for the Earth and a general planet p, the earth-planet distances 
#         is directly calculable from phase angle
#     Args:
#         alpha (float) - angle formed between sun-planet and planet-Earth vectors ranging from 0 to pi in deg
#         a_p (float) - planet semi-major axis in AU
#     Returns:
#         d (float) - Earth to planet distance at given phase angle in AU
#     """
#     a_earth = 1. #in AU
#     #alpha_crit = np.arcsin(a_earth/a_p)
#     #alpha_crit2 = np.arctan2(a_earth,a_p)

#     alpha_max = alpha_crit_fromEarth(a_p)*np.pi/180. #in deg
#     assert np.all(alpha < alpha_max), "an alpha is above the maximum possible alpha"

#     if a_p > a_earth:
#         #There are two distances satisfying this solution
#     else:


#     inds = np.arange(len(alpha))
#     oneEightyInds = np.where(alpha == 180.)[0]
#     zeroInds = np.where(alpha == 0.)[0]
#     inner = a_p*np.sin(alpha*np.pi/180.)/a_earth
#     inner[oneEightyInds] = 0.
#     inner[zeroInds] = 0.
#     # if inner > 1.: #Angle is actually complement
#     #     inner = a_p*np.sin(np.pi - alpha*np.pi/180.)/a_earth
#     assert np.all(a_p*np.sin(alpha*np.pi/180.)/a_earth >= -1.), 'arcsin below range'
#     assert np.all(a_p*np.sin(alpha*np.pi/180.)/a_earth <= 1.), 'arcsin above range'
#     d = np.sqrt(a_p**2. + a_earth**2. - 2.*a_p*a_earth*np.cos( 180.*np.pi/180. - alpha*np.pi/180. - np.arcsin(inner)))
#     return d

def elongation_max(a_p):
    """
    Return:
        elongation (float) - maximum elongation in radians
    """
    a_earth = 1. #AU
    if a_p > a_earth:
        elongation = np.pi
    else:
        elongation = np.arctan2(a_p,a_earth)

    return elongation

def d_D_test():
    return d

def d_alpha_test(alpha,a_p):
    a_earth = 1.
    d = np.sqrt(a_p**2. + a_earth**2. + 2.*a_p*a_earth*(np.cos(alpha)*np.sqrt(1.-a_p**2./a_earth**2.*np.sin(alpha)**2.) - a_p/a_earth*np.sin(alpha)**2.))
    return d

alpha = np.linspace(start=0., stop=alpha_crit_fromEarth(1.5),num=30)
d = d_alpha_test(alpha,1.5)

plt.close(999)
plt.figure(num=999)
plt.plot(1.+d*np.cos(alpha),d*np.sin(alpha))
plt.show(block=False)

print(saltyburrito)

def phi_lambert(alpha):
    """ Lambert phase function as presented in Garrett2016
    Args:
        alpha (float) - phase angle
    Returns:
        Phi (float) - phase function value between 0 and 1
    """
    phi = (np.sin(alpha) + (np.pi-alpha)*np.cos(alpha))/np.pi
    return phi

def transitionStart(x,a,b):
    """ Smoothly transition from one 0 to 1
    """
    s = 0.5+0.5*np.tanh((x-a)/b)
    return s
def transitionEnd(x,a,b):
    """ Smoothly transition from one 1 to 0
    Smaller b is sharper step
    a is midpoint, s(a)=0.5
    """
    s = 0.5-0.5*np.tanh((x-a)/b)
    return s

#### Planet Visual Magnitudes From Mallama 2018 #################################################################################
#Mercury
#r distance of planet from sun
#d distance of planet from Earth
#V = 5.*np.log10(r*d) - 0.613 + 6.3280e-02*alpha - 1.6336e-03*alpha**2. + 3.3644e-05*alpha**3. - 3.4265e-07*alpha**4. + 1.6893e-09*alpha**5. - 3.0334e-12*alpha**6.
#planProp['mercury'] = {'numV':1,'alpha_mins':[0.],'alpha_maxs':[180.],'V':[5.*np.log10(r*d) - 0.613 + 6.3280e-02*alpha - 1.6336e-03*alpha**2. + 3.3644e-05*alpha**3. - 3.4265e-07*alpha**4. + 1.6893e-09*alpha**5. - 3.0334e-12*alpha**6.]}
def V_magMercury(alpha,a_p,d):
    """ Valid from 0 to 180 deg
    """
    V = 5.*np.log10(a_p*d) - 0.613 + 6.3280e-02*alpha - 1.6336e-03*alpha**2. + 3.3644e-05*alpha**3. - 3.4265e-07*alpha**4. + 1.6893e-09*alpha**5. - 3.0334e-12*alpha**6.
    return V

#Venus
#0<alpha<163.7
#V = 5.*np.log10(r*d) - 4.384 - 1.044e-03*alpha + 3.687e-04*alpha**2. - 2.814e-06*alpha**3. + 8.938e-09*alpha**4.
#163.7<alpha<179
#V = 5.*np.log10(r*d) + 236.05828 - 2.81914e-00*alpha + 8.39034e-03*alpha**2.
#planProp['venus'] = {'numV':2,'alpha_mins':[0.,163.7],'alpha_maxs':[163.7,179.],'V':[5.*np.log10(r*d) - 4.384 - 1.044e-03*alpha + 3.687e-04*alpha**2. - 2.814e-06*alpha**3. + 8.938e-09*alpha**4.,\
#    5.*np.log10(r*d) + 236.05828 - 2.81914e-00*alpha + 8.39034e-03*alpha**2.]}
def V_magVenus_1(alpha, a_p, d):
    """ Valid from 0 to 163.7 deg
    """
    V = 5.*np.log10(a_p*d) - 4.384 - 1.044e-03*alpha + 3.687e-04*alpha**2. - 2.814e-06*alpha**3. + 8.938e-09*alpha**4.
    return V
def V_magVenus_2(alpha, a_p, d):
    """ Valid from 163.7 to 179 deg
    """
    V = 5.*np.log10(a_p*d) + 236.05828 - 2.81914e-00*alpha + 8.39034e-03*alpha**2.
    return V


#Earth
#V = 5.*np.log10(r*d) - 3.99 - 1.060e-3*alpha + 2.054e-4*alpha**2.
#planProp['earth'] = {'numV':1,'alpha_mins':[0.],'alpha_maxs':[180.],'V':[5.*np.log10(r*d) - 3.99 - 1.060e-3*alpha + 2.054e-4*alpha**2.]}
def V_magEarth(alpha,a_p,d):
    """ Valid from 0 to 180 deg
    """
    V = 5.*np.log10(a_p*d) - 3.99 - 1.060e-3*alpha + 2.054e-4*alpha**2.
    return V

#Mars
#alpha<=50
#V = 5.*np.log10(r*d) - 1.601 + 0.02267*alpha - 0.0001302*alpha**2.+ L(λe) + L(LS)
#alpha > 50 approximated by average dimming magnitudes for mercury and Earth
#V = 5.*np.log10(r*d) - 0.367 - 0.02573*alpha + 0.0003445*alpha**2. + L(λe) + L(Ls)
#Find L
#planProp['mars'] = {'numV':2,'alpha_mins':[0.,50.],'alpha_maxs':[50.,180.],'V':[5.*np.log10(r*d) - 1.601 + 0.02267*alpha - 0.0001302*alpha**2.+ L(λe) + L(LS),\
#    5.*np.log10(r*d) - 0.367 - 0.02573*alpha + 0.0003445*alpha**2. + L(λe) + L(Ls)]}
def V_magMars_1(alpha,a_p,d):
    """ Valid from 0 to 50 deg
    """
    V = 5.*np.log10(a_p*d) - 1.601 + 0.02267*alpha - 0.0001302*alpha**2.+ 0. + 0.#L(λe) + L(LS)
    return V
def V_magMars_2(alpha,a_p,d):
    """ Valid from 50 to 180 deg
    """
    V = 5.*np.log10(a_p*d) - 0.367 - 0.02573*alpha + 0.0003445*alpha**2. + 0. + 0. #L(λe) + L(Ls)
    return V

#Jupiter
#alpha<12
# V = 5.*np.log10(r*d) - 9.395 - 3.7e-04*alpha + 6.16e-04*alpha**2.
#12<alpha<130
# V = 5.*np.log10(r*d) - 9.428 - 2.5*np.log10(1.0 - 1.507*(alpha/180.)\
#     - 0.363*(alpha/180.)**2. - 0.062*(alpha/180.)**3.\
#     + 2.809*(alpha/180.)**4. - 1.876*(alpha/180.)**5.)
#no data beyond 130 deg
#planProp['jupiter'] = {'numV':2,'alpha_mins':[0.,12.],'alpha_maxs':[12.,130.],'V':[5.*np.log10(r*d) - 9.395 - 3.7e-04*alpha + 6.16e-04*alpha**2.,\
#    5.*np.log10(r*d) - 9.428 - 2.5*np.log10(1.0 - 1.507*(alpha/180.) - 0.363*(alpha/180.)**2. - 0.062*(alpha/180.)**3.+ 2.809*(alpha/180.)**4. - 1.876*(alpha/180.)**5.)]}

def V_magJupiter_1(alpha,a_p,d):
    """ Valid from 0 to 12 deg
    """
    V = 5.*np.log10(a_p*d) - 9.395 - 3.7e-04*alpha + 6.16e-04*alpha**2.
    return V
def V_magJupiter_2(alpha,a_p,d):
    """ Valid from 12 to 130 deg
    """
    V = 5.*np.log10(a_p*d) - 9.428 - 2.5*np.log10(1.0 - 1.507*(alpha/180.) - 0.363*(alpha/180.)**2. - 0.062*(alpha/180.)**3.+ 2.809*(alpha/180.)**4. - 1.876*(alpha/180.)**5.)
    return V

#Saturn
#V = 5.*np.log10(r*d) - 8.914 - 1.825*np.sin(beta) + 0.026*alpha \
#    - 0.378*np.sin(beta)*np.exp(-2.25*alpha)

#6<alpha<150 this approximates the globe of saturn only, not the rings
#V = 5.*np.log10(r*d) - 8.94 + 2.446e-4*alpha \
#    + 2.672e-4*alpha**2. - 1.505e-6*alpha**3. + 4.767e-9*alpha**4.

#Not enough data to include saturn's rings for alpha>6.5
def V_magSaturn_1(alpha,a_p,d,beta):
    """ Valid alpha from 0 to 6.5 deg
    Valid beta from 0 to 27
    Globe and Rings
    beta in deg
    """
    V = 5.*np.log10(a_p*d) - 8.914-1.825*np.sin(beta*np.pi/180.) + 0.026*alpha - 0.378*np.sin(beta*np.pi/180.)*np.exp(-2.25*alpha)
    return V
def V_magSaturn_2(alpha,a_p,d):
    """ Valid alpha from 0 to 6.5 deg
    Saturn Globe Only Earth Observations
    """
    V = 5.*np.log10(a_p*d) - 8.95 - 3.7e-04*alpha +6.16e-04*alpha**2.
    return V
def V_magSaturn_3(alpha,a_p,d):
    """ Valid alpha from 6 to 150. deg
    Saturn Globe Only Pioneer Observations
    """
    V = 5.*np.log10(a_p*d) - 8.94 + 2.446e-4*alpha + 2.672e-4*alpha**2. - 1.505e-6*alpha**3. + 4.767e-9*alpha**2.
    return V    

#Uranus
#f = 0.0022927 #flattening of the planet
#phi = #planetocentric latitude
#Phi ranges from -82 to 82
#phi_prime = np.arctan2(np.tan(phi),(1.-f)**2.) #planetographic latitude
#V = 5.*np.log10(r*d) - 7.110 - 8.4E-04*phi + 6.587E-3*alpha + 1.045E-4*alpha**2.
def phiprime_phi(phi):
    """ Valid for phi from -82 to 82 deg
    Returns:
        phiprime (float) - in deg
    """
    f = 0.0022927
    phiprime = np.arctan2(np.tan(phi*np.pi/180.),(1.-f)**2.)*180./np.pi
    return phiprime
def V_magUranus(alpha,a_p,d,phi):
    """ Valid for alpha 0 to 154 deg
    phi in deg
    """
    V = 5.*np.log10(a_p*d) - 7.110 - 8.4e-04*phiprime_phi(phi) + 6.587e-3*alpha + 1.045e-4*alpha**2.
    return V

#Neptune
def V_magNeptune(alpha,a_p,d):
    """ Valid for alpha 0 to 133.14 deg
    """
    V = 5.*np.log10(a_p*d) - 7.00 + 7.944e-3*alpha + 9.617e-5*alpha**2.
    return V

#### Possible From Earth Alphas Range
alpha_max_mercury = alpha_crit_fromEarth(planProp['mercury']['a']*u.m.to('AU'))
alphasmax_mercury = np.linspace(start=0.,stop=alpha_max_mercury*180./np.pi,num=100.,endpoint=True)
alpha_max_venus = alpha_crit_fromEarth(planProp['venus']['a']*u.m.to('AU'))
alphasmax_venus = np.linspace(start=0.,stop=alpha_max_venus*180./np.pi,num=100.,endpoint=True)
alpha_max_earth = alpha_crit_fromEarth(planProp['earth']['a']*u.m.to('AU'))
alphasmax_earth = np.linspace(start=0.,stop=alpha_max_earth*180./np.pi,num=100.,endpoint=True)
alpha_max_mars = alpha_crit_fromEarth(planProp['mars']['a']*u.m.to('AU'))
alphasmax_mars = np.linspace(start=0.,stop=alpha_max_mars*180./np.pi,num=100.,endpoint=True)
alpha_max_jupiter = alpha_crit_fromEarth(planProp['jupiter']['a']*u.m.to('AU'))
alphasmax_jupiter = np.linspace(start=0.,stop=alpha_max_jupiter*180./np.pi,num=100.,endpoint=True)
alpha_max_saturn = alpha_crit_fromEarth(planProp['saturn']['a']*u.m.to('AU'))
alphasmax_saturn = np.linspace(start=0.,stop=alpha_max_saturn*180./np.pi,num=100.,endpoint=True)
alpha_max_uranus = alpha_crit_fromEarth(planProp['uranus']['a']*u.m.to('AU'))
alphasmax_uranus = np.linspace(start=0.,stop=alpha_max_uranus*180./np.pi,num=100.,endpoint=True)
alpha_max_neptune = alpha_crit_fromEarth(planProp['neptune']['a']*u.m.to('AU'))
alphasmax_neptune = np.linspace(start=0.,stop=alpha_max_neptune*180./np.pi,num=100.,endpoint=True)
V_magsMercury_Earth = V_magMercury(alphasmax_mercury,planProp['mercury']['a']*u.m.to('AU'),d_planet_earth_alpha(alphasmax_mercury,planProp['mercury']['a']*u.m.to('AU')))
V_magsVenus_Earth = V_magVenus_1(alphasmax_venus,planProp['venus']['a']*u.m.to('AU'),d_planet_earth_alpha(alphasmax_venus,planProp['venus']['a']*u.m.to('AU')))
V_magsEarth_Earth = V_magEarth(alphasmax_earth,planProp['earth']['a']*u.m.to('AU'),d_planet_earth_alpha(alphasmax_earth,planProp['earth']['a']*u.m.to('AU')))
V_magsMars_Earth = V_magMars_1(alphasmax_mars,planProp['mars']['a']*u.m.to('AU'),d_planet_earth_alpha(alphasmax_mars,planProp['mars']['a']*u.m.to('AU')))
V_magsJupiter_Earth = V_magJupiter_1(alphasmax_jupiter,planProp['jupiter']['a']*u.m.to('AU'),d_planet_earth_alpha(alphasmax_jupiter,planProp['jupiter']['a']*u.m.to('AU')))
V_magsSaturn_Earth = V_magSaturn_1(alphasmax_saturn,planProp['saturn']['a']*u.m.to('AU'),d_planet_earth_alpha(alphasmax_saturn,planProp['saturn']['a']*u.m.to('AU')),beta=0.)
V_magsUranus_Earth = V_magUranus(alphasmax_uranus,planProp['uranus']['a']*u.m.to('AU'),d_planet_earth_alpha(alphasmax_uranus,planProp['uranus']['a']*u.m.to('AU')),phi=-82.)
V_magsNeptune_Earth = V_magNeptune(alphasmax_neptune,planProp['neptune']['a']*u.m.to('AU'),d_planet_earth_alpha(alphasmax_neptune,planProp['neptune']['a']*u.m.to('AU')))

plt.close(10)
fig10 = plt.figure(num=10)
plt.plot(alphasmax_mercury,V_magsMercury_Earth,color='gray',label='mercury')
plt.plot(alphasmax_venus,V_magsVenus_Earth,color='yellow',label='venus')
plt.plot(alphasmax_earth,V_magsEarth_Earth,color='blue',label='earth')
plt.plot(alphasmax_mars,V_magsMars_Earth,color='red',label='mars')
plt.plot(alphasmax_jupiter,V_magsJupiter_Earth,color='orange',label='jupiter')
plt.plot(alphasmax_saturn,V_magsSaturn_Earth,color='gold',label='saturn')
plt.plot(alphasmax_uranus,V_magsUranus_Earth,color='blue',label='uranus',linestyle='--')
plt.plot(alphasmax_neptune,V_magsNeptune_Earth,color='cyan',label='neptune')
plt.xlim([0.,90.])
plt.ylim([-10.,10.])
plt.ylabel('Visual Apparent Magnitude', weight='bold')
plt.xlabel('Phase Angle in deg', weight='bold')
plt.legend()
plt.show(block=False)


#### Full Range of Vmag equations in Mallama
alphas_mercury = np.linspace(start=0.,stop=180.,num=np.ceil(180./3.),endpoint=True)
alphas_venus1 = np.linspace(start=0.,stop=163.7,num=np.ceil(163.7/3.),endpoint=True)
alphas_venus2 = np.linspace(start=163.7,stop=179.,num=np.ceil((179.-163.7)/3.),endpoint=True)
alphas_earth = np.linspace(start=0.,stop=180.,num=np.ceil(180./3.),endpoint=True)
alphas_mars1 = np.linspace(start=0.,stop=50.,num=np.ceil(50./3.),endpoint=True)
alphas_mars2 = np.linspace(start=50.,stop=180.,num=np.ceil((180.-50.)/3.),endpoint=True)
alphas_jupiter1 = np.linspace(start=0.,stop=12.,num=np.ceil(12./3.),endpoint=True)
alphas_jupiter2 = np.linspace(start=12.,stop=130.,num=np.ceil((130.-12.)/3.),endpoint=True)
alphas_saturn1 = np.linspace(start=0.,stop=6.5,num=np.ceil(6.5/3.),endpoint=True)
betas_saturn1 = np.linspace(start=0.,stop=27.,num=np.ceil(27./3.),endpoint=True)
alphas_saturn2 = np.linspace(start=0.,stop=6.5,num=np.ceil(6.5/3.),endpoint=True)
alphas_saturn3 = np.linspace(start=0.,stop=6.5,num=np.ceil(6.5/3.),endpoint=True)
alphas_saturn4 = np.linspace(start=6.,stop=150.,num=np.ceil((150.-6.5)/3.),endpoint=True)
alphas_uranus = np.linspace(start=0.,stop=154.,num=np.ceil(154./3.),endpoint=True)
phis_uranus = np.linspace(start=-82,stop=82.,num=np.ceil((82+82)/3.),endpoint=True)
alphas_neptune = np.linspace(start=0.,stop=133.14,num=np.ceil(133.14/3.),endpoint=True)
V_magsMercury = V_magMercury(alphas_mercury,planProp['mercury']['a']*u.m.to('AU'))
V_magsVenus_1 = V_magVenus_1(alphas_venus1,planProp['venus']['a']*u.m.to('AU'))
V_magsVenus_2 = V_magVenus_2(alphas_venus2,planProp['venus']['a']*u.m.to('AU'))
V_magsEarth = V_magEarth(alphas_earth,planProp['earth']['a']*u.m.to('AU'))
V_magsMars_1 = V_magMars_1(alphas_mars1,planProp['mars']['a']*u.m.to('AU'))
V_magsMars_2 = V_magMars_2(alphas_mars2,planProp['mars']['a']*u.m.to('AU'))
V_magsJupiter_1 = V_magJupiter_1(alphas_jupiter1,planProp['jupiter']['a']*u.m.to('AU'))
V_magsJupiter_2 = V_magJupiter_2(alphas_jupiter2,planProp['jupiter']['a']*u.m.to('AU'))
V_magsSaturn_1 = V_magSaturn_1(alphas_saturn1,planProp['saturn']['a']*u.m.to('AU'),beta=0.)
V_magsSaturn_2 = V_magSaturn_1(alphas_saturn2,planProp['saturn']['a']*u.m.to('AU'),beta=27.)
V_magsSaturn_3 = V_magSaturn_2(alphas_saturn3,planProp['saturn']['a']*u.m.to('AU'))
V_magsSaturn_4 = V_magSaturn_3(alphas_saturn4,planProp['saturn']['a']*u.m.to('AU'))
V_magsUranus_1 = V_magUranus(alphas_uranus,planProp['uranus']['a']*u.m.to('AU'),phi=-82.)
V_magsUranus_2 = V_magUranus(alphas_uranus,planProp['uranus']['a']*u.m.to('AU'),phi=0.)
V_magsUranus_3 = V_magUranus(alphas_uranus,planProp['uranus']['a']*u.m.to('AU'),phi=82.)
V_magsNeptune = V_magNeptune(alphas_neptune,planProp['neptune']['a']*u.m.to('AU'))

#### Plot Raw plane Visual Apparent Magnitudes ######################################################################
plt.close(1)
fig1 = plt.figure(num=1)
plt.plot(alphas_mercury,V_magsMercury,color='gray')
plt.plot(alphas_venus1,V_magsVenus_1,color='yellow', marker='x')
plt.plot(alphas_venus2,V_magsVenus_2,color='yellow',linestyle='--')
plt.plot(alphas_earth,V_magsEarth,color='blue')
plt.plot(alphas_mars1,V_magsMars_1,color='red', marker='x')
plt.plot(alphas_mars2,V_magsMars_2,color='red',linestyle='--')
plt.plot(alphas_jupiter1,V_magsJupiter_1,color='orange', marker='x')
plt.plot(alphas_jupiter2,V_magsJupiter_2,color='orange', linestyle='--')
plt.plot(alphas_saturn1,V_magsSaturn_1,color='gold', marker='x')
plt.plot(alphas_saturn2,V_magsSaturn_2,color='gold',linestyle='--')
plt.plot(alphas_saturn3,V_magsSaturn_3,color='gold',linestyle='.')
plt.plot(alphas_saturn4,V_magsSaturn_4,color='gold',linestyle='.-')
plt.plot(alphas_uranus,V_magsUranus_1,color='blue', marker='x')
plt.plot(alphas_uranus,V_magsUranus_2,color='blue', linestyle='--')
plt.plot(alphas_uranus,V_magsUranus_3,color='blue',linestyle='.')
plt.plot(alphas_neptune,V_magsNeptune,color='cyan')
plt.show(block=False)
######################################################################################################################





#Saturn geometric functions
"""The light reflected by Saturn and Saturn's rings is a combination of geometric functions.
The central body is a sphere with, under some viewing angles, some portion of the body obstructed by Saturn's rings.
The rings are effectively a flat disk with, under some viewing angles, some portion of the rings obstructed by Saturn.
"""
# rings = dict()
# rings['D'] = {'widths':[66000.*1000.,74000.*1000.],'opticalDepth':10e-3,'albedo':}
# rings['C'] = {'widths':[74490.*1000.,91983.*1000.],'opticalDepth':0.1,'albedo':0.2}
# rings['B'] = {'widths':[91983.*1000.,117516.*1000.],'opticalDepth':2.75,'albedo':0.5}
# rings['CassiniDivision'] = {'widths':[117516.*1000.,122053.*1000.],'opticalDepth':0.15,'albedo':0.2}
# rings['A'] = {'widths':[122053.*1000.,136774.*1000.],'opticalDepth':0.5,'albedo':0.5}
# rings['F'] = {'widths':[140200.*1000.,140250.*1000.],'opticalDepth':0.5,'albedo':}
# rings['G'] = {'widths':[166000.*1000.,173000.*1000.],'opticalDepth':10e-6,'albedo':}
# rings['E'] = {'widths':[180000.*1000.,450000.*1000.],'opticalDepth':10e-5,'albedo':}
#https://www.aanda.org/articles/aa/pdf/2015/11/aa26673-15.pdf
#https://iopscience.iop.org/article/10.1086/426050/pdf




