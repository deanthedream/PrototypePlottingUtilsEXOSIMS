# Plot dmag vs s for a specific planet given KOE

import matplotlib.pyplot as plt
import numpy as np
import eqnsEXOSIMS2020
from EXOSIMS.util.deltaMag import deltaMag
#DELETE import EXOSIMS.Prototypes.PlanetPhysicalModel as PPM #where Phiu_lambert is
from EXOSIMS.util.phiLambert import phiLambert
import astropy.units as u

#ellipse area A=a*b*pi

#### Circumference of ellipse
#Solve complete elliptic integral of the second kind (wikipedia)
#https://en.wikipedia.org/wiki/Elliptic_integral#Complete_elliptic_integral_of_the_second_kind

def elliptic_integral_of_second_kind_power_series(k,nmax=10):
    """Note: E(e) = 1/4 circumference of ellipse
    Args:
        k (float) - 
        nmax (int) - number to expand to
    """
    Ek = np.pi/2.*np.sum([(np.math.factorial(2.*n)/(2.**(2.*n)*np.math.factorial(n)**2.))**2. * (k**(2.*n))/(1.-2.*n) for n in np.arange(nmax)])
    return Ek

def circ_from_E(a,e):
    """Uses power series expansion of elliptic integral of second kind
    Returns:
        c (float):
            curcimference
    """
    c = 4.*a*elliptic_integral_of_second_kind_power_series(e)
    return c

def r_koe(a,e,nu):
    """
    Args:
        a (float):
            semi-major axis
        e (float):
            semi-minor axis
        nu (float):
            true anomaly
    Returns:
        r (float):
            orbital radius
    """
    r = a*(1.-e**2.)/(1.+e*np.cos(nu))
    return r

def dr_koe(a,e,nu):
    """
    Args:
        a (float):
            semi-major axis
        e (float):
            semi-minor axis
        nu (float):
            true anomaly
    Returns:
        dr (float):
            delta orbital radius
    """
    dr = (e*(e**2.-1.)*a*np.sin(nu))/(e*np.cos(nu)+1.)**2.
    return dr

def s_koe(a,e,nu,inc,omega):
    """ Equation from paper of sep vs koe (except Omega)
    Returns:
        s (float):
            planet-star separation
    """
    s = r_koe(a,e,nu)*np.sqrt(np.cos(omega+nu)**2. + np.sin(omega+nu)**2.*np.cos(inc)**2.)
    return s

def plotdmagvss(sma,eccen,inc,omega,Omega,ax=None,num=None):
    #need a mechanism for calculating evenly spaced distribution of nu
    circ = circ_from_E(sma,eccen)
    b = np.sqrt(sma**2.*(1.-eccen**2.)) #semi-minor axis
    area = np.pi*sma*b #area inside the ellipse

    #Note: We only need to evenly distribute over area bc equal area
    #in equal time is one of Kepler's laws
    numPts=200 #number of points to get
    dA = area/numPts #each dA aiming to achieve
    rs = np.zeros(numPts)#All the r to use
    nus = np.zeros(numPts)#All the nu to use
    xs = np.zeros(numPts)
    ys = np.zeros(numPts)
    zs = np.zeros(numPts)
    rs[0] = r_koe(sma,eccen,0.) #Set initial r
    nus[0] = 0.
    #EXAMPLE nus[1] = nus[0] + 2.*dA/(rs[0]*(rs[0]-dr_koe(a,e,nus[0])))
    xs[0] = x_koe(rs[0],inc,omega,Omega,nus[0])
    ys[0] = y_koe(rs[0],inc,omega,Omega,nus[0])
    zs[0] = z_koe(rs[0],inc,omega,Omega,nus[0])
    #EXAMPLE nus[1] = nus[0] + 2.*dA/(rs[0]*(rs[0]-dr_koe(a,e,nus[0])))
    for i in np.arange(numPts-1)+1:
        nus[i] = nus[i-1] + 2.*dA/(rs[i-1]*(rs[i-1]-dr_koe(sma,eccen,nus[i-1])))
        rs[i] = r_koe(sma,eccen,nus[i])
        xs[i] = x_koe(rs[i],inc,omega,Omega,nus[i])
        ys[i] = y_koe(rs[i],inc,omega,Omega,nus[i])
        zs[i] = z_koe(rs[i],inc,omega,Omega,nus[i])

    ss = s_koe(sma,eccen,nus,inc,omega)
    betas = np.arcsin(zs/rs) #From my paper
    Phis = phiLambert(betas*u.rad)
    dmags = deltaMag(p=1.,Rp=1.*u.earthRad,d=rs*u.AU,Phi=Phis)
    if ax == None:
        plt.figure()
        plt.plot(ss,dmags)
        plt.xlabel('s')
        plt.ylabel('dmag')
        plt.show(block=False)
        return None
    else:
        ax.plot(ss,dmags)
        #ax.scatter(ss,dmags,s=4)
        #E = np.pi/2.-omega#-omega/2.#-omega
        #nu = np.arccos((np.cos(E)-eccen)/(1.-eccen*np.cos(E))) + omega/2.*np.cos(inc)
        nu = np.arccos(-eccen)
        s = s_koe(sma,eccen,nu,inc,omega)
        r = r_koe(sma,eccen,nu)
        z = z_koe(r,inc,omega,Omega,nu)
        beta = np.arcsin(z/r) #From my paper
        Phi = phiLambert(beta*u.rad)
        dmag = deltaMag(p=1.,Rp=1.*u.earthRad,d=r*u.AU,Phi=Phi)
        ax.scatter(s,dmag)

        ra = sma*(1.+eccen)
        #DELETEax.plot([ra*np.sin(inc),ra*np.sin(inc)],[20.,23])
        ax.set_xlim([0.,ra])
        plt.show(block=False)
        return ax

def x_koe(r,inc,omega,Omega,nu):
    return r*(np.cos(Omega)*np.cos(omega+nu)-np.sin(Omega)*np.sin(omega+nu)*np.cos(inc))

def y_koe(r,inc,omega,Omega,nu):
    return r*(np.sin(Omega)*np.cos(omega+nu)+np.cos(Omega)*np.sin(omega+nu)*np.cos(inc))

def z_koe(r,inc,omega,Omega,nu):
    return r*np.sin(inc)*np.sin(omega+nu)

def plotxyvskoe(sma,eccen,inc,omega,Omega,ax=None,num=None):
    #need a mechanism for calculating evenly spaced distribution of nu
    circ = circ_from_E(sma,eccen)
    b = np.sqrt(sma**2.*(1.-eccen**2.)) #semi-minor axis
    area = np.pi*sma*b #area inside the ellipse

    #Note: We only need to evenly distribute over area bc equal area
    #in equal time is one of Kepler's laws
    numPts=200 #number of points to get
    dA = area/numPts #each dA aiming to achieve
    rs = np.zeros(numPts)#All the r to use
    nus = np.zeros(numPts)#All the nu to use
    xs = np.zeros(numPts)
    ys = np.zeros(numPts)
    zs = np.zeros(numPts)
    rs[0] = r_koe(sma,eccen,0.) #Set initial r
    nus[0] = 0.
    xs[0] = x_koe(rs[0],inc,omega,Omega,nus[0])
    ys[0] = y_koe(rs[0],inc,omega,Omega,nus[0])
    zs[0] = z_koe(rs[0],inc,omega,Omega,nus[0])
    #EXAMPLE nus[1] = nus[0] + 2.*dA/(rs[0]*(rs[0]-dr_koe(a,e,nus[0])))
    for i in np.arange(numPts-1)+1:
        nus[i] = nus[i-1] + 2.*dA/(rs[i-1]*(rs[i-1]-dr_koe(sma,eccen,nus[i-1])))
        rs[i] = r_koe(sma,eccen,nus[i])
        xs[i] = x_koe(rs[i],inc,omega,Omega,nus[i])
        ys[i] = y_koe(rs[i],inc,omega,Omega,nus[i])
        zs[i] = z_koe(rs[i],inc,omega,Omega,nus[i])

    ss = s_koe(sma,eccen,nus,inc,omega)
    betas = np.arcsin(ss/rs) #From my paper
    Phis = phiLambert(betas*u.rad)
    dmags = deltaMag(p=1.,Rp=1.*u.earthRad,d=rs*u.AU,Phi=Phis)
    if ax == None:
        plt.figure()
        plt.plot(xs,ys)
        ra = sma*(1.+eccen)
        plt.ylim([-ra,ra])
        plt.xlim([-ra,ra])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show(block=False)
        return None
    else:
        ax.plot(xs,ys)
        #E = np.pi/2.-omega#-omega/2.#-omega
        #nu = np.arccos((np.cos(E)-eccen)/(1.-eccen*np.cos(E))) + omega/2.*np.cos(inc)
        #Plot a specific point on orbit
        nu = np.arccos(-eccen)
        r = r_koe(sma,eccen,nu)#DELETE
        x = x_koe(r,inc,omega,Omega,nu)#DELETE
        y = y_koe(r,inc,omega,Omega,nu)#DELETE
        z = z_koe(r,inc,omega,Omega,nu)#DELETE
        ax.scatter(x,y)#DELETE
        #Plot vector from Foci to Origin
        c = 
        ra = sma*(1.+eccen)
        ax.set_ylim([-ra,ra])
        ax.set_xlim([-ra,ra])
        plt.show(block=False)
        return ax

#DELETE E = elliptic_integral_of_second_kind_power_series(e)

#### Good dMag vs s Plots for several orbits
fig0, ax0 = plt.subplots(nrows=2,ncols=3,num=500)
ax0[0,0] = plotdmagvss(sma=1.,eccen=0.2,inc=0.,       omega=0.,Omega=0.,ax=ax0[0,0])
ax0[0,1] = plotdmagvss(sma=1.,eccen=0.2,inc=np.pi/10.,omega=0.,Omega=0.,ax=ax0[0,1])
ax0[0,2] = plotdmagvss(sma=1.,eccen=0.2,inc=np.pi/2., omega=0.,Omega=0.,ax=ax0[0,2])
ax0[1,0] = plotdmagvss(sma=1.,eccen=0.2,inc=0.,       omega=np.pi/10.,Omega=0.,ax=ax0[1,0])
ax0[1,1] = plotdmagvss(sma=1.,eccen=0.2,inc=np.pi/10.,omega=np.pi/10.,Omega=0.,ax=ax0[1,1])
ax0[1,2] = plotdmagvss(sma=1.,eccen=0.2,inc=np.pi/2., omega=np.pi/10.,Omega=0.,ax=ax0[1,2])
fig0.subplots_adjust(hspace=0.,wspace=0.)
#plt.close('all')

#### Associated s,dmag plots for these orbits
fig00, ax00 = plt.subplots(nrows=4,ncols=4,num=5000)
ax00[0,0] = plotdmagvss(sma=1.,eccen=0.6,inc=0.,       omega=0.,Omega=0.,ax=ax00[0,0])
ax00[0,1] = plotdmagvss(sma=1.,eccen=0.6,inc=np.pi/6.,omega=0.,Omega=0.,ax=ax00[0,1])
ax00[0,2] = plotdmagvss(sma=1.,eccen=0.6,inc=np.pi/3.,omega=0.,Omega=0.,ax=ax00[0,2])
ax00[0,3] = plotdmagvss(sma=1.,eccen=0.6,inc=np.pi/2., omega=0.,Omega=0.,ax=ax00[0,3]) #edge on
ax00[1,0] = plotdmagvss(sma=1.,eccen=0.6,inc=0.,       omega=np.pi/6.,Omega=0.,ax=ax00[1,0])
ax00[1,1] = plotdmagvss(sma=1.,eccen=0.6,inc=np.pi/6.,omega=np.pi/6.,Omega=0.,ax=ax00[1,1])
ax00[1,2] = plotdmagvss(sma=1.,eccen=0.6,inc=np.pi/3.,omega=np.pi/6.,Omega=0.,ax=ax00[1,2])
ax00[1,3] = plotdmagvss(sma=1.,eccen=0.6,inc=np.pi/2., omega=np.pi/6.,Omega=0.,ax=ax00[1,3]) #edge on
ax00[2,0] = plotdmagvss(sma=1.,eccen=0.6,inc=0.,       omega=np.pi/3.,Omega=0.,ax=ax00[2,0])
ax00[2,1] = plotdmagvss(sma=1.,eccen=0.6,inc=np.pi/6.,omega=np.pi/3.,Omega=0.,ax=ax00[2,1])
ax00[2,2] = plotdmagvss(sma=1.,eccen=0.6,inc=np.pi/3.,omega=np.pi/3.,Omega=0.,ax=ax00[2,2])
ax00[2,3] = plotdmagvss(sma=1.,eccen=0.6,inc=np.pi/2., omega=np.pi/3,Omega=0.,ax=ax00[2,3]) #edge on
ax00[3,0] = plotdmagvss(sma=1.,eccen=0.6,inc=0.,       omega=75./180.*np.pi,Omega=0.,ax=ax00[3,0])
ax00[3,1] = plotdmagvss(sma=1.,eccen=0.6,inc=np.pi/6.,omega=75./180.*np.pi,Omega=0.,ax=ax00[3,1])
ax00[3,2] = plotdmagvss(sma=1.,eccen=0.6,inc=np.pi/3.,omega=75./180.*np.pi,Omega=0.,ax=ax00[3,2])
ax00[3,3] = plotdmagvss(sma=1.,eccen=0.6,inc=np.pi/2., omega=75./180.*np.pi,Omega=0.,ax=ax00[3,3]) #edge on
fig00.subplots_adjust(hspace=0.,wspace=0.)
#plt.close('all')

#### Associated x,y plots for these orbits
fig1, ax = plt.subplots(nrows=4,ncols=4,num=501)
ax[0,0] = plotxyvskoe(sma=1.,eccen=0.6,inc=0.,       omega=0.,Omega=0.,ax=ax[0,0])
ax[0,1] = plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/6.,omega=0.,Omega=0.,ax=ax[0,1])
ax[0,2] = plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/3.,omega=0.,Omega=0.,ax=ax[0,2])
ax[0,3] = plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/2., omega=0.,Omega=0.,ax=ax[0,3]) #edge on
ax[1,0] = plotxyvskoe(sma=1.,eccen=0.6,inc=0.,       omega=np.pi/6.,Omega=0.,ax=ax[1,0])
ax[1,1] = plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/6.,omega=np.pi/6.,Omega=0.,ax=ax[1,1])
ax[1,2] = plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/3.,omega=np.pi/6.,Omega=0.,ax=ax[1,2])
ax[1,3] = plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/2., omega=np.pi/6.,Omega=0.,ax=ax[1,3]) #edge on
ax[2,0] = plotxyvskoe(sma=1.,eccen=0.6,inc=0.,       omega=np.pi/3.,Omega=0.,ax=ax[2,0])
ax[2,1] = plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/6.,omega=np.pi/3.,Omega=0.,ax=ax[2,1])
ax[2,2] = plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/3.,omega=np.pi/3.,Omega=0.,ax=ax[2,2])
ax[2,3] = plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/2., omega=np.pi/3.,Omega=0.,ax=ax[2,3]) #edge on
ax[3,0] = plotxyvskoe(sma=1.,eccen=0.6,inc=0.,       omega=75./180.*np.pi,Omega=0.,ax=ax[3,0])
ax[3,1] = plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/6.,omega=75./180.*np.pi,Omega=0.,ax=ax[3,1])
ax[3,2] = plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/3.,omega=75./180.*np.pi,Omega=0.,ax=ax[3,2])
ax[3,3] = plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/2., omega=75./180.*np.pi,Omega=0.,ax=ax[3,3]) #edge on
fig1.subplots_adjust(hspace=0.,wspace=0.)
#plt.close('all')

#### Plot s vs nu
def plotsvskoe(sma,eccen,inc,omega,Omega,ax=None,num=None):
    #need a mechanism for calculating evenly spaced distribution of nu
    circ = circ_from_E(sma,eccen)
    b = np.sqrt(sma**2.*(1.-eccen**2.)) #semi-minor axis
    area = np.pi*sma*b #area inside the ellipse

    #Note: We only need to evenly distribute over area bc equal area
    #in equal time is one of Kepler's laws
    numPts=200 #number of points to get
    dA = area/numPts #each dA aiming to achieve
    rs = np.zeros(numPts)#All the r to use
    nus = np.zeros(numPts)#All the nu to use
    #xs = np.zeros(numPts)
    #ys = np.zeros(numPts)
    #zs = np.zeros(numPts)
    rs[0] = r_koe(sma,eccen,0.) #Set initial r
    nus[0] = 0.
    #xs[0] = x_koe(rs[0],inc,omega,Omega,nus[0])
    #ys[0] = y_koe(rs[0],inc,omega,Omega,nus[0])
    #zs[0] = z_koe(rs[0],inc,omega,Omega,nus[0])
    #EXAMPLE nus[1] = nus[0] + 2.*dA/(rs[0]*(rs[0]-dr_koe(a,e,nus[0])))
    for i in np.arange(numPts-1)+1:
        nus[i] = nus[i-1] + 2.*dA/(rs[i-1]*(rs[i-1]-dr_koe(sma,eccen,nus[i-1])))
        rs[i] = r_koe(sma,eccen,nus[i])
        #xs[i] = x_koe(rs[i],inc,omega,Omega,nus[i])
        #ys[i] = y_koe(rs[i],inc,omega,Omega,nus[i])
        #zs[i] = z_koe(rs[i],inc,omega,Omega,nus[i])

    ss = s_koe(sma,eccen,nus,inc,omega)
    #betas = np.arcsin(ss/rs) #From my paper
    #Phis = phiLambert(betas*u.rad)
    #dmags = deltaMag(p=1.,Rp=1.*u.earthRad,d=rs*u.AU,Phi=Phis)
    if ax == None:
        plt.figure()
        plt.plot(nus,ss)
        ra = sma*(1.+eccen)
        plt.ylim([0.,ra])
        plt.xlim([0.,2.*np.pi])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show(block=False)
        return None
    else:
        ax.plot(nus,ss)
        ra = sma*(1.+eccen)
        ax.set_ylim([0.,ra])
        ax.set_xlim([0.,2.*np.pi])
        plt.show(block=False)
        return ax

#### Associated x,y plots for these orbits
fig3, ax3 = plt.subplots(nrows=4,ncols=4,num=503)
ax3[0,0] = plotsvskoe(sma=1.,eccen=0.6,inc=0.,       omega=0.,Omega=0.,ax=ax3[0,0])
ax3[0,1] = plotsvskoe(sma=1.,eccen=0.6,inc=np.pi/6.,omega=0.,Omega=0.,ax=ax3[0,1])
ax3[0,2] = plotsvskoe(sma=1.,eccen=0.6,inc=np.pi/3.,omega=0.,Omega=0.,ax=ax3[0,2])
ax3[0,3] = plotsvskoe(sma=1.,eccen=0.6,inc=np.pi/2., omega=0.,Omega=0.,ax=ax3[0,3]) #edge on
ax3[1,0] = plotsvskoe(sma=1.,eccen=0.6,inc=0.,       omega=np.pi/6.,Omega=0.,ax=ax3[1,0])
ax3[1,1] = plotsvskoe(sma=1.,eccen=0.6,inc=np.pi/6.,omega=np.pi/6.,Omega=0.,ax=ax3[1,1])
ax3[1,2] = plotsvskoe(sma=1.,eccen=0.6,inc=np.pi/3.,omega=np.pi/6.,Omega=0.,ax=ax3[1,2])
ax3[1,3] = plotsvskoe(sma=1.,eccen=0.6,inc=np.pi/2., omega=np.pi/6.,Omega=0.,ax=ax3[1,3]) #edge on
ax3[2,0] = plotsvskoe(sma=1.,eccen=0.6,inc=0.,       omega=np.pi/3.,Omega=0.,ax=ax3[2,0])
ax3[2,1] = plotsvskoe(sma=1.,eccen=0.6,inc=np.pi/6.,omega=np.pi/3.,Omega=0.,ax=ax3[2,1])
ax3[2,2] = plotsvskoe(sma=1.,eccen=0.6,inc=np.pi/3.,omega=np.pi/3.,Omega=0.,ax=ax3[2,2])
ax3[2,3] = plotsvskoe(sma=1.,eccen=0.6,inc=np.pi/2., omega=np.pi/3.,Omega=0.,ax=ax3[2,3]) #edge on
ax3[3,0] = plotsvskoe(sma=1.,eccen=0.6,inc=0.,       omega=75./180.*np.pi,Omega=0.,ax=ax3[3,0])
ax3[3,1] = plotsvskoe(sma=1.,eccen=0.6,inc=np.pi/6.,omega=75./180.*np.pi,Omega=0.,ax=ax3[3,1])
ax3[3,2] = plotsvskoe(sma=1.,eccen=0.6,inc=np.pi/3.,omega=75./180.*np.pi,Omega=0.,ax=ax3[3,2])
ax3[3,3] = plotsvskoe(sma=1.,eccen=0.6,inc=np.pi/2., omega=75./180.*np.pi,Omega=0.,ax=ax3[3,3]) #edge on
fig3.subplots_adjust(hspace=0.,wspace=0.)
#plt.close('all')

#### Test Omega 
# plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/10.,omega=0.,Omega=0.)
# plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/10.,omega=0.,Omega=np.pi/10.)
# plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/10.,omega=0.,Omega=np.pi/6.) #edge on
# plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/10.,omega=np.pi/6.,Omega=0.)
# plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/10.,omega=np.pi/6.,Omega=np.pi/10.)
# plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/10.,omega=np.pi/6.,Omega=np.pi/6.) #edge on
# test confirms changing Omega changes rotation of X,Y ellipse
