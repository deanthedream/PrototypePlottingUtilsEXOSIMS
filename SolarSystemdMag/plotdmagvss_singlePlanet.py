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

def plotdmagvss(sma,eccen,inc,omega,Omega):
    #need a mechanism for calculating evenly spaced distribution of nu
    circ = circ_from_E(sma,eccen)
    b = np.sqrt(sma**2.*(1.-eccen**2.)) #semi-minor axis
    area = np.pi*sma*b #area inside the ellipse

    #Note: We only need to evenly distribute over area bc equal area
    #in equal time is one of Kepler's laws
    numPts=100 #number of points to get
    dA = area/numPts #each dA aiming to achieve
    rs = np.zeros(numPts)#All the r to use
    nus = np.zeros(numPts)#All the nu to use
    rs[0] = r_koe(sma,eccen,0.) #Set initial r
    nus[0] = 0.
    #EXAMPLE nus[1] = nus[0] + 2.*dA/(rs[0]*(rs[0]-dr_koe(a,e,nus[0])))
    for i in np.arange(numPts-1)+1:
        nus[i] = nus[i-1] + 2.*dA/(rs[i-1]*(rs[i-1]-dr_koe(sma,eccen,nus[i-1])))
        rs[i] = r_koe(sma,eccen,nus[i])

    ss = s_koe(sma,eccen,nus,inc,omega)
    betas = np.arcsin(ss/rs) #From my paper
    Phis = phiLambert(betas)
    dmags = deltaMag(p=1.,Rp=1.*u.earthRad,d=rs*u.AU,Phi=Phis)
    plt.figure()
    plt.plot(ss,dmags)
    #plt.scatter(ss,dmags)
    plt.xlabel('s')
    plt.ylabel('dmag')
    plt.show(block=False)

def x_koe(r,inc,omega,Omega,nu):
    return r*(np.cos(Omega)*np.cos(omega+nu)-np.sin(Omega)*np.sin(omega+nu)*np.cos(inc))

def y_koe(r,inc,omega,Omega,nu):
    return r*(np.sin(Omega)*np.cos(omega+nu)+np.cos(Omega)*np.sin(omega+nu)*np.cos(inc))

def z_koe(r,inc,omega,Omega,nu):
    return r*np.sin(inc)*np.sin(omega+nu)

def plotxyvskoe(sma,eccen,inc,omega,Omega):
    #need a mechanism for calculating evenly spaced distribution of nu
    circ = circ_from_E(sma,eccen)
    b = np.sqrt(sma**2.*(1.-eccen**2.)) #semi-minor axis
    area = np.pi*sma*b #area inside the ellipse

    #Note: We only need to evenly distribute over area bc equal area
    #in equal time is one of Kepler's laws
    numPts=100 #number of points to get
    dA = area/numPts #each dA aiming to achieve
    rs = np.zeros(numPts)#All the r to use
    nus = np.zeros(numPts)#All the nu to use
    xs = np.zeros(numPts)
    ys = np.zeros(numPts)
    rs[0] = r_koe(sma,eccen,0.) #Set initial r
    nus[0] = 0.
    xs[0] = x_koe(rs[0],inc,omega,Omega,nus[0])
    ys[0] = y_koe(rs[0],inc,omega,Omega,nus[0])
    #EXAMPLE nus[1] = nus[0] + 2.*dA/(rs[0]*(rs[0]-dr_koe(a,e,nus[0])))
    for i in np.arange(numPts-1)+1:
        nus[i] = nus[i-1] + 2.*dA/(rs[i-1]*(rs[i-1]-dr_koe(sma,eccen,nus[i-1])))
        rs[i] = r_koe(sma,eccen,nus[i])
        xs[i] = x_koe(rs[i],inc,omega,Omega,nus[i])
        ys[i] = y_koe(rs[i],inc,omega,Omega,nus[i])

    ss = s_koe(sma,eccen,nus,inc,omega)
    betas = np.arcsin(ss/rs) #From my paper
    Phis = phiLambert(betas)
    dmags = deltaMag(p=1.,Rp=1.*u.earthRad,d=rs*u.AU,Phi=Phis)
    plt.figure()
    #plt.plot(ss,dmags)
    #plt.scatter(ss,dmags)
    plt.plot(xs,ys)
    ra = sma*(1.+eccen)
    plt.ylim([-ra,ra])
    plt.xlim([-ra,ra])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show(block=False)


#DELETE E = elliptic_integral_of_second_kind_power_series(e)

#### Good dMag vs s Plots for several orbits
plotdmagvss(sma=1.,eccen=0.2,inc=0.,       omega=0.,Omega=0.)
plotdmagvss(sma=1.,eccen=0.2,inc=np.pi/10.,omega=0.,Omega=0.)
plotdmagvss(sma=1.,eccen=0.2,inc=np.pi/2., omega=0.,Omega=0.)
plotdmagvss(sma=1.,eccen=0.2,inc=0.,       omega=np.pi/10.,Omega=0.)
plotdmagvss(sma=1.,eccen=0.2,inc=np.pi/10.,omega=np.pi/10.,Omega=0.)
plotdmagvss(sma=1.,eccen=0.2,inc=np.pi/2., omega=np.pi/10.,Omega=0.)
plt.close('all')

#### Associated x,y plots for these orbits
plotxyvskoe(sma=1.,eccen=0.6,inc=0.,       omega=0.,Omega=0.)
plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/6.,omega=0.,Omega=0.)
plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/3.,omega=0.,Omega=0.)
plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/2., omega=0.,Omega=0.) #edge on
plotxyvskoe(sma=1.,eccen=0.6,inc=0.,       omega=np.pi/6.,Omega=0.)
plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/6.,omega=np.pi/6.,Omega=0.)
plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/3.,omega=np.pi/6.,Omega=0.)
plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/2., omega=np.pi/6.,Omega=0.) #edge on
plotxyvskoe(sma=1.,eccen=0.6,inc=0.,       omega=np.pi/3.,Omega=0.)
plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/6.,omega=np.pi/3.,Omega=0.)
plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/3.,omega=np.pi/3.,Omega=0.)
plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/2., omega=np.pi/3.,Omega=0.) #edge on
plt.close('all')
#hmmmm... it doesn't look like changing inclination has any effect here and everything is baked in


#### Test Omega 
# plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/10.,omega=0.,Omega=0.)
# plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/10.,omega=0.,Omega=np.pi/10.)
# plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/10.,omega=0.,Omega=np.pi/6.) #edge on
# plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/10.,omega=np.pi/6.,Omega=0.)
# plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/10.,omega=np.pi/6.,Omega=np.pi/10.)
# plotxyvskoe(sma=1.,eccen=0.6,inc=np.pi/10.,omega=np.pi/6.,Omega=np.pi/6.) #edge on
# test confirms changing Omega changes rotation of X,Y ellipse
