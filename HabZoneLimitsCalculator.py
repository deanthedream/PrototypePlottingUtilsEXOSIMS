#simple Habitable Zone Calculator

import astropy.units as u
import numpy as np



Tmax = 100.+273.
Tmin = 273.
Rmin = 0.7*u.earthRad.to('m')
Rmax = 1.4*u.earthRad.to('m')
STotalEnergyOut = 1344*(4.*np.pi*(1.*u.AU.to('m'))**2.) #total solar energy radiating out in all directions

sb = 5.670367*10**-8 #stefan boltzmann constant W/(m^2 k^4)

#Flux Out for all four corners
q0 = sb*Tmin**4.*4.*np.pi*Rmin**2. #in W
q1 = sb*Tmin**4.*4.*np.pi*Rmax**2.
q2 = sb*Tmax**4.*4.*np.pi*Rmin**2.
q3 = sb*Tmax**4.*4.*np.pi*Rmax**2.

#Calculating r_ave in AU
r_ave0 = np.sqrt(STotalEnergyOut*Rmin**2./(4.*q0))*u.m.to('AU')
r_ave1 = np.sqrt(STotalEnergyOut*Rmax**2./(4.*q1))*u.m.to('AU')
r_ave2 = np.sqrt(STotalEnergyOut*Rmin**2./(4.*q2))*u.m.to('AU')
r_ave3 = np.sqrt(STotalEnergyOut*Rmax**2./(4.*q3))*u.m.to('AU')

print(r_ave0)
print(r_ave1)
print(r_ave2)
print(r_ave3)

#L, flux at the given earth orbital radii
L0 = STotalEnergyOut/(4.*np.pi*(r_ave0*u.AU.to('m'))**2.)
L1 = STotalEnergyOut/(4.*np.pi*(r_ave1*u.AU.to('m'))**2.)
L2 = STotalEnergyOut/(4.*np.pi*(r_ave2*u.AU.to('m'))**2.)
L3 = STotalEnergyOut/(4.*np.pi*(r_ave3*u.AU.to('m'))**2.)

print(L0/1344)
print(L1/1344)
print(L2/1344)
print(L3/1344)

