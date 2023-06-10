# Make ECEF to Geocentric, Geodetic, parametric latitude functions diagram



import numpy as np
import matplotlib.pyplot as plt

x_offset = 1.
y_offset = 1.

a = 10.
b = 5.

ths = np.linspace(start=0.,stop=2.*np.pi,num=500)

fig = plt.figure(num=1,figsize=(6,10))

#plot ellipse
x_ellipse = a*np.cos(ths)
y_ellipse = b*np.sin(ths)
plt.plot(x_ellipse,y_ellipse,color='k')

#plot greater circle
x_circle = a*np.cos(ths)
y_circle = a*np.sin(ths)
plt.plot(x_circle,y_circle,color='k',linestyle='dotted',linewidth=0.5)

#Compute Point A
beta = 35.*np.pi/180.#np.pi/4.
A_x = a*np.cos(beta)
A_y = a*np.sin(beta)
plt.plot([0,A_x],[0,A_y],color='k')
plt.plot([A_x,A_x],[0,A_y],color='k',linewidth=0.5)
plt.scatter(A_x,A_y)
plt.text(A_x-0.1,A_y+0.3,s='A')

#Point Gamma
r = a*b/np.sqrt(a**2*np.sin(beta)**2 + b**2*np.cos(beta)**2)
B_x = r*np.cos(beta)
B_y = r*np.sin(beta)
plt.text(B_x-0.1,B_y-0.45,s='B')


def geocentric_from_parametric(beta,a,b):#ok
    return np.arctan(b/a*np.tan(beta))

def geocentric_from_geodetic(phi,a,b):#ok
    return np.arctan(b**2/a**2*np.tan(phi))

def geodetic_from_geocentric(psi,a,b):#ok
    return np.arctan(a**2/b**2*np.tan(psi))

def geodetic_from_parametric(beta,a,b):#ok
    return np.arctan(a/b*np.tan(beta))

def parametric_from_geocentric(psi,a,b):#ok
    return np.arctan(a/b*np.tan(psi))

def parametric_from_geodetic(phi,a,b):#ok
    return np.arctan(b/a*np.tan(phi))

a0 = 2.
b0 = 1.
beta0 = 35.*np.pi/180.
psi0 = geocentric_from_parametric(beta0,a0,b0)
assert(beta0 == parametric_from_geocentric(psi0,a0,b0))
phi0 = geodetic_from_parametric(beta0,a0,b0)
assert(beta0 == parametric_from_geodetic(phi0,a0,b0))
assert(psi0 == geocentric_from_geodetic(phi0,a0,b0))
assert(phi0 == geodetic_from_geocentric(psi0,a0,b0))

assert(beta0 == parametric_from_geodetic(geodetic_from_parametric(beta0,a0,b0),a0,b0))
assert(beta0 == parametric_from_geocentric(geocentric_from_parametric(beta0,a0,b0),a0,b0))
assert(psi0 == geocentric_from_geodetic(geodetic_from_geocentric(psi0,a0,b0),a0,b0))



#Compute Point P
psi = geocentric_from_parametric(beta,a,b)
#psi = np.arctan(b/a/np.cos(beta)*np.sqrt(1-np.cos(beta)**2))
r = a*b/np.sqrt(a**2*np.sin(psi)**2 + b**2*np.cos(psi)**2)
P_x = r*np.cos(psi)
P_y = r*np.sin(psi)
plt.plot([0,P_x],[0,P_y],color='k')
plt.scatter(P_x,P_y)
plt.text(P_x+0.3,P_y+0,s='P')

#Plot x
plt.plot([0,P_x],[P_y,P_y],color='k',linewidth=0.5)
plt.text(0.25*P_x,P_y+0.05,s='x')

#Plot H
phi = geodetic_from_geocentric(psi,a,b)
t = P_y/np.sin(phi)
plt.plot([P_x-t*np.cos(phi),P_x],[0,P_y],color='k')
t2 = (P_x-t*np.cos(phi))/np.cos(phi)
plt.plot([0,P_x-t*np.cos(phi)],[-t2*np.sin(phi),0],color='k')
plt.text(0-0.5,-t2*np.sin(phi)-0.3,s='H')
D_x = P_x-t*np.cos(phi)
D_y = 0
plt.text(D_x,D_y-0.5,s="D")

#Plot Line Perpendicular to P
t2 = 20
plt.plot([P_x-t2*np.cos(phi-np.pi/2.),P_x],[P_y-t2*np.sin(phi-np.pi/2.),P_y],color='k',linestyle='dotted',linewidth=0.5)
plt.plot([P_x,P_x+t2*np.cos(phi-np.pi/2.)],[P_y,P_y+t2*np.sin(phi-np.pi/2.)],color='k',linestyle='dotted',linewidth=0.5)

#Plot Perpendicular Box
l = 0.35
plt.plot([P_x-l*np.cos(phi),P_x-l*np.cos(phi)+l*np.cos(phi-np.pi/2.)],[P_y-l*np.sin(phi),P_y-l*np.sin(phi)+l*np.sin(phi-np.pi/2.)],color='k',linewidth=0.5)
plt.plot([P_x-l*np.cos(phi)+l*np.cos(phi-np.pi/2.),P_x+l*np.cos(phi-np.pi/2.)],[P_y-l*np.sin(phi)+l*np.sin(phi-np.pi/2.),P_y+l*np.sin(phi-np.pi/2.)],color='k',linewidth=0.5)



#Plot Phi Angle Line
nus2 = np.linspace(start=0,stop=phi,num=50)
beta_x = 1*np.cos(nus2) + D_x#-0.5
beta_y = 1*np.sin(nus2) + D_y
plt.plot(beta_x,beta_y,color='k')
i = 11
plt.arrow(beta_x[-i], beta_y[-i], -(beta_x[-3]-beta_x[-1]), -(beta_y[-3]-beta_y[-1]), shape='full', lw=3,
   length_includes_head=True, head_width=.05, color='k',head_starts_at_zero=True)
plt.text(beta_x[25]+0.2,beta_y[25],s=r'$\phi$')


#x-axis
plt.plot([0,0],[-1.1*a,1.5*a],color='black',linewidth=0.5)

#y-axis
plt.plot([-1.1*a,1.5*a],[0,0],color='black',linewidth=0.5)


plt.text(a+0.2,0-0.5,s='E')
plt.text(0-0.5,0-0.5,s='O')
plt.text(A_x+0.1,0-0.5,s='M')
plt.text(0-0.5,b+0.3,s='Q')


nus = np.linspace(start=0,stop=beta,num=50)
beta_x = 2*np.cos(nus)
beta_y = 2*np.sin(nus)
plt.plot(beta_x,beta_y,color='k')
i = 11
plt.arrow(beta_x[-i], beta_y[-i], -(beta_x[-3]-beta_x[-1]), -(beta_y[-3]-beta_y[-1]), shape='full', lw=3,
   length_includes_head=True, head_width=.05, color='k',head_starts_at_zero=True)
plt.text(beta_x[25]+0.3,beta_y[25]-0.3,s=r'$\beta$')

nus = np.linspace(start=0,stop=psi,num=50)
beta_x = 3*np.cos(nus)
beta_y = 3*np.sin(nus)
plt.plot(beta_x,beta_y,color='k')
i = 11
plt.arrow(beta_x[-i], beta_y[-i], -(beta_x[-3]-beta_x[-1]), -(beta_y[-3]-beta_y[-1]), shape='full', lw=3,
   length_includes_head=True, head_width=.05, color='k',head_starts_at_zero=True)
plt.text(beta_x[25]+0.3,beta_y[25]-0.1,s=r'$\psi$')


#Additional Labeling Text
plt.text(0-0.4,-2.5,s='b')
plt.text(4.3,0-0.35,s='a')

plt.text(5.4,1.5,s='r')
plt.text(8.3,0.8,s='y')



plt.axis('square')

plt.xlim([-1.,13])
#plt.ylim([-1.07*b,10.5])
plt.ylim([-10.5,10.5])
plt.axis('off')
plt.show(block=False)
plt.savefig("ECEF_geodesics_conversion.png",dpi=300)




################################################################
#NEED TO REDO EVERYTHING WITH ACTUAL SV
#Adding SV

#SV Point
SV_x = 1.1*a*np.cos(beta)
SV_y = 1.1*a*np.sin(beta)
plt.plot([A_x,SV_x],[A_y,SV_y],color='k',linewidth=2)
plt.plot([P_x,SV_x],[P_y,SV_y],color='k',linewidth=2)
plt.text(SV_x+0.2,SV_y+0.2,s='SV')

# t3 = (P_x/np.cos(beta)-P_y/np.sin(beta))/(np.sin(phi)/np.sin(beta) - np.cos(phi)/np.cos(beta))
# SV_x = P_x + t3*np.cos(phi)
# SV_y = P_y + t3*np.sin(phi)
# plt.plot([A_x,SV_x],[A_y,SV_y],color='k',linewidth=2)
# plt.plot([P_x,SV_x],[P_y,SV_y],color='k',linewidth=2)
# plt.text(SV_x+0.2,SV_y+0.2,s='SV')

plt.show(block=False)
plt.savefig("ECEF_geodesics_conversion_SV_BAD.png",dpi=300)





#######################################################################

def lat_lon_from_ECEF(x,y,z,e_oplus,R_oplus,tol=0.001):
    """
    ALGORITHM 12 from Vallado'
    Computes the Geodetic latitude of some object at some altitude above the surface of the earth given in ECEF XYZ
    """
    r_delta_sat = np.sqrt(x**2. + y**2.)
    delta = np.arctan2(z,r_delta_sat)


    alpha = np.arctan2(z,r_delta_sat)#ok
    lam = alpha
    phi_gd = delta
    r_delta = r_delta_sat

    phi_gd_old = phi_gd
    for i in np.arange(1000):
        c_oplus = R_oplus/np.sqrt(1.-e_oplus**2.*np.sin(phi_gd_old)**2.)
        phi_gd = np.arctan2(z+c_oplus*e_oplus**2.*np.sin(phi_gd_old),r_delta)
        if np.abs(phi_gd-phi_gd_old) < tol:
            break
        phi_gd_old = phi_gd
        
    #Compute h_ellp
    if phi_gd < 0.99*np.pi/2. and phi_gd > -0.99*np.pi/2.:
        h_ellp = r_delta/np.cos(phi_gd) - c_oplus
    else:
        s_oplus = (R_oplus*(1.-e_oplus**2.))/np.sqrt(1.-e_oplus**2.*np.sin(phi_gd)**2.)
        h_ellp = z/np.sin(phi_gd)-s_oplus

    return (phi_gd, lam, h_ellp)










#inputs
SV_x = 9.
SV_y = 6.5
e = np.sqrt(1.-b**2./a**2.)
fig = plt.figure(num=2,figsize=(6,10))
(phi_gd, lam, h_ellp) = lat_lon_from_ECEF(SV_x,0.,SV_y,e,a,tol=0.001) #in this case, SV_y is actually z in ECEF

#plot ellipse
plt.plot(x_ellipse,y_ellipse,color='k')

#plot greater circle
plt.plot(x_circle,y_circle,color='k',linestyle='dotted',linewidth=0.5)


#Plot O to SV
plt.plot([0,SV_x],[0.,SV_y],color='k')
psi_sv = np.arctan2(SV_y,SV_x)

#Plot G
bigR = np.sqrt(SV_x**2.+SV_y**2.)
plt.scatter(SV_x*a/bigR,SV_y*a/bigR)
plt.text(SV_x*a/bigR-0.1,SV_y*a/bigR+0.3,s='G')



#Compute Point A
beta = parametric_from_geodetic(phi_gd,a,b)
#SV_beta = 35.*np.pi/180.#np.pi/4.
A_x = a*np.cos(beta)
A_y = a*np.sin(beta)
plt.plot([0,A_x],[0,A_y],color='k')
plt.scatter(A_x,A_y)
plt.text(A_x-0.1,A_y+0.3,s='A')




#Point Gamma
r = a*b/np.sqrt(a**2.*np.sin(beta)**2. + b**2.*np.cos(beta)**2.)
B_x = r*np.cos(beta)
B_y = r*np.sin(beta)
plt.text(B_x-0.1,B_y-0.45,s='B')


#Compute Point P
#psi = geocentric_from_parametric(beta,a,b)
psi = geocentric_from_geodetic(phi_gd,a,b)
#psi = np.arctan(b/a/np.cos(beta)*np.sqrt(1-np.cos(beta)**2))
r = a*b/np.sqrt(a**2.*np.sin(psi)**2. + b**2.*np.cos(psi)**2.)
P_x = r*np.cos(psi)
P_y = r*np.sin(psi)
plt.plot([0,P_x],[0,P_y],color='k')
plt.scatter(P_x,P_y)
plt.text(P_x+0.3,P_y+0,s='P')

#Plot x
plt.plot([0,P_x],[P_y,P_y],color='k',linewidth=0.5)
plt.text(0.25*P_x,P_y+0.05,s='x')

#Plot y
plt.plot([P_x,P_x],[0,P_y],color='k',linewidth=0.5)
plt.plot([P_x,P_x],[P_y,A_y],color='k',linewidth=0.5)
plt.text(7.5,0.8,s='y')

#Plot H
phi = geodetic_from_geocentric(psi,a,b)
t = P_y/np.sin(phi)
plt.plot([P_x-t*np.cos(phi),P_x],[0,P_y],color='k')
t2 = (P_x-t*np.cos(phi))/np.cos(phi)
plt.plot([0,P_x-t*np.cos(phi)],[-t2*np.sin(phi),0],color='k')
plt.text(0-0.5,-t2*np.sin(phi)-0.3,s='H')
D_x = P_x-t*np.cos(phi)
D_y = 0
plt.text(D_x,D_y-0.5,s="D")

#Plot Line Perpendicular to P
t2 = 20
plt.plot([P_x-t2*np.cos(phi-np.pi/2.),P_x],[P_y-t2*np.sin(phi-np.pi/2.),P_y],color='k',linestyle='dotted',linewidth=0.5)
plt.plot([P_x,P_x+t2*np.cos(phi-np.pi/2.)],[P_y,P_y+t2*np.sin(phi-np.pi/2.)],color='k',linestyle='dotted',linewidth=0.5)

#Plot Perpendicular Box
l = 0.35
plt.plot([P_x-l*np.cos(phi),P_x-l*np.cos(phi)+l*np.cos(phi-np.pi/2.)],[P_y-l*np.sin(phi),P_y-l*np.sin(phi)+l*np.sin(phi-np.pi/2.)],color='k',linewidth=0.5)
plt.plot([P_x-l*np.cos(phi)+l*np.cos(phi-np.pi/2.),P_x+l*np.cos(phi-np.pi/2.)],[P_y-l*np.sin(phi)+l*np.sin(phi-np.pi/2.),P_y+l*np.sin(phi-np.pi/2.)],color='k',linewidth=0.5)



#Plot Phi Angle Line
nus2 = np.linspace(start=0,stop=phi,num=50)
beta_x = 1*np.cos(nus2) + D_x#-0.5
beta_y = 1*np.sin(nus2) + D_y
plt.plot(beta_x,beta_y,color='k')
i = 11
plt.arrow(beta_x[-i], beta_y[-i], -(beta_x[-3]-beta_x[-1]), -(beta_y[-3]-beta_y[-1]), shape='full', lw=3,
   length_includes_head=True, head_width=.05, color='k',head_starts_at_zero=True)
plt.text(beta_x[25]+0.2,beta_y[25],s=r'$\phi$')


#x-axis
plt.plot([0,0],[-1.1*a,1.5*a],color='black',linewidth=0.5)

#y-axis
plt.plot([-1.1*a,1.5*a],[0,0],color='black',linewidth=0.5)


plt.text(a+0.2,0-0.5,s='E')
plt.text(0-0.5,0-0.5,s='O')
plt.text(A_x+0.1,0-0.5,s='M')
plt.text(0-0.5,b+0.3,s='Q')


nus = np.linspace(start=0,stop=beta,num=50)
beta_x = 2*np.cos(nus)
beta_y = 2*np.sin(nus)
plt.plot(beta_x,beta_y,color='k')
i = 11
plt.arrow(beta_x[-i], beta_y[-i], -(beta_x[-3]-beta_x[-1]), -(beta_y[-3]-beta_y[-1]), shape='full', lw=3,
   length_includes_head=True, head_width=.05, color='k',head_starts_at_zero=True)
plt.text(beta_x[25]+0.3,beta_y[25]-0.3,s=r'$\beta$')

nus = np.linspace(start=0,stop=psi,num=50)
beta_x = 3*np.cos(nus)
beta_y = 3*np.sin(nus)
plt.plot(beta_x,beta_y,color='k')
i = 11
plt.arrow(beta_x[-i], beta_y[-i], -(beta_x[-3]-beta_x[-1]), -(beta_y[-3]-beta_y[-1]), shape='full', lw=3,
   length_includes_head=True, head_width=.05, color='k',head_starts_at_zero=True)
plt.text(beta_x[25]+0.3,beta_y[25]-0.1,s=r'$\psi$')


#Plot psi_SV arc
nus = np.linspace(start=0,stop=psi_sv,num=50)
beta_x = 4.*np.cos(nus)
beta_y = 4.*np.sin(nus)
plt.plot(beta_x,beta_y,color='k')
i = 7
plt.arrow(beta_x[-i], beta_y[-i], -(beta_x[-3]-beta_x[-1]), -(beta_y[-3]-beta_y[-1]), shape='full', lw=3,
   length_includes_head=True, head_width=.05, color='k',head_starts_at_zero=True)
plt.text(beta_x[45]+0.3,beta_y[45]+0.1,s=r'$\psi_{sv}$')




#Plot P to SV
plt.plot([P_x,SV_x],[P_y,SV_y],color='k')


#SV Text
plt.text(SV_x+0.2,SV_y+0.2,s='SV')

#Plot O to A
#beta = parametric_from_geodetic(phi_gd,a,b)
# r = a*b/np.sqrt(a**2*np.sin(beta)**2 + b**2*np.cos(beta)**2)
# A_x = r*np.cos(beta)
# A_y = r*np.sin(beta)
# plt.plot([0,A_x],[0,A_y],color='k')


#Additional Labeling Text
plt.text(0-0.4,-2.5,s='b')
plt.text(4.3,0-0.35,s='a')

plt.text(5.1,2.0,s='r')




#Descriptors
#textstr = '\n'.join((r'$\mu=%.2f$' % (mu, ), r'$\mathrm{median}=%.2f$' % (median, ),r'$\sigma=%.2f$' % (sigma, )))
textstr = '\n'.join((r'$\beta$' + ": parametric/reduced lat.",
    r'$\psi$' + ": centric lat.",
    r'$\psi_{sv}$' + ": SV centric lat.",
    r'$\phi$' + ": detic lat.",
    "P: nadir"))
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='lightgrey', alpha=1.0)
# place a text box in upper left in axes coords
plt.text(1.5, -2., textstr, fontsize=12,
        verticalalignment='top', bbox=props) #transform=ax.transAxes,



plt.axis('square')

plt.xlim([-1.,13])
#plt.ylim([-1.07*b,10.5])
plt.ylim([-10.5,10.5])
plt.axis('off')
plt.show(block=False)
plt.savefig("ECEF_geodesics_conversion_WithSV.png",dpi=300)




