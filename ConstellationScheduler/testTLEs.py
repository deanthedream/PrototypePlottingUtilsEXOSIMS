# Sample Examples

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from skyfield.sgp4lib import TEME_to_ITRF
from jplephem.spk import SPK
from astropy.time import Time
import time
import os

#### INPUTS
spkpath = './de432s.bsp'
kernel = SPK.open(spkpath) # smaller file, covers mission time range
step = 4000

#### Create array of mjd times in range
# we are going to get positions between these two dates
utc = ['Jun 21, 2024', 'Jun 22, 2024'] #Earth shoud be almost entirely in the x direction relative to the sun
utc = ['2024-06-21T00:00:00.00', '2024-06-22T00:00:00']
utc = ['2024-06-21T00:00:00.00', '2025-06-21T00:00:00']
jdlims = Time(utc,format='isot',scale='tai')
jdtimes = np.linspace(start=jdlims.jd[0],stop=jdlims.jd[1],num=50)

#Create array of sun-earth-moon positions vs time
#currentTime = Time(koTimes,format='mjd',scale='tai')  # scale must be tai to account for leap seconds
#jdtime = np.array(currentTime.jd, ndmin=1)
rs_earth_sun = list()
rs_earth_moon = list()
rshat_earth_sun = list()
rshat_earth_moon = list()
for i in np.arange(len(jdtimes)):
    jdtime = jdtimes[i]
    #vectors in heliocentric ecliptic frame, origin at the sun center, fundamental plane in the plane of the earth's equator, x-axis toward vernal equinox
    r_solarbarycenter_sun = kernel[0,10].compute(jdtime)
    r_solarbarycenter_earthbarycenter = kernel[0,3].compute(jdtime)
    r_earthbarycenter_moon = kernel[3,301].compute(jdtime)
    r_earthbarycenter_earth = kernel[3,399].compute(jdtime)

    r_earth_sun = -r_earthbarycenter_earth - r_solarbarycenter_earthbarycenter + r_solarbarycenter_sun #vector from Earth center to sun center
    r_earth_moon = -r_earthbarycenter_earth + r_earthbarycenter_moon #vector from Earth center to moon

    rs_earth_sun.append(r_earth_sun)
    rs_earth_moon.append(r_earth_moon)
    rshat_earth_sun.append(r_earth_sun/np.linalg.norm(r_earth_sun))
    rshat_earth_moon.append(r_earth_moon/np.linalg.norm(r_earth_moon))
    #Earth Centered Inertial (geocentric equatorial coordinates), origin at earth center, fundamental plane in the plane of the Eath's equator, x-axis towards vernal equinox
    #i think they Hheliocentric ecliptic and earthc entered inertial are the same but translated in space (just the origins are different, xhat_sun=xhat_ECI...)


#### Generate Obs
r_earth = 6371.0 #km

def genRandomLatLon(num=1,lat_low=-np.pi/2.,lat_high=np.pi/2.,lon_low=0,lon_high=2.*np.pi):
    """ Generates a random latitude and longitude
    """
    lons = np.random.uniform(low=lon_low,high=lon_high,size=num)
    #lats = np.arccos(np.sin(np.random.uniform(low=np.sin(lat_low),high=np.sin(lat_high),size=num)))
    lats = np.arccos(np.random.uniform(low=np.sin(lat_low),high=np.sin(lat_high),size=num))-np.pi/2.
    return lons, lats

def lonLats_to_xyz(r,lons,lats):
    """ Computes x,y,z position from lons and lats
    """
    return r*np.asarray([np.cos(lons)*np.cos(lats),np.sin(lons)*np.cos(lats),np.sin(lats)])

num_locs = 400#50#400#200    
lons, lats = genRandomLatLon(num_locs,lat_low=-np.pi/2.,lat_high=np.pi/2.,lon_low=0.,lon_high=2.*np.pi) #Full sphere
#Randomly generate locations
r_locs = lonLats_to_xyz(r_earth+1000.,lons,lats)


#### At Given Time, plot SV and positions that are visible
fig4 = plt.figure(num=455888999999999999,figsize=(8,8))
ax3 = fig4.add_subplot(111, projection='3d',computed_zorder=False)
ax3.set_box_aspect(aspect = (1,1,1))#set_aspect('equal')
#plot invisible box to bound
ax3.scatter(r_locs[0],r_locs[1],r_locs[2],color='blue')
#Plot wireframe of earth
plt.show(block=False)






#### SV
sv = list()
for i in np.arange(num_locs):
    print(i/num_locs)
    for j in np.arange(num_locs):
        if i==j:
            sv.append([])
        else:
            data = dict()
            data['r_ij'] = r_locs[:,j]-r_locs[:,i] #vector from i to j
            data['rhat_ij'] = data['r_ij']/np.linalg.norm(data['r_ij'])
            #Sun Keepout
            data['sunKOang'] = list()
            data['insunKO'] = list()
            for k in np.arange(len(jdtimes)):
                ang = np.arccos(np.dot(rshat_earth_sun[k],data['rhat_ij']))
                data['sunKOang'].append(ang)
                if ang < 20.*np.pi/180.: #Maximum solar exclusion limit
                    data['insunKO'].append(1)
                else:
                    data['insunKO'].append(0) #Not in keepout, target visible
            #Moon Keepout
            data['moonKOang'] = list()
            data['inmoonKO'] = list()
            for k in np.arange(len(jdtimes)):
                ang = np.arccos(np.dot(rshat_earth_moon[k],data['rhat_ij']))
                data['moonKOang'].append(ang)
                if ang < 7.*np.pi/180.: #Maximum moon exclusion limit
                    data['inmoonKO'].append(1)
                else:
                    data['inmoonKO'].append(0) #Not in keepout, target visible
            
            #tangent height

            sv.append(data)





#Compute Tangent Point and Tangent Height
def tangentPointline_spherespace(p0_c,p1_c):
    """ Computes the Tangent Point on the line
    """
    d = np.linalg.norm(p1_c-p0_c)
    uhat = (p1_c-p0_c)/d
    e = uhat[0]
    f = uhat[1]
    g = uhat[2]
    x0 = p0_c[0]
    y0 = p0_c[1]
    z0 = p0_c[2]

    t = (-x0*e-y0*f-z0*g)/(e**2.+f**2.+g**2.)

    tpt_c = p0_c+t*uhat
    return tpt_c

def pt_to_ptc(p0, p1, a, b, c):
    """Scales Points in ellipsoid space to points in sphere space
    """
    p0_c = np.asarray([p0[0]/a,p0[1]/b,p0[2]/c])
    p1_c = np.asarray([p1[0]/a,p1[1]/b,p1[2]/c])
    return p0_c, p1_c

def tangentPointline_ellipsespace(p0,p1,a,b,c):
    """Computes tangent point of the line in ellipsoid space
    p0, p1
    a,b,c of ellipsoid
    """
    p0_c, p1_c = pt_to_ptc(p0, p1, a, b, c)
    tpt_c = tangentPointline_spherespace(p0_c,p1_c)
    tpt = np.asarray([tpt_c[0]*a,tpt_c[1]*b,tpt_c[2]*c])
    return tpt

def tangentPointellipse_ellipsespace(p0,p1,a,b,c):
    """
    """
    p0_c, p1_c = pt_to_ptc(p0, p1, a, b, c)
    tpt_c = tangentPointline_spherespace(p0_c,p1_c)
    tpt_onCircle = tpt_c/np.linalg.norm(tpt_c)

    tpt = np.asarray([tpt_onCircle[0]*a,tpt_onCircle[1]*b,tpt_onCircle[2]*c])
    return tpt

def tangentHeight(tpt_c,a,b,c):
    n_c = tpt_c/np.linalg.norm(tpt_c)
    #L_c = n_c
    L = np.asarray([n_c[0]*a,n_c[1]*b,n_c[2]*c])

    height = np.linalg.norm(tpt-L)
    return height

def nhat(x0,y0,z0,a,b,c):
    """ Surface normal vector to point on ellipsoid in ellipsoid space
    """
    return np.asarray([x0/a**2,y0/b**2,z0/c**2])/np.sqrt((x0**2/a**4+y0**2/b**4+z0**2/c**4))

def uhat(p0,p1):
    return (p1-p0)/np.linalg.norm(p1-p0)

def x_tangentpoint_from_line(p0,p1,p2,p3):
    x0,y0,z0 = p0[0],p0[1],p0[2]
    x1,y1,z1 = p1[0],p1[1],p1[2]
    x2,y2,z2 = p2[0],p2[1],p2[2]
    x3,y3,z3 = p3[0],p3[1],p3[2]
    #P0x = p0[0]
    #P0y = p0[1]
    #x0 = L[0]
    #y0 = L[1]
    #e = uhat[0]
    #f = uhat[1]

    #the two lines are parallel
    if x1==x0 and x2==x3:
        if x0==x2: #the two lines are identical
            return x0
        else: #the two lines will never intersect
            return np.nan

    if x1==x0:
        return x0
    else:
        m_line = (y1-y0)/(x1-x0)
        
    if x2==x3:
        return x3
    else:
        m_tan = (y3-y2)/(x3-x2)

    b_line = y0-m_line*x0
    b_tan = y2-m_tan*x2

    x = (b_tan-b_line)/(m_line-m_tan)
        #c_lt = y0-nhat[1]/nhat[0]*x0 #from y=mx+c_lt
        #c_pp = P0y-nhat[1]/nhat[0]*P0x

        #t = (p0[1]-L[1] - nhat[1]/nhat[0]*L[0]+nhat[1]/nhat[0]*L[0])/(nhat[1]/nhat[0]*e-f)
        #x = (c_pp-c_lt)/(nhat[1]-nhat[0]-f/e)
        #dx = (y0 - P0y)/(f/e-nhat[1]/nhat[0])
        #t = (x-P0x)/e
    return x

def xyz_tangentpoint_from_line(p0,p1,p2,p3):
    """
    p0 is the SV
    p1 is the target
    p2 is a point along the tangent line
    """
    x0,y0,z0 = p0[0],p0[1],p0[2]
    x1,y1,z1 = p1[0],p1[1],p1[2]
    x2,y2,z2 = p2[0],p2[1],p2[2]
    x3,y3,z3 = p3[0],p3[1],p3[2]
    #P0x = p0[0]
    #P0y = p0[1]
    #x0 = L[0]
    #y0 = L[1]
    #e = uhat[0]
    #f = uhat[1]

    #the two lines are parallel
    if x1==x0 and x2==x3:
        if x0==x2: #the two lines are identical
            return x0
        else: #the two lines will never intersect
            return np.nan

    if x1==x0:
        return x0
    else:
        m_line = (y1-y0)/(x1-x0)
        
    if x2==x3:
        return x3
    else:
        m_tan = (y3-y2)/(x3-x2)

    b_line = y0-m_line*x0
    b_tan = y2-m_tan*x2

    #Compute x
    x = (b_tan-b_line)/(m_line-m_tan)

    #Compute y
    y = (x-x0)*m_line + y0

    #Compute z
    m_line2 = (z1-z0)/(x1-x0)
    z = m_line2*(x-x0) + +z0

    return x, y, z

def tangentHeight(p0,p1,a,b,c):
    """ Computes the Tangent Height of the Observation
    p0 - space vehicle location
    p1 - target location
    a, b, c ellipsoid parameters
    """
    tptellipse_ellipseSpace = tangentPointellipse_ellipsespace(p0,p1,a,b,c)
    nhat0 = nhat(tptellipse_ellipseSpace[0],tptellipse_ellipseSpace[1],tptellipse_ellipseSpace[2],a,b,c)
    x,y,z = xyz_tangentpoint_from_line(p0,p1,tptellipse_ellipseSpace,tptellipse_ellipseSpace+1*nhat0)
    tangentPoint = np.asarray([x,y,z])
    sgn = -1.
    if np.linalg.norm(tangentPoint) > np.linalg.norm(tptellipse_ellipseSpace):
        sgn = 1.
    return sgn*np.linalg.norm(tangentPoint - tptellipse_ellipseSpace)


def lunarKOVisible(p0,p1,r_earth_moon,KOangle=6*np.pi/180.):
    """
    look vector from p0 to p1
    """
    rhat_SV_targ = (p1-p0)/np.linalg.norm(p1-p0) #spacecraft to target vector
    rhat_SV_moon = (r_earth_moon-p0)/np.linalg.norm(r_earth_moon-p0) #spacecraft to moon vector
    angle = np.arccos(np.dot(rhat_SV_targ,rhat_SV_moon))
    if angle < KOangle:
        return False
    else:
        return True

def solarKOVisible(p0,p1,r_earth_sun,KOangle=10*np.pi/180.):
    rhat_SV_targ = (p1-p0)/np.linalg.norm(p1-p0) #spacecraft to target vector
    rhat_SV_sun = (r_earth_sun-p0)/np.linalg.norm(r_earth_sun-p0) #spacecraft to sun vector
    angle = np.arccos(np.dot(rhat_SV_targ,rhat_SV_sun))
    if angle < KOangle:
        return False
    else:
        return True

def tangentHeightVisible(p0,p1,a,b,c,heightLimit=200.):
    height = tangentHeight(p0,p1,a,b,c)

    if height < heightLimit:
        return False
    else:
        return True

def solarZenithAngleVisible(p0,p1,a,b,c,r_earth_sun,zenithAngleLimit = 90*np.pi/180.):
    tptellipse_ellipseSpace = tangentPointellipse_ellipsespace(p0,p1,a,b,c)
    rhat_tpt_sun = (r_earth_sun-tptellipse_ellipseSpace)/np.linalg.norm(r_earth_sun-tptellipse_ellipseSpace)
    nhat0 = nhat(p0[0],p0[1],p0[2],a,b,c)
    zenithAngle = np.arccos(np.dot(rhat_tpt_sun,nhat0))
    if zenithAngle < zenithAngleLimit:
        return False
    else:
        return True

def solarPhaseAngleVisible(p0,p1,a,b,c,r_earth_sun,solarPhaseAngleLimit=160*np.pi/180.):
    #tptellipse_ellipseSpace = tangentPointellipse_ellipsespace(p0,p1,a,b,c)
    rhat_tpt_sun = (r_earth_sun-tptellipse_ellipseSpace)/np.linalg.norm(r_earth_sun-tptellipse_ellipseSpace)
    solarPhaseAngle = np.arccos(np.dot(-uhat(p0,p1),rhat_tpt_sun))
    if solarPhaseAngle > solarPhaseAngleLimit:
        return False
    else:
        return True




utc = ['2024-06-21T00:00:00.00', '2025-06-21T00:00:00']
a, b, c = 6371., 6371., 6371.
jd = Time([utc[0]],format='isot',scale='tai')
jdtime = jd.jd[0]

def isVisible(p0,p1,kernel,jdtime,a,b,c):
    """
    p0 is the source point
    p1 is the target point
    """
    #r_earth_sun
    #vectors in heliocentric ecliptic frame, origin at the sun center, fundamental plane in the plane of the earth's equator, x-axis toward vernal equinox
    r_solarbarycenter_sun = kernel[0,10].compute(jdtime)
    r_solarbarycenter_earthbarycenter = kernel[0,3].compute(jdtime)
    r_earthbarycenter_moon = kernel[3,301].compute(jdtime)
    r_earthbarycenter_earth = kernel[3,399].compute(jdtime)

    r_earth_sun = -r_earthbarycenter_earth - r_solarbarycenter_earthbarycenter + r_solarbarycenter_sun #vector from Earth center to sun center
    r_earth_moon = -r_earthbarycenter_earth + r_earthbarycenter_moon #vector from Earth center to moon



    isVisibleLunar = lunarKOVisible(p0,p1,r_earth_moon)
    isVisibleSolar = solarKOVisible(p0,p1,r_earth_sun)
    isVisibleTangentHeight = tangentHeightVisible(p0,p1,a,b,c)
    isVisibleSolarZenithAngle = solarZenithAngleVisible(p0,p1,a,b,c,r_earth_sun)
    isVisibleSolarPhaseAngle = solarPhaseAngleVisible(p0,p1,a,b,c,r_earth_sun)
    visible = bool(isVisibleLunar*isVisibleSolar*isVisibleTangentHeight*isVisibleSolarZenithAngle*isVisibleSolarPhaseAngle)
    return visible, isVisibleLunar, isVisibleSolar, isVisibleTangentHeight, isVisibleSolarZenithAngle, isVisibleSolarPhaseAngle 



#### Verify Tangent Height Calculations ################3
p0 = np.asarray([-20,0,0])
p1 = np.asarray([0,20,0])
a = 10
b = 5
c = 1
plt.figure(2342356236)
xs = np.linspace(start=0,stop=-a)
ys = np.sqrt(b**2.*(1.-xs**2./a**2.))
plt.plot(xs,ys,color='black')
plt.plot([p0[0],p1[0]],[p0[1],p1[1]],color='orange')
#pt0_c, pt1_c = pt_to_ptc(p0, p1, a, b, c)
#tptline_ellipseSpace = tangentPointline_ellipsespace(p0,p1,a,b,c)
#plt.scatter(tptline_ellipseSpace[0],tptline_ellipseSpace[1],color='red')

#The Tangent Point on the Ellipse
tptellipse_ellipseSpace = tangentPointellipse_ellipsespace(p0,p1,a,b,c)
plt.scatter(tptellipse_ellipseSpace[0],tptellipse_ellipseSpace[1],color='blue')
nhat0 = nhat(tptellipse_ellipseSpace[0],tptellipse_ellipseSpace[1],tptellipse_ellipseSpace[2],a,b,c)
uhat0 = uhat(p0,p1)
# t = t_tangentLine_line(nhat0, p0, tptellipse_ellipseSpace, uhat0)
#x = x_tangentLine_line(nhat0, p0, tptellipse_ellipseSpace, uhat0)
x = x_tangentpoint_from_line(p0,p1,tptellipse_ellipseSpace,tptellipse_ellipseSpace+1*nhat0)
#tptline_ellipseSpace = p0+uhat0*t
plt.scatter(x,p0[1] + (x-p0[0])*uhat0[1]/uhat0[0],color='red')
#plt.scatter(tptline_ellipseSpace[0],tptline_ellipseSpace[1],color='red')

#The tangent point!!!!!
x,y,z = xyz_tangentpoint_from_line(p0,p1,tptellipse_ellipseSpace,tptellipse_ellipseSpace+1*nhat0)
plt.scatter(x,y,color='green')

plt.gca().axis('equal')
plt.show(block=False)
plt.close(2342356236)



p0 = np.asarray([20,0,0])
p1 = np.asarray([0,20,0])
a = 10
b = 5
c = 1
plt.figure(23423562342236)
xs = np.linspace(start=0,stop=-a)
ys = np.sqrt(b**2.*(1.-xs**2./a**2.))
plt.plot(xs,ys,color='black')
plt.plot([p0[0],p1[0]],[p0[1],p1[1]],color='orange')
#pt0_c, pt1_c = pt_to_ptc(p0, p1, a, b, c)
#tptline_ellipseSpace = tangentPointline_ellipsespace(p0,p1,a,b,c)
#plt.scatter(tptline_ellipseSpace[0],tptline_ellipseSpace[1],color='red')

#The Tangent Point on the Ellipse
tptellipse_ellipseSpace = tangentPointellipse_ellipsespace(p0,p1,a,b,c)
plt.scatter(tptellipse_ellipseSpace[0],tptellipse_ellipseSpace[1],color='blue')
nhat0 = nhat(tptellipse_ellipseSpace[0],tptellipse_ellipseSpace[1],tptellipse_ellipseSpace[2],a,b,c)
uhat0 = uhat(p0,p1)
# t = t_tangentLine_line(nhat0, p0, tptellipse_ellipseSpace, uhat0)
#x = x_tangentLine_line(nhat0, p0, tptellipse_ellipseSpace, uhat0)
x = x_tangentpoint_from_line(p0,p1,tptellipse_ellipseSpace,tptellipse_ellipseSpace+1*nhat0)
#tptline_ellipseSpace = p0+uhat0*t
plt.scatter(x,p0[1] + (x-p0[0])*uhat0[1]/uhat0[0],color='red')
#plt.scatter(tptline_ellipseSpace[0],tptline_ellipseSpace[1],color='red')

#The tangent point!!!!!
x,y,z = xyz_tangentpoint_from_line(p0,p1,tptellipse_ellipseSpace,tptellipse_ellipseSpace+1*nhat0)
plt.scatter(x,y,color='green')

plt.gca().axis('equal')
plt.show(block=False)
plt.close(23423562342236)


p0 = np.asarray([15,0,0])
p1 = np.asarray([0,-20,0])
a = 10
b = 5
c = 1
plt.figure(23423562323442236)
xs = np.linspace(start=0,stop=-a)
ys = np.sqrt(b**2.*(1.-xs**2./a**2.))
plt.plot(xs,ys,color='black')
plt.plot([p0[0],p1[0]],[p0[1],p1[1]],color='orange')
#pt0_c, pt1_c = pt_to_ptc(p0, p1, a, b, c)
#tptline_ellipseSpace = tangentPointline_ellipsespace(p0,p1,a,b,c)
#plt.scatter(tptline_ellipseSpace[0],tptline_ellipseSpace[1],color='red')

#The Tangent Point on the Ellipse
tptellipse_ellipseSpace = tangentPointellipse_ellipsespace(p0,p1,a,b,c)
plt.scatter(tptellipse_ellipseSpace[0],tptellipse_ellipseSpace[1],color='blue')
nhat0 = nhat(tptellipse_ellipseSpace[0],tptellipse_ellipseSpace[1],tptellipse_ellipseSpace[2],a,b,c)
uhat0 = uhat(p0,p1)
# t = t_tangentLine_line(nhat0, p0, tptellipse_ellipseSpace, uhat0)
#x = x_tangentLine_line(nhat0, p0, tptellipse_ellipseSpace, uhat0)
x = x_tangentpoint_from_line(p0,p1,tptellipse_ellipseSpace,tptellipse_ellipseSpace+1*nhat0)
#tptline_ellipseSpace = p0+uhat0*t
plt.scatter(x,p0[1] + (x-p0[0])*uhat0[1]/uhat0[0],color='red')
#plt.scatter(tptline_ellipseSpace[0],tptline_ellipseSpace[1],color='red')

#The tangent point!!!!!
x,y,z = xyz_tangentpoint_from_line(p0,p1,tptellipse_ellipseSpace,tptellipse_ellipseSpace+1*nhat0)
plt.scatter(x,y,color='green')

plt.gca().axis('equal')
plt.show(block=False)
plt.close(23423562323442236)
########################################################



#### Plot Earth at different times #######################
fig3 = plt.figure(num=45383421345341113,figsize=(8,8))
ax3 = fig3.add_subplot(111, projection='3d',computed_zorder=False)
ax3.set_box_aspect(aspect = (1,1,1))#set_aspect('equal')
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
r_earth = 6371. #in km
for i in np.arange(len(rs_earth_sun)):
    #plot invisible box to bound
    ax3.scatter([-1.5*r_earth,-1.5*r_earth,1.5*r_earth,1.5*r_earth,-1.5*r_earth,-1.5*r_earth,1.5*r_earth,1.5*r_earth],[-1.5*r_earth,1.5*r_earth,-1.5*r_earth,1.5*r_earth,-1.5*r_earth,1.5*r_earth,-1.5*r_earth,1.5*r_earth],[-1.5*r_earth,-1.5*r_earth,-1.5*r_earth,-1.5*r_earth,1.5*r_earth,1.5*r_earth,1.5*r_earth,1.5*r_earth],alpha=0.)
    #Plot wireframe of earth
    ax3.plot_wireframe(r_earth*x, r_earth*y, r_earth*z, color="lightgrey",zorder=10)
    #plot origin
    ax3.scatter(0,0,0,color='blue')
    #plot sun and moon vectors
    ax3.scatter(1.5*r_earth*rshat_earth_sun[i][0],1.5*r_earth*rshat_earth_sun[i][1],1.5*r_earth*rshat_earth_sun[i][2],color='yellow')
    ax3.scatter(1.5*r_earth*rshat_earth_moon[i][0],1.5*r_earth*rshat_earth_moon[i][1],1.5*r_earth*rshat_earth_moon[i][2],color='grey')
    ax3.plot([0,1.5*r_earth*rshat_earth_sun[i][0]],[0,1.5*r_earth*rshat_earth_sun[i][1]],[0,1.5*r_earth*rshat_earth_sun[i][2]],color='yellow')
    ax3.plot([0,1.5*r_earth*rshat_earth_moon[i][0]],[0,1.5*r_earth*rshat_earth_moon[i][1]],[0,1.5*r_earth*rshat_earth_moon[i][2]],color='grey')
    ax3.set_title("JD: " + str(jdtimes[i]))
    plt.show(block=False)
    #time.sleep(0.2)
    plt.pause(0.01)
    plt.cla()
####



#At Given Time, plot SV and positions that are visible
fig3 = plt.figure(num=4555555545341113,figsize=(8,8))
ax3 = fig3.add_subplot(111, projection='3d',computed_zorder=False)
ax3.set_box_aspect(aspect = (1,1,1))#set_aspect('equal')
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
r_earth = 6371. #in km
i=0
#plot invisible box to bound
ax3.scatter([-1.5*r_earth,-1.5*r_earth,1.5*r_earth,1.5*r_earth,-1.5*r_earth,-1.5*r_earth,1.5*r_earth,1.5*r_earth],[-1.5*r_earth,1.5*r_earth,-1.5*r_earth,1.5*r_earth,-1.5*r_earth,1.5*r_earth,-1.5*r_earth,1.5*r_earth],[-1.5*r_earth,-1.5*r_earth,-1.5*r_earth,-1.5*r_earth,1.5*r_earth,1.5*r_earth,1.5*r_earth,1.5*r_earth],alpha=0.)
#Plot wireframe of earth
ax3.plot_wireframe(r_earth*x, r_earth*y, r_earth*z, color="grey",zorder=10)
#plot origin
ax3.scatter(0,0,0,color='blue')
#plot sun and moon vectors
ax3.scatter(1.5*r_earth*rshat_earth_sun[i][0],1.5*r_earth*rshat_earth_sun[i][1],1.5*r_earth*rshat_earth_sun[i][2],color='yellow')
ax3.scatter(1.5*r_earth*rshat_earth_moon[i][0],1.5*r_earth*rshat_earth_moon[i][1],1.5*r_earth*rshat_earth_moon[i][2],color='grey')
ax3.plot([0,1.5*r_earth*rshat_earth_sun[i][0]],[0,1.5*r_earth*rshat_earth_sun[i][1]],[0,1.5*r_earth*rshat_earth_sun[i][2]],color='yellow')
ax3.plot([0,1.5*r_earth*rshat_earth_moon[i][0]],[0,1.5*r_earth*rshat_earth_moon[i][1]],[0,1.5*r_earth*rshat_earth_moon[i][2]],color='grey')

#Plot Look Vectors
utc = ['2024-07-04T00:00:00.00', '2025-06-21T00:00:00']
a, b, c = 6371., 6371., 6371.
jd = Time([utc[0]],format='isot',scale='tai')
jdtime = jd.jd[0]
numPlotted = 0
numTanHeight = 0
numVisible = 0
vis0 = list()
#for i in np.arange(num_locs):
i=50
vis1 = list()
for j in np.arange(num_locs):
    if not (i==j):
        visible, isVisibleLunar, isVisibleSolar, isVisibleTangentHeight, isVisibleSolarZenithAngle, isVisibleSolarPhaseAngle  = \
            isVisible(r_locs[:,i],r_locs[:,j],kernel,jdtime,a,b,c)
        print(str(i) + ", " + str(visible) + ", "+ str(isVisibleLunar) + ", " +str(isVisibleSolar) + ", " +str(isVisibleTangentHeight) + ", "+ str(isVisibleSolarZenithAngle) + ", "+ str(isVisibleSolarPhaseAngle))
        vis1.append([visible, isVisibleLunar, isVisibleSolar, isVisibleTangentHeight, isVisibleSolarZenithAngle, isVisibleSolarPhaseAngle])
        if visible:
            color='green'#'lightgreen'
            numVisible += 1
            print(i/num_locs)
            #color='green'
        elif isVisibleTangentHeight: #the SV is not obscured by the planet
            if not isVisibleLunar: #a small and unique keepout
                color='grey'
            elif not isVisibleSolar: #A larger lunar keepout
                color='goldenrod'
            elif not isVisibleSolarPhaseAngle: #a larger solar keepout
                color='gold'
            elif not isVisibleSolarZenithAngle: #A solar zenith angle, draws so many lines
                color='mediumpurple'
            else:
                continue #don't plot anything
        else:
            numTanHeight += 1
            #color='purple'
            continue #don't plot anything
        ax3.plot([r_locs[0,i],r_locs[0,j]],[r_locs[1,i],r_locs[1,j]],[r_locs[2,i],r_locs[2,j]],color=color,alpha=1.)

        numPlotted += 1
    else:
        vis1.append([])
plt.show(block=False)



####    ##################################
#0 - visible
#1 - lunar
#2 - solar
#3 - solar phase
#4 - zenith
#5 - tangent height
#from itertools import combinations
from itertools import product
combos = list(product([0,1],repeat=5))
combos = np.asarray(combos)
combos_ij = list()
for i in np.arange(combos.shape[0]):
    combos_ij.append(list())


utc = ['2024-07-04T00:00:00.00', '2025-01-21T00:00:00']
jdlims = Time(utc,format='isot',scale='tai')
jdtimes = np.linspace(start=jdlims.jd[0],stop=jdlims.jd[1],num=100)
jds = Time(utc,format='isot',scale='tai')
jdtime = jds.jd[0]

for k in np.arange(len(jds)): #iterate over times to evaluate at
    jdtime = jds.jd[k]
    #### Create array of all visiblity status between two points
    visibility = np.full((num_locs, num_locs, 6), False)
    for i in np.arange(num_locs):#Iterate over number of points
        for j in np.arange(num_locs):#Iterate over number of points
            if not (i==j): #If theyre not the same point, comput evisibility
                visible, isVisibleLunar, isVisibleSolar, isVisibleTangentHeight, isVisibleSolarZenithAngle, isVisibleSolarPhaseAngle  = \
                    isVisible(r_locs[:,i],r_locs[:,j],kernel,jds.jd[k],a,b,c)
                #print("(" + str(i) + "," + str(j) + "), " + str(visible) + ", "+ str(isVisibleLunar) + ", " +str(isVisibleSolar) + ", " +str(isVisibleTangentHeight) + ", "+ str(isVisibleSolarZenithAngle) + ", "+ str(isVisibleSolarPhaseAngle))
                visibility[i,j,0] = visible
                visibility[i,j,1] = isVisibleLunar
                visibility[i,j,2] = isVisibleSolar
                visibility[i,j,3] = isVisibleTangentHeight
                visibility[i,j,4] = isVisibleSolarZenithAngle
                visibility[i,j,5] = isVisibleSolarPhaseAngle
                ind = np.where(np.all(combos == np.asarray([int(isVisibleLunar), int(isVisibleSolar), int(isVisibleTangentHeight), int(isVisibleSolarZenithAngle), int(isVisibleSolarPhaseAngle)]),axis=1))[0][0]
                combos_ij[ind].append((i,j,jdtime))

#Enumerate the number of each combination that has occured
lenCombos = list()
for i in np.arange(len(combos_ij)):
    lenCombos.append(len(combos_ij[i]))
    if len(combos_ij[i]) == 0:
        print(combos[i])




# #### EARTH, MOON, SUN POSITIONS
# spice.furnsh("./naif0009.tls")
# # get et values one and two, we could vectorize str2et
# etOne = spice.str2et(utc[0])
# etTwo = spice.str2et(utc[1])
# print("ET One: {}, ET Two: {}".format(etOne, etTwo))
# # get times
# times = [x*(etTwo-etOne)/step + etOne for x in range(step)]

# #Run spkpos as a vectorized function
# #positions, lightTimes = spice.spkpos('Cassini', times, 'J2000', 'NONE', 'SATURN BARYCENTER')
# positions_sun, lightTimes = spice.spkpos('10', times, 'J2000', 'NONE', '399') #399 Earth center, #10 sun center, #301 Moon center
# positions_moon, lightTimes = spice.spkpos('301', times, 'J2000', 'NONE', '399')
# """
#     targ       I   Target body name.
#    et         I   Observer epoch. in seconds past J2000
#    ref        I   Reference frame of output position vector. i.e J2000, 
#    abcorr     I   Aberration correction flag. "NONE" means no correction will be applied
#    obs        I   Observing body name. #some valid bodies https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/18_spk.pdf
# """


# # Positions is a 3xN vector of XYZ positions
# print("Positions: ")
# print(positions[0])

# # Light times is a N vector of time
# print("Light Times: ")
# print(lightTimes[0])

#ts = np.arange(180)*10 #maxt was arount 1751
#for t in np.arange(len(ts)):

#### TLE FROM KOE
def TLE_from_KOE(satnum, epochyr, epochdays, ndot, bstar, inclination, raan, eccentricity, perigee, meanAnomaly, meanMotion):
    """ Returns a list containing two lines defining the TLE
    inclination in degrees
    raan in degrees
    perigee in degrees
    meanAnomaly in degrees
    meanMotion in revolutions per day
    """
    tle = list()

    tle1 = "1 "
    tle1 += '{:5d}'.format(satnum)#tle1 += "%05d".format(satnum)#String.format("%05d", satnum)
    tle1 += "U"
    tle1 += "          "
    tle1 += '{:2d}'.format(epochyr) #"%02d".format(epochyr)
    tle1 += '{:12.8f}'.format(epochdays).replace(' ','0') #"%03.8f".format(epochdays)
    tle1 += "  "
    tle1 += ("{:.8f}".format(ndot))[1:]#"%.8f" .substring(1) #9
    tle1 += "  "
    tmpstring = "{:.4e}".format(bstar) #"%5e" bstar
    y = tmpstring.split("e")
    bstar_exp = str(int(y[1])) #Integer.toString(Integer.parseInt(y[1])+1)
    bstar_float = float(y[0]) #Double.parseDouble(y[0])
    if bstar_float > 0:
        tle1 += " "
    if int(y[1]) < 0:
        bstar_sgn = '-'
    else:
        bstar_sgn = '+'
    tle1 += "{:.4f}".format(bstar_float).replace(".","") + bstar_sgn + bstar_exp #String.format("%.4f",bstar_float).replace(".", "") + bstar_exp
    tle1 += " "
    tle1 += " 00000-0"
    tle1 += " "
    tle1 += "0"
    tle1 += " "
    tle1 += " 0000"


    tle2 = "2 "
    tle2 += '{:5d}'.format(satnum) #String.format("%05d", satnum)
    tle2 += " "
    #DecimalFormat df1 = new DecimalFormat("000.0000")
    tmpInc = "{:8.4f}".format(inclination).replace(' ','0') #df1.format(Math.toDegrees(inclination))
    if tmpInc[:2] == "00":
        tmpInc = "  " + tmpInc[2:]
    elif tmpInc[0] == "0":
        tmpInc = " " + tmpInc[1:]
    tle2 += tmpInc
    tle2 += " "
    tle2 += "{:8.4f}".format(raan).replace(' ','0') #df1.format(Math.toDegrees(raan))
    tle2 += " "
    tle2 += "{:.7f}".format(eccentricity).replace('0.','') #String.format("%.7f",eccentricity).replace("0.","")
    tle2 += " "
    perigeeString = "{:8.4f}".format(perigee).replace(' ','0') #df1.format(Math.toDegrees(perigee))
    if perigeeString[:2] == "00":
        perigeeString = "  " + perigeeString[2:]
    elif perigeeString[0] == "0":
        perigeeString = " " + perigeeString[1:]
    tle2 += perigeeString
    tle2 += " "
    tle2 += "{:8.4f}".format(meanAnomaly).replace(' ','0') #df1.format(Math.toDegrees(meanAnomaly))
    tle2 += " "
    #df2 = new DecimalFormat("00.00000000")
    #meanMotionString = df2.format(meanMotion)
    tle2 += "{:11.10}".format(meanMotion).replace(' ','0')
    tle2 += "00000"
    #Integer checksumVal2 = (Integer) calculateChecksum(line2)%10
    #tle2 += checksumVal2.toString()
    tle2 += str(calculateChecksum(tle2)%10)


    tle.append(tle1)
    tle.append(tle2)
    return tle

def calculateChecksum(tleLine):
    checksum = 0 #int checksum = 0;
    #char[] chars = tleLine.toCharArray();
    for c in tleLine:
        if c == " " or c == "." or c == "+" or c.isalnum(): #character.isletter(c):
            continue
        elif c == "-":
            checksum +=1
        elif c.isdigit(): #character.isDigit(c):
            checksum += int(c) #Character.getNumericValue(c)
        else:
            assert error, "something went wrong"
        #    throw new build exception

    checksum -= int(tleLine[-1])
    #checksum -= Character.getNumericValue(chars[chars.length - 1])
    checksum %= 10

    return checksum









#### Position in ITRS to TLE
i=50
n=1

def XYZ_to_TLE(X,Y,Z,jd):
    n=1
    # X = r_locs[0,i]
    # Y = r_locs[1,i]
    # Z = r_locs[2,i]
    #out_lines = xyz_vxvyvz_to_TLE(r_locs[0,i],r_locs[1,i],r_locs[2,i],,,)
    #we can assume the following
    eccentricity=0.
    # inclination
    C = 0.5*(np.cos(0.)-np.cos(np.pi))
    I = (np.arccos(np.cos(0.) - 2.*C*np.random.uniform(size=n)))
    # longitude of the ascending node
    O = np.random.uniform(low=0., high=2.*np.pi, size=n)
    # argument of periapse
    w = 0.# np.random.uniform(low=0., high=2.*np.pi, size=n)
    #inclination = 
    #raan = 
    #perigee = 
    #for circular orbits, true anomaly does not exist, use u=v+w to compute argument of latitude
    #X = r(cos(Ω) cos(ω + ν) − sin(Ω) sin(ω + ν) cos(i))
    #Y = r(sin(Ω) cos(ω + ν) + cos(Ω) sin(ω + ν) cos(i))
    #Z = r sin(i) sin(ω + ν).
    meanAnomaly = np.arctan2(Z*np.cos(O),X*np.sin(I)+Z*np.sin(O)*np.cos(I))
    satnum=0
    epochyr=int(str(jd.datetime[0].year)[-2:])
    epochdays=jd.datetime[0].timetuple().tm_yday + (jd.datetime[0].timetuple().tm_hour+(jd.datetime[0].timetuple().tm_min+jd.datetime[0].timetuple().tm_sec/60.)/60.)/24.
    ndot=0.
    bstar=0.
    sma = np.sqrt((X)**2.+(Y)**2.+(Z)**2.) #in km
    M=5.972*10**24. #in kg
    G=6.6743*10**-11./(1000**3.) #in km3
    meanMotion=np.sqrt(G*M/sma**3.)*(24.*60.*60.)/(2.*np.pi) # in rev per day (is this the right day?)

    tleLines = TLE_from_KOE(satnum, epochyr, epochdays, ndot, bstar, I[0]*(180./np.pi), O[0]*(180./np.pi), eccentricity, w*(180./np.pi), meanAnomaly[0]*(180./np.pi), meanMotion)
    return tleLines


# def xyz_vxvyvz_to_TLE(x,y,z,vx,vy,vz):
#     """ A function to convert from xyz and vx,vy,vz to a TLE
#     """

#     lines = list()
#     lines.append(line1)
#     lines.append(line2)
#     return lines


############################################################
# #### A Check to ensure the TLE function works as intended
# #tle = "ORBCOMM-X [-]           
# tle1 = "1 21576U 91050C   22237.17288983  .00000077  00000+0  37081-4 0  9994"
# tle2 = "2 21576  98.4766 302.3208 0004616  54.6548 305.5072 14.41800866634069"
# satnum = 21576
# epochyr = 22
# epochdays = 237.17288983
# ndot = .00000077
# bstar = 00000+0
# inclination = 98.4766
# raan = 302.3208
# eccentricity = 0.0004616
# perigee = 54.6548
# meanAnomaly = 305.5072
# meanMotion = 14.41800866
# tle = TLE_from_KOE(satnum, epochyr, epochdays, ndot, bstar, inclination*(180./np.pi), raan*(180./np.pi), eccentricity, perigee*(180./np.pi), meanAnomaly*(180./np.pi), meanMotion)
# ############################################################################




#### Print Combos Out To Files #######################################################
for i in np.arange(len(combos)):
    #Create the filename
    filename = "combos" + "".join(str(x) for x in combos[i]) + ".txt"

    #if file exists, delete it
    if os.path.exists(filename):
        os.remove(filename)

    for j in np.arange(len(combos_ij[i])):
        datatuple = combos_ij[i][j]
        #datatuple[0] gets turned into TLE
        tleLine = XYZ_to_TLE(r_locs[0,datatuple[0]],r_locs[1,datatuple[0]],r_locs[2,datatuple[0]],jd)
        #datatuple[1] is the xyz point
        xyz = r_locs[:,datatuple[1]]
        #datatuple[2] jdtime in tai


        #Write out to file
        with open(filename, "a+") as file:
            file.write(tleLine[0] + "\n")
            file.write(tleLine[1] + "\n")
            file.write(str(xyz[0]) + ","+str(xyz[1])+","+str(xyz[2])+","+str(jd.jd[0])+"\n")
#######################################################################################
