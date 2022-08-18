# Sample Examples

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from skyfield.sgp4lib import TEME_to_ITRF
from jplephem.spk import SPK
from astropy.time import Time
import time

#### INPUTS
spkpath = './de432s.bsp'
kernel = SPK.open(spkpath) # smaller file, covers mission time range
step = 4000
# we are going to get positions between these two dates
utc = ['Jun 21, 2024', 'Jun 22, 2024'] #Earth shoud be almost entirely in the x direction relative to the sun
utc = ['2024-06-21T00:00:00.00', '2024-06-22T00:00:00']
utc = ['2024-06-21T00:00:00.00', '2025-06-21T00:00:00']
jdlims = Time(utc,format='isot',scale='tai')
jdtimes = np.linspace(start=jdlims.jd[0],stop=jdlims.jd[1],num=100)

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
r_earth = 6371.0

def genRandomLatLon(num=1,lat_low=-np.pi/2.,lat_high=np.pi/2.,lon_low=0,lon_high=2.*np.pi):
    """ Generates a random latitude and longitude
    """
    lons = np.random.uniform(low=lon_low,high=lon_high,size=num)
    lats = np.arccos(2.*np.random.uniform(low=np.sin(lat_low),high=np.sin(lat_high),size=num)-1.)
    return lons, lats

def lonLats_to_xyz(r,lons,lats):
    """ Computes x,y,z position from lons and lats
    """
    return r*np.asarray([np.cos(lats)*np.sin(lons),np.cos(lats)*np.sin(lons),np.sin(lats)])

num_locs = 200    
lons, lats = genRandomLatLon(num_locs,lat_low=-np.pi/2.,lat_high=np.pi/2.,lon_low=0.,lon_high=2.*np.pi) #Full sphere
#Randomly generate locations
r_locs = lonLats_to_xyz(r_earth+200,lons,lats)


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

#TODO verify these calculations!!!, is it tpt_c[0]*a pr tpt_c[0]/a?????
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





p0 = np.asarray([-20,0,0])
p1 = np.asarray([0,20,0])
a = 10
b = 5
c = 1


plt.figure(2342356236)
xs = np.linspace(start=0,stop=-a)
ys = np.sqrt(b**2.*(1.-xs**2./a**2.))
plt.plot(xs,ys)
plt.plot([p0[0],p1[0]],[p0[1],p1[1]])
#pt0_c, pt1_c = pt_to_ptc(p0, p1, a, b, c)
tptline_ellipseSpace = tangentPointline_ellipsespace(p0,p1,a,b,c)
plt.scatter(tptline_ellipseSpace[0],tptline_ellipseSpace[1])
tptellipse_ellipseSpace = tangentPointellipse_ellipsespace(p0,p1,a,b,c)
plt.scatter(tptellipse_ellipseSpace[0],tptellipse_ellipseSpace[1])
plt.show(block=False)




fig3 = plt.figure(num=45383421345341113,figsize=(8,8))
ax3 = fig3.add_subplot(111, projection='3d',computed_zorder=False)
ax3.set_box_aspect(aspect = (1,1,1))#set_aspect('equal')
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
r_earth = 6731. #in km
for i in np.arange(len(rs_earth_sun)):
    #plot invisible box to bound
    ax3.scatter([-1.5*r_earth/1000.,-1.5*r_earth/1000.,1.5*r_earth/1000.,1.5*r_earth/1000.,-1.5*r_earth/1000.,-1.5*r_earth/1000.,1.5*r_earth/1000.,1.5*r_earth/1000.],[-1.5*r_earth/1000.,1.5*r_earth/1000.,-1.5*r_earth/1000.,1.5*r_earth/1000.,-1.5*r_earth/1000.,1.5*r_earth/1000.,-1.5*r_earth/1000.,1.5*r_earth/1000.],[-1.5*r_earth/1000.,-1.5*r_earth/1000.,-1.5*r_earth/1000.,-1.5*r_earth/1000.,1.5*r_earth/1000.,1.5*r_earth/1000.,1.5*r_earth/1000.,1.5*r_earth/1000.],alpha=0.)
    #Plot wireframe of earth
    ax3.plot_wireframe(r_earth/1000.*x, r_earth/1000.*y, r_earth/1000.*z, color="grey",zorder=10)
    #plot origin
    ax3.scatter(0,0,0,color='blue')
    #plot sun and moon vectors
    ax3.scatter(1.5*r_earth/1000.*rshat_earth_sun[i][0],1.5*r_earth/1000.*rshat_earth_sun[i][1],1.5*r_earth/1000.*rshat_earth_sun[i][2],color='yellow')
    ax3.scatter(1.5*r_earth/1000.*rshat_earth_moon[i][0],1.5*r_earth/1000.*rshat_earth_moon[i][1],1.5*r_earth/1000.*rshat_earth_moon[i][2],color='grey')
    ax3.plot([0,1.5*r_earth/1000.*rshat_earth_sun[i][0]],[0,1.5*r_earth/1000.*rshat_earth_sun[i][1]],[0,1.5*r_earth/1000.*rshat_earth_sun[i][2]],color='yellow')
    ax3.plot([0,1.5*r_earth/1000.*rshat_earth_moon[i][0]],[0,1.5*r_earth/1000.*rshat_earth_moon[i][1]],[0,1.5*r_earth/1000.*rshat_earth_moon[i][2]],color='grey')
    plt.show(block=False)
    #time.sleep(0.2)
    plt.pause(0.01)
    plt.cla()







#### EARTH, MOON, SUN POSITIONS
spice.furnsh("./naif0009.tls")
# get et values one and two, we could vectorize str2et
etOne = spice.str2et(utc[0])
etTwo = spice.str2et(utc[1])
print("ET One: {}, ET Two: {}".format(etOne, etTwo))
# get times
times = [x*(etTwo-etOne)/step + etOne for x in range(step)]

#Run spkpos as a vectorized function
#positions, lightTimes = spice.spkpos('Cassini', times, 'J2000', 'NONE', 'SATURN BARYCENTER')
positions_sun, lightTimes = spice.spkpos('10', times, 'J2000', 'NONE', '399') #399 Earth center, #10 sun center, #301 Moon center
positions_moon, lightTimes = spice.spkpos('301', times, 'J2000', 'NONE', '399')
"""
    targ       I   Target body name.
   et         I   Observer epoch. in seconds past J2000
   ref        I   Reference frame of output position vector. i.e J2000, 
   abcorr     I   Aberration correction flag. "NONE" means no correction will be applied
   obs        I   Observing body name. #some valid bodies https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/18_spk.pdf
"""


# Positions is a 3xN vector of XYZ positions
print("Positions: ")
print(positions[0])

# Light times is a N vector of time
print("Light Times: ")
print(lightTimes[0])





#ts = np.arange(180)*10 #maxt was arount 1751
#for t in np.arange(len(ts)):








#### TODO Position in ITRS to TLE


def xyz_vxvyvz_to_TLE(x,y,z,vx,vy,vz):
    """ A function to convert from xyz and vx,vy,vz to a TLE
    """

    lines = list()
    lines.append(line1)
    lines.append(line2)
    return lines
