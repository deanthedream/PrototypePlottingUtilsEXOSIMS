#### Ringed Planet Model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
#from quaternion import *

#Cylinder Equation
def planet_cyl(r_kstar,d_z,R,r_ksunbot1,r_ksunbot2,phi):
    """ Produces a point on the surface of a 3D cylinder
    Args:
        r_kstar (numpy array) - 3D array describing unit vector from star to planet
        d_z (float) - distance along direction of r_kstar
        R (float) - radius of the planet
        r_ksunbot1 (numpy array) - 3D array describing a vector mutually orthogonal to r_ksunbot2 and r_kstar
        r_ksunbot2 (numpy array) - 3D array describing a vector mutually orthogonal to r_ksunbot1 and r_kstar
        phi (float) - angle
    """
    S_cyl = r_kstar * d_z + r_ksunbot1 * R * np.sin(phi) + r_ksunbot2 * R * np.cos(phi)
    return S_cyl

# Circle Equation
def ring_circ(R_r,r_Rbot1,r_Rbot2,phi):
    """ Produces a point on the 3D circle
    Args:
        R_r (float) - circle radius
        r_Rbot1 (numpy array) - 3D array describing a vector mutually orthogonal to r_Rbot2 and the ring plane vector
        r_Rbot2 (numpy array) - 3D array describing a vector mutually orthogonal to r_Rbot1 and the ring plane vector
        phi (float) - angle
    returns:
        S_circ (numpy array) - 3d numpy array defining a single point in space
    """
    S_circ = R_r * r_Rbot1 * np.sin(phi)  + R_r * r_Rbot2 * np.cos(phi)
    return S_circ

def generate_twoVectPerpToVect(vect):
    """ Generate two 3D vectors perpendicular to a 3D vector
    Args:
        vect (nmupy array) - a 3D vector
    Retunrs:
        vect1 (numpy array) - a 3D vector mutually orthogonal to vect and vect2 
        vect2 (numpy array) - a 3D vector mutually orthogonal to vect and vect1
    """
    vect = vect/np.linalg.norm(vect)
    vect_tmp = np.asarray([0.,0.,1.]) # vector along pole axis
    if not (vect == vect_tmp).all(): # Ensure vect_tmp is not coincident with vect
        vect1 = np.cross(vect_tmp,vect)/np.linalg.norm(np.cross(vect_tmp,vect)) # vector along lon. line
    else:
        vect1 = np.asarray([1.,0.,0.])
    vect1 = vect1/np.linalg.norm(vect1)
    vect2 = np.cross(vect,vect1)/np.linalg.norm(np.cross(vect,vect1))
    return vect1, vect2

def calc_CriticalThetas(R, R_rmin, R_rmax):
    """ Calculates Critical Angles where a planet/ring will begin obstructing one another
    Args:
        R (float) - planet radius
        R_rmin (float) - ring inner radius
        R_rmax (float) - ring outer radius
    Return:
        theta_crit1 (float) - angle in radians between ring normal and either r_view or r_kstar
        theta_crit2 (float) - angle in radians
    """
    theta_crit1 = np.arccos(R/R_rmin)
    theta_crit2 = np.arccos(R/R_rmax)
    return theta_crit1, theta_crit2

def calc_ellipseCircleIntersections(a,b,R):
    """ Calculates ellipse and circle intersection angles
    Assuming th=0 is along a semi-major axis of the ellipse
    http://mathworld.wolfram.com/Circle-EllipseIntersection.html
    Args:
        a (float) - circle semi-major axis
        b (float) - circle semi-minor axis
        R (float) - planet radius
    Returns:
        th1 (float) - top right
        th2 (float) - top left
        th3 (float) - bottom left
        th4 (float) - bottom right
    """
    x1 = a*np.sqrt((R**2.-b**2.)/(a**2.-b**2.))
    x2 = -x1
    y1 = b*np.sqrt((a**2.-R**2.)/(a**2.-b**2.))
    y2 = -y1

    th1 = np.arctan2(y1,x1)
    th2 = np.arctan2(y1,x2)
    th3 = np.arctan2(y2,x2)
    th4 = np.arctan2(y2,x1)
    return th1, th2, th3, th4

def calc_ellipticityAngle(r_r,r_view):
    """Calculates the smallest angle from a viewing direction and a surface normal vector of a 3D circle
    Args:
        r_r (numpy array) - ring surface normal vector
        r_view (numpy array) - viewing direction
    Returns:
        r_r2 (numpy array) - ring surface normal forming smallest angle with r_view
        phi_ellipse (float) - angle between in ring plane an 
    """
    if np.dot(r_r,r_view) >= 0.:
        r_r2 = r_r
    else:# np.dot(-r_r,r_view) < 0
        r_r2 = -r_r
    phi_ellipse = np.arccos(np.dot(r_r2,r_view)/np.linalg.norm(r_r2)/np.linalg.norm(r_view)) #angle formed between ring plane and viewing angle
    return r_r2, phi_ellipse

def calc_AreaUnderEllipseSection(a,b,th1,th2):
    """ Calculates the Area under an ellipse between two angles starting from a semi-major axis direction
    From: https://keisan.casio.com/exec/system/1343722259
    Args:
        a (float) - ellipse semi-major axis
        b (float) - ellipse semi-minor axis
        th1 (float) - angle #1 defined from a direction in rad
        th2 (float) - angle #2 defined from a direction in rad
    Returns:
        area (float) - area between two angles of ellipse
    """
    F2 = a*b/2.*(th2-np.arctan2((b-a)*np.sin(2.*th2),b+a+(b-a)*np.cos(2.*th2)))
    F1 = a*b/2.*(th1-np.arctan2((b-a)*np.sin(2.*th1),b+a+(b-a)*np.cos(2.*th1)))
    area = F2 - F1
    return area

#### Inputs ##################################################################################
#Viewing direction
r_view = np.asarray([0.,-1.,0.]) #must be unit vector
r_kstar = np.asarray([1.,0.,0.]) #Vector from the star to the planet
r_r = np.sin(np.pi/6.)*np.asarray([1.,0.,0.]) + np.cos(np.pi/6.)*np.asarray([0.,np.cos(45.*np.pi/180.),np.sin(45.*np.pi/180.)]) #ring plane normal vector
R  = 10. #planet radius #Depricating
R_r= 1.5*R #a ring radius
R_rmin = 1.5*R #minimum ring radius
R_rmax = 2.5*R
##############################################################################################

#### Maximal Ring SA #########################################################################
SA_ringInner = np.pi*R_rmin**2.
SA_ringOuter = np.pi*R_rmax**2.
SA_ring = SA_ringOuter - SA_ringInner #Rough total area of ring in top down view
##############################################################################################

#### Ring Ellipticity - Viewing Direction ####################################################
r_r2, phi_ellipse_view = calc_ellipticityAngle(r_r,r_view)
b_Rrmin_view = np.cos(phi_ellipse_view)*R_r #minor axis of apparent ellipse of ring
#### Ring Ellipticity - Star Direction
r_r3, phi_ellipse_kstar = calc_ellipticityAngle(r_r,r_kstar)
b_Rrmin_kstar = np.cos(phi_ellipse_kstar)*R_r #minor axis of apparent ellipse of ring
################################################################################################

#### Plot Cylinder and circle

#Genetate Cylinder
d_z = np.linspace(start=-2.5*R, stop=2.5*R,num=30,endpoint=True) #various distances along r_kstar_hat direction to define distance
r_ksunbot1, r_ksunbot2 = generate_twoVectPerpToVect(r_kstar) #Two perp vect perp to R_kstar
phi1 = np.linspace(start=0.,stop=2*np.pi,num=30) #various angles about r_kstar to define the circle

p0 = np.asarray([0.,0.,0.]) #starting point
d_z_mesh, phi1_mesh = np.meshgrid(d_z,phi1)
X_cyl, Y_cyl, Z_cyl = [p0[i] + r_kstar[i] * d_z_mesh + R * np.sin(phi1_mesh) * r_ksunbot1[i] + R * np.cos(phi1_mesh) * r_ksunbot2[i] for i in [0, 1, 2]]

#### Generate R_rmin circle
r_Rbot1, r_Rbot2 = generate_twoVectPerpToVect(r_r) #Two perp vect in ring plane perp to r_r
phi2 = np.linspace(start=0.,stop=2*np.pi,num=30) #various angles about r_r
#for plotting
S_circs = list()
for phi in phi2:
    S_circs.append(ring_circ(R_rmin,r_Rbot1, r_Rbot2,phi))
S_circs = np.asarray(S_circs)

#### Generate R_rmax circle
S_circs_max = list()
for phi in phi2:
    S_circs_max.append(ring_circ(R_rmax,r_Rbot1, r_Rbot2,phi))
S_circs_max = np.asarray(S_circs_max)

#### Direction of semi-minor and semi-major on actual ring plane
r_ellipsea1 = R_rmin*np.cross(r_r2,r_view)/np.linalg.norm(np.cross(r_r2,r_view))
r_ellipseb1 = R_rmin*np.cross(r_ellipsea1,r_r2)/np.linalg.norm(r_ellipsea1)
#r_tmp1 = r_view*np.dot(r_r2,r_view) #component of r_r in direction of r_view
#r_tmp2 = r_r2-r_tmp1 #component vector of r_r perpendicular to r_view
#r_tmp3 = -np.sin(np.pi/2.-phi_ellipse_view)*r_tmp2/np.linalg.norm(r_tmp2) #component vector to 
#r_ringb1 = R_r*(r_tmp2+r_tmp3)
#r_tmp2 = r_view*np.sqrt(1.-np.linalg.norm(r_tmp1)**2.)
#r_ellipseb1 = R_r*(r_tmp1+r_tmp2)
#r_ellipseb2 = -r_ellipseb1

##### Component Vector of Ellipse perpendicular to r_view in apparent ellipse direction of b
r_ellipseb1_proj = (r_ellipseb1-np.dot(r_view,r_ellipseb1)*r_view)
r_ellipsea1_proj = (r_ellipsea1-np.dot(r_view,r_ellipsea1)*r_view)
S_circ_viewproj = list()
for phi in phi2:
    S_circ_viewproj.append(np.cos(phi)*r_ellipseb1_proj + np.sin(phi)*r_ellipsea1_proj)
S_circ_viewproj = np.asarray(S_circ_viewproj)

#### Minimum phi_ellipse_view Angle where the planet obstructs ring visibility ##############################
#NOTE THIS MUST BE GENERALIZED SO IT CAN BE USED FOR ILLUMINATION OBSTRCUTION
theta = phi_ellipse_view
theta_crit1, theta_crit1 = calc_CriticalThetas(R, R_rmin, R_rmax)
########################################################################################################

#### Planet Obstructing Ring Points ####################################################################
#In the viewing plane, the following angles represent the intersections in the viewing plane.
#Orientation should not matter
th1, th2, th3, th4 = calc_ellipseCircleIntersections(R_rmin,np.linalg.norm(r_ellipseb1_proj),R)
#Calculate points projected on the view plane
intpt_viewproj1 = np.cos(th1)*r_ellipsea1_proj/np.linalg.norm(r_ellipsea1_proj)*R + np.sin(th1)*r_ellipseb1_proj/np.linalg.norm(r_ellipseb1_proj)*R
intpt_viewproj2 = np.cos(th2)*r_ellipsea1_proj/np.linalg.norm(r_ellipsea1_proj)*R + np.sin(th2)*r_ellipseb1_proj/np.linalg.norm(r_ellipseb1_proj)*R
intpt_viewproj3 = np.cos(th3)*r_ellipsea1_proj/np.linalg.norm(r_ellipsea1_proj)*R + np.sin(th3)*r_ellipseb1_proj/np.linalg.norm(r_ellipseb1_proj)*R
intpt_viewproj4 = np.cos(th4)*r_ellipsea1_proj/np.linalg.norm(r_ellipsea1_proj)*R + np.sin(th4)*r_ellipseb1_proj/np.linalg.norm(r_ellipseb1_proj)*R
#Calculate points on ellipse projected on the view plane
########################################################################################################

#### Generate Circle of sphere 
r_Rbot1_viewproj, r_Rbot2_viewproj = generate_twoVectPerpToVect(-r_view) #Two perp vect in ring plane perp to r_r
#for plotting
S_circs_proj = list()
for phi in phi2:
    S_circs_proj.append(ring_circ(R,r_Rbot1_viewproj, r_Rbot2_viewproj,phi))
S_circs_proj = np.asarray(S_circs_proj)


#Bounding box edges
maxBounds = np.max([np.max(np.abs(X_cyl)),np.max(np.abs(Y_cyl)),np.max(np.abs(Z_cyl)),np.max(np.abs(S_circs))])

plt.close(1)
fig1 = plt.figure(num=1)
ax1= fig1.add_subplot(111, projection= '3d')
#ax1.set_aspect('equal')
ax1.plot(S_circs[:,0],S_circs[:,1],S_circs[:,2],color='red') #plot R_rmin circle
ax1.plot(S_circs_max[:,0],S_circs_max[:,1],S_circs_max[:,2],color='red') #plot R_rmax circle
ax1.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.2, color='blue')#, rstride=rstride, cstride=cstride)
ax1.plot([-20.*r_kstar[0],0.],[-20.*r_kstar[1],0.],[-20.*r_kstar[2],0.],color='cyan') #plot sun to star vector
ax1.scatter([maxBounds,maxBounds,maxBounds,maxBounds,-maxBounds,-maxBounds,-maxBounds,-maxBounds],\
        [maxBounds,maxBounds,-maxBounds,-maxBounds,maxBounds,maxBounds,-maxBounds,-maxBounds],\
        [maxBounds,-maxBounds,maxBounds,-maxBounds,maxBounds,-maxBounds,maxBounds,-maxBounds],alpha=0.)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.view_init(elev=180./np.pi*np.arcsin(r_view[2]), azim=180./np.pi*np.arcsin(r_view[1]/np.sqrt(r_view[0]**2. + r_view[1]**2.)))#set viewing angle
#ax1.plot([0.,r_ellipseb1[0]],[0.,r_ellipseb1[1]],[0.,r_ellipseb1[2]],color='cyan') #plot projected component vectors of apparent ellipse
ax1.plot([0.,r_ellipseb1[0]],[0.,r_ellipseb1[1]],[0.,r_ellipseb1[2]],color='cyan') #b1 in ring plane
ax1.plot([0.,r_ellipseb1_proj[0]],[-20.,-20.],[0.,r_ellipseb1_proj[2]],color='orange') #b1 in viewing plane
ax1.plot([0.,r_ellipsea1_proj[0]],[-20.,-20.],[0.,r_ellipsea1_proj[2]],color='orange') #a1 in viewing plane
ax1.scatter([intpt_viewproj1[0],intpt_viewproj2[0],intpt_viewproj3[0],intpt_viewproj4[0]],\
        [-20.,-20.,-20.,-20.],\
        [intpt_viewproj1[2],intpt_viewproj2[2],intpt_viewproj3[2],intpt_viewproj4[2]],color='orange') #a1 in viewing plane
ax1.plot(S_circs_proj[:,0],S_circs_proj[:,1]-20.,S_circs_proj[:,2],color='blue')
ax1.plot(S_circ_viewproj[:,0],-20.+S_circ_viewproj[:,1],S_circ_viewproj[:,2],color='orange')
# ax1.plot([R_r*r_ellipseb1[0],R_r*r_ellipseb1[0]+R_r*r_ellipseb1[0]],\
#         [R_r*r_ellipseb1[1],R_r*r_ellipseb1[1]+R_r*r_ellipseb1[1]],\
#         [R_r*r_ellipseb1[2],R_r*r_ellipseb1[2]+R_r*r_ellipseb1[2]],color='cyan') #plot projected component vectors of apparent ellipse


#### Plot Sphere For Reference ####################################################
#### PLOT BLUE
u = np.linspace(np.pi/2., -np.pi/2., num=100)
v = np.linspace(0, np.pi, num=100)
theta = np.linspace(0, 2*np.pi, num=100)

x = R * np.outer(np.cos(u), np.sin(v))
y = R * np.outer(np.sin(u), np.sin(v))
z = R * np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_surface(x, y, z,  rstride=4, cstride=4, color='blue', linewidth=0, alpha=0.3)

#### PLOT YELLOW
u = np.linspace(np.pi/2., 3.*np.pi/2., num=100)
v = np.linspace(0, np.pi, num=100)
theta = np.linspace(0, 2*np.pi, num=100)

x = R * np.outer(np.cos(u), np.sin(v))
y = R * np.outer(np.sin(u), np.sin(v))
z = R * np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_surface(x, y, z+0.0,  rstride=4, cstride=4, color='yellow', linewidth=0, alpha=0.3)
#####################################################################


ax1.plot([0.,5.*r_r[0]],[0.,5.*r_r[1]],[0.,5.*r_r[2]],color='black') #circle normal vector
ax1.plot([0.,20.*r_view[0]],[0.,20.*r_view[1]],[0.,20.*r_view[2]],color='black') #viewing direction vector

plt.show(block=False)


#### Ring Bond Albedo ############################################################################################################
"""Realistically, an observed reflected light intensity of a planet will vary in intensity across angle of emittance and spectrum
We assume a lambert reflectance model that is uniform in spectrum
In a Lambert reflectance model, the emittance of reflected light is uniform across all azimuth and elevation angles 
i.e. if 1 W/m^2 is incident on a 1 m^2 surface Area flat plate, 1 W/Sr will be emitted
"""
#Thoughts:
#The phase function is intended to simulate the change in total planet reflected light as a function of phase angle.
#planet dmag=-2.5*np.log10(p*(Rp/d).decompose()**2*Phi).value
#Here p is the geometric albedo, Rp is the planet radius, and Phi is the phase function value.
#Combined, the value in the log10 is the planet flux relative to a 1 flux star

#The total reflected light of the ring is complex because it is composed of a large body of non-uniform particles of varying
#composition which absorb some light, reflect some light, and allow some light to pass through
#Arnaldo says look at Beer-Lambert Law

#Our treatment of the rings' reflected light

a_bond_ring = 0.342 #from Hanel 1983 Albedo, Internal Heat Flux, and Energy Balance of Saturn
a_bond_body = 0.342
#### Maximal Total Energy Incident on Ring ####
#1. Calculate total surface Area of ring projected onto star-planet vector plane
A_ellipse_starproj = np.pi*R_rmin*b_Rrmin_kstar
#2. Calculate total illuminated area obstructed by planet in star-planet vector plane
th1, th2, th3, th4 = calc_ellipseCircleIntersections(R_rmin,b_Rrmin_kstar,R)
A_ringIlluminationObstructed = R**2.*(th2-th1)/2. - calc_AreaUnderEllipseSection(R_rmin,b_Rrmin_kstar,th1,th2)
#3. Calculate total incident Energy to ring
FluxSaturn = 1366.*(1.**2.)/(9.6**2.) #Min 9AU, Max 10.1AU, AVG 9.6AU
Qdot_ring = FluxSaturn*(A_ellipse_starproj-A_ringIlluminationObstructed)
#4. Calculate total area of ring obstructing body - used in future
#### Maximal Energy Reflected From Ring ####
#1. Calculate total surface Area of ring projected onto r_view vector plane
A_ellipse_viewproj = np.pi*R_rmin*b_Rrmin_view
#2. Calculate total viewed area obstructed by planet in r_view vector plane
#3. Calculate total reflected flux loss due to Lambert reflectance model
#dI
#################################################################################################################################




#### Total Saturn + Ring Flux Calculation ########################################################################################
totalFlux = maximal_ringFlux + maximal_saturnFlux
#### Vis Mag of Saturn + Ring
dMagSaturnSystem = -2.5*np.log10(totalFlux/1.) #Assuming a 1. magnitude star this allows us to ignore the solid angle
    #d**2. division and simply keep the telescope aperture
##################################################################################################################################
