#### Ringed Planet Model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import matplotlib.gridspec as gridspec
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
    """ Calculates Critical Angle between ring plane and planet
    where a planet/ring will begin obstructing one another
    if theta >= theta_crit1, then ring is unobstructed
    if theta < theta_crit1 and theta >= theta_crit2, then inner ring circle obstructed and outer is not
    if theta < theta_crit2, then the outer and inner ring cirlces are obstructed.
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
        th2 (float) - angle #2 defined from a direction in rad larger angle
    Returns:
        area (float) - area between two angles of ellipse
    """
    F2 = a*b/2.*(th2-np.arctan2((b-a)*np.sin(2.*th2),b+a+(b-a)*np.cos(2.*th2)))
    F1 = a*b/2.*(th1-np.arctan2((b-a)*np.sin(2.*th1),b+a+(b-a)*np.cos(2.*th1)))
    area = F2 - F1
    return area

def planeLineIntersect(r_plane, p0, r_line, D=0.):
    """
    Args:
        r_plane (numpy array) - normal vector describing intersection plane (assume D=0)
        p0 (numpy array) - Line starting point
        r_line (numpy array) - vector describing the line
        D (float) - RHS of plane equation
    Returns:
        pt (numpy array) - Intersection point between line and plane
        t (float) - the magnitude of t to make that intersection
    """
    A = np.asarray([[r_plane[0],r_plane[1],r_plane[2],0.],[1.,0.,0.,-r_line[0]],[0.,1.,0,-r_line[1]],[0.,0.,1.,-r_line[2]]])
    b = np.asarray([[D],[p0[0]],[p0[1]],[p0[2]]])
    out = np.linalg.solve(A,b)
    pt = out.T[0][:3]
    t = out.T[0][-1]
    return pt, t

def check_angles(beta,epsilon,omega):
    """Checks angles between r_r, r_view, and r_kstar to ensure they are feasible
    Returns:
        ok (boolean) - true if ok, false if not ok
    """
    if beta > np.pi or epsilon > np.pi/2. or omega > np.pi/2.:
        #beta cannot exceed 180 deg, epsilon and omega cannot exceed 90 deg
        return False
    
    return ok

#### Inputs ##################################################################################
beta = 90.*np.pi/2. #minimum angle between r_view and r_kstar
epsilon = 45.*np.pi/2. #minimum angle between r_r and r_kstar
omega = 45.*np.pi/2. #angle between r_r and r_view
ok = check_angles(beta,epsilon,omega)

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
b_Rrmin_view = np.cos(phi_ellipse_view)*R_rmin #minor axis of apparent ellipse of Rrmin ring
b_Rrmax_view = np.cos(phi_ellipse_view)*R_rmax #minor axis of apparent ellipse of Rrmax ring
#### Ring Ellipticity - Star Direction #######################################################
r_r3, phi_ellipse_kstar = calc_ellipticityAngle(r_r,r_kstar)
b_Rrmin_kstar = np.cos(phi_ellipse_kstar)*R_rmin#minor axis of apparent ellipse of Rrmin ring
b_Rrmax_kstar = np.cos(phi_ellipse_kstar)*R_rmax#minor axis of apparent ellipse of Rrmax ring
##############################################################################################

#### Direction of semi-minor and semi-major of ring in actual ring plane
r_ellipsea1_view = R_rmin*np.cross(r_r2,r_view)/np.linalg.norm(np.cross(r_r2,r_view))
r_ellipseb1_view = R_rmin*np.cross(r_ellipsea1_view,r_r2)/np.linalg.norm(r_ellipsea1_view)
##### Component Vector of Ellipse perpendicular to r_view in apparent ellipse direction of b
r_ellipseb1_viewproj = (r_ellipseb1_view-np.dot(r_view,r_ellipseb1_view)*r_view)
r_ellipsea1_viewproj = (r_ellipsea1_view-np.dot(r_view,r_ellipsea1_view)*r_view)

#### Planet Obstructing Ring Points ####################################################################
#In the viewing plane, the following angles represent the intersections in the viewing plane.
#Orientation should not matter
th1, th2, th3, th4 = calc_ellipseCircleIntersections(R_rmin,np.linalg.norm(r_ellipseb1_viewproj),R)
#Calculate points projected on the view plane
intpt_viewproj1 = np.cos(th1)*r_ellipsea1_viewproj/np.linalg.norm(r_ellipsea1_viewproj)*R + np.sin(th1)*r_ellipseb1_viewproj/np.linalg.norm(r_ellipseb1_viewproj)*R
intpt_viewproj2 = np.cos(th2)*r_ellipsea1_viewproj/np.linalg.norm(r_ellipsea1_viewproj)*R + np.sin(th2)*r_ellipseb1_viewproj/np.linalg.norm(r_ellipseb1_viewproj)*R
intpt_viewproj3 = np.cos(th3)*r_ellipsea1_viewproj/np.linalg.norm(r_ellipsea1_viewproj)*R + np.sin(th3)*r_ellipseb1_viewproj/np.linalg.norm(r_ellipseb1_viewproj)*R
intpt_viewproj4 = np.cos(th4)*r_ellipsea1_viewproj/np.linalg.norm(r_ellipsea1_viewproj)*R + np.sin(th4)*r_ellipseb1_viewproj/np.linalg.norm(r_ellipseb1_viewproj)*R
#Calculate points on ellipse projected on the view plane
########################################################################################################

#### Planet Obstructing Ring Points In 3D ####################
pt_rmincirc_viewproj1, t = planeLineIntersect(r_r, intpt_viewproj1, r_view)
pt_rmincirc_viewproj2, t = planeLineIntersect(r_r, intpt_viewproj2, r_view)
pt_rmincirc_viewproj3, t = planeLineIntersect(r_r, intpt_viewproj3, r_view)
pt_rmincirc_viewproj4, t = planeLineIntersect(r_r, intpt_viewproj4, r_view)
##############################################################


#### Direction of semi-minor and semi-major of ring in actual ring plane
r_ellipsea1_kstar = R_rmin*np.cross(r_r2,r_kstar)/np.linalg.norm(np.cross(r_r2,r_kstar))
r_ellipseb1_kstar = R_rmin*np.cross(r_ellipsea1_kstar,r_r2)/np.linalg.norm(r_ellipsea1_kstar)
##### Component Vector of Ellipse perpendicular to r_view in apparent ellipse direction of b
r_ellipseb1_kstarproj = (r_ellipseb1_kstar-np.dot(r_kstar,r_ellipseb1_kstar)*r_kstar)
r_ellipsea1_kstarproj = (r_ellipsea1_kstar-np.dot(r_kstar,r_ellipsea1_kstar)*r_kstar)

#### Planet Obstructing Ring Points ####################################################################
#In the viewing plane, the following angles represent the intersections in the viewing plane.
#Orientation should not matter
th1, th2, th3, th4 = calc_ellipseCircleIntersections(R_rmin,np.linalg.norm(r_ellipseb1_kstarproj),R)
#Calculate points projected on the view plane
intpt_kstarproj1 = np.cos(th1)*r_ellipsea1_kstarproj/np.linalg.norm(r_ellipsea1_kstarproj)*R + np.sin(th1)*r_ellipseb1_kstarproj/np.linalg.norm(r_ellipseb1_kstarproj)*R
intpt_kstarproj2 = np.cos(th2)*r_ellipsea1_kstarproj/np.linalg.norm(r_ellipsea1_kstarproj)*R + np.sin(th2)*r_ellipseb1_kstarproj/np.linalg.norm(r_ellipseb1_kstarproj)*R
intpt_kstarproj3 = np.cos(th3)*r_ellipsea1_kstarproj/np.linalg.norm(r_ellipsea1_kstarproj)*R + np.sin(th3)*r_ellipseb1_kstarproj/np.linalg.norm(r_ellipseb1_kstarproj)*R
intpt_kstarproj4 = np.cos(th4)*r_ellipsea1_kstarproj/np.linalg.norm(r_ellipsea1_kstarproj)*R + np.sin(th4)*r_ellipseb1_kstarproj/np.linalg.norm(r_ellipseb1_kstarproj)*R
#Calculate points on ellipse projected on the view plane
########################################################################################################

#### Planet Obstructing Ring Points In 3D ####################
pt_rmincirc_kstarproj1, t = planeLineIntersect(r_r, intpt_kstarproj1, r_kstar)
pt_rmincirc_kstarproj2, t = planeLineIntersect(r_r, intpt_kstarproj2, r_kstar)
pt_rmincirc_kstarproj3, t = planeLineIntersect(r_r, intpt_kstarproj3, r_kstar)
pt_rmincirc_kstarproj4, t = planeLineIntersect(r_r, intpt_kstarproj4, r_kstar)
##############################################################



#### Minimum phi_ellipse_view Angle where the planet obstructs ring visibility ##############################
theta_crit1, theta_crit2 = calc_CriticalThetas(R, R_rmin, R_rmax)
#############################################################################################################

#### Illumination Conditions ################################################################################
if phi_ellipse_kstar >= theta_crit1:
    #then ring is UNobstructed
    rkstar_state = 0
    #No cylinder/ring intersections
    #1. Calc total surface Area of ring projected onto star-planet vector plane
    A_ellipse_starproj = np.pi*R_rmax*b_Rrmax_kstar - np.pi*R_rmin*b_Rrmin_kstar
    #2. Calc total illuminated area obstructed by planet in star-planet vector plane
    #SKIP
    #3. Calc total incident Energy to ring
    FluxSaturn = 1366.*(1.**2.)/(9.6**2.) #Min 9AU, Max 10.1AU, AVG 9.6AU
    Qdot_ring = FluxSaturn*A_ellipse_starproj
    #4. Calc SA of planet
    A_planet = np.pi*R**2.
    #5. Calc total incident Energy on planet
    Qdot_planet = FluxSaturn*A_planet
elif phi_ellipse_kstar < theta_crit1 and phi_ellipse_kstar >= theta_crit2:
    #then inner ring circle obstructed and outer is not
    rkstar_state = 1
    #Inner ring/cylinder intersection
    #1. Calc total surface Area of ring projected onto star-planet vector plane
    A_ellipse_starproj = np.pi*R_rmax*b_Rrmax_kstar - np.pi*R_rmin*b_Rrmin_kstar
    #2. Calc total illuminated area obstructed by planet in star-planet vector plane
    th1, th2, th3, th4 = calc_ellipseCircleIntersections(R_rmin,b_Rrmin_kstar,R)
    A_ringIlluminationObstructed = R**2.*(th2-th1)/2. - calc_AreaUnderEllipseSection(R_rmin,b_Rrmin_kstar,th1,th2)
    #3. Calc total incident Energy to ring
    FluxSaturn = 1366.*(1.**2.)/(9.6**2.) #Min 9AU, Max 10.1AU, AVG 9.6AU
    Qdot_ring = FluxSaturn*(A_ellipse_starproj-A_ringIlluminationObstructed)
    #4. Calc SA of planet
    A_planet = np.pi*R**2. - A_ringIlluminationObstructed
    #5. Calc total incident Energy on planet
    Qdot_planet = FluxSaturn*A_planet
elif phi_ellipse_kstar < theta_crit2:
    #then the outer and inner ring cirlces are obstructed.
    rkstar_state = 2
    #Inner and Outer ring/cylinder intersections
    #1. Calc total surface Area of ring projected onto star-planet vector plane
    A_ellipse_starproj = np.pi*R_rmax*b_Rrmax_kstar - np.pi*R_rmin*b_Rrmin_kstar
    #2. Calc total illuminated area obstructed by planet in star-planet vector plane
    th1_min_starproj, th2_min_starproj, th3_min_starproj, th4_min_starproj = calc_ellipseCircleIntersections(R_rmin,b_Rrmin_kstar,R) #Thetas of inner intersections
    th1_max_starproj, th2_max_starproj, th3_max_starproj, th4_max_starproj = calc_ellipseCircleIntersections(R_rmax,b_Rrmax_kstar,R) #Thetas of outer intersections
    area_1 = calc_AreaUnderEllipseSection(R_rmin,b_Rrmin_kstar,th2_max_starproj,th2_min_starproj)
    area_1p2 = calc_AreaUnderEllipseSection(R_rmax,b_Rrmax_kstar,th2_max_starproj,th2_min_starproj)
    area_2 = area_1p2 - area_1
    area_3p7 = calc_AreaUnderEllipseSection(R_rmax,b_Rrmax_kstar,th1_max_starproj,th2_max_starproj)
    area_7 = calc_AreaUnderEllipseSection(R_rmin,b_Rrmin_kstar,th2_min_starproj,th2_max_starproj)
    area_3 = area_3p7 - area_7
    area_6 = calc_AreaUnderEllipseSection(R_rmin,b_Rrmin_kstar,th1_min_starproj,th1_max_starproj)
    area_5p6 = calc_AreaUnderEllipseSection(R_rmax,b_Rrmax_kstar,th1_min_starproj,th1_max_starproj)
    area_5 = area_5p6 - area_6
    A_ringIlluminationObstructed = R**2.*(th2-th1)/2. - (area_2+area_3+area_5)
    #3. Calc total incident Energy to ring
    Qdot_ring = FluxSaturn*(A_ellipse_starproj-A_ringIlluminationObstructed)
    #4. Calc SA of planet
    A_planet = np.pi*R**2. - A_ringIlluminationObstructed
    #5. Calc total incident Energy on planet
    Qdot_planet = FluxSaturn*A_planet
else:
    print(error2)
#### Viewing Conditions #######################################
if phi_ellipse_view >= theta_crit1:
    #then ring is UNobstructed
    view_state = 0
    #No cylinder/ring intersections
    #1. Calc total surface Area of ring projected onto star-planet vector plane
    A_ellipse_viewproj = np.pi*R_rmax*b_Rrmax_view - np.pi*R_rmin*b_Rrmin_view
    #2. Calculate total viewed area obstructed by planet in r_view vector plane
    #3. Calculate total reflected flux loss due to Lambert reflectance model
elif phi_ellipse_view < theta_crit1 and phi_ellipse_view >= theta_crit2:
    #then inner ring circle obstructed and outer is not
    view_state = 1
    #Inner ring/cylinder intersection
elif phi_ellipse_view < theta_crit2:
    #then the outer and inner ring cirlces are obstructed.
    view_state = 2
    #Inner and Outer ring/cylinder intersections
else:
    print(error1)
#############################################################################################################

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

#MAKE USE CASES FOR ALL THESE THINGS DEPENDING UPON THE PHI ELLIPSE AND CRITICAL ELLIPSE ANGLES


#4. Calculate total area of ring obstructing the body - used in future
#### Maximal Energy Reflected From Ring ####
#1. Calculate total surface Area of ring projected onto r_view vector plane
A_ellipse_viewproj = np.pi*R_rmin*b_Rrmin_view

#dI
#################################################################################################################################


#### Total Saturn + Ring Flux Calculation ########################################################################################
#totalFlux = maximal_ringFlux + maximal_saturnFlux
#### Vis Mag of Saturn + Ring
#dMagSaturnSystem = -2.5*np.log10(totalFlux/1.) #Assuming a 1. magnitude star this allows us to ignore the solid angle
    #d**2. division and simply keep the telescope aperture
##################################################################################################################################


#### Generate 3D Cylinder and Circle Geometries For Plotting #####################################################
#Genetate 3D Cylinder
d_z = np.linspace(start=-2.5*R, stop=2.5*R,num=30,endpoint=True) #various distances along r_kstar_hat direction to define distance
r_ksunbot1, r_ksunbot2 = generate_twoVectPerpToVect(r_kstar) #Two perp vect perp to R_kstar
phi1 = np.linspace(start=0.,stop=2*np.pi,num=30) #various angles about r_kstar to define the circle
p0 = np.asarray([0.,0.,0.]) #starting point
d_z_mesh, phi1_mesh = np.meshgrid(d_z,phi1)
X_cyl, Y_cyl, Z_cyl = [p0[i] + r_kstar[i] * d_z_mesh + R * np.sin(phi1_mesh) * r_ksunbot1[i] + R * np.cos(phi1_mesh) * r_ksunbot2[i] for i in [0, 1, 2]]

#### Generate 3D R_rmin circle
r_Rbot1, r_Rbot2 = generate_twoVectPerpToVect(r_r) #Two perp vect in ring plane perp to r_r
phi2 = np.linspace(start=0.,stop=2*np.pi,num=30) #various angles about r_r
#for plotting
S_circs = list()
for phi in phi2:
    S_circs.append(ring_circ(R_rmin,r_Rbot1, r_Rbot2,phi))
S_circs = np.asarray(S_circs)

#### Generate 3D R_rmax circle
S_circs_max = list()
for phi in phi2:
    S_circs_max.append(ring_circ(R_rmax,r_Rbot1, r_Rbot2,phi))
S_circs_max = np.asarray(S_circs_max)

#### Generate Circle of sphere Projected onto Viewing Plane
r_Rbot1_viewproj, r_Rbot2_viewproj = generate_twoVectPerpToVect(-r_view) #Two perp vect in ring plane perp to r_r
S_circs_viewproj = list()
for phi in phi2:
    S_circs_viewproj.append(ring_circ(R,r_Rbot1_viewproj, r_Rbot2_viewproj,phi))
S_circs_viewproj = np.asarray(S_circs_viewproj)

#### Generate Projected Ellipse in viewing direction
S_circ_viewproj = list()
for phi in phi2:
    S_circ_viewproj.append(np.cos(phi)*r_ellipseb1_viewproj + np.sin(phi)*r_ellipsea1_viewproj)
S_circ_viewproj = np.asarray(S_circ_viewproj)

#### Generate Circle of sphere Projected onto Illumination Plane
r_Rbot1_kstarproj, r_Rbot2_kstarproj = generate_twoVectPerpToVect(-r_kstar) #Two perp vect in ring plane perp to r_r
S_circs_kstarproj = list()
for phi in phi2:
    S_circs_kstarproj.append(ring_circ(R,r_Rbot1_kstarproj, r_Rbot2_kstarproj,phi))
S_circs_kstarproj = np.asarray(S_circs_kstarproj)

#### Generate Projected Ellipse in Illumination direction
S_circ_kstarproj = list()
for phi in phi2:
    S_circ_kstarproj.append(np.cos(phi)*r_ellipseb1_kstarproj + np.sin(phi)*r_ellipsea1_kstarproj)
S_circ_kstarproj = np.asarray(S_circ_kstarproj)
###############################################################################################################


#Bounding box edges
maxBounds = np.max([np.max(np.abs(X_cyl)),np.max(np.abs(Y_cyl)),np.max(np.abs(Z_cyl)),np.max(np.abs(S_circs))])

plt.close(1)
fig = plt.figure(num=1, figsize=(10,10))
numRows = 2
numCols = 2
height_ratios = ([1,1])
width_ratios = ([1,1])
gs = gridspec.GridSpec(numRows,numCols, width_ratios=width_ratios, height_ratios=height_ratios)
gs.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')

ax0 = plt.subplot(gs[0], projection='3d')
ax1 = plt.subplot(gs[1], projection='3d')
ax2 = plt.subplot(gs[2], projection='3d')
ax3 = plt.subplot(gs[3], projection='3d')

#### ax3 #########################################################################################
ax3.plot(S_circs[:,0],S_circs[:,1],S_circs[:,2],color='red') #plot R_rmin circle
ax3.plot(S_circs_max[:,0],S_circs_max[:,1],S_circs_max[:,2],color='red') #plot R_rmax circle
ax3.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.2, color='blue')#, rstride=rstride, cstride=cstride)
ax3.plot([-20.*r_kstar[0],0.],[-20.*r_kstar[1],0.],[-20.*r_kstar[2],0.],color='cyan') #plot sun to star vector
ax3.scatter([maxBounds,maxBounds,maxBounds,maxBounds,-maxBounds,-maxBounds,-maxBounds,-maxBounds],\
        [maxBounds,maxBounds,-maxBounds,-maxBounds,maxBounds,maxBounds,-maxBounds,-maxBounds],\
        [maxBounds,-maxBounds,maxBounds,-maxBounds,maxBounds,-maxBounds,maxBounds,-maxBounds],alpha=0.)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.view_init(elev=180./np.pi*np.arcsin(r_view[2]), azim=180./np.pi*np.arcsin(r_view[1]/np.sqrt(r_view[0]**2. + r_view[1]**2.)))#set viewing angle
#### Plot view proj ellipse and intersection points
ax3.plot([0.,r_ellipseb1_view[0]],[0.,r_ellipseb1_view[1]],[0.,r_ellipseb1_view[2]],color='cyan') #b1 in ring plane
ax3.plot([0.,r_ellipseb1_viewproj[0]],[-20.,-20.],[0.,r_ellipseb1_viewproj[2]],color='orange') #b1 in viewing plane
ax3.plot([0.,r_ellipsea1_viewproj[0]],[-20.,-20.],[0.,r_ellipsea1_viewproj[2]],color='orange') #a1 in viewing plane
ax3.scatter([intpt_viewproj1[0],intpt_viewproj2[0],intpt_viewproj3[0],intpt_viewproj4[0]],\
        [-20.,-20.,-20.,-20.],\
        [intpt_viewproj1[2],intpt_viewproj2[2],intpt_viewproj3[2],intpt_viewproj4[2]],color='orange') #a1 in viewing plane
ax0.scatter([pt_rmincirc_viewproj1[0],pt_rmincirc_viewproj2[0],pt_rmincirc_viewproj3[0],pt_rmincirc_viewproj4[0]],\
        [pt_rmincirc_viewproj1[1],pt_rmincirc_viewproj2[1],pt_rmincirc_viewproj3[1],pt_rmincirc_viewproj4[1]],\
        [pt_rmincirc_viewproj1[2],pt_rmincirc_viewproj2[2],pt_rmincirc_viewproj3[2],pt_rmincirc_viewproj4[2]],color='orange',marker='x') #a1 in viewing plane
ax3.plot(S_circs_viewproj[:,0],S_circs_viewproj[:,1]-20.,S_circs_viewproj[:,2],color='blue')
ax3.plot(S_circ_viewproj[:,0],-20.+S_circ_viewproj[:,1],S_circ_viewproj[:,2],color='orange')
#### PLOT BLUE HEMISPHERE For Reference 
u = np.linspace(np.pi/2., -np.pi/2., num=100)
v = np.linspace(0, np.pi, num=100)
theta = np.linspace(0, 2*np.pi, num=100)
x = R * np.outer(np.cos(u), np.sin(v))
y = R * np.outer(np.sin(u), np.sin(v))
z = R * np.outer(np.ones(np.size(u)), np.cos(v))
ax3.plot_surface(x, y, z,  rstride=4, cstride=4, color='blue', linewidth=0, alpha=0.3)
#### PLOT YELLOW HEMISPHERE For Reference
u = np.linspace(np.pi/2., 3.*np.pi/2., num=100)
v = np.linspace(0, np.pi, num=100)
theta = np.linspace(0, 2*np.pi, num=100)
x = R * np.outer(np.cos(u), np.sin(v))
y = R * np.outer(np.sin(u), np.sin(v))
z = R * np.outer(np.ones(np.size(u)), np.cos(v))
ax3.plot_surface(x, y, z+0.0,  rstride=4, cstride=4, color='yellow', linewidth=0, alpha=0.3)
####
ax3.plot([0.,5.*r_r[0]],[0.,5.*r_r[1]],[0.,5.*r_r[2]],color='black') #circle normal vector
ax3.plot([0.,20.*r_view[0]],[0.,20.*r_view[1]],[0.,20.*r_view[2]],color='black') #viewing direction vector
ax3.set_title('Seen From Viewing Direction')
####################################################################################################

#### ax0 #########################################################################################
ax0.plot(S_circs[:,0],S_circs[:,1],S_circs[:,2],color='red') #plot R_rmin circle
ax0.plot(S_circs_max[:,0],S_circs_max[:,1],S_circs_max[:,2],color='red') #plot R_rmax circle
ax0.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.2, color='blue')#, rstride=rstride, cstride=cstride)
ax0.plot([-20.*r_kstar[0],0.],[-20.*r_kstar[1],0.],[-20.*r_kstar[2],0.],color='cyan') #plot sun to star vector
ax0.scatter([maxBounds,maxBounds,maxBounds,maxBounds,-maxBounds,-maxBounds,-maxBounds,-maxBounds],\
        [maxBounds,maxBounds,-maxBounds,-maxBounds,maxBounds,maxBounds,-maxBounds,-maxBounds],\
        [maxBounds,-maxBounds,maxBounds,-maxBounds,maxBounds,-maxBounds,maxBounds,-maxBounds],alpha=0.)
ax0.set_xlabel('X')
ax0.set_ylabel('Y')
ax0.set_zlabel('Z')
ax0.view_init(elev=180./np.pi*np.arcsin(r_kstar[2]), azim=180.+180./np.pi*np.arcsin(-r_kstar[1]/np.sqrt(r_kstar[0]**2. + r_kstar[1]**2.)))#set viewing angle
#### Plot view proj ellipse and intersection points
ax0.plot([0.,r_ellipseb1_view[0]],[0.,r_ellipseb1_view[1]],[0.,r_ellipseb1_view[2]],color='cyan') #b1 in ring plane
ax0.plot([0.,r_ellipseb1_viewproj[0]],[-20.,-20.],[0.,r_ellipseb1_viewproj[2]],color='orange') #b1 in viewing plane
ax0.plot([0.,r_ellipsea1_viewproj[0]],[-20.,-20.],[0.,r_ellipsea1_viewproj[2]],color='orange') #a1 in viewing plane
ax0.scatter([intpt_viewproj1[0],intpt_viewproj2[0],intpt_viewproj3[0],intpt_viewproj4[0]],\
        [-20.,-20.,-20.,-20.],\
        [intpt_viewproj1[2],intpt_viewproj2[2],intpt_viewproj3[2],intpt_viewproj4[2]],color='orange') #a1 in viewing plane
ax0.scatter([pt_rmincirc_viewproj1[0],pt_rmincirc_viewproj2[0],pt_rmincirc_viewproj3[0],pt_rmincirc_viewproj4[0]],\
        [pt_rmincirc_viewproj1[1],pt_rmincirc_viewproj2[1],pt_rmincirc_viewproj3[1],pt_rmincirc_viewproj4[1]],\
        [pt_rmincirc_viewproj1[2],pt_rmincirc_viewproj2[2],pt_rmincirc_viewproj3[2],pt_rmincirc_viewproj4[2]],color='orange',marker='x') #a1 in viewing plane
ax0.plot(S_circs_viewproj[:,0],S_circs_viewproj[:,1]-20.,S_circs_viewproj[:,2],color='blue')
ax0.plot(S_circ_viewproj[:,0],-20.+S_circ_viewproj[:,1],S_circ_viewproj[:,2],color='orange')
#### Plot kstar proj ellipse and intersection points
ax0.plot([0.,r_ellipseb1_kstar[0]],[0.,r_ellipseb1_kstar[1]],[0.,r_ellipseb1_kstar[2]],color='cyan') #b1 in ring plane
ax0.plot([-20.,-20.],[0.,r_ellipseb1_kstarproj[1]],[0.,r_ellipseb1_kstarproj[2]],color='cyan') #b1 in viewing plane
ax0.plot([-20.,-20.],[0.,r_ellipsea1_kstarproj[1]],[0.,r_ellipsea1_kstarproj[2]],color='cyan') #a1 in viewing plane
ax0.scatter([-20.,-20.,-20.,-20.],\
        [intpt_kstarproj1[1],intpt_kstarproj2[1],intpt_kstarproj3[1],intpt_kstarproj4[1]],\
        [intpt_kstarproj1[2],intpt_kstarproj2[2],intpt_kstarproj3[2],intpt_kstarproj4[2]],color='cyan') #a1 in viewing plane
ax0.scatter([pt_rmincirc_kstarproj1[0],pt_rmincirc_kstarproj2[0],pt_rmincirc_kstarproj3[0],pt_rmincirc_kstarproj4[0]],\
        [pt_rmincirc_kstarproj1[1],pt_rmincirc_kstarproj2[1],pt_rmincirc_kstarproj3[1],pt_rmincirc_kstarproj4[1]],\
        [pt_rmincirc_kstarproj1[2],pt_rmincirc_kstarproj2[2],pt_rmincirc_kstarproj3[2],pt_rmincirc_kstarproj4[2]],color='cyan',marker='x') #a1 in viewing plane
ax0.plot(-20.+S_circs_kstarproj[:,0],S_circs_kstarproj[:,1],S_circs_kstarproj[:,2],color='blue')
ax0.plot(-20.+S_circ_kstarproj[:,0],S_circ_kstarproj[:,1],S_circ_kstarproj[:,2],color='cyan')
#### PLOT BLUE HEMISPHERE For Reference 
u = np.linspace(np.pi/2., -np.pi/2., num=100)
v = np.linspace(0, np.pi, num=100)
theta = np.linspace(0, 2*np.pi, num=100)
x = R * np.outer(np.cos(u), np.sin(v))
y = R * np.outer(np.sin(u), np.sin(v))
z = R * np.outer(np.ones(np.size(u)), np.cos(v))
ax0.plot_surface(x, y, z,  rstride=4, cstride=4, color='blue', linewidth=0, alpha=0.3)
#### PLOT YELLOW HEMISPHERE For Reference
u = np.linspace(np.pi/2., 3.*np.pi/2., num=100)
v = np.linspace(0, np.pi, num=100)
theta = np.linspace(0, 2*np.pi, num=100)
x = R * np.outer(np.cos(u), np.sin(v))
y = R * np.outer(np.sin(u), np.sin(v))
z = R * np.outer(np.ones(np.size(u)), np.cos(v))
ax0.plot_surface(x, y, z+0.0,  rstride=4, cstride=4, color='yellow', linewidth=0, alpha=0.3)
####
ax0.plot([0.,5.*r_r[0]],[0.,5.*r_r[1]],[0.,5.*r_r[2]],color='black') #circle normal vector
ax0.plot([0.,20.*r_view[0]],[0.,20.*r_view[1]],[0.,20.*r_view[2]],color='black') #viewing direction vector
ax0.set_title('Seen From Illumination Direction')
####################################################################################################
plt.show(block=False)


