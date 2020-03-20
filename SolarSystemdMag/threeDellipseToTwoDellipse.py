#3D ellipse to 2D ellipse

import sympy as sp
import numpy as np

from sympy.physics.vector import *
N = ReferenceFrame('N')


#All angles in radians!!!!
a, e, inc, W, w, v = sp.symbols('a e inc W w v', real=True, positive=True)
p = sp.Symbol('p', real=True,positive=True)
R = sp.Symbol('R', real=True,positive=True)
A, B = sp.symbols('A B', real=True, positive=True)
Phi = sp.Symbol('Phi',real=True,positive=True)
alpha = sp.Symbol('alpha', real=True, positive=True)

#### The common point between two ellipses is the orbiting Foci
#In the projected ellipse, the Foci may not be the orbiting Foci

#r
eqnr = a*(1-e**2)/(1+e*sp.cos(v))
print('r')
print(eqnr)

#### XYZ of 3D ellipse
eqnX = eqnr*(sp.cos(W)*sp.cos((w+v)) - sp.sin(W)*sp.sin((w+v))*sp.cos(inc))
eqnY = eqnr*(sp.sin(W)*sp.cos((w+v)) + sp.cos(W)*sp.sin((w+v))*sp.cos(inc))
eqnZ = eqnr*(sp.sin(inc)*sp.sin((w+v)))
print('X')
print(eqnX)
print('Y')
print(eqnY)
print('Z')
print(eqnZ)

#### Generic Point p on the 3D ellipse
p_3DEllipse = sp.Matrix([eqnX,eqnY,eqnZ])

#### XY of 2D ellipse in terms of KOE is eqnX and eqnY

#### The 3D vector along the semi-MAJOR axis in the direction of perigee is when v=0
X_perigee_vect_3D_orbit = eqnX.subs(v,0.)
Y_perigee_vect_3D_orbit = eqnY.subs(v,0.)
Z_perigee_vect_3D_orbit = eqnZ.subs(v,0.)
majorAxisVect_3D_ellipse = sp.Matrix([X_perigee_vect_3D_orbit,Y_perigee_vect_3D_orbit,Z_perigee_vect_3D_orbit])

#### The 3D vector along the semi-MAJOR axis in the direction of apogee is when v=np.pi
#UNNECESSARY SINCE WE HAVE SMA and ECCEN
X_apogee_vect_3D_orbit = eqnX.subs(v,np.pi)
Y_apogee_vect_3D_orbit = eqnY.subs(v,np.pi)
p_3DEllipse_apogee = sp.Matrix([X_apogee_vect_3D_orbit,Y_apogee_vect_3D_orbit,0])

#Unit vector from 3D Ellipse Foci to 3D Ellipse Origin

U_3DFoci_to_origin = -p_3DEllipse_apogee/p_3DEllipse_apogee.norm()
#Distance from 3D Ellipse Foci to 3D Ellipse Origin: c**2 = a**2-b**2
B_3D = a*sp.sqrt(1-e**2)
C_3DFoci_to_origin = sp.sqrt(a**2-B_3D**2)

#### The 3D unit vector describing orbital plane
#NEED TO FIGURE OUT HOW TO DO SYMPY CROSS PRODUCT BS OR JUST USE NUMPY??
normalVect_3D_ellipse = majorAxisVect_3D_ellipse.cross(p_3DEllipse.subs(v,np.pi/2.)) #np.pi/2. was used, but any value of v is acceptable
# normalVect_3D_ellipse = sp.cross([X_perigee_vect_3D_orbit,Y_perigee_vect_3D_orbit,Z_perigee_vect_3D_orbit],\
#             [eqnX.subs(v,np.pi/2.),eqnY.subs(v,np.pi/2.),eqnZ.subs(v,np.pi/2.)]) #np.pi/2. was used, but any value of v is acceptable
normalVect_3D_ellipse = normalVect_3D_ellipse/normalVect_3D_ellipse.norm()

#### The 3D unit vector along the semi-MINOR axis is given by
minorAxisUnitVect_3D_ellipse = majorAxisVect_3D_ellipse.cross(normalVect_3D_ellipse)

#### The 3D vector along the semi-MINOR axis is therefore
minorAxisVect_3D_ellipse = B_3D*minorAxisUnitVect_3D_ellipse

#DELETEAssumption the center of a 3D ellipse is the same as the center of it's projection
#Assume a 3D Ellipse center projected onto a 2D plane has the same center as the projected ellipse
#### From 2D Projected Ellipse Center to Projected 3D Ellipse Foci
#A (major or minor) axis must lie along the Ellipse center-foci line
C_2DFoci_to_origin = sp.sqrt(U_3DFoci_to_origin[0]**2+U_3DFoci_to_origin[0]**2)
projected_perigee_distance = sp.sqrt(majorAxisVect_3D_ellipse[0]**2+majorAxisVect_3D_ellipse[1]**2)
axis1_projected = C_2DFoci_to_origin+projected_perigee_distance #this is either the semi-major axis or semi-minor axis
axis1Unit_projected = sp.Matrix([X_perigee_vect_3D_orbit, Y_perigee_vect_3D_orbit, 0])#DELETE/np.linalg.norm([X_perigee_vect_3D_orbit, Y_perigee_vect_3D_orbit, 0])
axis1Unit_projected = axis1Unit_projected/axis1Unit_projected.norm()

#Axis 2 unit vector
axis2Unit_projected = axis1Unit_projected.cross(sp.Matrix([0,0,1]))

#Ellipse origin location
projectedEllipseCenter = -axis1Unit_projected*C_2DFoci_to_origin

#Axis 2 length
#Solve the following for axis2_projected
axis2_projected = 
[eqnX,eqnY,0] == projectedEllipseCenter + axis2_projected*axis2Unit_projected
#Aha! Solve for axis2_projected for only one variable! (Easy?)

#solve for length of axis 2 by 
#LEAVING OFF HERE! NEED TO FIND OUT HOW TO CALCULATE THE SEMI-MINOR AXIS LENGTH

#Distance from orbit Foci to perigee sqrt(xp**2.+yp**2.) 

#### Projected Semi-major axis

#### Projected Semi-minor axis

#### Eccentricty of Projected Ellipse

#### Canonical Coordinates


