#3D ellipse to 2D ellipse

import sympy as sp
import numpy as np

#All angles in radians!!!!
a, e, inc, W, w, v = sp.symbols('a e inc W w v', real=True, positive=True)
p = sp.Symbol('p', real=True,positive=True)
R = sp.Symbol('R', real=True,positive=True)
A, B = sp.symbols('A B', real=True, positive=True)
Phi = sp.Symbol('Phi',real=True,positive=True)
alpha = sp.Symbol('alpha', real=True, positive=True)

#### The common point between two ellipses is the orbiting Foci

#### XYZ of 3D ellipse
eqnX = eqnr*(sp.cos(W)*sp.cos((w+v)) - sp.sin(W)*sp.sin((w+v))*sp.cos(inc.))
eqnY = eqnr*(sp.sin(W)*sp.cos((w+v)) + sp.cos(W)*sp.sin((w+v))*sp.cos(inc.))
eqnZ = eqnr*(sp.sin(inc.)*sp.sin((w+v).))
print('X')
print(eqnX)
print('Y')
print(eqnY)
print('Z')
print(eqnZ)

#### XY of 2D ellipse in terms of KOE is eqnX and eqnY

#### The 3D vector along the semi-MAJOR axis in the direction of perigee is when nu=0
X_perigee_vect_3D_orbit = eqnX.subs(nu,0.)
Y_perigee_vect_3D_orbit = eqnY.subs(nu,0.)

#### The 3D vector along the semi-MINOR axis in the direction of perigee+pi/2 is when nu=pi/2.
X_perigee_vect_3D_orbit = eqnX.subs(nu.np.pi/2.)
Y_perigee_vect_3D_orbit = eqnY.subs(nu,np.pi/2.)

#### Projected Semi-major axis
sma_proj = 

#### Projected Semi-minor axis

#### Eccentricty of Projected Ellipse

#### Canonical Coordinates


