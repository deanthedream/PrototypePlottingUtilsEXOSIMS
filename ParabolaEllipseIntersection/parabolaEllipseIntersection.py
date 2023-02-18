
import numpy as np

import matplotlib.pyplot as plt



#### Generate Initial System

#ellipse
a = 2
b = 1
#parabola


#General conic equation Unrotated
Ax^2 + Bxy + Cy^2 + D x+ Ey + F =0


(1) If B^2-4AC = 0

, the conic is a parabola.

(2) If B^2-4AC < 0

, the conic is an ellipse.

(3) If B^2-4AC > 0
, the conic is a hyperbola.


#General Conic Rotated
Ap = A*np.cos(theta)**2 + 0.5*B*np.sin(2*theta) + C*np.sin(theta)**2
Bp = -A*np.sin(2*theta) + C*np.sin(2*theta) + B*np.cos(2*theta)
Cp = A*np.sin(theta)**2 - *B*np.sin(2*theta) + C*np.sin(theta)**2
Dp = D*np.cos(theta)+E*np.sin(theta)
Ep = -D*np.sin(theta) + E*np.cos(theta)
Fp = F

Ap x^2 + Bp xy + Cp y^2 + Dp x+ Ep y + Fp =0


