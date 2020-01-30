

from sympy import * 
#Symbol, symbols
#from sympy import sin, cos, asin, acos, tan, atan2, exp
#from sympy import power
#from sympy import log
#from mpmath import *
import numpy as np

a, e, i, W, w, v = symbols('a e i W w v')
p = Symbol('p')
R = Symbol('R')
A, B = symbols('A B')

#r
eqnr = a*(1.-power.Pow(e,2.))/(1.+e*cos(v))
print('r')
print(eqnr)

#XYZ
eqnX = eqnr*(cos(W)*cos(w+v) - sin(W)*sin(w+v)*cos(i))
eqnY = eqnr*(sin(W)*cos(w+v) - cos(W)*sin(w+v)*cos(i))
eqnZ = eqnr*(sin(i)*sin(w+v))
print('X')
print(eqnX)
print('Y')
print(eqnY)
print('Z')
print(eqnZ)

#alpha
eqnAlpha = acos(eqnZ/sqrt(eqnX**2.+eqnY**2.+eqnZ**2.))
print('Alpha')
print(eqnAlpha)

#### PHASE FUNCTIONS ##############
#LAMBERT
eqnLAMBERT = (sin(eqnAlpha*np.pi/180.) + (np.pi-eqnAlpha)*cos(eqnAlpha*np.pi/180.))/np.pi

#START, STOP
eqnSTART = 0.5+0.5*tanh((eqnAlpha-A)/B)
eqnEND = 0.5-0.5*tanh((eqnAlpha-A)/B)

#MERCURY
phaseMERCURY = 10.**(-0.4*(6.3280e-02*eqnAlpha - 1.6336e-03*eqnAlpha**2. + 3.3644e-05*eqnAlpha**3. - 3.4265e-07*eqnAlpha**4. + 1.6893e-09*eqnAlpha**5. - 3.0334e-12*eqnAlpha**6.))
print(phaseMERCURY)

#VENUS
eqnPhase1Venus = 10.**(-0.4*(- 1.044e-03*eqnAlpha + 3.687e-04*eqnAlpha**2. - 2.814e-06*eqnAlpha**3. + 8.938e-09*eqnAlpha**4.))
tmpPhase = 10.**(-0.4*( - 2.81914e-00*eqnAlpha + 8.39034e-03*eqnAlpha**2.))
h1 = tmpPhase.subs(eqnAlpha,163.7)
h2 = 10.**(-0.4*( - 2.81914e-00*163.7 + 8.39034e-03*163.7**2.)) - 10.**(-0.4*( - 2.81914e-00*179. + 8.39034e-03*179.**2.))
tmpDifference = tmpPhase.subs(eqnAlpha, 163.7) - h1/h2*(10.**(-0.4*( - 2.81914e-00*163.7 + 8.39034e-03*163.7**2.)))
eqnPhase2Venus = tmpPhase + tmpDifference
phaseVENUS = eqnEND.subs(A,163.7).subs(B,5.)*eqnPhase1Venus + \
        eqnSTART.subs(A,163.7).subs(B,5.)*eqnEND.subs(A,179.).subs(B,0.5)*eqnPhase2Venus + \
        eqnSTART.subs(A,179.).subs(B,0.5)*eqnLAMBERT

#EARTH
phaseEARTH = 10.**(-0.4*(- 1.060e-3*eqnAlpha + 2.054e-4*eqnAlpha**2.))

#MARS
eqnPhase1Mars = 10.**(-0.4*(0.02267*eqnAlpha - 0.0001302*eqnAlpha**2.+ 0. + 0.)) #L(λe) + L(LS)
eqnPhase2Mars = eqnPhase1Mars.subs(eqnAlpha,50.)/10.**(-0.4*(- 0.02573*50. + 0.0003445*50.**2.)) * 10.**(-0.4*(- 0.02573*eqnAlpha + 0.0003445*eqnAlpha**2. + 0. + 0.)) #L(λe) + L(Ls)
phaseMARS = eqnEND.subs(A,50.).subs(B,5.)*eqnPhase1Mars + \
        transitionStart(alpha,50.,5.)*eqnPhase2Mars

#JUPITER

#SATURN

#URANUS

#NEPTUNE

