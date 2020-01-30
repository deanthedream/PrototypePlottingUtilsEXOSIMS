

#from sympy import * 
import sympy as sp
#Symbol, symbols
#from sympy import sp.sin, sp.cos, asp.sin, asp.cos, tan, atan2, exp
#from sympy import power
#from sympy import log
#from mpmath import *
#DELTEfrom mpmath import log10
import numpy as np

a, e, i, W, w, v = sp.symbols('a e i W w v', real=True)
p = sp.Symbol('p', real=True)
R = sp.Symbol('R', real=True)
A, B = sp.symbols('A B', real=True)

#r
eqnr = a*(1.-sp.power.Pow(e,2.))/(1.+e*sp.cos(v))
print('r')
print(eqnr)

#XYZ
eqnX = eqnr*(sp.cos(W)*sp.cos(w+v) - sp.sin(W)*sp.sin(w+v)*sp.cos(i))
eqnY = eqnr*(sp.sin(W)*sp.cos(w+v) - sp.cos(W)*sp.sin(w+v)*sp.cos(i))
eqnZ = eqnr*(sp.sin(i)*sp.sin(w+v))
print('X')
print(eqnX)
print('Y')
print(eqnY)
print('Z')
print(eqnZ)

#alpha
eqnAlpha = sp.acos(eqnZ/sp.sqrt(eqnX**2.+eqnY**2.+eqnZ**2.))
print('Alpha')
print(eqnAlpha)

#### PHASE FUNCTIONS ##############
#LAMBERT
eqnLAMBERT = (sp.sin(eqnAlpha*np.pi/180.) + (np.pi-eqnAlpha)*sp.cos(eqnAlpha*np.pi/180.))/np.pi

#START, STOP
eqnSTART = 0.5+0.5*sp.tanh((eqnAlpha-A)/B)
eqnEND = 0.5-0.5*sp.tanh((eqnAlpha-A)/B)

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
        eqnSTART.subs(A,50.).subs(B,5.)*eqnPhase2Mars

#JUPITER
eqnPhase1Jupiter = 10.**(-0.4*(- 3.7e-04*eqnAlpha + 6.16e-04*eqnAlpha**2.))
tmpDifference = eqnPhase1Jupiter.subs(eqnAlpha,12.) - 10.**(-0.4*(- 2.5*sp.log(1.0 - 1.507*(12./180.) - 0.363*(12./180.)**2. - 0.062*(12./180.)**3.+ 2.809*(12./180.)**4. - 1.876*(12./180.)**5.,10)))
eqnPhase2Jupiter = tmpDifference + 10.**(-0.4*(- 2.5*sp.log(1.0 - 1.507*(eqnAlpha/180.) - 0.363*(eqnAlpha/180.)**2. - 0.062*(eqnAlpha/180.)**3.+ 2.809*(eqnAlpha/180.)**4. - 1.876*(eqnAlpha/180.)**5.,10)))

phaseJUPITER = eqnEND.subs(A,12.).subs(B,5.)*eqnPhase1Jupiter + \
	eqnSTART.subs(A,12.).subs(B,5.)*eqnEND.subs(A,130.).subs(B,5.)*eqnPhase2Jupiter + \
	eqnSTART.subs(A,130.).subs(B,5.)*eqnLAMBERT

#SATURN
eqnPhase2Saturn = 10.**(-0.4*(- 3.7e-04*eqnAlpha +6.16e-04*eqnAlpha**2.))
eqnDifference = eqnPhase2Saturn.subs(eqnAlpha,6.5) - 10.**(-0.4*(2.446e-4*6.5 + 2.672e-4*6.5**2. - 1.505e-6*6.5**3. + 4.767e-9*6.5**2.))
eqnPhase3Saturn = eqnDifference + 10.**(-0.4*(2.446e-4*eqnAlpha + 2.672e-4*eqnAlpha**2. - 1.505e-6*eqnAlpha**3. + 4.767e-9*eqnAlpha**2.))
phaseSATURN = eqnEND.subs(A,6.5).subs(B,5.)*eqnPhase2Saturn + \
                eqnSTART.subs(A,6.5).subs(B,5.)*eqnEND.subs(A,150.).subs(B,5.)*eqnPhase3Saturn + \
                eqnSTART.subs(A,150.).subs(B,5.)*eqnLAMBERT

#URANUS
phi = Symbol('phi', real=True)
f = Symbol('f', real=True)
#f = 0.0022927
eqnPhiUranus = sp.atan2(sp.tan(phi*np.pi/180.),(1.-f)**2.)*180./np.pi
eqnPhase1Uranus = 10.**(-0.4*(- 8.4e-04*eqnPhiUranus.subs(f,0.0022927).subs(phi,-82.) + 6.587e-3*eqnAlpha + 1.045e-4*eqnAlpha**2.))
phaseURANUS = eqnEND.subs(A,154.).subs(B,5.)*eqnPhase1Uranus + \
        eqnSTART.subs(A,154.).subs(B,5.)*eqnLAMBERT

#NEPTUNE
eqnPhaseNeptune = 10.**(-0.4*(7.944e-3*eqnAlpha + 9.617e-5*eqnAlpha**2.))
phaseNEPTUNE = eqnEND.subs(A,133.14).subs(B,5.)*eqnPhaseNeptune + \
        eqnSTART.subs(A,133.14).subs(B,5.)*eqnLAMBERT
        
