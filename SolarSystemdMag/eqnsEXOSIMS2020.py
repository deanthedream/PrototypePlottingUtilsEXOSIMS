

#from sympy import * 
import sympy as sp
#Symbol, symbols
#from sympy import sp.sin, sp.cos, asp.sin, asp.cos, tan, atan2, exp
#from sympy import power
#from sympy import log
#from mpmath import *
#DELTEfrom mpmath import log10
import numpy as np

a, e, inc, W, w, v = sp.symbols('a e inc W w v', real=True, positive=True)
p = sp.Symbol('p', real=True,positive=True)
R = sp.Symbol('R', real=True,positive=True)
A, B = sp.symbols('A B', real=True, positive=True)
Phi = sp.Symbol('Phi',real=True,positive=True)
alpha = sp.Symbol('alpha', real=True, positive=True)

#r
eqnr = a*(1.-e**2.)/(1.+e*sp.cos(v*np.pi/180))
print('r')
print(eqnr)

#XYZ
eqnX = eqnr*(sp.cos(W*np.pi/180)*sp.cos((w+v)*np.pi/180) - sp.sin(W*np.pi/180)*sp.sin((w+v)*np.pi/180)*sp.cos(inc*np.pi/180.))
eqnY = eqnr*(sp.sin(W*np.pi/180)*sp.cos((w+v)*np.pi/180) + sp.cos(W*np.pi/180)*sp.sin((w+v)*np.pi/180)*sp.cos(inc*np.pi/180.))
eqnZ = eqnr*(sp.sin(inc*np.pi/180.)*sp.sin((w+v)*np.pi/180.))
print('X')
print(eqnX)
print('Y')
print(eqnY)
print('Z')
print(eqnZ)

#For Paper
thisIsOne = sp.simplify(((eqnX/eqnr)**2+(eqnY/eqnr)**2+(eqnZ/eqnr)**2))
thetaEquation = sp.simplify(eqnY/eqnX)
FORPAPER_eqnS_eqnr = sp.simplify(sp.sqrt((eqnX/eqnr)**2.+(eqnY/eqnr)**2.))

#s
eqnS = sp.sqrt(eqnX**2.+eqnY**2.)
eqnSAlpha = a*sp.sin(alpha*np.pi/180.)

#alpha in deg
eqnAlpha1 = sp.acos(eqnZ/sp.sqrt(eqnX**2.+eqnY**2.+eqnZ**2.))*180./np.pi
eqnAlpha2 = sp.asin(eqnS/sp.sqrt(eqnX**2.+eqnY**2.+eqnZ**2.))*180./np.pi
print('Alpha')
print(eqnAlpha1)

#dmag
eqnDmagInside = p*(R/a)**2.*Phi
eqnDmag = -2.5*sp.log(eqnDmagInside,10)

#### PHASE FUNCTIONS ##############
#LAMBERT
eqnLAMBERT = (sp.sin(alpha*np.pi/180.) + (np.pi-alpha*np.pi/180.)*sp.cos(alpha*np.pi/180.))/np.pi

#START, STOP
eqnSTART = 0.5+0.5*sp.tanh((alpha-A)/B)
eqnEND = 0.5-0.5*sp.tanh((alpha-A)/B)

#MERCURY
phaseMERCURY = 10.**(-0.4*(6.3280e-02*alpha - 1.6336e-03*alpha**2. + 3.3644e-05*alpha**3. - 3.4265e-07*alpha**4. + 1.6893e-09*alpha**5. - 3.0334e-12*alpha**6.))
print('phaseMERCURY')
print(phaseMERCURY)

#VENUS
eqnPhase1Venus = 10.**(-0.4*(- 1.044e-03*alpha + 3.687e-04*alpha**2. - 2.814e-06*alpha**3. + 8.938e-09*alpha**4.)) #OK alpha in deg
tmpPhase = 10.**(-0.4*( - 2.81914e-00*alpha + 8.39034e-03*alpha**2.))
h1 = eqnPhase1Venus.subs(alpha,163.7).evalf() #OK
h2 = 10.**(-0.4*( - 2.81914e-00*163.7 + 8.39034e-03*163.7**2.)) - 10.**(-0.4*( - 2.81914e-00*179. + 8.39034e-03*179.**2.)) #OK
tmpDifference = eqnPhase1Venus.subs(alpha, 163.7).evalf() - h1/h2*(10.**(-0.4*( - 2.81914e-00*163.7 + 8.39034e-03*163.7**2.)))
eqnPhase2Venus = tmpPhase*h1/h2 + tmpDifference
phaseVENUS = eqnEND.subs(A,163.7).subs(B,5.)*eqnPhase1Venus + \
        eqnSTART.subs(A,163.7).subs(B,5.)*eqnEND.subs(A,179.).subs(B,0.3)*eqnPhase2Venus + \
        eqnSTART.subs(A,179.).subs(B,0.3)*eqnLAMBERT
print('phaseVENUS')
print(phaseVENUS)

#EARTH
phaseEARTH = 10.**(-0.4*(- 1.060e-3*alpha + 2.054e-4*alpha**2.))
print('phaseEARTH')
print(phaseEARTH)

#MARS
eqnPhase1Mars = 10.**(-0.4*(0.02267*alpha - 0.0001302*alpha**2.+ 0. + 0.)) #L(λe) + L(LS)
eqnPhase2Mars = eqnPhase1Mars.subs(alpha,50.)/10.**(-0.4*(- 0.02573*50. + 0.0003445*50.**2.)) * 10.**(-0.4*(- 0.02573*alpha + 0.0003445*alpha**2. + 0. + 0.)) #L(λe) + L(Ls)
phaseMARS = eqnEND.subs(A,50.).subs(B,5.)*eqnPhase1Mars + \
        eqnSTART.subs(A,50.).subs(B,5.)*eqnPhase2Mars
print('phaseMARS')
print(phaseMARS)

#JUPITER
eqnPhase1Jupiter = 10.**(-0.4*(- 3.7e-04*alpha + 6.16e-04*alpha**2.))
tmpDifference = eqnPhase1Jupiter.subs(alpha,12.) - 10.**(-0.4*(- 2.5*sp.log(1.0 - 1.507*(12./180.) - 0.363*(12./180.)**2. - 0.062*(12./180.)**3.+ 2.809*(12./180.)**4. - 1.876*(12./180.)**5.,10)))
eqnPhase2Jupiter = tmpDifference + 10.**(-0.4*(- 2.5*sp.log(1.0 - 1.507*(alpha/180.) - 0.363*(alpha/180.)**2. - 0.062*(alpha/180.)**3.+ 2.809*(alpha/180.)**4. - 1.876*(alpha/180.)**5.,10)))

phaseJUPITER = eqnEND.subs(A,12.).subs(B,5.)*eqnPhase1Jupiter + \
	eqnSTART.subs(A,12.).subs(B,5.)*eqnEND.subs(A,130.).subs(B,5.)*eqnPhase2Jupiter + \
	eqnSTART.subs(A,130.).subs(B,5.)*eqnLAMBERT
print('phaseJUPITER')
print(phaseJUPITER)

#SATURN
eqnPhase2Saturn = 10.**(-0.4*(- 3.7e-04*alpha +6.16e-04*alpha**2.))
eqnDifference = eqnPhase2Saturn.subs(alpha,6.5) - 10.**(-0.4*(2.446e-4*6.5 + 2.672e-4*6.5**2. - 1.505e-6*6.5**3. + 4.767e-9*6.5**4.))
eqnPhase3Saturn = eqnDifference + 10.**(-0.4*(2.446e-4*alpha + 2.672e-4*alpha**2. - 1.505e-6*alpha**3. + 4.767e-9*alpha**4.))
phaseSATURN = eqnEND.subs(A,6.5).subs(B,5.)*eqnPhase2Saturn + \
                eqnSTART.subs(A,6.5).subs(B,5.)*eqnEND.subs(A,150.).subs(B,5.)*eqnPhase3Saturn + \
                eqnSTART.subs(A,150.).subs(B,5.)*eqnLAMBERT
print('phaseSATURN')
print(phaseSATURN)

#URANUS
phi_uranus = sp.Symbol('phi_uranus', real=True)
f = sp.Symbol('f', real=True, positive=True)
#f = 0.0022927
eqnPhiUranus = sp.atan2(sp.tan(phi_uranus*np.pi/180.),(1.-f)**2.)*180./np.pi
eqnPhase1Uranus = 10.**(-0.4*(- 8.4e-04*eqnPhiUranus.subs(f,0.0022927).subs(phi_uranus,-82.) + 6.587e-3*alpha + 1.045e-4*alpha**2.))
phaseURANUS = eqnEND.subs(A,154.).subs(B,5.)*eqnPhase1Uranus + \
        eqnSTART.subs(A,154.).subs(B,5.)*eqnLAMBERT
print('phaseURANUS')
print(phaseURANUS)

#NEPTUNE
eqnPhaseNeptune = 10.**(-0.4*(7.944e-3*alpha + 9.617e-5*alpha**2.))
phaseNEPTUNE = eqnEND.subs(A,133.14).subs(B,5.)*eqnPhaseNeptune + \
        eqnSTART.subs(A,133.14).subs(B,5.)*eqnLAMBERT
print('phaseNEPTUNE')
print(phaseNEPTUNE)


#### List of Planet Phase Functions
symbolicPhases = [sp.Abs(phaseMERCURY), sp.Abs(phaseVENUS), sp.Abs(phaseEARTH), sp.Abs(phaseMARS), sp.Abs(phaseJUPITER),\
    sp.Abs(phaseSATURN), sp.Abs(phaseURANUS), sp.Abs(phaseNEPTUNE)]



