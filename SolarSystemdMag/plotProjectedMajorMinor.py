
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import itertools
import time

#### ds/dnu
omega, xxx, inc, nu, theta = sp.symbols('omega, xxx, inc, nu, theta', real=True)
W = sp.symbols('W', real=True)
sma, eccen, sep = sp.symbols('sma, eccen, sep', real=True, positive=True)


thetax, thetay, thetaz = sp.symbols('thetax thetay thetaz',real=True)
Rx = sp.Matrix([[1,0,0],[0,sp.cos(thetax),-sp.sin(thetax)],[0,sp.sin(thetax),sp.cos(thetax)]])
Ry = sp.Matrix([[sp.cos(thetay),0,sp.sin(thetay)],[0,1,0],[-sp.sin(thetay),0,sp.cos(thetay)]])
Rz = sp.Matrix([[sp.cos(thetaz),-sp.sin(thetaz),0],[sp.sin(thetaz),sp.cos(thetaz),0],[0,0,1]])

b = sp.symbols('b',real=True, positive=True)


#semi-minor axis vector
bvect = sp.Matrix([[0],[b],[0]])
bFinal = Rx*Ry*Rz*bvect
bFinal = bFinal.subs(thetaz,0).subs(b,0.5)

#semi-major axis vector
avect = sp.Matrix([[sma],[0],[0]])
aFinal = Rx*Ry*Rz*avect
aFinal = aFinal.subs(thetaz,0).subs(sma,1.)

thetaxs = [0.,np.pi/6.,np.pi/3.]#,np.pi/2.]
thetays = [0.,np.pi/6.,np.pi/3.]#,np.pi/2.]
plt.figure()
for i,j in itertools.product(np.arange(len(thetaxs)),np.arange(len(thetays))): # zip(np.arange(len(thetaxs)),np.arange(len(thetays))):
    print(str(i) + " " + str(j))
    thx = thetaxs[i]
    thy = thetays[j]
    A = 1.
    G = 0.
    R = (thx)/(np.pi/2.)
    B = (thy)/(np.pi/2.)
    #avect
    ax = aFinal[0].subs(thetax,thx).subs(thetay,thy)
    ay = aFinal[1].subs(thetax,thx).subs(thetay,thy)
    plt.plot([0,ax],[0,ay],color=(R,G,B,A))
    #bvect
    bx = bFinal[0].subs(thetax,thx).subs(thetay,thy)
    by = bFinal[1].subs(thetax,thx).subs(thetay,thy)
    plt.plot([0,bx],[0,by],color=(R,G,B,A),linewidth=4.0)
    
    #Plot ellipse
    E = np.linspace(start=0,stop=2.*np.pi,num=50)
    plt.plot(np.sqrt(bx**2.+by**2.)*, ,color=(R,G,B,A))

    plt.show(block=False) 
    plt.pause(0.75)

