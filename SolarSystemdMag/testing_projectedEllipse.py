import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
import numpy.random as random
from sys import getsizeof
import time

#### Randomly Generate Orbits
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))
filename = 'HabEx_CKL2_PPKL2.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
PPop = sim.PlanetPopulation
n = 10**4 #Dean's nice computer can go up to 10**8 what can atuin go up to?
inc, W, w = PPop.gen_angles(n,None)
inc = inc.to('rad').value
inc[np.where(inc>np.pi/2)[0]] = np.pi - inc[np.where(inc>np.pi/2)[0]]
W = W.to('rad').value
w = w.to('rad').value
sma, e, p, Rp = PPop.gen_plan_params(n)
sma = sma.to('AU').value

#### Calculate Projected Ellipse Angles and Minor Axis
start0 = time.time()
dmajorp, dminorp, Psi, psi, theta_OpQ_X, theta_OpQp_X, dmajorp_v2, dminorp_v2, Psi_v2, psi_v2 = projected_apbpPsipsi(sma,e,W,w,inc)
stop0 = time.time()
print('stop0: ' + str(stop0-start0))
#DELETEO = projected_Op(sma,e,W,w,inc)
#DELETEc_3D_projected = projected_projectedLinearEccentricity(sma,e,W,w,inc)
#3D Ellipse Center
start1 = time.time()
Op = projected_Op(sma,e,W,w,inc)
stop1 = time.time()
print('stop1: ' + str(stop1-start1))

# Checks
assert np.all(dmajorp < sma), "Not all Semi-major axis of the projected ellipse are less than the original 3D ellipse"
assert np.all(dminorp < dmajorp), "All projected Semi-minor axes are less than all projected semi-major axes"

#### Plotting Projected Ellipse
def plotProjectedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, num):
    plt.close(num)
    fig = plt.figure(num=num)
    ca = plt.gca()
    ca.axis('equal')
    ## Central Sun
    plt.scatter([0],[0],color='orange')
    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
    x_3Dellipse = r[0,0,:]
    y_3Dellipse = r[1,0,:]
    plt.plot(x_3Dellipse,y_3Dellipse,color='black')

    #plot 3D Ellipse Center
    plt.scatter(Op[0][ind],Op[1][ind],color='black')
    print('a: ' + str(np.round(sma[ind],2)) + ' e: ' + str(np.round(e[ind],2)) + ' W: ' + str(np.round(W[ind],2)) + ' w: ' + str(np.round(w[ind],2)) + ' i: ' + str(np.round(inc[ind],2)) +\
         ' Psi: ' + str(np.round(Psi[ind],2)) + ' psi: ' + str(np.round(psi[ind],2)))# + ' theta: ' + str(np.round(theta[ind],2)))
    #print(dmajorp[ind]*np.cos(theta[ind]))#print(dmajorp[ind]*np.cos(theta[ind]))#print(dminorp[ind]*np.cos(theta[ind]+np.pi/2))#print(dminorp[ind]*np.sin(theta[ind]+np.pi/2))

    ang2 = (theta_OpQ_X[ind]+theta_OpQp_X[ind])/2
    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='purple',linestyle='-')
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='purple',linestyle='-')
    plt.show(block=False)
    ####

ind = random.randint(low=0,high=n)
plotProjectedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, num=877)

#### Derotate Ellipse
start2 = time.time()
x, y, Phi = derotatedEllipse(theta_OpQ_X, theta_OpQp_X, Op)
stop2 = time.time()
print('stop2: ' + str(stop2-start2))
a = dmajorp
b = dminorp
mx = np.abs(x) #x converted to a strictly positive value
my = np.abs(y) #y converted to a strictly positive value

def plotDerotatedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, a, b, num=879):
    plt.close(num)
    fig = plt.figure(num=num)
    ca = plt.gca()
    ca.axis('equal')
    plt.scatter([0],[0],color='orange')
    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
    x_3Dellipse = r[0,0,:]
    y_3Dellipse = r[1,0,:]
    plt.plot(x_3Dellipse,y_3Dellipse,color='black')
    plt.scatter(Op[0][ind],Op[1][ind],color='black')
    ang2 = (theta_OpQ_X[ind]+theta_OpQp_X[ind])/2
    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='purple',linestyle='-')
    #new plot stuff
    Erange = np.linspace(start=0.,stop=2*np.pi,num=400)
    plt.plot([-a[ind],a[ind]],[0,0],color='purple',linestyle='--') #major
    plt.plot([0,0],[-b[ind],b[ind]],color='purple',linestyle='--') #minor
    xellipsetmp = a[ind]*np.cos(Erange)
    yellipsetmp = b[ind]*np.sin(Erange)
    plt.plot(xellipsetmp,yellipsetmp,color='black')
    plt.scatter(x[ind],y[ind],color='orange',marker='x')

    c_ae = a[ind]*np.sqrt(1-b[ind]**2/a[ind]**2)
    plt.scatter([-c_ae,c_ae],[0,0],color='blue')

    plt.show(block=False)

start3 = time.time()
plotDerotatedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, a, b, num=880)
stop3 = time.time()
print('stop3: ' + str(stop3-start3))

#### Calculate X,Y Position of Minimum and Maximums with Quartic
start4 = time.time()
xreal, imag = quarticSolutions(a, b, mx, my)
stop4 = time.time()
print('stop4: ' + str(stop4-start4))
start5 = time.time()
yreal = ellipseYFromX(xreal, a, b)
stop5 = time.time()
print('stop5: ' + str(stop5-start5))

#### Calculate Separations
start6 = time.time()
s_mp, s_absmin, s_absmax = calculateSeparations(xreal, yreal, mx, my)
stop6 = time.time()
print('stop6: ' + str(stop6-start6))

#### Calculate Min Max Separation Points
start7 = time.time()
minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, s_mplminSeps, s_mplmaxSeps = sepsMinMaxLminLmax(s_absmin, s_absmax, s_mp, xreal, yreal, x, y)
stop7 = time.time()
print('stop7: ' + str(stop7-start7))
#################################################################################


#### Memory Usage
memories = [getsizeof(inc),getsizeof(W),getsizeof(w),getsizeof(sma),getsizeof(e),getsizeof(p),getsizeof(Rp),getsizeof(dmajorp),getsizeof(dminorp),getsizeof(Psi),getsizeof(psi),getsizeof(theta_OpQ_X),\
getsizeof(theta_OpQp_X),getsizeof(dmajorp_v2),getsizeof(dminorp_v2),getsizeof(Psi_v2),getsizeof(psi_v2),getsizeof(Op),getsizeof(x),getsizeof(y),getsizeof(Phi),getsizeof(a),getsizeof(b),\
getsizeof(mx),getsizeof(my),getsizeof(xreal),getsizeof(imag),getsizeof(yreal),getsizeof(s_mp),getsizeof(s_absmin),getsizeof(s_absmax),getsizeof(minSepPoints_x),getsizeof(minSepPoints_y),\
getsizeof(maxSepPoints_x),getsizeof(maxSepPoints_y),getsizeof(lminSepPoints_x),getsizeof(lminSepPoints_y),getsizeof(lmaxSepPoints_x),getsizeof(lmaxSepPoints_y),getsizeof(minSep),\
getsizeof(maxSep),getsizeof(s_mplminSeps),getsizeof(s_mplmaxSeps)]
totalMemoryUsage = np.sum(memories)
print('Total Data Used: ' + str(totalMemoryUsage/10**9) + ' GB')
####


num=960
plt.close(num)
fig = plt.figure(num=num)
ca = plt.gca()
ca.axis('equal')
#DELETEplt.scatter([xreal[ind,0],xreal[ind,1],xreal[ind,2],xreal[ind,3]], [yreal[ind,0],yreal[ind,1],yreal[ind,2],yreal[ind,3]], color='purple')
plt.scatter([0],[0],color='orange')
## 3D Ellipse
vs = np.linspace(start=0,stop=2*np.pi,num=300)
r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
x_3Dellipse = r[0,0,:]
y_3Dellipse = r[1,0,:]
plt.plot(x_3Dellipse,y_3Dellipse,color='black')
plt.scatter(Op[0][ind],Op[1][ind],color='black')
ang2 = (theta_OpQ_X[ind]+theta_OpQp_X[ind])/2
dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='purple',linestyle='-')
plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='purple',linestyle='-')
plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='purple',linestyle='-')
plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='purple',linestyle='-')
#new plot stuff
Erange = np.linspace(start=0.,stop=2*np.pi,num=400)
plt.plot([-a[ind],a[ind]],[0,0],color='purple',linestyle='--') #major
plt.plot([0,0],[-b[ind],b[ind]],color='purple',linestyle='--') #minor
xellipsetmp = a[ind]*np.cos(Erange)
yellipsetmp = b[ind]*np.sin(Erange)
plt.plot(xellipsetmp,yellipsetmp,color='black')
plt.scatter(x[ind],y[ind],color='orange',marker='x')

c_ae = a[ind]*np.sqrt(1-b[ind]**2/a[ind]**2)
plt.scatter([-c_ae,c_ae],[0,0],color='blue')

#Plot Min Sep Circle
x_circ = minSep[ind]*np.cos(vs)
y_circ = minSep[ind]*np.sin(vs)
plt.plot(x[ind]+x_circ,y[ind]+y_circ,color='cyan')

#Plot Max Sep Circle
x_circ2 = maxSep[ind]*np.cos(vs)
y_circ2 = maxSep[ind]*np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='red')

#Plot lminSep Circle
x_circ2 = s_mplminSeps[ind]*np.cos(vs)
y_circ2 = s_mplminSeps[ind]*np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='magenta')
#Plot lmaxSep Circle
x_circ2 = s_mplmaxSeps[ind]*np.cos(vs)
y_circ2 = s_mplmaxSeps[ind]*np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='gold')

#Plot Min Sep Ellipse Intersection
plt.scatter(minSepPoints_x[ind],minSepPoints_y[ind],color='cyan')
#Plot Max Sep Ellipse Intersection
plt.scatter(maxSepPoints_x[ind],maxSepPoints_y[ind],color='red')
#### Plot Local Min
plt.scatter(lminSepPoints_x[ind], lminSepPoints_y[ind],color='magenta')
#### Plot Local Max Points
plt.scatter(lmaxSepPoints_x[ind], lmaxSepPoints_y[ind],color='gold')

#### r Intersection test
x_circ2 = np.cos(vs)
y_circ2 = np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='green')

plt.show(block=False)




#### Plot separation vs vs parameter
num=961
plt.close(num)
fig = plt.figure(num=num)
xellipsetmp = a[ind]*np.cos(Erange)
yellipsetmp = b[ind]*np.sin(Erange)
septmp = (x[ind] - xellipsetmp)**2 + (y[ind] - yellipsetmp)**2
plt.plot(Erange,septmp,color='black')
plt.plot([0,2.*np.pi],[0,0],color='black',linestyle='--') #0 sep line
plt.xlim([0,2.*np.pi])
plt.ylabel('Projected Separation in AU')
plt.xlabel('Projected Ellipse E (rad)')
plt.show(block=False)
####



def checkResiduals(A,B,C,D,xreals2,inds,numSols):
    residual_0 = xreals2[inds,0]**4 + A[inds]*xreals2[inds,0]**3 + B[inds]*xreals2[inds,0]**2 + C[inds]*xreals2[inds,0] + D[inds]
    residual_1 = xreals2[inds,1]**4 + A[inds]*xreals2[inds,1]**3 + B[inds]*xreals2[inds,1]**2 + C[inds]*xreals2[inds,1] + D[inds]
    residual_2 = np.zeros(residual_0.shape)
    residual_3 = np.zeros(residual_0.shape)
    if numSols > 2:
        residual_2 = xreals2[inds,2]**4 + A[inds]*xreals2[inds,2]**3 + B[inds]*xreals2[inds,2]**2 + C[inds]*xreals2[inds,2] + D[inds]
        if numSols > 3:
            residual_3 = xreals2[inds,3]**4 + A[inds]*xreals2[inds,3]**3 + B[inds]*xreals2[inds,3]**2 + C[inds]*xreals2[inds,3] + D[inds]
    residual = np.asarray([residual_0, residual_1, residual_2, residual_3]).T
    isAll = np.all((np.real(residual) < 1e-7)*(np.imag(residual) < 1e-7))
    maxRealResidual = np.max(np.real(residual))
    maxImagResidual = np.max(np.imag(residual))
    return residual, isAll, maxRealResidual, maxImagResidual




#### Testing ellipse_to_Quartic solution
r = np.ones(len(a),dtype='complex128')
a.astype('complex128')
b.astype('complex128')
mx.astype('complex128')
my.astype('complex128')
r.astype('complex128')
A = -4*a**2*mx/(a**2 - b**2)
B = 2*a**2*(a**2*b**2 - a**2*r**2 + 3*a**2*mx**2 + a**2*my**2 - b**4 + b**2*r**2 - b**2*mx**2 + b**2*my**2)/(a**4 - 2*a**2*b**2 + b**4)
C = 4*a**4*mx*(-b**2 + r**2 - mx**2 - my**2)/(a**4 - 2*a**2*b**2 + b**4)
D = a**4*(b**4 - 2*b**2*r**2 + 2*b**2*mx**2 - 2*b**2*my**2 + r**4 - 2*r**2*mx**2 - 2*r**2*my**2 + mx**4 + 2*mx**2*my**2 + my**4)/(a**4 - 2*a**2*b**2 + b**4)
# A = -4*a**2*x/(a**2 - b**2)
# B = 2*a**2*(a**2*b**2 - a**2*r**2 + 3*a**2*x**2 + a**2*y**2 - b**4 + b**2*r**2 - b**2*x**2 + b**2*y**2)/(a**4 - 2*a**2*b**2 + b**4)
# C = 4*a**4*x*(-b**2 + r**2 - x**2 - y**2)/(a**4 - 2*a**2*b**2 + b**4)
# D = a**4*(b**4 - 2*b**2*r**2 + 2*b**2*x**2 - 2*b**2*y**2 + r**4 - 2*r**2*x**2 - 2*r**2*y**2 + x**4 + 2*x**2*y**2 + y**4)/(a**4 - 2*a**2*b**2 + b**4)


p0 = (-3*A**2/8+B)**3
p1 = (A*(A**2/8-B/2)+C)**2
p2 = -A*(A*(3*A**2/256-B/16)+C/4)+D
p3 = -3*A**2/8+B
p4 = 2*A*(A**2/8-B/2)
p5 = -p0/108-p1/8+p2*p3/3
p6 = (p0/216+p1/16-p2*p3/6+np.sqrt(p5**2/4+(-p2-p3**2/12)**3/27))**(1/3)
p7 = A**2/4-2*B/3
p8 = (2*p2+p3**2/6)/(3*p6)
#, (-2*p2-p3**2/6)/(3*p6)
p9 = np.sqrt(-2*p5**(1/3)+p7)
p10 = np.sqrt(2*p6+p7+p8)
p11 = A**2/2-4*B/3

#otherwise case
x0 = -A/4 - p10/2 - np.sqrt(p11 - 2*p6 - p8 + (2*C + p4)/p10)/2
x1 = -A/4 - p10/2 + np.sqrt(p11 - 2*p6 - p8 + (2*C + p4)/p10)/2
x2 = -A/4 + p10/2 - np.sqrt(p11 - 2*p6 - p8 + (-2*C - p4)/p10)/2
x3 = -A/4 + p10/2 + np.sqrt(p11 - 2*p6 - p8 + (-2*C - p4)/p10)/2
zeroInds = np.where(p2 + p3**2/12 == 0)[0] #piecewise condition
if len(zeroInds) != 0:
    x0[zeroInds] = -A[zeroInds]/4 - p9[zeroInds]/2 - np.sqrt(p11[zeroInds] + 2*np.cbrt(p5[zeroInds]) + (2*C[zeroInds] + p4[zeroInds])/p9[zeroInds])/2
    x1[zeroInds] = -A[zeroInds]/4 - p9[zeroInds]/2 + np.sqrt(p11[zeroInds] + 2*np.cbrt(p5[zeroInds]) + (2*C[zeroInds] + p4[zeroInds])/p9[zeroInds])/2
    x2[zeroInds] = -A[zeroInds]/4 + p9[zeroInds]/2 - np.sqrt(p11[zeroInds] + 2*np.cbrt(p5[zeroInds]) + (-2*C[zeroInds] - p4[zeroInds])/p9[zeroInds])/2
    x3[zeroInds] = -A[zeroInds]/4 + p9[zeroInds]/2 + np.sqrt(p11[zeroInds] + 2*np.cbrt(p5[zeroInds]) + (-2*C[zeroInds] - p4[zeroInds])/p9[zeroInds])/2
tmpxreals2 = np.asarray([x0, x1, x2, x3]).T
xreals2 = np.asarray([x0, x1, x2, x3]).T
# xreals2[np.abs(np.imag(xreals2)) > 1e-4] = np.nan #There is evidence from below that the residual resulting from entiring solutions with 3e-5j results in 0+1e-20j therefore we will nan above 1e-4
# xreals2 = np.real(xreals2)
yreals2 = ellipseYFromX(xreals2, a, b)
# seps2_0 = np.sqrt((xreals2[:,0]-x)**2 + (yreals2[:,0]-y)**2)
# seps2_1 = np.sqrt((xreals2[:,1]-x)**2 + (yreals2[:,1]-y)**2)
# seps2_2 = np.sqrt((xreals2[:,2]-x)**2 + (yreals2[:,2]-y)**2)
# seps2_3 = np.sqrt((xreals2[:,3]-x)**2 + (yreals2[:,3]-y)**2)
seps2_0 = np.sqrt((xreals2[:,0]-mx)**2 + (yreals2[:,0]-my)**2)
seps2_1 = np.sqrt((xreals2[:,1]-mx)**2 + (yreals2[:,1]-my)**2)
seps2_2 = np.sqrt((xreals2[:,2]-mx)**2 + (yreals2[:,2]-my)**2)
seps2_3 = np.sqrt((xreals2[:,3]-mx)**2 + (yreals2[:,3]-my)**2)
seps2 = np.asarray([seps2_0,seps2_1,seps2_2,seps2_3]).T

delta = 256*D**3 - 192*A*C*D**2 - 128*B**2*D**2 + 144*B*C**2*D - 27*C**4\
        + 144*A**2*B*D**2 - 6*A**2*C**2*D - 80*A*B**2*C*D + 18*A*B*C**3 + 16*B**4*D\
        - 4*B**3*C**2 - 27*A**4*D**2 + 18*A**3*B*C*D - 4*A**3*C**3 - 4*A**2*B**3*D + A**2*B**2*C**2 #verified against wikipedia multiple times
assert 0 == np.count_nonzero(np.imag(delta)), 'All delta are real'
delta = np.real(delta)
P = 8*B - 3*A**2
assert 0 == np.count_nonzero(np.imag(P)), 'Not all P are real'
P = np.real(P)
D2 = 64*D - 16*B**2 + 16*A**2*B - 16*A*C - 3*A**4 #is 0 if the quartic has 2 double roots 
assert 0 == np.count_nonzero(np.imag(D2)), 'Not all D2 are real'
D2 = np.real(D2)
R = A**3 + 8*C* - 4*A*B
assert 0 == np.count_nonzero(np.imag(R)), 'Not all R are real'
R = np.real(R)
delta_0 = B**2 - 3*A*C + 12*D
assert 0 == np.count_nonzero(np.imag(delta_0)), 'Not all delta_0 are real'
delta_0 = np.real(delta_0)

# #Number of Solutions 
# deltagt0 = delta > 0
# deltaIndsgt0 = np.where(delta > 0)[0]
# Plt0 = P < 0
# Pindslt0 = np.where(P < 0)[0]
# D2lt0 = D2 < 0
# D2indslt0 = np.where(D2 < 0)[0]

#we are currently omitting all of these potential calculations so-long-as the following assert is never true
assert ~np.any(p2+p3**2/12 == 0), 'Oops, looks like the sympy piecewise was true once!'

#### Root Types For Each Planet ##########################
# If delta > 0 and P < 0 and D < 0 four roots all real or none
#allRealDistinctInds = np.where((delta > 0)*(P < 0)*(D2 < 0))[0] #METHOD 1, out of 10000, this found 1638, missing ~54
allRealDistinctInds = np.where(np.all(np.abs(np.imag(xreals2)) < 1e-9, axis=1))[0] #This found 1692
residual_allreal, isAll_allreal, maxRealResidual_allreal, maxImagResidual_allreal = checkResiduals(A,B,C,D,xreals2,allRealDistinctInds,4)
assert maxRealResidual_allreal < 1e-9, 'At least one all real residual is too large'
# If delta < 0, two distinct real roots, two complex
twoRealDistinctInds = np.where(delta < 0)[0]

# If delta > 0 and (P < 0 or D < 0)
allImagInds = np.where((delta > 0)*((P > 0)|(D2 > 0)))[0]

# If delta == 0, multiple root
realDoubleRootTwoRealRootsInds = np.where((delta == 0)*(P < 0)*(D2 < 0)*(delta_0 != 0))[0] #delta=0 and P<0 and D2<0
realDoubleRootTwoComplexInds = np.where((delta == 0)*((D2 > 0)|((P > 0)*((D2 != 0)|(R != 0)))))[0] #delta=0 and (D>0 or (P>0 and (D!=0 or R!=0)))
tripleRootSimpleRootInds = np.where((delta == 0)*(delta_0 == 0)*(D2 !=0))[0]
twoRealDoubleRootsInds = np.where((delta == 0)*(D2 == 0)*(P < 0))[0]
twoComplexDoubleRootsInds = np.where((delta == 0)*(D2 == 0)*(P > 0)*(R == 0))[0]
fourIdenticalRealRootsInds = np.where((delta == 0)*(D2 == 0)*(delta_0 == 0))[0]

#### Double checking root classification
#twoRealDistinctInds #check that 2 of the 4 imags are below thresh
numUnderThresh = np.sum(np.abs(np.imag(xreals2[twoRealDistinctInds])) > 1e-11, axis=1)
indsUnderThresh = np.where(numUnderThresh != 2)[0]
indsThatDontBelongIntwoRealDistinctInds = twoRealDistinctInds[indsUnderThresh]
twoRealDistinctInds = np.delete(twoRealDistinctInds,indsThatDontBelongIntwoRealDistinctInds) #deletes the desired inds from aray
#np.count_nonzero(numUnderThresh < 2)

#### All Real Distinct Inds

###########################


#The 1e-5 here gave me the number as the Imag count
allRealDistinctInds2 = np.where(np.all(np.abs(np.imag(xreals2)) > 1e-5, axis=1))[0]
allRealDistinctInds2 = np.where(np.all(np.abs(np.imag(xreals2)) > 1e-9, axis=1))[0]


#Number of Solutions of Each Type
numRootInds = [twoRealDistinctInds,allRealDistinctInds,allImagInds,realDoubleRootTwoRealRootsInds,realDoubleRootTwoComplexInds,\
    tripleRootSimpleRootInds,twoRealDoubleRootsInds,twoComplexDoubleRootsInds,fourIdenticalRealRootsInds]

#Number of Roots of Each Type
lenNumRootsInds = [len(numRootInds[i]) for i in np.arange(len(numRootInds))]

# Calculate Residuals
# residual_0 = xreals2[:,0]**4 + A*xreals2[:,0]**3 + B*xreals2[:,0]**2 + C*xreals2[:,0] + D
# residual_1 = xreals2[:,1]**4 + A*xreals2[:,1]**3 + B*xreals2[:,1]**2 + C*xreals2[:,1] + D
# residual_2 = xreals2[:,2]**4 + A*xreals2[:,2]**3 + B*xreals2[:,2]**2 + C*xreals2[:,2] + D
# residual_3 = xreals2[:,3]**4 + A*xreals2[:,3]**3 + B*xreals2[:,3]**2 + C*xreals2[:,3] + D
# residual = np.asarray([residual_0, residual_1, residual_2, residual_3]).T
# #assert np.all((np.real(residual) < 1e-7)*(np.imag(residual) < 1e-7)), 'All residual are not less than 1e-7'
# del residual_0, residual_1, residual_2, residual_3
residual_all, isAll_all, maxRealResidual_all, maxImagResidual_all = checkResiduals(A,B,C,D,xreals2,np.arange(len(A)),4)


#### NEED TO TEST MECHANISMS FOR STRIPING



xfinal = np.zeros(xreals2.shape) + np.nan
# case 1 Two Real Distinct Inds
#find 2 xsols with smallest imag part
xreals2[twoRealDistinctInds[0]]
ximags2 = np.imag(xreals2[twoRealDistinctInds])
ximags2smallImagInds = np.argsort(np.abs(ximags2),axis=1)[:,0:2] #sorts from smallest magnitude to largest magnitude
xrealsTwoRealDistinct = np.asarray([xreals2[twoRealDistinctInds,ximags2smallImagInds[:,0]], xreals2[twoRealDistinctInds,ximags2smallImagInds[:,1]]]).T
xfinal[twoRealDistinctInds,0:2]= np.real(xrealsTwoRealDistinct)
#Check residuals
# residual_0 = xrealsTwoRealDistinct[:,0]**4 + A[twoRealDistinctInds]*xrealsTwoRealDistinct[:,0]**3 + B[twoRealDistinctInds]*xrealsTwoRealDistinct[:,0]**2 + C[twoRealDistinctInds]*xrealsTwoRealDistinct[:,0] + D[twoRealDistinctInds]
# residual_1 = xrealsTwoRealDistinct[:,1]**4 + A[twoRealDistinctInds]*xrealsTwoRealDistinct[:,1]**3 + B[twoRealDistinctInds]*xrealsTwoRealDistinct[:,1]**2 + C[twoRealDistinctInds]*xrealsTwoRealDistinct[:,1] + D[twoRealDistinctInds]
# residual = np.asarray([residual_0, residual_1]).T
# assert np.all((np.real(residual) < 1e-8)*(np.imag(residual) < 1e-8)), 'All residual, Two Real Distinct, are not less than 1e-8'
# del residual_0, residual_1
residual_case1, isAll_case1, maxRealResidual_case1, maxImagResidual_case1 = checkResiduals(A,B,C,D,xfinal,twoRealDistinctInds,2)
#The following does not work
# residual_0 = np.real(xrealsTwoRealDistinct[:,0])**4 + A[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,0])**3 + B[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,0])**2 + C[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,0]) + D[twoRealDistinctInds]
# residual_1 = np.real(xrealsTwoRealDistinct[:,1])**4 + A[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,1])**3 + B[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,1])**2 + C[twoRealDistinctInds]*np.real(xrealsTwoRealDistinct[:,1]) + D[twoRealDistinctInds]
# residual = np.asarray([residual_0, residual_1]).T
# assert np.all((np.real(residual) < 1e-8)*(np.imag(residual) < 1e-8)), 'All residual are not less than 1e-8'
# del residual_0, residual_1
indsOfRebellious_0 = np.where(np.real(residual_case1[:,0]) > 1e-1)[0]
indsOfRebellious_1 = np.where(np.real(residual_case1[:,1]) > 1e-1)[0]
indsOfRebellious = np.unique(np.concatenate((indsOfRebellious_0,indsOfRebellious_1)))
xrealIndsOfRebellious = twoRealDistinctInds[indsOfRebellious]

#residual = tmpxreals2[twoRealDistinctInds[0]]**4 + A[twoRealDistinctInds[0]]*tmpxreals2[twoRealDistinctInds[0]]**3 + B[twoRealDistinctInds[0]]*tmpxreals2[twoRealDistinctInds[0]]**2 + C[twoRealDistinctInds[0]]*tmpxreals2[twoRealDistinctInds[0]] + D[twoRealDistinctInds[0]]
#residual2 = xreals2[twoRealDistinctInds[0]]**4 + A[twoRealDistinctInds[0]]*xreals2[twoRealDistinctInds[0]]**3 + B[twoRealDistinctInds[0]]*xreals2[twoRealDistinctInds[0]]**2 + C[twoRealDistinctInds[0]]*xreals2[twoRealDistinctInds[0]] + D[twoRealDistinctInds[0]]
#residual3 = np.real(xreals2[twoRealDistinctInds[0]])**4 + A[twoRealDistinctInds[0]]*np.real(xreals2[twoRealDistinctInds[0]])**3 + B[twoRealDistinctInds[0]]*np.real(xreals2[twoRealDistinctInds[0]])**2 + C[twoRealDistinctInds[0]]*np.real(xreals2[twoRealDistinctInds[0]]) + D[twoRealDistinctInds[0]]

#currently getting intersection points that are not physically possible

# case 2 All Real Distinct Inds
xfinal[allRealDistinctInds] = np.real(xreals2[allRealDistinctInds])
# residual_0 = xfinal[allRealDistinctInds,0]**4 + A[allRealDistinctInds]*xfinal[allRealDistinctInds,0]**3 + B[allRealDistinctInds]*xfinal[allRealDistinctInds,0]**2 + C[allRealDistinctInds]*xfinal[allRealDistinctInds,0] + D[allRealDistinctInds]
# residual_1 = xfinal[allRealDistinctInds,1]**4 + A[allRealDistinctInds]*xfinal[allRealDistinctInds,1]**3 + B[allRealDistinctInds]*xfinal[allRealDistinctInds,1]**2 + C[allRealDistinctInds]*xfinal[allRealDistinctInds,1] + D[allRealDistinctInds]
# residual_2 = xfinal[allRealDistinctInds,2]**4 + A[allRealDistinctInds]*xfinal[allRealDistinctInds,2]**3 + B[allRealDistinctInds]*xfinal[allRealDistinctInds,2]**2 + C[allRealDistinctInds]*xfinal[allRealDistinctInds,2] + D[allRealDistinctInds]
# residual_3 = xfinal[allRealDistinctInds,3]**4 + A[allRealDistinctInds]*xfinal[allRealDistinctInds,3]**3 + B[allRealDistinctInds]*xfinal[allRealDistinctInds,3]**2 + C[allRealDistinctInds]*xfinal[allRealDistinctInds,3] + D[allRealDistinctInds]
# residual = np.asarray([residual_0, residual_1, residual_2, residual_3]).T
# assert np.all((np.real(residual) < 1e-7)*(np.imag(residual) < 1e-7)), 'All residual, All Real Distinct, are not less than 1e-7'
# del residual_0, residual_1, residual_2, residual_3
residual_case2, isAll_case2, maxRealResidual_case2, maxImagResidual_case2 = checkResiduals(A,B,C,D,xfinal,allRealDistinctInds,4)

# case 3 All Imag Inds
#NO REAL ROOTS
#xreals2[allImagInds[0]]

# case 4 a real double root and 2 real solutions (2 real solutions which are identical and 2 other real solutions)
#xreals2[realDoubleRootTwoRealRootsInds[0]]
ximags2 = np.imag(xreals2[realDoubleRootTwoRealRootsInds])
ximags2smallImagInds = np.argsort(np.abs(ximags2),axis=1)[:,0:2] #sorts from smallest magnitude to largest magnitude
xrealDoubleRootTwoRealRoots = np.asarray([xreals2[realDoubleRootTwoRealRootsInds,ximags2smallImagInds[:,0]], xreals2[realDoubleRootTwoRealRootsInds,ximags2smallImagInds[:,1]]]).T
xfinal[realDoubleRootTwoRealRootsInds,0:2] = np.real(xrealDoubleRootTwoRealRoots)



# case 5 a real double root
#xreals2[realDoubleRootTwoComplexInds[0]]
ximags2 = np.imag(xreals2[realDoubleRootTwoComplexInds])
ximags2smallImagInds = np.argsort(np.abs(ximags2),axis=1)[:,0:2] #sorts from smallest magnitude to largest magnitude
xrealDoubleRootTwoComplex = np.asarray([xreals2[realDoubleRootTwoComplexInds,ximags2smallImagInds[:,0]], xreals2[realDoubleRootTwoComplexInds,ximags2smallImagInds[:,1]]]).T
xfinal[realDoubleRootTwoComplexInds,0:2] = np.real(xrealDoubleRootTwoComplex)

yfinal = ellipseYFromX(xfinal, a, b)
s_mpr, s_absminr, s_absmaxr = calculateSeparations(xfinal, yfinal, x, y)
#TODO need to do what I did for the sepsMinMaxLminLmax function for x, y coordinate determination


#### Notes
#If r < smin, then all imag
#if r > smin and r > slmin, then 2 real.
#if r > slmin and r < slmax, then 4 real.
#if r < smax and r > slmax, then 2 real.
#if r > smax, then all imag.



minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, s_mplminSeps, s_mplmaxSeps = sepsMinMaxLminLmax(s_absmin, s_absmax, s_mp, xreal, yreal, x, y)





#outputs
#nWith4SolutionsIMAG = np.count_nonzero(np.count_nonzero(imag,axis=1)==0)


np.count_nonzero(np.imag(x0))

# tmp1 = 2b**3-9abc+27c**2+27a**2d-72bd
# tmp2 = b**2-3*a*c+12*d
# tmp3 = sp.sqrt(-4*tmp2**3+tmp1**2)
# expression = -a/4-(1/2){sp.sqrt{a**2/4-2*b/3+(2**(1/3)tmp2)(3*(tmp1+tmp3)**(1/3))
# +((tmp1+sp.sqrt(-4*tmp2**3+tmp1**2))/54)**(1/3)}}-(1/2)*sp.sqrt(a**2/2-4*b/3-
# (2**(1/3)*tmp2)/(3*(tmp1+tmp3)**(1/3))-
# ((tmp1+tmp3)/54)**(1/3)-(-a**3+4*a*b-8*c)/(4*sp.sqrt(a**2/4-2*b/3+(2**(1/3)(tmp2))/(3*(tmp1+tmp3)**(1/3))+((tmp1+tmp3)/54)**(1/3))))

shouldBeZero = x0**4 + A*x0**3 + B*x0**2 + C*x0 + D
