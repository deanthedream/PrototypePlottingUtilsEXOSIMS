import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
import numpy.random as random

#### Randomly Generate Orbits
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))
filename = 'HabEx_CKL2_PPKL2.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
PPop = sim.PlanetPopulation
n = 10**4
inc, W, w = PPop.gen_angles(n,None)
inc = inc.to('rad').value
inc[np.where(inc>np.pi/2)[0]] = np.pi - inc[np.where(inc>np.pi/2)[0]]
W = W.to('rad').value
w = w.to('rad').value
sma, e, p, Rp = PPop.gen_plan_params(n)
sma = sma.to('AU').value

####
dmajorp, dminorp, Psi, psi, theta_OpQ_X, theta_OpQp_X, dmajorp_v2, dminorp_v2, Psi_v2, psi_v2 = projected_apbpPsipsi(sma,e,W,w,inc)
O = projected_Op(sma,e,W,w,inc)
#DELETE theta = projected_BpAngle(sma,e,W,w,inc)
c_3D_projected = projected_projectedLinearEccentricity(sma,e,W,w,inc)
#3D Ellipse Center
Op = projected_Op(sma,e,W,w,inc)

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

ind = random.randint(low=0,high=n)#2
plotProjectedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, num=877)
#DELETEplotProjectedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp_v2, dminorp_v2, Op, num=878)

#### Derotate Ellipse
x, y, Phi = derotatedEllipse(theta_OpQ_X, theta_OpQp_X, Op)
a = dmajorp
b = dminorp
#DELETEa_v2 = dmajorp_v2 #max difference between a_v2 and a is 1e-14
#DELETEb_v2 = dminorp_v2 #max difference between b_v2 and b is 9e-15 #CONCLUSION, _V2 IS USELESS WASTE OF TIME

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

plotDerotatedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, a, b, num=880)
#plotDerotatedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp_v2, dminorp_v2, Op, a_v2, b_v2, num=881)

mx = np.abs(x)
my = np.abs(y)

def roots_vec(p):
    p = np.atleast_1d(p)
    n = p.shape[-1]
    A = np.zeros(p.shape[:1] + (n-1, n-1), float)
    A[...,1:,:-1] = np.eye(n-2)
    A[...,0,:] = -p[...,1:]/p[...,None,0]
    return np.linalg.eigvals(A)

def roots_loop(p):
    r = []
    for pp in p:
        r.append(np.roots(pp))
    return r


# #From ipynb ellipse_to_quartic DE
# A = 4*a**4 - 8*a**2*b**2 + 4*b**4
# B = -8*a**4*mx + 8*a**2*b**2*mx
# C = -8*a**6 + 16*a**4*b**2 + 4*a**4*mx**2 - 8*a**2*b**4 + 4*a**2*b**2*my**2
# D = 16*a**6*mx - 16*a**4*b**2*mx
# E = 4*a**8 - 8*a**6*b**2 - 8*a**6*mx**2 + 4*a**4*b**4 - 4*a**4*b**2*my**2
# F = -8*a**8*mx + 8*a**6*b**2*mx
# G = 4*a**8*mx**2

# parr = np.asarray([A,B,C,D,E,F,G]).T #[B/A,C/A,D/A,E/A]
# out = np.asarray(roots_loop(parr))


#TRY 3
xe3 = -a**2*mx/(2*my-a**2)
ye3 = np.sqrt(b**2*(1-xe3**2/a**2))
s3 = np.sqrt((xe3-mx)**2 + (ye3-my)**2)
#s1_v2 = (xe1_v2-mx)**2 + (ye_1_v2-my)**2


#################### THIS WORKS
# TRY 4 QUARTIC ROOTS
A = -(2 - 2*b**2/a**2)**2/a**2
B = 4*mx*(2 - 2*b**2/a**2)/a**2
C = -4*my**2*b**2/a**4 - 4*mx**2/a**2 + (2 - 2*b**2/a**2)**2
D = -4*mx*(2-2*b**2/a**2)
E = 4*mx**2

parr = np.asarray([A,B,C,D,E]).T #[B/A,C/A,D/A,E/A]
out = np.asarray(roots_loop(parr))
real = np.real(out)
x0 = real[:,0]
x1 = real[:,1]
x2 = real[:,2]
x3 = real[:,3]
print('Number of nan in x0: ' + str(np.count_nonzero(np.isnan(x0))))
print('Number of nan in x1: ' + str(np.count_nonzero(np.isnan(x1))))
print('Number of nan in x2: ' + str(np.count_nonzero(np.isnan(x2))))
print('Number of nan in x3: ' + str(np.count_nonzero(np.isnan(x3))))
print('Number of non-zero in x0: ' + str(np.count_nonzero(x0 != 0)))
print('Number of non-zero in x1: ' + str(np.count_nonzero(x1 != 0)))
print('Number of non-zero in x2: ' + str(np.count_nonzero(x2 != 0)))
print('Number of non-zero in x3: ' + str(np.count_nonzero(x3 != 0)))
print('Number of non-zero in x0+x1+x2+x3: ' + str(np.count_nonzero((x0 != 0)*(x1 != 0)*(x2 != 0)*(x3 != 0))))
imag = np.imag(out)
x0_i = imag[:,0]
x1_i = imag[:,1]
x2_i = imag[:,2]
x3_i = imag[:,3]
print('Number of non-zero in x0_i: ' + str(np.count_nonzero(x0_i != 0)))
print('Number of non-zero in x1_i: ' + str(np.count_nonzero(x1_i != 0)))
print('Number of non-zero in x2_i: ' + str(np.count_nonzero(x2_i != 0)))
print('Number of non-zero in x3_i: ' + str(np.count_nonzero(x3_i != 0)))
print('Number of non-zero in x0_i+x1_i+x2_i+x3_i: ' + str(np.count_nonzero((x0_i != 0)*(x1_i != 0)*(x2_i != 0)*(x3_i != 0))))
y0 = np.sqrt(b**2*(1-x0**2/a**2))
y1 = np.sqrt(b**2*(1-x1**2/a**2))
y2 = np.sqrt(b**2*(1-x2**2/a**2))
y3 = np.sqrt(b**2*(1-x3**2/a**2))
s0_min = np.sqrt((x0-mx)**2 + (y0-my)**2)
s1_min = np.sqrt((x1-mx)**2 + (y1-my)**2)
s2_min = np.sqrt((x2-mx)**2 + (y2-my)**2)
s3_min = np.sqrt((x3-mx)**2 + (y3-my)**2)

minSepsInds = np.nanargmin(np.asarray([s0_min,s1_min,s2_min,s3_min]),axis=0)
minSeps = np.asarray([s0_min,s1_min,s2_min,s3_min]).T[:,minSepsInds][:,0] #Minimum Planet-StarSeparations
minSeps_x = np.asarray([x0,x1,x2,x3]).T[:,minSepsInds][:,0] #Minimum Planet-StarSeparations x coord
minSeps_y = np.asarray([y0,y1,y2,y3]).T[:,minSepsInds][:,0] #Minimum Planet-StarSeparations y coord

s0_max = np.sqrt((x0+mx)**2 + (y0+my)**2)
s1_max = np.sqrt((x1+mx)**2 + (y1+my)**2)
s2_max = np.sqrt((x2+mx)**2 + (y2+my)**2)
s3_max = np.sqrt((x3+mx)**2 + (y3+my)**2)
maxSepsInds = np.nanargmax(np.asarray([s0_max,s1_max,s2_max,s3_max]),axis=0)
maxSeps = np.asarray([s0_max,s1_max,s2_max,s3_max]).T[:,maxSepsInds][:,0] #Maximum Planet-StarSeparations
maxSeps_x = np.asarray([x0,x1,x2,x3]).T[:,maxSepsInds][:,0] #Maximum Planet-StarSeparations x coord
maxSeps_y = np.asarray([y0,y1,y2,y3]).T[:,maxSepsInds][:,0] #Maximum Planet-StarSeparations y coord

#################################################################################



num=960
plt.close(num)
fig = plt.figure(num=num)
ca = plt.gca()
ca.axis('equal')
plt.scatter([x0[ind],x1[ind],x2[ind],x3[ind]], [y0[ind],y1[ind],y2[ind],y3[ind]], color='purple')
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

#finding index where sun pos belongs
bool1 = x > 0
bool2 = y > 0
#quadIndex = ~bool1*bool2 + ~bool1*~bool2*2 + bool1*~bool2*3 #+ bool1*bool2

#Plot Min Sep Ellipse Intersection
plt.scatter(minSeps_x[ind]*(2*bool1[ind]-1),minSeps_y[ind]*(2*bool2[ind]-1),color='red')

#Plot Min Sep Circle
x_circ = minSeps[ind]*np.cos(vs)
y_circ = minSeps[ind]*np.sin(vs)
plt.plot(x[ind]+x_circ,y[ind]+y_circ,color='red')

#Plot Max Sep Ellipse Intersection
plt.scatter(maxSeps_x[ind]*(-2*bool1[ind]+1),maxSeps_y[ind]*(-2*bool2[ind]+1),color='red')

#Plot Max Sep Circle
x_circ2 = maxSeps[ind]*np.cos(vs)
y_circ2 = maxSeps[ind]*np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='red')
plt.show(block=False)


