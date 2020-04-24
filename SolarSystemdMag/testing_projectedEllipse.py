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
n = 10**3
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

ind = random.randint(low=0,high=n)
plotProjectedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, num=877)

#### Derotate Ellipse
x, y, Phi = derotatedEllipse(theta_OpQ_X, theta_OpQp_X, Op)
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

plotDerotatedEllipse(ind, sma, e, W, w, inc, theta_OpQ_X, theta_OpQp_X, dmajorp, dminorp, Op, a, b, num=880)



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

#### QUARTIC ROOTS #### THIS WORKS
A = -(2 - 2*b**2/a**2)**2/a**2
B = 4*mx*(2 - 2*b**2/a**2)/a**2
C = -4*my**2*b**2/a**4 - 4*mx**2/a**2 + (2 - 2*b**2/a**2)**2
D = -4*mx*(2-2*b**2/a**2)
E = 4*mx**2

parr = np.asarray([A,B,C,D,E]).T #[B/A,C/A,D/A,E/A]
out = np.asarray(roots_loop(parr))
xreal = np.real(out)
# x0 = xreal[:,0]
# x1 = xreal[:,1]
# x2 = xreal[:,2]
# x3 = xreal[:,3]
print('Number of nan in x0: ' + str(np.count_nonzero(np.isnan(xreal[:,0]))))
print('Number of nan in x1: ' + str(np.count_nonzero(np.isnan(xreal[:,1]))))
print('Number of nan in x2: ' + str(np.count_nonzero(np.isnan(xreal[:,2]))))
print('Number of nan in x3: ' + str(np.count_nonzero(np.isnan(xreal[:,3]))))
print('Number of non-zero in x0: ' + str(np.count_nonzero(xreal[:,0] != 0)))
print('Number of non-zero in x1: ' + str(np.count_nonzero(xreal[:,1] != 0)))
print('Number of non-zero in x2: ' + str(np.count_nonzero(xreal[:,2] != 0)))
print('Number of non-zero in x3: ' + str(np.count_nonzero(xreal[:,3] != 0)))
print('Number of non-zero in x0+x1+x2+x3: ' + str(np.count_nonzero((xreal[:,0] != 0)*(xreal[:,1] != 0)*(xreal[:,2] != 0)*(xreal[:,3] != 0))))
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
yreal = np.asarray([np.sqrt(b**2*(1-xreal[:,0]**2/a**2)), np.sqrt(b**2*(1-xreal[:,1]**2/a**2)), np.sqrt(b**2*(1-xreal[:,2]**2/a**2)), np.sqrt(b**2*(1-xreal[:,3]**2/a**2))]).T
# y0 = np.sqrt(b**2*(1-xreal[:,0]**2/a**2))
# y1 = np.sqrt(b**2*(1-xreal[:,1]**2/a**2))
# y2 = np.sqrt(b**2*(1-xreal[:,2]**2/a**2))
# y3 = np.sqrt(b**2*(1-xreal[:,3]**2/a**2))
# s0_min = np.sqrt((x0-mx)**2 + (y0-my)**2)
# s1_min = np.sqrt((x1-mx)**2 + (y1-my)**2)
# s2_min = np.sqrt((x2-mx)**2 + (y2-my)**2)
# s3_min = np.sqrt((x3-mx)**2 + (y3-my)**2)
s_min = np.asarray([np.sqrt((xreal[:,0]-mx)**2 + (yreal[:,0]-my)**2), np.sqrt((xreal[:,1]-mx)**2 + (yreal[:,1]-my)**2), np.sqrt((xreal[:,2]-mx)**2 + (yreal[:,2]-my)**2), np.sqrt((xreal[:,3]-mx)**2 + (yreal[:,3]-my)**2)])

#### Minimum Separations and x, y of minimum separation
minSepInd = np.nanargmin(s_min,axis=0)
minSep = s_min.T[:,minSepInd][:,0] #Minimum Planet-StarSeparations
minSep_x = xreal[:,minSepInd][:,0] #Minimum Planet-StarSeparations x coord
minSep_y = yreal[:,minSepInd][:,0] #Minimum Planet-StarSeparations y coord
minSepMask = np.zeros((len(minSepInd),4), dtype=bool) 
minSepMask[np.arange(len(minSepInd)),minSepInd] = 1 #contains 1's where the minimum separation occurs
minNanMask = np.isnan(s_min.T) #places true where value is nan
countMinNans = np.sum(minNanMask,axis=0) #number of Nans in each 4
freqMinNans = np.unique(countMinNans, return_counts=True)
minAndNanMask = minSepMask + minNanMask #Array of T/F of minSep and NanMask
countMinAndNanMask = np.sum(minAndNanMask,axis=1) #counting number of minSep and NanMask for each star
freqMinAndNanMask = np.unique(countMinAndNanMask,return_counts=True) #just gives the quantity of 1,2,3 accounted
minIndsOf3 = np.where(countMinAndNanMask == 3)[0] #planetInds where 3 of the 4 soultions are accounted for
minIndsOf2 = np.where(countMinAndNanMask == 2)[0]
minIndsOf1 = np.where(countMinAndNanMask == 1)[0]
# #For 3
# #need to use minAndNanMask to find index of residual False term for the minIndsOf3 stars
# for32smallest = np.nanmax(minSeps[minIndsOf3],axis=1)#*~minAndNanMask[minIndsOf3]
# #For 2
# #For 1
# #order each of these?

#Note: Last 2 indicies will be Nan
sminOrderInds = np.argsort(s_min, axis=0)


#Sort arrays
minSeps = s_min[sminOrderInds,np.arange(len(minSepInd))].T
#DELETEminSeps = s_min.T[:,sminOrderInds][:,0]
minSeps_x = xreal[np.arange(len(minSepInd)),sminOrderInds]
minSeps_y = yreal[np.arange(len(minSepInd)),sminOrderInds]



#### Maximum Separations and x,y of maximum separation
# s0_max = np.sqrt((x0+mx)**2 + (y0+my)**2)
# s1_max = np.sqrt((x1+mx)**2 + (y1+my)**2)
# s2_max = np.sqrt((x2+mx)**2 + (y2+my)**2)
# s3_max = np.sqrt((x3+mx)**2 + (y3+my)**2)
s_max = np.asarray([np.sqrt((xreal[:,0]+mx)**2 + (yreal[:,0]+my)**2), np.sqrt((xreal[:,1]+mx)**2 + (yreal[:,1]+my)**2), np.sqrt((xreal[:,2]+mx)**2 + (yreal[:,2]+my)**2), np.sqrt((xreal[:,3]+mx)**2 + (yreal[:,3]+my)**2)])
#CHANGED THESE TWO LINES TO S_MAX FROM S_MIN
maxSepInd = np.nanargmax(s_max,axis=0)
maxSep = s_max.T[:,maxSepInd][:,0] #Maximum Planet-StarSeparations
maxSep_x = xreal[:,maxSepInd][:,0] #Maximum Planet-StarSeparations x coord
maxSep_y = yreal[:,maxSepInd][:,0] #Maximum Planet-StarSeparations y coord
maxSepMask = np.zeros((len(maxSepInd),4), dtype=bool) 
maxSepMask[np.arange(len(maxSepInd)),maxSepInd] = 1 #contains 1's where the maximum separation occurs
maxNanMask = np.isnan(s_max.T)
#countMaxNans = np.sum(maxNanMask,axis=0) #number of Nans in each 4 #EQUAL TO countMinNans
#freqMaxNans = np.unique(countMaxNans, return_counts=True) #EQUAL TO countMaxNans
#DO THE SAME WITH NAN MASKS AS I DO FOR MIN
#Note: Last 2 indicies will be Nan
smaxOrderInds = np.argsort(s_max, axis=0)

#Sort arrays
maxSeps = s_max[smaxOrderInds,np.arange(len(maxSepInd))].T
#maxSeps = maxSeps[smaxOrderInds]
maxSeps_x = xreal[np.arange(len(maxSepInd)),smaxOrderInds]
#maxSeps_x = maxSeps_x[smaxOrderInds]
maxSeps_y = yreal[np.arange(len(maxSepInd)),smaxOrderInds]#maxSep_y[smaxOrderInds]

#Quadrant Star Belongs to
bool1 = x > 0
bool2 = y > 0

#### Min Sep Point (Point on plot of Min Sep)
minSepPoint_x = minSeps_x[0,ind]*(2*bool1[ind]-1)
minSepPoint_y = minSeps_y[0,ind]*(2*bool2[ind]-1)

#### Max Sep Point (Point on plot of max sep)
maxSepPoint_x = maxSeps_x[0,ind]*(-2*bool1[ind]+1)
maxSepPoint_y = maxSeps_y[0,ind]*(-2*bool2[ind]+1)

# DELETE #TODO Parse out local min and local max
# tmp = np.asarray([s0_max,s1_max,s2_max,s3_max])
# SepNans = np.isnan(np.asarray([s0_max,s1_max,s2_max,s3_max]))
# countNans = np.sum(SepNans,axis=0)

#np.unique(countNans, return_counts=True) #tells me the frequency and number of Nans in solutions
# zeroNanInds = np.where(countNans == 0)[0]
# oneNanInds = np.where(countNans == 1)[0]
# twoNanInds = np.where(countNans == 2)[0]
#Do Operations for twoNans
#Do Operations for 1 Nan
#Do Operations for No Nan


#################################################################################



num=960
plt.close(num)
fig = plt.figure(num=num)
ca = plt.gca()
ca.axis('equal')
plt.scatter([xreal[ind,0],xreal[ind,1],xreal[ind,2],xreal[ind,3]], [yreal[ind,0],yreal[ind,1],yreal[ind,2],yreal[ind,3]], color='purple')
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

#Plot Min Sep Ellipse Intersection
plt.scatter(minSepPoint_x,minSepPoint_y,color='red')

# #Plot Min Sep Circle
# x_circ = minSep[ind]*np.cos(vs)
# y_circ = minSep[ind]*np.sin(vs)
# plt.plot(x[ind]+x_circ,y[ind]+y_circ,color='red')
#Plot Min Sep Circle
x_circ = minSeps[ind,0]*np.cos(vs)
y_circ = minSeps[ind,0]*np.sin(vs)
plt.plot(x[ind]+x_circ,y[ind]+y_circ,color='red')
x_circ = minSeps[ind,1]*np.cos(vs)
y_circ = minSeps[ind,1]*np.sin(vs)
plt.plot(x[ind]+x_circ,y[ind]+y_circ,color='green')
x_circ = minSeps[ind,2]*np.cos(vs)
y_circ = minSeps[ind,2]*np.sin(vs)
plt.plot(x[ind]+x_circ,y[ind]+y_circ,color='blue')
x_circ = minSeps[ind,3]*np.cos(vs)
y_circ = minSeps[ind,3]*np.sin(vs)
plt.plot(x[ind]+x_circ,y[ind]+y_circ,color='orange')


#Plot Max Sep Ellipse Intersection
plt.scatter(maxSepPoint_x,maxSepPoint_y,color='red')

#Plot Max Sep Circle
x_circ2 = maxSeps[ind,0]*np.cos(vs)
y_circ2 = maxSeps[ind,0]*np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='cyan')
x_circ2 = maxSeps[ind,1]*np.cos(vs)
y_circ2 = maxSeps[ind,1]*np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='magenta')
x_circ2 = maxSeps[ind,2]*np.cos(vs)
y_circ2 = maxSeps[ind,2]*np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='brown')
x_circ2 = maxSeps[ind,3]*np.cos(vs)
y_circ2 = maxSeps[ind,3]*np.sin(vs)
plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='gold')
plt.show(block=False)


