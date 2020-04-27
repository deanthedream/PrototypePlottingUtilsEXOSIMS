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
def quarticCoefficients(a,b,mx,my):
    """ Calculates coefficients to the quartic polynomial from projected ellipse semi parameters and
    projected strictly positive star coordinates
    Args:
        a (numpy array):
            semi-major axis of projected ellipse with length n planets
        b (numpy array):
            semi-minor axis of projected ellipse with length n planets
        mx (numpy array):
            positive x coordinate of projected star position with length n planets
        my (numpy array):
            positive y coordinate of projected star position with length n planets
    Returns:
        parr (numpy array):
            quartic coefficients for each star with shape n planets by 5 coefficients
    """
    A = -(2 - 2*b**2/a**2)**2/a**2
    B = 4*mx*(2 - 2*b**2/a**2)/a**2
    C = -4*my**2*b**2/a**4 - 4*mx**2/a**2 + (2 - 2*b**2/a**2)**2
    D = -4*mx*(2-2*b**2/a**2)
    E = 4*mx**2
    return np.asarray([A, B, C, D, E]).T

def quarticSolutions(a,b,mx,my):
    """ Runs and separates out real and imaginary components of the quartic solutions
    Args:
        a (numpy array):
            semi-major axis of projected ellipse with length n planets
        b (numpy array):
            semi-minor axis of projected ellipse with length n planets
        mx (numpy array):
            positive x coordinate of projected star position with length n planets
        my (numpy array):
            positive y coordinate of projected star position with length n planets
    Returns:
        xreal (numpy array):
            real component of x solutions to the quartic
        imag (numpy array):
            imaginary component of x solutions to the quartic
    """
    #DELETEparr = np.asarray([A,B,C,D,E]).T #[B/A,C/A,D/A,E/A]
    #DELETEparr = quarticCoefficients(a,b,mx,my)
    out = np.asarray(roots_loop(quarticCoefficients(a,b,mx,my)))
    xreal = np.real(out)
    # print('Number of nan in x0: ' + str(np.count_nonzero(np.isnan(xreal[:,0]))))
    # print('Number of nan in x1: ' + str(np.count_nonzero(np.isnan(xreal[:,1]))))
    # print('Number of nan in x2: ' + str(np.count_nonzero(np.isnan(xreal[:,2]))))
    # print('Number of nan in x3: ' + str(np.count_nonzero(np.isnan(xreal[:,3]))))
    # print('Number of non-zero in x0: ' + str(np.count_nonzero(xreal[:,0] != 0)))
    # print('Number of non-zero in x1: ' + str(np.count_nonzero(xreal[:,1] != 0)))
    # print('Number of non-zero in x2: ' + str(np.count_nonzero(xreal[:,2] != 0)))
    # print('Number of non-zero in x3: ' + str(np.count_nonzero(xreal[:,3] != 0)))
    # print('Number of non-zero in x0+x1+x2+x3: ' + str(np.count_nonzero((xreal[:,0] != 0)*(xreal[:,1] != 0)*(xreal[:,2] != 0)*(xreal[:,3] != 0))))
    imag = np.imag(out)
    # x0_i = imag[:,0]
    # x1_i = imag[:,1]
    # x2_i = imag[:,2]
    # x3_i = imag[:,3]
    # print('Number of non-zero in x0_i: ' + str(np.count_nonzero(x0_i != 0)))
    # print('Number of non-zero in x1_i: ' + str(np.count_nonzero(x1_i != 0)))
    # print('Number of non-zero in x2_i: ' + str(np.count_nonzero(x2_i != 0)))
    # print('Number of non-zero in x3_i: ' + str(np.count_nonzero(x3_i != 0)))
    # print('Number of non-zero in x0_i+x1_i+x2_i+x3_i: ' + str(np.count_nonzero((x0_i != 0)*(x1_i != 0)*(x2_i != 0)*(x3_i != 0))))
    
    return xreal, imag

def ellipseYFromX(xreal, a, b):
    """ Calculates y values in the positive quadrant 
    Args:
        xreal (numpy array):
            shape n planets by 4
        a (numpy array):
            semi-major axis of projected ellipse with length n planets
        b (numpy array):
            semi-minor axis of projected ellipse with length n planets
    return:
        yreal (numpy array):
            numpy array of ellipse quadrant 1 y values, shape n planets by 4
    """
    return np.asarray([np.sqrt(b**2*(1-xreal[:,0]**2/a**2)), np.sqrt(b**2*(1-xreal[:,1]**2/a**2)), np.sqrt(b**2*(1-xreal[:,2]**2/a**2)), np.sqrt(b**2*(1-xreal[:,3]**2/a**2))]).T #yreal

xreal, imag = quarticSolutions(a,b,mx,my)
yreal = ellipseYFromX(xreal, a, b)

#Calculate Possible Separation Combinations for Point
s_mm = np.asarray([np.sqrt((xreal[:,0]-mx)**2 + (yreal[:,0]-my)**2), np.sqrt((xreal[:,1]-mx)**2 + (yreal[:,1]-my)**2), np.sqrt((xreal[:,2]-mx)**2 + (yreal[:,2]-my)**2), np.sqrt((xreal[:,3]-mx)**2 + (yreal[:,3]-my)**2)]).T
s_pp = np.asarray([np.sqrt((xreal[:,0]+mx)**2 + (yreal[:,0]+my)**2), np.sqrt((xreal[:,1]+mx)**2 + (yreal[:,1]+my)**2), np.sqrt((xreal[:,2]+mx)**2 + (yreal[:,2]+my)**2), np.sqrt((xreal[:,3]+mx)**2 + (yreal[:,3]+my)**2)]).T
s_pm = np.asarray([np.sqrt((xreal[:,0]+mx)**2 + (yreal[:,0]-my)**2), np.sqrt((xreal[:,1]+mx)**2 + (yreal[:,1]-my)**2), np.sqrt((xreal[:,2]+mx)**2 + (yreal[:,2]-my)**2), np.sqrt((xreal[:,3]+mx)**2 + (yreal[:,3]-my)**2)]).T
s_mp = np.asarray([np.sqrt((xreal[:,0]-mx)**2 + (yreal[:,0]+my)**2), np.sqrt((xreal[:,1]-mx)**2 + (yreal[:,1]+my)**2), np.sqrt((xreal[:,2]-mx)**2 + (yreal[:,2]+my)**2), np.sqrt((xreal[:,3]-mx)**2 + (yreal[:,3]+my)**2)]).T

#Using Abs because some terms are negative???
s_absmin = np.asarray([np.sqrt((np.abs(xreal[:,0])-mx)**2 + (np.abs(yreal[:,0])-my)**2), np.sqrt((np.abs(xreal[:,1])-mx)**2 + (np.abs(yreal[:,1])-my)**2), np.sqrt((np.abs(xreal[:,2])-mx)**2 + (np.abs(yreal[:,2])-my)**2), np.sqrt((np.abs(xreal[:,3])-mx)**2 + (np.abs(yreal[:,3])-my)**2)]).T
s_absmax = np.asarray([np.sqrt((np.abs(xreal[:,0])+mx)**2 + (np.abs(yreal[:,0])+my)**2), np.sqrt((np.abs(xreal[:,1])+mx)**2 + (np.abs(yreal[:,1])+my)**2), np.sqrt((np.abs(xreal[:,2])+mx)**2 + (np.abs(yreal[:,2])+my)**2), np.sqrt((np.abs(xreal[:,3])+mx)**2 + (np.abs(yreal[:,3])+my)**2)]).T

#DELETEs_min = np.asarray([np.sqrt((xreal[:,0]-mx)**2 + (yreal[:,0]-my)**2), np.sqrt((xreal[:,1]-mx)**2 + (yreal[:,1]-my)**2), np.sqrt((xreal[:,2]-mx)**2 + (yreal[:,2]-my)**2), np.sqrt((xreal[:,3]-mx)**2 + (yreal[:,3]-my)**2)])
#DELETEs_max = np.asarray([np.sqrt((xreal[:,0]+mx)**2 + (yreal[:,0]+my)**2), np.sqrt((xreal[:,1]+mx)**2 + (yreal[:,1]+my)**2), np.sqrt((xreal[:,2]+mx)**2 + (yreal[:,2]+my)**2), np.sqrt((xreal[:,3]+mx)**2 + (yreal[:,3]+my)**2)])
s_lmin = np.asarray([np.sqrt((xreal[:,0]-mx)**2 + (yreal[:,0]+my)**2), np.sqrt((xreal[:,1]-mx)**2 + (yreal[:,1]+my)**2), np.sqrt((xreal[:,2]-mx)**2 + (yreal[:,2]+my)**2), np.sqrt((xreal[:,3]-mx)**2 + (yreal[:,3]+my)**2)])
s_lmax = np.asarray([np.sqrt((xreal[:,0]+mx)**2 + (yreal[:,0]-my)**2), np.sqrt((xreal[:,1]+mx)**2 + (yreal[:,1]-my)**2), np.sqrt((xreal[:,2]+mx)**2 + (yreal[:,2]-my)**2), np.sqrt((xreal[:,3]+mx)**2 + (yreal[:,3]-my)**2)])

#### Minimum Separations and x, y of minimum separation
minSepInd = np.nanargmin(s_mm,axis=1)
minSep = s_mm[np.arange(len(minSepInd)),minSepInd] #Minimum Planet-StarSeparations
minSep_x = xreal[:,minSepInd][:,0] #Minimum Planet-StarSeparations x coord
minSep_y = yreal[:,minSepInd][:,0] #Minimum Planet-StarSeparations y coord
minSepMask = np.zeros((len(minSepInd),4), dtype=bool) 
minSepMask[np.arange(len(minSepInd)),minSepInd] = 1 #contains 1's where the minimum separation occurs
minNanMask = np.isnan(s_mm) #places true where value is nan
#
countMinNans = np.sum(minNanMask,axis=0) #number of Nans in each 4
freqMinNans = np.unique(countMinNans, return_counts=True)
minAndNanMask = minSepMask + minNanMask #Array of T/F of minSep and NanMask
countMinAndNanMask = np.sum(minAndNanMask,axis=1) #counting number of minSep and NanMask for each star
freqMinAndNanMask = np.unique(countMinAndNanMask,return_counts=True) #just gives the quantity of 1,2,3 accounted
minIndsOf3 = np.where(countMinAndNanMask == 3)[0] #planetInds where 3 of the 4 soultions are accounted for
minIndsOf2 = np.where(countMinAndNanMask == 2)[0]
minIndsOf1 = np.where(countMinAndNanMask == 1)[0]

#### Maximum Separations and x,y of maximum separation
maxSepInd = np.nanargmax(s_absmax,axis=1)
maxSep = s_absmax[np.arange(len(maxSepInd)),maxSepInd] #Maximum Planet-StarSeparations
maxSep_x = xreal[:,maxSepInd][:,0] #Maximum Planet-StarSeparations x coord
maxSep_y = yreal[:,maxSepInd][:,0] #Maximum Planet-StarSeparations y coord
maxSepMask = np.zeros((len(maxSepInd),4), dtype=bool) 
maxSepMask[np.arange(len(maxSepInd)),maxSepInd] = 1 #contains 1's where the maximum separation occurs
maxNanMask = np.isnan(s_absmax)
#
countMaxNans = np.sum(maxNanMask,axis=0) #number of Nans in each 4
freqMaxNans = np.unique(countMaxNans, return_counts=True)
maxAndNanMask = maxSepMask + maxNanMask #Array of T/F of maxSep and NanMask
countMaxAndNanMask = np.sum(maxAndNanMask,axis=1) #counting number of maxSep and NanMask for each star
freqMaxAndNanMask = np.unique(countMaxAndNanMask,return_counts=True) #just gives the quantity of 1,2,3 accounted
maxIndsOf3 = np.where(countMaxAndNanMask == 3)[0] #planetInds where 3 of the 4 soultions are accounted for
maxIndsOf2 = np.where(countMaxAndNanMask == 2)[0]
maxIndsOf1 = np.where(countMaxAndNanMask == 1)[0]

#Sort arrays Minimum
sminOrderInds = np.argsort(s_absmin, axis=1) #argsort sorts from min to max with last terms being nan
minSeps = s_absmin[np.arange(len(minSepInd)),sminOrderInds[:,0]]
minSeps_x = xreal[np.arange(len(minSepInd)),sminOrderInds[:,0]]
minSeps_y = yreal[np.arange(len(minSepInd)),sminOrderInds[:,0]]

#Sort arrays Maximum
smaxOrderInds = np.argsort(-s_absmax, axis=1) #-argsort sorts from max to min with last terms being nan #Note: Last 2 indicies will be Nan
maxSeps = s_absmax[np.arange(len(maxSepInd)),smaxOrderInds[:,0]]
maxSeps_x = np.abs(xreal[np.arange(len(maxSepInd)),smaxOrderInds[:,0]])
maxSeps_y = yreal[np.arange(len(maxSepInd)),smaxOrderInds[:,0]]

#Masking
mask = np.zeros((len(maxSepInd),4), dtype=bool)
assert ~np.any(sminOrderInds[:,0] == smaxOrderInds[:,0]), 'Exception: A planet has smin == smax'
mask[np.arange(len(minSepInd)),sminOrderInds[:,0]] = 1 #contains 1's where the minimum separation occurs
mask[np.arange(len(maxSepInd)),smaxOrderInds[:,0]] = 1 #contains 1's where the maximum separation occurs
assert np.all(np.isnan(s_absmin) == np.isnan(s_absmax)), 'Exception: absmin and absmax have different nan values'
mask += np.isnan(s_absmin) #Adds nan solutions to mask
countMinMaxNanMask = np.sum(mask,axis=1) #counting number of stars with min, max, and nan for each star
#freqMinMaxNanMask = np.unique(countMinMaxNanMask,return_counts=True) #just gives the quantity of 1,2,3 accounted

#Quadrant Star Belongs to
bool1 = x > 0
bool2 = y > 0
#Quadrant 1 if T,T
#Quadrant 2 if F,T
#Quadrant 3 if F,F
#Quadrant 4 if T,F

#### Account For Stars with Local Minimum and Maximum
inds_accnt2 = np.where(countMinMaxNanMask == 2)[0] #planetInds where 2 of the 4 soultions are accounted for
inds_accnt4 = np.where(countMinMaxNanMask == 4)[0] #planetInds where all 4 solutions are accounted for
#For Stars with 2 Inds Accounted For
s_mptmp = ~mask*s_mp
s_mptmp[mask] = np.nan

#Sort arrays Local Minimum
s_mpOrderInds = np.argsort(s_mptmp, axis=1) #argsort sorts from min to max with last terms being nan
s_mplminSeps = s_mptmp[np.arange(len(s_mptmp)),s_mpOrderInds[:,0]]
lminSeps_x = xreal[np.arange(len(s_mptmp)),s_mpOrderInds[:,0]]
lminSeps_y = yreal[np.arange(len(s_mptmp)),s_mpOrderInds[:,0]]
s_mplmaxSeps = s_mptmp[np.arange(len(s_mptmp)),s_mpOrderInds[:,1]]
lmaxSeps_x = np.abs(xreal[np.arange(len(s_mptmp)),s_mpOrderInds[:,1]])
lmaxSeps_y = yreal[np.arange(len(s_mptmp)),s_mpOrderInds[:,1]]

#### Min Sep Point (Point on plot of Min Sep)
minSepPoint_x = minSeps_x[ind]*(2*bool1[ind]-1)
minSepPoint_y = minSeps_y[ind]*(2*bool2[ind]-1)

#### Max Sep Point (Point on plot of max sep)
maxSepPoint_x = maxSeps_x[ind]*(-2*bool1[ind]+1)
maxSepPoint_y = maxSeps_y[ind]*(-2*bool2[ind]+1)

#### Local Min Sep Point
lminSepPoint_x = lminSeps_x[ind]*(2*bool1[ind]-1)
lminSepPoint_y = lminSeps_y[ind]*(-2*bool2[ind]+1)

#### Local Max Sep Point
lmaxSepPoint_x = lmaxSeps_x[ind]*(2*bool1[ind]-1)
lmaxSepPoint_y = lmaxSeps_y[ind]*(-2*bool2[ind]+1)
#################################################################################



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
plt.scatter(minSepPoint_x,minSepPoint_y,color='cyan')
#Plot Max Sep Ellipse Intersection
plt.scatter(maxSepPoint_x,maxSepPoint_y,color='red')
#### Plot Local Min
plt.scatter(lminSepPoint_x, lminSepPoint_y,color='magenta')
#### Plot Local Max Points
plt.scatter(lmaxSepPoint_x, lmaxSepPoint_y,color='gold')

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
