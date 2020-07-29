

import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib import ticker
from matplotlib import colors
from matplotlib import cm
from scipy.optimize import minimize

#### Maximum Latency
maxLat = (4.*35786.-2.*6371.)/299792.458
print('maximum latency (s) is: ' + str(maxLat))

####
aveLat = 0.01
nSC = 5.
pComm = 0.995

O = ((0.4349 - aveLat)/0.4349)/nSC*pComm
mO = ((0.4349 - aveLat)/0.4349)/(nSC-1.)*pComm
pO = ((0.4349 - aveLat)/0.4349)/(nSC+1.)*pComm

#Percent Deltas
mDelta = mO/O
pDelta = pO/O

# Exponent Calculations
a = np.log(pDelta)/np.log((0.4349-0.01-0.005)/0.4349)
b = np.log(pDelta)/np.log(0.995)
c = 1.
####

#### Theoretical Minimum Latency
#Orbit altitude above the Earth
A = 450 #in km
#Angular Distance From Source To Sink
theta = np.pi/2.
#Earth Radius
Re = 6371 #in km
# Total Distance
D = theta*Re + theta*A + 2.*A
#Theoretical Minimum Latency
theoreticalMinimumLatency = D/299792.458
print('The Theoretical Minimum Latency is: ' + str(theoreticalMinimumLatency))


def theoreticalMinimumLatency_circularOrbits(A,theta):
    """
    Args:
        A (float) - Altitude of orbit above the surface of the Earth in km
        theta (float) - Angular distance between source and sink in radians
    Returns:
        theoreticalMinimumLatency (float) - the theoretical minimum latency in seconds
    """
    D = theta*6371. + theta*A + 2.*A
    return D/299792.458

orbitalRadii = np.logspace(start=np.log10(400),stop=np.log10(35786.-6371.),num=300)
angularDistances = np.linspace(start=0.,stop=np.pi*0.99,num=300)
theoreticalMinimumLat = np.zeros((len(orbitalRadii),len(angularDistances)))
for i in np.arange(len(orbitalRadii)):
    theoreticalMinimumLat[i] = theoreticalMinimumLatency_circularOrbits(orbitalRadii[i],angularDistances)

fig = plt.figure(num=654684321)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
#cmap = cm.bwr
bounds = [10**-4,10**-3,10**-2,10**-1,10**0]
#norm = colors.BoundaryNorm(bounds, cmap.N)
clev = np.logspace(start=np.log10(theoreticalMinimumLat.min()),stop=np.log10(theoreticalMinimumLat.max()),num=100) #Adjust the .001 to get finer gradient
plt.contourf(orbitalRadii,angularDistances,theoreticalMinimumLat,clev,cmap='bwr',locator=ticker.LogLocator())
cbar = plt.colorbar(ticks=bounds)
cbar.set_label(label='Theoretical Minimum Communication Latency (s)',weight='bold')
plt.contour(orbitalRadii,angularDistances,theoreticalMinimumLat,locator=ticker.LogLocator(),colors='black')
#cm.ScalarMappable(cmap=cmap, norm=norm),ticks=bounds,
plt.xscale('log')
plt.xlabel('Circular Orbit Altitude (km)', weight='bold')
plt.ylabel('Angular Distance (rad)', weight='bold')
plt.show(block=False)
####




#### Theoretical Minimum Latency with Circular Orbit Points

A = 400 #in km
theta = 1.5*(np.pi)/2.
orbitalRadii2 = np.logspace(start=np.log10(400),stop=np.log10(35786.-6371.),num=100)
angularDistances2 = np.linspace(start=0.,stop=np.pi*0.95,num=100)

def theoreticalMinimumLatency_circularOrbitPoints(A,theta):
    Re = 6371 # in km
    pt1 = np.asarray([Re,0]) #furthest right on circle
    pt2 = np.asarray([Re*np.cos(theta),Re*np.sin(theta)]) #furthest left on circle

    #inputs
    # theta: Angle between 1 and 2
    # theta must not be 180 deg
    pt3 = (pt1+pt2)/2. #midpoint between pt1 and pt2 (part of origin,midpoint,tangent-tangent line)

    #line tangent to pt 1 is just a vertical line passing through (Re,0)
    m = pt3[1]/pt3[0] #slope of line from origin to tangent-tangent intersection point
    pt4 = np.asarray([pt1[0],m*pt1[0]]) #tangent-tangent intersection point, second term is 
    pt4d = np.linalg.norm(pt4) #distance of tangent-tangent intersection point from Earth
    minDist = 0.

    #Case 1 A single spacecraft at this altitude can Handle the Comm
    if pt4d <= Re+A:
        optNumSC = 1
        optSCLocs = np.asarray([[(Re+A)*pt4/pt4d]])
        minDist = 2.*np.linalg.norm(optSCLocs-pt1) #uses symmetry
    elif pt4d > Re+A:
        #1 Test if Two SC is acceptable solution
        pt5 = np.asarray([Re,np.sqrt((Re+A)**2.-Re**2.)])
        mpt6 = -pt2[0]/pt2[1] #slope of the line tangent to pt2
        bpt6 = pt2[1] - mpt6*pt2[0] #y-intercept of the line tangent to pt2
        # a,b,c are the coefficients of the solution to the above line and constellation orbit
        b = 2.*mpt6*bpt6
        a = 1.+mpt6**2.
        c = bpt6**2. - (Re+A)**2.
        #pt6 on constellation circle
        x6 = (-b+np.sqrt(b**2-4*a*c))/(2*a)
        pt6 = np.asarray([x6,mpt6*x6+bpt6])
        pt7y = mpt6*Re+bpt6

        #if pt7y < np.sqrt((Re + A)**2. - Re**2.):
        if np.linalg.norm((pt5+pt6)/2.) > Re: #Checking if the midpoint of two SC on the circular orbit is inside the Earth.
            #If it is not inside the Earth, then we need two spacecraft
            #Then 2 points is an acceptable solution
            optNumSC = 2
            optSCLocs = np.asarray([pt5,pt6]) #these are not yet optimal
            def objfun_twoSC(x,Re,A,theta):
                #Re,A,theta = args
                pt1 = np.asarray([Re,0]) #furthest right on circle
                pt2 = np.asarray([Re*np.cos(theta),Re*np.sin(theta)]) #furthest left on circle
                (x5,y5,x6,y6) = x
                return np.sqrt((x6-pt2[0])**2 + (y6-pt2[1])**2) + np.sqrt((x6-x5)**2+(y6-y5)**2) + np.sqrt((x5-pt1[0])**2+(y5-pt1[1])**2)
            def con_scOnOrbit1(x,Re,A,theta):
                #spacecraft 5 rests onorbit
                return np.sqrt(x[0]**2 + x[1]**2) - (Re+A)
            def con_scOnOrbit2(x,Re,A,theta):
                #spacecraft 6 rests on surface
                return np.sqrt(x[2]**2 + x[3]**2) - (Re+A)
            def con_commGTRe15(x,Re,A,theta): #Comm is not obstructed by Earth
                pt1 = np.asarray([Re,0]) #furthest right on circle
                return np.linalg.norm((np.asarray([x[0],x[1]])+pt1)/2.) - Re
            def con_commGTRe26(x,Re,A,theta): #Comm is not obstructed by Earth
                pt2 = np.asarray([Re*np.cos(theta),Re*np.sin(theta)]) #furthest left on circle
                return np.linalg.norm((np.asarray([x[2],x[3]])+pt2)/2.) - Re
            def con_commGTRe56(x,Re,A,theta): #Comm is not obstructed by Earth
                return np.linalg.norm((np.asarray([x[2],x[3]])+np.asarray([x[0],x[1]]))/2.) - Re  
            x0 = optSCLocs.flatten()
            out = minimize(objfun_twoSC,x0,constraints=[{'type':'eq','fun':con_scOnOrbit1,'args':(Re,A,theta)},{'type':'eq','fun':con_scOnOrbit2,'args':(Re,A,theta)},{'type':'ineq','fun':con_commGTRe15,'args':(Re,A,theta)},
                        {'type':'ineq','fun':con_commGTRe26,'args':(Re,A,theta)},{'type':'ineq','fun':con_commGTRe56,'args':(Re,A,theta)}],args=(Re,A,theta))
            #assert out.success == True, 'Optimization did not finish properly'
            if out.success == False:
                print(saltyburrito)
            optSCLocs = np.asarray([[out.x[0],out.x[1]],[out.x[2],out.x[3]]])
            minDist = out.fun

        else: #If the pt5,pt6 midpoint is inside the Earth, we need more spacecraft
            # Requires More than 2 points
            phi = 2.*np.arccos(Re/(Re+A)) #maximum angle formed by line tangent to Earth
            Gamma = np.arccos(pt6[0]/(Re + A))-np.arccos(pt5[0]/(Re + A)) #angle formed between pt5 and pt6
            numAngles = int(np.ceil(Gamma/phi)) #number of angles
            optNumSC = int(numAngles-1+2)
            assert numAngles > 1, 'Number of angles must be greater than 1 otherwise the previous if should have executed'
            def objfun_nSC(x,Re,A,theta,numAngles):
                pt6gnd = np.sqrt((x[-2]-pt2[0])**2+(x[-1]-pt2[1])**2) 
                pt5gnd = np.sqrt((x[0]-Re)**2+(x[1]-0.)**2)
                others = np.sum([np.sqrt((x[2*(ind+1)]-x[2*ind])**2+(x[2*(ind+1)+1]-x[2*ind+1])**2) for ind in np.arange(numAngles)])
                return pt5gnd + others + pt6gnd
            def con_scOnOrbit(x,Re,A,theta,numAngles,index):
                xs = x[index*2]
                ys = x[index*2+1]
                return xs**2 + ys**2 - (Re+A)**2
            optSCLocs = list()
            optSCLocs.append(pt5)
            for i in np.arange(numAngles-1):
                ang = np.arccos(pt5[0]/np.linalg.norm(pt5)) + (i+1)*Gamma/numAngles
                optSCLocs.append([(Re+A)*np.cos(ang),(Re+A)*np.sin(ang)])
            optSCLocs.append(pt6)
            optSCLocs = np.asarray(optSCLocs)
            x0 = optSCLocs.flatten()
            cons = list()
            cons.append({'type':'eq','fun':con_scOnOrbit,'args':(Re,A,theta,numAngles,0)})
            cons.append({'type':'eq','fun':con_scOnOrbit,'args':(Re,A,theta,numAngles,-1)})
            for i in np.arange(numAngles-1):
                cons.append({'type':'eq','fun':con_scOnOrbit,'args':(Re,A,theta,numAngles,i+1)})
            out = minimize(objfun_nSC,x0,constraints=cons,args=(Re,A,theta,numAngles),options={'maxiter':1000,'ftol':10})
            if out.success == False:
                print(saltyburrito)
            #assert out.success == True, 'Optimization did not finish properly'
            optSCLocs = np.asarray([[out.x[2*i],out.x[2*i+1]] for i in np.arange(int(len(out.x)/2))])
            minDist = out.fun
    return optNumSC, optSCLocs, minDist

theoreticalMinimumLatpoints = np.zeros((len(orbitalRadii2),len(angularDistances2)))
minDists = np.zeros((len(orbitalRadii2),len(angularDistances2)))
for (i,j) in itertools.product(np.arange(len(orbitalRadii2)),np.arange(len(angularDistances2))):
    print(i,j)
    optNumSC, optSCLocs, minDist = theoreticalMinimumLatency_circularOrbitPoints(orbitalRadii2[i],angularDistances2[j])
    minDists[i,j] = minDist
    theoreticalMinimumLatpoints[i,j] = minDists[i,j]/299792.458
    if (i,j) == (2,80):
        saved_optSCLocs = optSCLocs



#### For Debugging
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]

(i,j)=(2,80)
(A,theta) = (orbitalRadii2[i],angularDistances2[j])
fig = plt.figure(num=1)
rad = np.linspace(start=0,stop=np.pi,num=100)
plt.plot(Re*np.cos(rad),Re*np.sin(rad),color='blue',zorder=10)
plt.plot((Re+A)*np.cos(rad),(Re+A)*np.sin(rad),color=colors['coral'],zorder=10)
Re = 6371 # in km
pt1 = np.asarray([Re,0])
pt2 = np.asarray([Re*np.cos(theta),Re*np.sin(theta)])
#inputs
# theta: Angle between 1 and 2
# theta must not be 180 deg
pt3 = (pt1+pt2)/2. #midpoint between pt1 and pt2 (part of origin,midpoint,tangent-tangent line)
#line tangent to pt 1 is just a vertical line passing through (Re,0)
m = pt3[1]/pt3[0] #slope of line from origin to tangent-tangent intersection point
pt4 = np.asarray([pt1[0],m*pt1[0]]) #tangent-tangent intersection point, second term is 
pt4d = np.linalg.norm(pt4) #distance of tangent-tangent intersection point from Earth
minDist = 0.
pt5 = np.asarray([Re,np.sqrt((Re+A)**2.-Re**2.)])
mpt6 = -pt2[0]/pt2[1] #slope of the line tangent to pt2
bpt6 = pt2[1] - mpt6*pt2[0] #y-intercept of the line tangent to pt2
# a,b,c are the coefficients of the solution to the above line and constellation orbit
b = 2.*mpt6*bpt6
a = 1.+mpt6**2.
c = bpt6**2. - (Re+A)**2.
#pt6 on constellation circle
x6 = (-b+np.sqrt(b**2-4*a*c))/(2*a)
pt6 = np.asarray([x6,mpt6*x6+bpt6])
pt7y = mpt6*Re+bpt6
phi = 2.*np.arccos(Re/(Re+A)) #maximum angle formed by line tangent to Earth
Gamma = np.arccos(pt6[0]/(Re + A))-np.arccos(pt5[0]/(Re + A)) #angle formed between pt5 and pt6
numAngles = int(np.ceil(Gamma/phi)) #number of angles
optNumSC = int(numAngles-1+2)
optSCLocs = list()
optSCLocs.append(pt5)
for i in np.arange(numAngles-1):
    ang = np.arccos(pt5[0]/np.linalg.norm(pt5)) + (i+1)*Gamma/numAngles
    optSCLocs.append([(Re+A)*np.cos(ang),(Re+A)*np.sin(ang)])
optSCLocs.append(pt6)
optSCLocs = np.asarray(optSCLocs)
#plt.scatter(pt5[0],pt5[1],color='black')
#plt.scatter(pt6[0],pt6[1],color='red')
plt.scatter(pt1[0],pt1[1],color='purple',marker='s',zorder=12)
plt.scatter(pt2[0],pt2[1],color='purple',marker='s',zorder=12)
for i in np.arange(len(optSCLocs)):
    plt.scatter(optSCLocs[i][0],optSCLocs[i][1],color='black',marker='o',zorder=12)
plt.gca().set_aspect('equal', adjustable='box')

for i in np.arange(saved_optSCLocs.shape[0]-1):
    plt.plot([saved_optSCLocs[i][0],saved_optSCLocs[i+1][0]],[saved_optSCLocs[i][1],saved_optSCLocs[i+1][1]],color='limegreen')

plt.plot([pt1[0],saved_optSCLocs[0][0]],[pt1[1],saved_optSCLocs[0][1]],color='limegreen')
plt.plot([pt2[0],saved_optSCLocs[-1][0]],[pt2[1],saved_optSCLocs[-1][1]],color='limegreen')
# ytmp = np.sqrt((Re+A)**2-Re**2)
# plt.scatter(Re,ytmp,color='red',marker='s')
# plt.plot([Re,Re],[0,ytmp],color='red')
# plt.plot([pt2[0],Re],[pt2[1],ytmp],color='red')


# tmp = np.asarray([5932.81872911, 3337.77611778, 5932.81872908, 3337.77611783])
# plt.scatter([tmp[0],tmp[2]],[tmp[1],tmp[3]],color='green')
# plt.plot([Re,tmp[0]],[0,tmp[1]],color='green')
# plt.plot([pt2[0],tmp[2]],[pt2[1],tmp[3]],color='green')
# plt.plot([tmp[0],tmp[2]],[tmp[1],tmp[3]],color='green')
plt.xlabel('X in AU', weight='bold')
plt.ylabel('Y in AU', weight='bold')
plt.show(block=False)

####


### For Plotting Results
fig = plt.figure(num=9989654684321)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
#cmap = cm.bwr
bounds = [10**-4,10**-3,10**-2,10**-1,10**0]
#norm = colors.BoundaryNorm(bounds, cmap.N)
clev = np.logspace(start=np.log10(theoreticalMinimumLat.min()),stop=np.log10(theoreticalMinimumLat.max()),num=100) #Adjust the .001 to get finer gradient
plt.contourf(orbitalRadii2,angularDistances2,theoreticalMinimumLatpoints,clev,cmap='bwr',locator=ticker.LogLocator())
cbar = plt.colorbar(ticks=bounds)
cbar.set_label(label='Theoretical Minimum Communication Latency (s)',weight='bold')
plt.contour(orbitalRadii2,angularDistances2,theoreticalMinimumLatpoints,locator=ticker.LogLocator(),colors='black')
#cm.ScalarMappable(cmap=cmap, norm=norm),ticks=bounds,
plt.xscale('log')
plt.xlabel('Circular Orbit Altitude (km)', weight='bold')
plt.ylabel('Angular Distance (rad)', weight='bold')
plt.show(block=False)