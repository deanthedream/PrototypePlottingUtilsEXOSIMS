import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt

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
a, e, p, Rp = PPop.gen_plan_params(n)
a = a.to('AU').value

####
dmajorp, dminorp, Psi, psi = projected_apbpPsipsi(a,e,W,w,inc)
O = projected_Op(a,e,W,w,inc)
theta = projected_BpAngle(a,e,W,w,inc)
c_3D_projected = projected_projectedLinearEccentricity(a,e,W,w,inc)

# Checks
assert np.all(dmajorp < a), "Not all Semi-major axis of the projected ellipse are less than the original 3D ellipse"
assert np.all(dminorp < dmajorp), "All projected Semi-minor axes are less than all projected semi-major axes"


#### Plotting Projected Ellipse
import numpy.random as random



### DELETE From thetas
# eqnr = a[ind]*(1-e[ind]**2)/(1+e[ind]*sp.cos(vs))
# eqnX = eqnr*(sp.cos(W[ind])*sp.cos(w[ind]+vs) - sp.sin(W[ind])*sp.sin(w[ind]+vs)*sp.cos(inc[ind]))
# eqnY = eqnr*(sp.sin(W[ind])*sp.cos(w[ind]+vs) + sp.cos(W[ind])*sp.sin(w[ind]+vs)*sp.cos(inc[ind]))
# eqnZ = eqnr*(sp.sin(inc[ind])*sp.sin(w[ind]+vs))
# eqnr = a[ind]*(1-e[ind]**2)/(1+e[ind])
# eqnX = eqnr*(np.cos(W[ind])*np.cos(w[ind]) - np.sin(W[ind])*np.sin(w[ind])*np.cos(inc[ind]))
# eqnY = eqnr*(np.sin(W[ind])*np.cos(w[ind]) + np.cos(W[ind])*np.sin(w[ind])*np.cos(inc[ind]))
# eqnZ = eqnr*(np.sin(inc[ind])*np.sin(w[ind]))
# rhat = np.asarray([[eqnX],[eqnY],[eqnZ]])/np.linalg.norm(np.asarray([[eqnX],[eqnY],[eqnZ]]), ord=2, axis=0, keepdims=True)
# theta =  np.arctan(rhat[1],rhat[0])
###

ind = random.randint(low=0,high=n)#2

plt.close(877)
fig = plt.figure(num=877)
ca = plt.gca()
ca.axis('equal')
## Central Sun
plt.scatter([0],[0],color='orange')
## 3D Ellipse
vs = np.linspace(start=0,stop=2*np.pi,num=300)
r = xyz_3Dellipse(a[ind],e[ind],W[ind],w[ind],inc[ind],vs)
x_3Dellipse = r[0,0,:]
y_3Dellipse = r[1,0,:]
plt.plot(x_3Dellipse,y_3Dellipse,color='black')

#3D Ellipse Center
Op = projected_Op(a,e,W,w,inc)
plt.scatter(Op[0][ind],Op[1][ind],color='black')
print('a: ' + str(np.round(a[ind],2)) + ' e: ' + str(np.round(e[ind],2)) + ' W: ' + str(np.round(W[ind],2)) + ' w: ' + str(np.round(w[ind],2)) + ' i: ' + str(np.round(inc[ind],2)) +\
     ' Psi: ' + str(np.round(Psi[ind],2)) + ' psi: ' + str(np.round(psi[ind],2)) + ' theta: ' + str(np.round(theta[ind],2)))
#print(dmajorp[ind]*np.cos(theta[ind]))#print(dmajorp[ind]*np.cos(theta[ind]))#print(dminorp[ind]*np.cos(theta[ind]+np.pi/2))#print(dminorp[ind]*np.sin(theta[ind]+np.pi/2))
#++
ang = Psi[ind]/2+psi[ind]
dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang)
dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang)
dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang+np.pi)
dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang+np.pi)
plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='blue',linestyle='-.')
plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='blue',linestyle='-.')
dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang+np.pi/2)
dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang+np.pi/2)
dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang-np.pi/2)
dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang-np.pi/2)
plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='blue',linestyle='-.')
plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='blue',linestyle='-.')
#a: 6.097704110164751 e: 0.36553661863447884 W: 4.266191939230118 w: 0.8802451672239122 i: 1.3946387796309012 Psi: 1.6850746074297651 theta: -0.564260076376564
#a: 3.9228410421458193 e: 0.3588279435807127 W: 2.9539995570747117 w: 4.629240940200019 i: 1.5585092161872263 Psi: 2.9634883254324995 theta: -0.0034592609918608725
#a: 0.14 e: 0.25 W: 1.79 w: 5.56 i: 0.37 Psi: 1.03 psi: 1.05 theta: 0.71

#--
ang2 = -Psi[ind]/2-psi[ind] #-(Psi[ind]/2-psi[ind]+np.pi)#Psi[ind]/2-psi[ind]
dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='red',linestyle='-.')
plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='red',linestyle='-.')
dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='red',linestyle='-.')
plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='red',linestyle='-.')
#a: 1.7168557322511873 e: 0.15097872093230147 W: 4.35207664182757 w: 2.0114757894338244 i: 1.4581301267904243 Psi: 2.251074008822032 theta: 0.34848476093192177
#a: 0.2473100405968065 e: 0.2786294655014428 W: 0.2785594333623774 w: 5.996993094854193 i: 1.1585321049317279 Psi: 0.5431460616655733 theta: 0.15380855742377286
#a: 2.0201747823741223 e: 0.3439775778804735 W: 4.277060608690321 w: 2.3006138270099625 i: 1.225596172950174 Psi: 1.6028361975845478 theta: 0.46220249281949455
#a: 0.14 e: 0.25 W: 1.79 w: 5.56 i: 0.37 Psi: 1.03 psi: 1.05 theta: 0.71

#+-
ang2 = Psi[ind]/2-psi[ind]#-(Psi[ind]/2-psi[ind]+np.pi)#Psi[ind]/2-psi[ind]
dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='cyan',linestyle='-')
plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='cyan',linestyle='-')
dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='cyan',linestyle='-')
plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='cyan',linestyle='-')
#a: 1.32 e: 0.35 W: 2.91 w: 0.24 i: 1.43 Psi: 0.45 psi: 0.45 theta: 0.19
#a: 0.34 e: 0.18 W: 3.96 w: 1.11 i: 1.07 Psi: 2.2 psi: 0.27 theta: -0.55
#a: 9.99 e: 0.23 W: 3.57 w: 0.91 i: 1.38 Psi: 1.8 psi: 0.47 theta: -0.37
#a: 1.42 e: 0.05 W: 5.72 w: 4.63 i: 0.72 Psi: 2.98 psi: 2.06 theta: -0.53

#-+
ang2 = -Psi[ind]/2+psi[ind]#-(Psi[ind]/2-psi[ind]+np.pi)#Psi[ind]/2-psi[ind]
dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='orange',linestyle='-')
plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='orange',linestyle='-')
dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='orange',linestyle='-')
plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='orange',linestyle='-')
#a: 3.206311303449458 e: 0.40195688119098727 W: 6.250887614928386 w: 2.2309377840245026 i: 1.1329266513869831 Psi: 1.696028781147292 theta: 0.3407282252669677
#a: 2.431569003923111 e: 0.18301448021027053 W: 1.5194248555533574 w: 2.1468861274839703 i: 1.3812975181405962 Psi: 1.9726351205409203 theta: -0.4919682982679808
#a: 4.61 e: 0.3 W: 2.65 w: 2.86 i: 0.92 Psi: 0.51 psi: 2.89 theta: -0.54
#a: 5.14 e: 0.38 W: 3.03 w: 5.4 i: 0.82 Psi: 1.54 psi: 0.56 theta: 0.54

plt.show(block=False)







# #worked in some cases ++/2
# dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(theta[ind]+Psi[ind]/2)
# dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(theta[ind]+Psi[ind]/2)
# dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(theta[ind]+np.pi+Psi[ind]/2)
# dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(theta[ind]+np.pi+Psi[ind]/2)
# plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='blue',linestyle='-')
# plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='blue',linestyle='-')
# dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(theta[ind]+np.pi/2+Psi[ind]/2)
# dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(theta[ind]+np.pi/2+Psi[ind]/2)
# dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(theta[ind]-np.pi/2+Psi[ind]/2)
# dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(theta[ind]-np.pi/2+Psi[ind]/2)
# plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='blue',linestyle='-')
# plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='blue',linestyle='-')
# #works for 
# #a: 0.3287338684024971 e: 0.22129849665590695W: 0.18412225028679155 w: 3.943660453131082 i: 0.8219836315395734
# #might work for
# #a: 0.2173228490751788 e: 0.4119212201801016W: 0.39619884023500146 w: 6.260702350195606 i: 0.6306936905963281

# #Worked in some cases -+/2
# dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(-(theta[ind]+Psi[ind]/2))
# dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(-(theta[ind]+Psi[ind]/2))
# dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(-(theta[ind]+np.pi+Psi[ind]/2))
# dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(-(theta[ind]+np.pi+Psi[ind]/2))
# plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='blue',linestyle='--')
# plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='blue',linestyle='--')
# dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(-(theta[ind]+np.pi/2+Psi[ind]/2))
# dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(-(theta[ind]+np.pi/2+Psi[ind]/2))
# dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(-(theta[ind]-np.pi/2+Psi[ind]/2))
# dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(-(theta[ind]-np.pi/2+Psi[ind]/2))
# plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='blue',linestyle='--')
# plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='blue',linestyle='--')
# #works for
# #a: 11.28214182804874 e: 0.08799724094434723W: 1.8895826156662348 w: 5.82230236430558 i: 0.6764296750680172
# #a: 1.0051017927275339 e: 0.45215311799322744W: 2.473129779337253 w: 0.3183485339135861 i: 0.9184975133671696
# #a: 0.18271163252812664 e: 0.34052581758474715W: 6.071943457806009 w: 3.338807591306464 i: 0.8366086555597616
# #a: 0.5582225292470047 e: 0.34894375278812206W: 4.9760567969645795 w: 4.00058819853595 i: 0.5383088698012133

# # +-/2
# dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(theta[ind]-Psi[ind]/2)
# dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(theta[ind]-Psi[ind]/2)
# dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(theta[ind]+np.pi-Psi[ind]/2)
# dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(theta[ind]+np.pi-Psi[ind]/2)
# plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='black',linestyle='-')
# plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='black',linestyle='-')
# dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(theta[ind]+np.pi/2-Psi[ind]/2)
# dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(theta[ind]+np.pi/2-Psi[ind]/2)
# dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(theta[ind]-np.pi/2-Psi[ind]/2)
# dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(theta[ind]-np.pi/2-Psi[ind]/2)
# plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='black',linestyle='-')
# plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='black',linestyle='-')


# #trying something --/2
# dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(-(theta[ind]-Psi[ind]/2))
# dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(-(theta[ind]-Psi[ind]/2))
# dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(-(theta[ind]+np.pi-Psi[ind]/2))
# dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(-(theta[ind]+np.pi-Psi[ind]/2))
# plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='orange',linestyle='--')
# plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='orange',linestyle='--')
# dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(-(theta[ind]+np.pi/2-Psi[ind]/2))
# dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(-(theta[ind]+np.pi/2-Psi[ind]/2))
# dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(-(theta[ind]-np.pi/2-Psi[ind]/2))
# dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(-(theta[ind]-np.pi/2-Psi[ind]/2))
# plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='orange',linestyle='--')
# plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='orange',linestyle='--')
# #a: 0.9154353802679572 e: 0.348475336139194W: 3.9385723669700967 w: 5.795319869799132 i: 0.5202491927004314

# #trying something --
# dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(-(theta[ind]-Psi[ind]))
# dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(-(theta[ind]-Psi[ind]))
# dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(-(theta[ind]+np.pi-Psi[ind]))
# dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(-(theta[ind]+np.pi-Psi[ind]))
# plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='orange',linestyle='-')
# plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='orange',linestyle='-')
# dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(-(theta[ind]+np.pi/2-Psi[ind]))
# dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(-(theta[ind]+np.pi/2-Psi[ind]))
# dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(-(theta[ind]-np.pi/2-Psi[ind]))
# dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(-(theta[ind]-np.pi/2-Psi[ind]))
# plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='orange',linestyle='-')
# plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='orange',linestyle='-')
# #nearly there
# #a: 9.133357208066457 e: 0.26795552473163003W: 2.487778842812044 w: 4.936607730927378 i: 1.5327105477663567

# #trying something +-
# dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos((theta[ind]-Psi[ind]))
# dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin((theta[ind]-Psi[ind]))
# dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos((theta[ind]+np.pi-Psi[ind]))
# dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin((theta[ind]+np.pi-Psi[ind]))
# plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='purple',linestyle='-')
# plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='purple',linestyle='-')
# dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos((theta[ind]+np.pi/2-Psi[ind]))
# dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin((theta[ind]+np.pi/2-Psi[ind]))
# dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos((theta[ind]-np.pi/2-Psi[ind]))
# dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin((theta[ind]-np.pi/2-Psi[ind]))
# plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='purple',linestyle='-')
# plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='purple',linestyle='-')

# #trying something -+
# dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(-(theta[ind]+Psi[ind]))
# dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(-(theta[ind]+Psi[ind]))
# dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(-(theta[ind]+np.pi+Psi[ind]))
# dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(-(theta[ind]+np.pi+Psi[ind]))
# plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='green',linestyle='-')
# plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='green',linestyle='-')
# dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(-(theta[ind]+np.pi/2+Psi[ind]))
# dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(-(theta[ind]+np.pi/2+Psi[ind]))
# dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(-(theta[ind]-np.pi/2+Psi[ind]))
# dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(-(theta[ind]-np.pi/2+Psi[ind]))
# plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='green',linestyle='-')
# plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='green',linestyle='-')

# #trying something ++
# dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos((theta[ind]+Psi[ind]))
# dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin((theta[ind]+Psi[ind]))
# dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos((theta[ind]+np.pi+Psi[ind]))
# dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin((theta[ind]+np.pi+Psi[ind]))
# plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='purple',linestyle='--')
# plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='purple',linestyle='--')
# dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos((theta[ind]+np.pi/2+Psi[ind]))
# dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin((theta[ind]+np.pi/2+Psi[ind]))
# dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos((theta[ind]-np.pi/2+Psi[ind]))
# dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin((theta[ind]-np.pi/2+Psi[ind]))
# plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='purple',linestyle='--')
# plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='purple',linestyle='--')

# dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(Psi[ind]+theta[ind]/2)
# dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(Psi[ind]+theta[ind]/2)
# dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(Psi[ind]+theta[ind]/2+np.pi)
# dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(Psi[ind]+theta[ind]/2+np.pi)
# plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='cyan',linestyle='--')
# plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='cyan',linestyle='--')
# dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(Psi[ind]+theta[ind]/2+np.pi/2)
# dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(Psi[ind]+theta[ind]/2+np.pi/2)
# dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(Psi[ind]+theta[ind]/2-np.pi/2)
# dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(Psi[ind]+theta[ind]/2-np.pi/2)
# plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='cyan',linestyle='--')
# plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='cyan',linestyle='--')


# # #Doing Reflections
# #trying something
# ang = 0.72
# dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang)
# dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang)
# dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang+np.pi)
# dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang+np.pi)
# plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='cyan',linestyle='-')
# plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='cyan',linestyle='-')
# dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang+np.pi/2)
# dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang+np.pi/2)
# dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang-np.pi/2)
# dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang-np.pi/2)
# plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='cyan',linestyle='-')
# plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='cyan',linestyle='-')


plt.show(block=False)


#ang=0.175
#a: 0.5162796543574061 e: 0.09038294953381557 W: 3.298421803279386 w: 4.457845246651879 i: 0.9098694117779238Psi: 2.628057456087888 theta: 0.5593952272451591

# ang=0.72
# a: 0.8425647998012417 e: 0.36455563297800353 W: 3.8640586967420463 w: 2.3927026770472732 i: 1.3750264231153055 Psi: 1.421405411764036 theta: 0.3674821766278237


#print('Psi: ' + str(Psi[ind]) + ' theta: ' + str(theta[ind]))




