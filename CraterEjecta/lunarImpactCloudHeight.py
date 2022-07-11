import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp # for multivariable functions



fdir = '/home/dean/Documents/exosims/PrototypePlottingUtilsEXOSIMS/CraterEjecta/'
filename = "Cintala1999Data.csv"

data = list()
with open(os.path.join(fdir,filename), newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        data.append(row)
        #print(', '.join(row))


#Table 1 from Ejection-velocity distributions from impacts into coarse-grained sand
#4202,4201,4198,4203,4204,4206,4207
Rdata = np.asarray([13.9,15.0,15.2,16.1,16.7,17.6,18.2])/100. #crater diameter in m
Udata = np.asarray([0.802,0.9,1.079,1.254,1.399,1.712,1.92])*1000. #projectile velocity in m/s
a = 4.76/2./1000. #impactor radius in m
rho_sand = 1.51*(1/1000.)*(100.)**3. #kg/m3 density of material impacting from Hansen2011
rho_lunar = 1.5*(1/1000.)*(100.)**3. #kg/m3 density of material impacting into https://en.wikipedia.org/wiki/Lunar_soil
delta = 2710 #kg/m3 #https://www.thyssenkrupp-materials.co.uk/density-of-aluminium.html
nu = 0.4 #a constant from Housen2011


xR = list()
xRa = list()
vU = list()
xarhodeltav = list()
#print(data)
#Data From cintala1999
x = list()
angle = list()
v = list()
for i in np.arange(len(data)-3)+3: #Iterate over each row

    for j in np.arange(6):
        if not (data[i][j*3] == ''):
            xval = float(data[i][j*3]) #in x0/R
            angleval = float(data[i][j*3+1]) #in deg
            vval = float(data[i][j*3+2]) #in cm/s
            
            x.append(xval)
            v.append(vval/100.)
            xR.append(xval*Rdata[j])
            xRa.append(xval*Rdata[j]/a)
            angle.append(angleval)
            vU.append(vval/100./Udata[j]) #v/U
            xarhodeltav.append(xval*Rdata[j]/a * ((rho_sand/delta)**nu))

            if xval > 0.9:
                print(i)





#Points of 4th order polynomial model fit in Figure 2 of Durda 2012.
polyPoints = np.asarray([[0.003588932247486437, 52.02076607792816],
    [0.016794934067157347, 52.32597347866487],
    [0.03650923591877553, 52.40227532884905],
    [0.055680111379384334, 52.40227532884905],
    [0.07665870555264329, 52.32597347866487],
    [0.09076500654526007, 52.27510557854209],
    [0.10215702372953325, 52.09706792811234],
    [0.12439848615108137, 51.74099262725283],
    [0.13994680381963123, 51.25774757608636],
    [0.15441254092621756, 51.003408075472436],
    [0.16779227969479607, 50.672766724674325],
    [0.18117144883403383, 50.29125747375343],
    [0.1923797605559375, 49.858880322709744],
    [0.2044929284855125, 49.47737107178885],
    [0.22148838949406172, 49.01955997068378],
    [0.22890268499236163, 48.9432581204996],
    [0.29163995168691137, 47.13744766614069],
    [0.308814845937775, 46.55246681472865],
    [0.3276163167709524, 45.86575016307105],
    [0.34551321621112885, 45.15359956135204],
    [0.3601586713746998, 44.79752426049254],
    [0.4854933332179441, 43.05774278215222],
    [0.5324765566429368, 43.05774278215222],
    [0.5616394078415328, 43.13368815009234],
    [0.5967494702159992, 43.817196461553415],
    [0.6161950563072737, 44.196923301254],
    [0.6507684834390726, 45.18421308447555],
    [0.6718375824911273, 45.86772139593661],
    [0.714518369002245, 47.46257412267911],
    [0.760440228270583, 49.13337221736172],
    [0.817177286240987, 52.095241567026335],
    [0.8658078352841209, 54.525493341110135],
    [0.9079553883607653, 56.727909011373576],
    [0.9349733994927879, 58.17087100223583],
    [0.9657725204328932, 59.68977836103821],
    [0.9900852435983144, 60.67706814425974],
    [0.8879621111501913, 55.66467386021191],
    [0.8404124866608299, 53.310367454068235],
    [0.7890783505551917, 50.57633420822396],
    [0.7388319426197147, 48.525809273840764],
    [0.6939876060969549, 46.627175075337796],
    [0.5097915986985391, 42.753961310391745],
    [0.4363542138497464, 43.43746962185282],
    [0.4628117770817407, 43.05774278215222],
    [0.4088165765160282, 43.817196461553415],
    [0.27709005870813375, 47.614464858559344]])


# def poly(coeffs,x): #returns y value of polynomial
#     return coeffs[4]*x**4 + coeffs[3]*x**3 + coeffs[2]*x**2 + coeffs[1]*x + coeffs[0]

# def erf(coeffs,x,y):
#     errors = np.abs(y - coeffs[4]*x**4 + coeffs[3]*x**3 + coeffs[2]*x**2 + coeffs[1]*x + coeffs[0])
#     errorout = np.sum(errors)
#     #print(errorout)
#     return errorout
#from scipy.optimize import minimize
#coeffs0 = np.asarray([0.00001,0.0001,0.0001,0.0001,45.])
#out = minimize(erf,coeffs0,args=(polyPoints[:,0],polyPoints[:,1]))





model = np.poly1d(np.polyfit(polyPoints[:,0], polyPoints[:,1], 4)) #The angular launch model


plt.figure(1)
plt.scatter(x,angle,s=1) #The data points from Cintala1999
plt.scatter(polyPoints[:,0],polyPoints[:,1],color='red',s=2,label='Poly Model Fit Points')

xpts = np.linspace(start=0.,stop=1.,num=1000)
#ypts = poly(out.x,xpts)
#plt.plot(xpts,ypts,color='red')

plt.plot(xpts,model(xpts),color='black',label='Poly Model Fit')
plt.xlim([0,1])
plt.ylim([0,100])
plt.xlabel(r'$x_0/R$')
plt.ylabel('Elevation Angle (deg)')
plt.show(block=False)


#
deltaAngle = angle - model(x)
std_angle = np.std(deltaAngle)
print("Delta Angle STD: " + str(std_angle))
print("Delta Analge Mean: " + str(np.mean(deltaAngle)))



#Compute Slope of ejection velocity vs position
model2 = np.poly1d(np.polyfit(np.log10(np.asarray(xarhodeltav)), np.log10(vU), 1))
#xspace = np.linspace(start=1,stop=100.,num=1000)
xspace = np.linspace(start=0.,stop=100.,num=1000)
yspace = model2(xspace)

plt.figure(3)
plt.scatter(np.asarray(xarhodeltav),vU,color='black',s=2)
plt.plot(10**xspace,10**yspace,color='red')
plt.ylabel('v/U')
plt.xlabel(r"$x/a (\rho/\delta)^\nu$")
plt.xscale('log')
plt.yscale('log')
plt.xlim([1,100])
plt.ylim([0.0001,0.1])
plt.show(block=False)


#Testing the exponential model for of log10(v/U) = model2[0]*log10(x/a*(rho/delta)**nu) + model2[1]
txspace = np.linspace(start=0.5,stop=2,num=1000)
tyspace = model2[0]*txspace + model2[1]
plt.figure(9999)
plt.scatter(np.log10(np.asarray(xarhodeltav)), np.log10(vU),s=1,color='blue')
#plt.plot(txspace,tyspace,color='red')
plt.plot(txspace,model2(txspace),color='orange')
plt.show(block=False)
print("slope: " + str(model2[0]))
print("offset: " + str(model2[1]))
vUdeltas = model2(np.log10(np.asarray(xarhodeltav))) - np.log10(vU)
std_vU = np.std(vUdeltas)
print("vU std: " + str(std_vU))


# plt.figure(4)
# plt.scatter(xR,vU,color='black')
# plt.ylabel('v/U')
# plt.xlabel(r"$x/R$")
# plt.xscale('log')
# plt.yscale('log')
# #plt.xlim([1,100])
# plt.ylim([0.0001,0.1])
# plt.show(block=False)


# plt.figure(5)
# plt.scatter(xRa,vU,color='black')
# plt.ylabel('v/U')
# plt.xlabel(r"$x/a$")
# plt.xscale('log')
# plt.yscale('log')
# #plt.xlim([1,100])
# plt.ylim([0.0001,0.1])
# plt.show(block=False)



#### Model of v/U vs x/a(rho/delta)^nu
def velFromX(x,m,b,U,a,rho,delta,nu):
    """
    """
    return U*10.**(m*np.log10(x/a*(rho/delta)**nu))*10.**b


def nominal_vU(x,m,b,a,rho,delta,nu):
    """
    m is the slope of the line in 9999
    b is the intersection with 0 in 9999
    """
    return 10.**(m*np.log10(x/a*(rho/delta)**nu))*10.**b

#Given x, a, rho, delta, nu we can extract a nominal v/U with
test_x = 10*a/((rho_sand/delta)**nu)
test_stdComponent = np.random.normal(loc=0.,scale=std_vU)
test_vU = np.log10(nominal_vU(test_x,model2[1],model2[0],a,rho_sand,delta,nu)) + test_stdComponent
test_vU2 = 10.**test_vU

plt.figure(3)
plt.scatter(test_x/a*(rho_sand/delta)**nu,test_vU2,marker='x',color='red')
plt.show(block=False)





#Moon mean orbital velocity #https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
#1.022 km/s
m_moon = 0.07346*10**24. #kg

#### Velocity of Objects Impacting Moon
G = 6.6743 * 10**-11. #m3/kg/s2
m_earth = 5.97219*10**24. #kg
r_moon_earth  = 384400*1000. #Earth to moon distance in meters
r_moon = 1737.4*1000. #radius of moon in km
a_g_atmoon = G*m_earth/(r_moon_earth**2.)
a_g_moonsurface = G*m_moon/(r_moon**2.) #acceleration at the surface of the moon



#Vis Viva #https://en.wikipedia.org/wiki/Hohmann_transfer_orbit
a_orbit = r_moon_earth/2. + 6731*1000
v_atMoon = np.sqrt(G*m_earth*(2./r_moon_earth - 1./a_orbit))

v_atEarth_hohmann = np.sqrt(G*m_earth*(2./(6731*1000) - 1./a_orbit))
v_LEO = np.sqrt(G*m_earth*(2./(6731*1000) - 1./(6731*1000)))
v_moon = np.sqrt(G*m_earth*(2./r_moon_earth - 1./r_moon_earth))
v_intercept = v_moon + v_atMoon
dV_leo_to_moon_hohmann = v_atEarth_hohmann - v_LEO

f9Mass_at_LEO = 22800. #kg to LEO
isp_merlin = 311. #s isp
mf = f9Mass_at_LEO/np.exp(dV_leo_to_moon_hohmann/(isp_merlin*9.81)) #final mass delivered to the moon

#Low Lunar Orbit Velocity
vel_LLO = np.sqrt(a_g_moonsurface*r_moon)


def calc_Ag(d):
    """ Calculate magnitude of acceleration due to gravity
    Args:
        d (float) - distance to center of body in m
    Returns:
        Ag (float) - Acceleration due to gravity in N
    """
    G = 6.67408*10**-11. #m^3/(kg s^2)
    mBody = m_moon
    Ag = G*mBody/d**2.
    return Ag

def model_3D_simple(t,states):
    """3D mortar model
    This model treats the mortar as a point mass
    Args:
        t (float) - time with shape m - here m=2
        states (array) - system states.
    Returns:
        x_dot (array) - derivatives of system states.
    """
    x,xd,y,yd,z,zd = states

    #Acceleration due to gravity component
    r_sc_body = np.asarray([x,y,z])#, x[4]]) #only consider y direction
    rhat = r_sc_body/np.linalg.norm(r_sc_body)
    Ag_sc_body = -calc_Ag(np.linalg.norm(r_sc_body))*rhat #simplification to ensure gravity isnt the issue

    #Acceleration due to drag
    rdot_sc_body = np.asarray([xd,yd,zd]) # velocity of the mortar relative to the body
    rdot_sc_bodysurface = rdot_sc_body # need to subtract planetary rotation from this. STANDIN VALUE

    #Sum All together
    xdot = np.zeros(6)
    xdot[0] = xd #xd
    xdot[1] = Ag_sc_body[0]#(F_cable[0])/(simProp.mMortar) #xdd gravity is not in x dir
    xdot[2] = yd #yd
    xdot[3] = Ag_sc_body[1]#-calc_Ag(y)#Ag_sc_body[0] #ydd
    xdot[4] = zd #zd
    xdot[5] = Ag_sc_body[2]#(2])/(simProp.mMortar) #zdd
    return xdot

R = 1738.1*1000. #https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
#http://pageperso.lif.univ-mrs.fr/~francois.denis/IAAM1/scipy-html-1.0.0/generated/scipy.integrate.solve_ivp.html
def event_hit_ground_3D_simple(t, y):
    """For 2D mortar point mass
    Note: this will only terminate when the sign switches from positive to negative (the object may start
    at a negative altitude)
    """
    r = np.asarray([y[0],y[2],y[4]])
    #rnorm = np.linalg.norm(r) # probe altitude off surface
    #lat = np.arcsin(y[2]/rnorm) #latitude in rad # elevation divided by surface vector magnitude
    #lon = np.arccos(r[0]/rnorm)-simProp.RR*t # x component divided by x-y place vector magnitude + account for planet rotation rate
    #surface_alt = plotMarsContour.altitude_from_lonlat(lon*180./np.pi, lat*180./np.pi, plotMarsContour.data) + simProp.R # calculation and data from plotMarsContour.py
    #print('rnorm: ' + str(rnorm) + ' Surface Alt: ' + str(surface_alt))
    #return 0 if rnorm < surface_alt else 1
    #return y[2] - R# - simProp.endY # this must be greater than 0 for the simulation to continue
    return np.linalg.norm(r) - r_moon
event_hit_ground_3D_simple.terminal = True
event_hit_ground_3D_simple.direction = -1. #causes termination when switching from positve to negative

#Sample Simulation
x0 = np.asarray([0.,0.,R,1000.,0.,1000.])
t_span = (0., 40.*60.) # interval of times in seconds
res = solve_ivp(model_3D_simple, t_span, x0, method='RK45', dense_output=True,\
                    events=[event_hit_ground_3D_simple],max_step=2.)#, options={'first_step':1.0})#t_eval=t_eval,\


outxyz = res.y

fig2 = plt.figure(num=8888,figsize=(8,8))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot(res.y[0],res.y[2],res.y[4])

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
plt.show(block=False)

plt.figure(2304857290384523)
plt.plot(res.y[2])
plt.show(block=False)






# fig2 = plt.figure(num=8888,figsize=(8,8))
# ax2 = fig2.add_subplot(111, projection='3d')
# ax2.plot(res.y[0],res.y[2],res.y[4])

# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax2.set_zlabel('z')
# plt.show(block=False)

plt.figure(2304857290384523)
plt.plot(res.y[2])
plt.show(block=False)








#### Lunar Particle Size Distribution
#Data taken from https://www.researchgate.net/publication/228342547_Characterization_of_Lunar_Dust_for_Toxicological_Studies_I_Particle_Size_Distribution/figures?lo=1
# Park J., Liu Y., Kihm K., Taylor L., Characterization of Lunar Dust for Toxicological Studies. I: Particle Size Distribution
lunarParticleDensityFunction = np.asarray([[0.019267024800402793, -0.0005462597216310261],
    [0.023911293974882227, 0.028708334356857046],
    [0.029977495537539048, 0.11510737335855303],
    [0.03777174547751707, 0.250208496845466],
    [0.0477259208158262, 0.2969983702815382],
    [0.06030487994538844, 0.3340479878997331],
    [0.07578716177230624, 0.4795383148944672],
    [0.09552573386591155, 0.47762650654323724],
    [0.12066201355605057, 0.6471436032845653],
    [0.1509144328089252, 0.6575682442089021],
    [0.1901901694467147, 0.7160460219285125],
    [0.23789536845568912, 0.6927044426842073],
    [0.3013455493177244, 0.7622219823932419],
    [0.37987653337450406, 0.7129075957283417],
    [0.47516611704070805, 0.6850205637690272],
    [0.6052230750802288, 0.6136298747085979],
    [0.756963537027727, 0.6247038660207931],
    [0.9520006942811188, 0.4844800223577883],
    [1.1939880272898589, 0.41828172021273735],
    [1.5131545672430442, 0.3046824505456759],
    [1.8879401875852742, 0.2599119058041254],
    [2.3863629799458534, 0.16254599313561147],
    [3.000540175442676, 0.11128315260922061],
    [3.83136686965094, 0.06911363370033885],
    [4.780059396603559, 0.047070352533835624],
    [6.025104175008473, 0.039314390691879364],
    [7.57507414566916, 0.024415171885564524],
    [9.548206031454747, 0.013412458104315572],
    [12.00434054681832, 0.0037080424008688606],
    [15.014574187899655, 0.0011456755680356867],
    [18.97378399095515, -0.00011637969742062104],
    [24.287231934339623, -0.00007772069758549272]])

lunarParticleDensityFunction[np.where(lunarParticleDensityFunction<0)[0]] = 0


#### cumulative mass ejected faster than v
def cumMassEjectedFasterThanV(v,m,U,rho,delta,nu,mu=0.41):
    return m*(v/U)*(rho/delta)**((3.*nu-1.)/(3.*mu))

#def M_crater():
#    return (3.*k)/(4.*np.pi)*rho/delta*((x/a)**3)

# def R_crater(M_crater,rho,k_crater=0.3):
#     R = (M_crater/k_crater)**(1./3.)
#     return R
# def R_crater(m,rho,delta,U,H2,Y=0.04*10**6.,nu=0.4,mu=0.41):
#     """ Y 0.04 MPa is the strength of small glass beads
#     """
#     R = (H2 * ((rho)/(delta))**((1.-3.*nu)/3.) * (Y/(rho*U**2.))**(-mu/2.)) / ((rho/m)**(1./3.))
#     return R

def R_crater1(m,rho,delta,U,H1,g=a_g_moonsurface,nu=0.4,mu=0.41):
    """ Y 0.04 MPa is the strength of small glass beads
    """
    R = H1 * (rho/delta)**((2.+mu-6.*nu)/(6.+3.*mu)) * (g*a/U**2.)**(-mu/(2.+mu)) * (rho/m)**(-1./3.)
    return R

#High Mass Impact
#### 0 Setup
U = v_intercept #m/s impact velocity polar hohman transfer directly from LEO into the moon using vis viva equation *assumes the moon is not moving (moon moves at nominally 1km/s)
m = mf
delta = 11.34 * (1/1000.)*(100.)**3. #g/cc impactor density #https://www.nuclear-power.com/nuclear-engineering/thermodynamics/thermodynamic-properties/what-is-density-physics/densest-materials-on-the-earth/
rho = rho_sand
nu = 0.4 #from Housen2011
Y = 0.04*10**6. #strength of lunar surface
mu = 0.41 # for sand
k = 0.3 #for sand
H2 = 0.4 #for SFA a sand/fly ash
H1 = 0.59 #sand
k_crater = 0.4 #from housen 2011
a = (m/delta/(4./3.*np.pi))**(1./3.) #

#### 1 generate total mass ejected by integrating cumMassEjectedFasterThanV from 0 to inf (or some large number)
#R = R_crater(m,rho,delta,U,H2,Y,nu,mu)
R = R_crater1(m,rho,delta,U,H1,g=a_g_moonsurface,nu=0.4,mu=0.41)
M_crater = k_crater*rho*R**3.

#2 Assume hemisphere crater

angles = np.linspace(start=0.,stop=2.*np.pi, endpoint=True, num=50)
#radii = np.linspace(start=1e-1,stop=R,num=10)
radii = np.logspace(start=np.log10(1.1*a),stop=np.log10(R),num=50,base=10.0)
resStorage = np.zeros(())

fig2 = plt.figure(num=8888,figsize=(8,8))
ax2 = fig2.add_subplot(111, projection='3d')
#for i in np.arange(len(angles)):
i=0
reses = list()
for j in np.arange(len(radii)):
    #Get Launch Angle
    LaunchAngle = model(radii[j]/R) + np.random.normal(loc=0.,scale=std_angle) #initial random launch angle
    LaunchVelocity_stdComponent = np.random.normal(loc=0.,scale=std_vU)
    LaunchVelocity_vU = np.log10(nominal_vU(radii[j],model2[1],model2[0],a,rho_sand,delta,nu)) + test_stdComponent
    LaunchVelocity_vU2 = 10.**LaunchVelocity_vU
    LaunchVelocity = LaunchVelocity_vU2*U
    print(LaunchVelocity)
    if LaunchVelocity > 2.38*1000.: #2.38km/s is escape velocity for moon
        LaunchVelocity = 2.2*1000.
    #LaunchVelocity = cumMassEjectedFasterThanV(v,m,U,rho,delta,nu,mu=0.41)

    r_unit = np.asarray([np.cos(angles[i]),0.,np.sin(angles[i])])
    r_impact = np.asarray([0.,r_moon,0.])
    r0 = r_impact + radii[j]*r_unit
    v0 = np.asarray([r_unit[0]*LaunchVelocity*np.cos(LaunchAngle*np.pi/180.),LaunchVelocity*np.sin(LaunchAngle*np.pi/180.),r_unit[2]*LaunchVelocity*np.cos(LaunchAngle*np.pi/180.)])

    

    x0 = np.asarray([0.,0.,R,1000.,0.,1000.])
    x0 = np.asarray([r0[0],v0[0],r0[1],v0[1],r0[2],v0[2]])
    #t_span = (0., 100.*60.) # interval of times in seconds
    t_span = (0.,1000.*60.)
    res = solve_ivp(model_3D_simple, t_span, x0, method='RK45', dense_output=True,\
                        events=[event_hit_ground_3D_simple],max_step=2.)#, options={'first_step':1.0})#t_eval=t_eval,\
    reses.append(res)

    ax2.plot(res.y[0],res.y[2],res.y[4])

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
plt.show(block=False)



##### Radii Where half the mass
R_half_mass = R/(2.)**(1./3.)

maxalt = 0.


import matplotlib.colors as mcol
import matplotlib.cm as cm
import matplotlib as mpl
cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
cpick = cm.ScalarMappable(cmap=cm1)
plt.figure(num=10980979087697)
for i in np.arange(len(reses)):
    #for j in np.arange(reses[i].y.shape[1]):
    alts = np.linalg.norm(np.asarray([reses[i].y[0],reses[i].y[2],reses[i].y[4]]),axis=0)-r_moon
    #alt = np.sqrt(reses[i].y[0,j]**2. + reses[i].y[2,j]**2. + reses[i].y[4,j]**2.) - r_moon
    if np.max(alts) > maxalt:
        maxalt = np.max(alts)
    plt.plot(reses[i].t/60.,alts/1000.,color=(1.-radii[i]/R,0.,radii[i]/R))#,color='black')
    plt.show(block=False)
    print(i)

#norm = mcol.Normalize(vmin=5, vmax=10)
#cb1 = mpl.colorbar.ColorbarBase(plt.gca(), cmap=cmap, norm=norm,
#                                orientation='horizontal')
#cb1.set_label('Some Units')

plt.colorbar(cpick,label=r'$x_0/R$')
#plt.colorbar(label=r'$x_0/R$')
plt.ylabel('Altitude (km)')
plt.xlabel('Time (min)')
plt.show(block=False)



### Downrange Distance vs Time
maxDist = 0.
plt.figure(1230980979896)
for i in np.arange(len(reses)):
    #for j in np.arange(reses[i].y.shape[1]):
    dist = np.linalg.norm(np.asarray([reses[i].y[0]-reses[i].y[0,0],reses[i].y[2]-reses[i].y[2,0],reses[i].y[4]-reses[i].y[4,0]]),axis=0)
    #alt = np.sqrt(reses[i].y[0,j]**2. + reses[i].y[2,j]**2. + reses[i].y[4,j]**2.) - r_moon
    if np.max(dist) > maxDist:
        maxDist = np.max(dist)
    plt.plot(reses[i].t/60.,dist/1000.,color=(1.-radii[i]/R,0.,radii[i]/R))#,color='black')
    print(i)
plt.xlabel("Time (min)")
plt.ylabel("Distance From Impact (km)")
plt.show(block=False)



