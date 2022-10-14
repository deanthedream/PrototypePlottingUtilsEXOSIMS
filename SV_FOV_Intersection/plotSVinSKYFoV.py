from sgp4.api import Satrec
import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import jday
import json
import datetime
import time
#from skyfield.api import load, EarthSatellite



with open('ALLSPACETRACKDATA.json') as f:
    data = json.load(f)



lines = list()
with open('3le') as f:
    lines = f.readlines()

n = len(lines)


#data ids
data_ids = list()
for i in np.arange(len(data)):
    try:
        data_ids.append(int(data[i]['NORAD_CAT_ID']))
    except:
        data_ids.append([])
data_ids = np.asarray(data_ids)

#Returns False if it would be filtered
def filter(data,data_ids,satnum):
    ind = np.where(data_ids==satnum)[0]
    #print(ind)

    #Must be a valid satellite id
    if ind.size == 0:
        return False

    #satnum is analyst object
    if satnum > 80000:
        return False

    #Remove those that decayed before today
    if data[ind[0]]['DECAY'] == None:
        (0) #do nothing
        #return False
    elif int(data[ind[0]]['DECAY'][:4]) < datetime.date.today().year-1:
        return False

    #Remove non SV
    if data[ind[0]]['OBJECT_TYPE'] == 'ROCKET BODY':
        return False
    elif  data[ind[0]]['OBJECT_TYPE'] == 'DEBRIS':
        return False

    return True



sats0 = list()
for i in np.arange(n/3,dtype=int):
    #print(str(i) + ' of ' + str(n/3))
    sat = Satrec.twoline2rv(lines[int(i)*3+1], lines[int(i)*3+2])
    try:
        int(sat.satnum)
    except:
        continue
    if filter(data,data_ids,int(sat.satnum)): #returns
        sats0.append(sat)

#Number of Satellites
jd, fr = jday(2022, 10, 3, 12, 0, 0)
rs = list()
sats = list()
for i in np.arange(len(sats0),dtype=int):
    e, r, v = sats0[int(i)].sgp4_array(np.asarray([jd],dtype=float), np.asarray([fr],dtype=float))
    #If the sv has nan entries for some reason
    if np.any(np.isnan(r[0])):
        continue
    rs.append(r[0])
    sats.append(sats0[int(i)])
N = len(sats)
print('N: ' + str(N))


#https://stackoverflow.com/questions/48265646/rotation-of-a-vector-python
def rotve(v,erot,angle):
    rotmeasure=np.linalg.norm(erot)
    erot=erot/rotmeasure;
    norme=np.dot(v,erot)
    vplane=v-norme*erot
    plnorm=np.linalg.norm(vplane)
    ep=vplane/plnorm
    eo=np.cross(erot,ep)
    vrot=(np.cos(angle)*ep+np.sin(angle)*eo)*plnorm+norme*erot
    return(vrot)


fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},num=1)


for j in np.arange(20):
    MM = 0#np.floor(j/2) #the whole number of minutes
    SS = j*30
    #             yyyy  MM dd  HH MM SS
    jd, fr = jday(2022, 10, 3, 12, MM, SS)
    print("jd: " + str(jd) + " MM: " + str(MM) + " SS: " + str(SS))


    #look vectors
    lat = 35.58*np.pi/180. #
    lon = 5.29*np.pi/180.
    earthRad = 6371 #m
    r_loc = earthRad*np.asarray([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]) #forms z vector
    #convert from ECEF to ECI https://space.stackexchange.com/questions/43187/is-this-commonly-attributed-eci-earth-centered-inertial-to-ecef-earth-centere
    theta_g = 280.46 + 360.9856123035484*(jd - 2451545.0) #J2000 Earth state #https://space.stackexchange.com/questions/38807/transform-eci-to-ecef
    r_loc = np.matmul([[np.cos(theta_g),-np.sin(theta_g),0.],[np.sin(theta_g),np.cos(theta_g),0.],[0.,0.,1.]],r_loc)
    #rotve(v,erot,angle)
    #r_loc = np.asarray([0,0,0])
    r_look = r_loc/np.linalg.norm(r_loc)#np.asarray([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])

    x_image = np.cross(r_look,np.asarray([0,0,1]))/np.linalg.norm(np.cross(r_look,np.asarray([0,0,1])))
    #y_image = np.cross(x_image,r_look)/np.linalg.norm(np.cross(x_image,r_look))

    r_look = rotve(r_look, x_image, -35.*np.pi/180.)
    #x_image = np.cross(r_look,np.asarray([0,0,1]))/np.linalg.norm(np.cross(r_look,np.asarray([0,0,1])))
    y_image = np.cross(x_image,r_look)/np.linalg.norm(np.cross(x_image,r_look))


    rs = list()
    for i in np.arange(N,dtype=int):
        e, r, v = sats[int(i)].sgp4_array(np.asarray([jd],dtype=float), np.asarray([fr],dtype=float))
        rs.append(r[0])


    FoVs = [5.,10.,15.] #angles to put tickmarks at
    radialAngles = list()
    #inds_in_FoV = list()
    azimuthAngles = list()
    for i in np.arange(N,dtype=int):
        radial_angle = np.arccos(np.dot(r_look,(rs[i]-r_loc)/np.linalg.norm(rs[i]-r_loc)))
        radialAngles.append(radial_angle)
        azimuth_angle = np.arctan2(np.dot(y_image,(rs[i]-r_loc)/np.linalg.norm(rs[i]-r_loc)),np.dot(x_image,(rs[i]-r_loc)/np.linalg.norm(rs[i]-r_loc)))
        azimuthAngles.append(azimuth_angle)

        # if radial_angle < np.max(FoVs)*np.pi/180.:
        #     inds_in_FoV.append(i)


    
    #plt.ion()


   
    ax.scatter(np.asarray(azimuthAngles), np.asarray(radialAngles)*180./np.pi,s=1)
    ax.set_rmax(15)
    ax.set_rticks([5,10,15])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    ax.set_title("SV in SKY", va='bottom')
    plt.show(block=False)
    
    
    #fig.clf()
    #fig.canvas.draw()
    #fig.canvas.flush_events()
    #plt.show(block=False)
    plt.pause(0.05)
    ax.cla()
    #time.sleep(0.1)
#plt.ioff()
#plt.show(block=False)
print("Done Polar Fig 1")





from scipy.spatial import Voronoi, voronoi_plot_2d, SphericalVoronoi, geometric_slerp
from mpl_toolkits.mplot3d import proj3d

MM = 0#np.floor(j/2) #the whole number of minutes
SS = 0#j*1
#             yyyy  MM dd  HH MM SS
jd, fr = jday(2022, 10, 3, 12, MM, SS)
print("jd: " + str(jd) + " MM: " + str(MM) + " SS: " + str(SS))


#look vectors
lat = 35.58*np.pi/180. #
lon = 5.29*np.pi/180.
earthRad = 6371 #m
r_loc = earthRad*np.asarray([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]) #forms z vector
#convert from ECEF to ECI https://space.stackexchange.com/questions/43187/is-this-commonly-attributed-eci-earth-centered-inertial-to-ecef-earth-centere
theta_g = 280.46 + 360.9856123035484*(jd - 2451545.0) #J2000 Earth state #https://space.stackexchange.com/questions/38807/transform-eci-to-ecef
r_loc = np.matmul([[np.cos(theta_g),-np.sin(theta_g),0.],[np.sin(theta_g),np.cos(theta_g),0.],[0.,0.,1.]],r_loc)
#r_loc = np.asarray([0,0,0])
r_look = r_loc/np.linalg.norm(r_loc)#np.asarray([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])

x_image = np.cross(r_look,np.asarray([0,0,1]))/np.linalg.norm(np.cross(r_look,np.asarray([0,0,1])))
#y_image = np.cross(x_image,r_look)/np.linalg.norm(np.cross(x_image,r_look))

ANG = -35.
r_look = rotve(r_look, x_image, ANG*np.pi/180.)
#x_image = np.cross(r_look,np.asarray([0,0,1]))/np.linalg.norm(np.cross(r_look,np.asarray([0,0,1])))
y_image = np.cross(x_image,r_look)/np.linalg.norm(np.cross(x_image,r_look))


rs = list()
for i in np.arange(N,dtype=int):
    e, r, v = sats[int(i)].sgp4_array(np.asarray([jd],dtype=float), np.asarray([fr],dtype=float))
    rs.append(r[0])
print("done rs")

# FoVs = [5.,10.,15.] #angles to put tickmarks at
# radialAngles = list()
# #inds_in_FoV = list()
# azimuthAngles = list()
# for i in np.arange(N,dtype=int):
#     radial_angle = np.arccos(np.dot(r_look,(rs[i]-r_loc)/np.linalg.norm(rs[i]-r_loc))) #azimuth?
#     radialAngles.append(radial_angle)
#      azimuth_angle = np.arctan2(np.dot(y_image,(rs[i]-r_loc)/np.linalg.norm(rs[i]-r_loc)),np.dot(x_image,(rs[i]-r_loc)/np.linalg.norm(rs[i]-r_loc)))
#     # azimuthAngles.append(azimuth_angle)
#     #azimuth_angle = np.arccos()# np.arctan2(np.dot(y_image,(rs[i]-r_loc)/np.linalg.norm(rs[i]-r_loc)),np.dot(x_image,(rs[i]-r_loc)/np.linalg.norm(rs[i]-r_loc)))
#     azimuthAngles.append(azimuth_angle)

#     # if radial_angle < np.max(FoVs)*np.pi/180.:
#     #     inds_in_FoV.append(i)

# points = np.zeros((N,2))
# for i in np.arange(N):
#     rhat = (rs[i]-r_loc)/np.linalg.norm(rs[i]-r_loc)
#     tmp = rhat - np.dot(rhat,x_image)
#     y_component = np.arccos(np.dot(r_look,tmp)/np.linalg.norm(tmp))
#     tmp = rhat - np.dot(rhat,y_image)
#     x_component = np.arccos(np.dot(r_look,tmp)/np.linalg.norm(tmp))
#     # tmp = (rs[i]-r_loc)/np.linalg.norm(rs[i]-r_loc)
#     # assert not np.any(np.isnan(tmp)), 'nan'
#     # x_component = tmp - y_image*np.dot(y_image,tmp)
#     # x_component = x_component/np.linalg.norm(x_component)
#     # assert not np.any(np.isnan(x_component)), 'nan y_component'
#     # y_component = tmp - x_image*np.dot(x_image,tmp)
#     # y_component = y_component/np.linalg.norm(y_component)
#     # assert not np.any(np.isnan(y_component)), 'nan y_component'
#     # points[i,0] = np.arccos(np.dot(tmp,x_component)/np.linalg.norm(x_component))
#     # print(np.sign(np.dot(tmp,x_component)))
#     # points[i,1] = np.arccos(np.dot(tmp,y_component)/np.linalg.norm(y_component))
#points = np.zeros((N,3))
points = list()
for i in np.arange(N):
    if np.any(np.isnan(rs[i])):
        continue
    #points[i] = (rs[i]-r_loc)/np.linalg.norm(rs[i]-r_loc)
    tmp = (rs[i]-r_loc)/np.linalg.norm(rs[i]-r_loc)
    for i in np.arange(len(points)):
        if np.all(np.linalg.norm(tmp - points[i])<=1e-6):
            continue
    points.append(tmp)
points = np.asarray(points)
print("points remaining: " + str(len(points)))




dots = list()
for i in np.arange(len(points)):
    dots.append(np.dot(points[i],r_look))
dots = np.asarray(dots)
indsClosest = np.argsort(dots)




#SPHERICAL voronoi diagram EXAMPLE
print('Starting SphericalVoronoi')
radius= 1.
origin = np.asarray([0.,0.,0.])
threshold = 1e-6
vor = SphericalVoronoi(points[indsClosest[:100]], radius, origin, threshold)#r_loc/np.linalg.norm(r_loc))
print('End SphericalVoronoi')

# import matplotlib.pyplot as plt
# fig = voronoi_plot_2d(vor)
# plt.show(block=False)


# sort vertices (optional, helpful for plotting)
vor.sort_vertices_of_regions()
t_vals = np.linspace(0, 1, 2000)
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
# plot the unit sphere for reference (optional)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax2.plot_surface(x, y, z, color='y', alpha=0.1)

# plot generator points
ax2.scatter(points[indsClosest[:100], 0], points[indsClosest[:100], 1], points[indsClosest[:100], 2], c='b')

# plot Voronoi vertices
ax2.scatter(vor.vertices[:, 0], vor.vertices[:, 1], vor.vertices[:, 2],c='g')

# indicate Voronoi regions (as Euclidean polygons)
for region in vor.regions:
   n = len(region)
   for i in range(n):
       start = vor.vertices[region][i]
       end = vor.vertices[region][(i + 1) % n]
       result = geometric_slerp(start, end, t_vals)
       ax2.plot(result[..., 0],result[..., 1],result[..., 2],c='k')
ax2.azim = 10
ax2.elev = 40
_ = ax2.set_xticks([])
_ = ax2.set_yticks([])
_ = ax2.set_zticks([])
fig2.set_size_inches(4, 4)
plt.show(block=False)
















#SPHERICAL voronoi diagram EXAMPLE
print('Starting SphericalVoronoi')
radius= 1.
origin = np.asarray([0.,0.,0.])
threshold = 1e-6
vor = SphericalVoronoi(points[indsClosest[:100]], radius, origin, threshold)#r_loc/np.linalg.norm(r_loc))
print('End SphericalVoronoi')

# import matplotlib.pyplot as plt
# fig = voronoi_plot_2d(vor)
# plt.show(block=False)


# sort vertices (optional, helpful for plotting)
vor.sort_vertices_of_regions()
t_vals = np.linspace(0, 1, 2000)
fig72 = plt.figure(num=72)
ax72 = fig72.add_subplot(111, projection='3d')
# plot the unit sphere for reference (optional)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax72.plot_surface(x, y, z, color='y', alpha=0.1)

# plot generator points
ax72.scatter(points[indsClosest[:100], 0], points[indsClosest[:100], 1], points[indsClosest[:100], 2], c='b')

# plot Voronoi vertices
ax72.scatter(vor.vertices[:, 0], vor.vertices[:, 1], vor.vertices[:, 2],c='g')

# indicate Voronoi regions (as Euclidean polygons)
for region in vor.regions:
   n = len(region)
   for i in range(n):
       start = vor.vertices[region][i]
       end = vor.vertices[region][(i + 1) % n]
       result = geometric_slerp(start, end, t_vals)
       ax72.plot(result[..., 0],result[..., 1],result[..., 2],c='k')
ax72.azim = 10
ax72.elev = 40
_ = ax72.set_xticks([])
_ = ax72.set_yticks([])
_ = ax72.set_zticks([])
fig72.set_size_inches(4, 4)
plt.show(block=False)










"""
phis angle between z axis and look vector in rad
thetas angle about z axis in rad
"""
def xyz_rloc_rlook_to_phitheta(r_loc,rlook,points):
    phis = np.zeros(points.shape[0])
    thetas = np.zeros(points.shape[0])

    x_tmp = np.cross(np.asarray([0.,0.,1.]),r_look)/np.linalg.norm(np.cross(np.asarray([0.,0.,1.]),r_look))
    y_tmp = np.cross(r_look,x_tmp)

    for i in np.arange(points.shape[0]):
        rhat = (points[i] - r_loc)/np.linalg.norm(points[i] - r_loc)
        phis[i] = np.arccos(np.dot(r_look,rhat)) #angle offset from z axis
        thetas[i] = np.arctan2(np.dot(rhat,y_tmp),np.dot(rhat,x_tmp))
        r_perp = rhat-r_look*np.dot(r_look,rhat)
    return phis, thetas



#Voronoi on 3d projection
#fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
maxAngle = 15.

for j in np.arange(1):
    MM = 0#np.floor(j/2) #the whole number of minutes
    SS = j*1
    #             yyyy  MM dd  HH MM SS
    jd, fr = jday(2022, 10, 3, 12, MM, SS)
    print("jd: " + str(jd) + " MM: " + str(MM) + " SS: " + str(SS))


    #look vectors
    lat = 35.58*np.pi/180. #
    lon = 5.29*np.pi/180.
    earthRad = 6371 #m
    r_loc = earthRad*np.asarray([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]) #forms z vector
    #convert from ECEF to ECI https://space.stackexchange.com/questions/43187/is-this-commonly-attributed-eci-earth-centered-inertial-to-ecef-earth-centere
    theta_g = 280.46 + 360.9856123035484*(jd - 2451545.0) #J2000 Earth state #https://space.stackexchange.com/questions/38807/transform-eci-to-ecef
    r_loc = np.matmul([[np.cos(theta_g),-np.sin(theta_g),0.],[np.sin(theta_g),np.cos(theta_g),0.],[0.,0.,1.]],r_loc)
    #rotve(v,erot,angle)
    #r_loc = np.asarray([0,0,0])
    r_look = r_loc/np.linalg.norm(r_loc)#np.asarray([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])


    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot([0.,r_loc[0]],[0.,r_loc[1]],[0.,r_loc[2]],c='k')
    ax3.plot([r_loc[0],r_loc[0]+r_look[0]*0.3*earthRad],[r_loc[1],r_loc[1]+r_look[1]*0.3*earthRad],[r_loc[2],r_loc[2]+r_look[2]*0.3*earthRad],c='r')
    ax3.plot([0,earthRad],[0,0],[0,0],c='b')
    ax3.plot([0,0],[0,earthRad],[0,0],c='b')
    ax3.plot([0,0],[0,0],[0,earthRad],c='b')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    plt.show(block=False)


    #ROTATION OF LOOK IS OK
    #Rotate r_look so it is looking at ecliptic
    x_image = np.cross(r_look,np.asarray([0,0,1]))/np.linalg.norm(np.cross(r_look,np.asarray([0,0,1])))
    #y_image = np.cross(x_image,r_look)/np.linalg.norm(np.cross(x_image,r_look))
    r_look = rotve(r_look, x_image, ANG*np.pi/180.)
    #x_image = np.cross(r_look,np.asarray([0,0,1]))/np.linalg.norm(np.cross(r_look,np.asarray([0,0,1])))
    #y_image = np.cross(x_image,r_look)/np.linalg.norm(np.cross(x_image,r_look))


    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(111, projection='3d')
    ax4.plot([0.,r_loc[0]],[0.,r_loc[1]],[0.,r_loc[2]],c='k')
    ax4.plot([r_loc[0],r_loc[0]+r_look[0]*0.3*earthRad],[r_loc[1],r_loc[1]+r_look[1]*0.3*earthRad],[r_loc[2],r_loc[2]+r_look[2]*0.3*earthRad],c='r')
    ax4.plot([0,earthRad],[0,0],[0,0],c='b')
    ax4.plot([0,0],[0,earthRad],[0,0],c='b')
    ax4.plot([0,0],[0,0],[0,earthRad],c='b')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')
    plt.show(block=False)



    rs = list()
    for i in np.arange(N,dtype=int):
        e, r, v = sats[int(i)].sgp4_array(np.asarray([jd],dtype=float), np.asarray([fr],dtype=float))
        #Require SV to be within FOV
        if np.arccos(np.dot((r[0]-r_loc)/np.linalg.norm(r[0]-r_loc),r_look)) <= maxAngle*np.pi/180.: #in rad
            rs.append(r[0])
    rs = np.asarray(rs)

    ax3.scatter(rs[:,0],rs[:,1],rs[:,2],c='b')
    ax4.scatter(rs[:,0],rs[:,1],rs[:,2],c='b')
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')
    plt.show(block=False)


    FoVs = [5.,10.,15.] #angles to put tickmarks at
    # radialAngles = list()
    # #inds_in_FoV = list()
    # azimuthAngles = list()
    # for i in np.arange(len(rs),dtype=int):
    #     tmp = (rs[i]-r_loc)/np.linalg.norm(rs[i]-r_loc)
    #     radial_angle = np.arccos(np.dot(r_look,tmp))
    #     radialAngles.append(radial_angle)
    #     azimuth_angle = np.arctan2(np.dot(y_image,tmp),np.dot(x_image,tmp))
    #     azimuthAngles.append(azimuth_angle)

        # if radial_angle < np.max(FoVs)*np.pi/180.:
        #     inds_in_FoV.append(i)


    #voronoi diagram
    rs = np.asarray(rs)
    points = np.zeros(rs.shape)
    for i in np.arange(rs.shape[0]):
        points[i] = (rs[i]-r_loc)/np.linalg.norm(rs[i]-r_loc)
    dots = list()
    for i in np.arange(len(points)):
        dots.append(np.dot(points[i],r_look))
    dots = np.asarray(dots)
    indsClosest = np.argsort(dots)


    fig6 = plt.figure(6)
    ax6 = fig6.add_subplot(111, projection='3d')
    #ax6.plot([0.,r_loc[0]],[0.,r_loc[1]],[0.,r_loc[2]],c='k')
    ax6.plot([0,r_look[0]],[0,r_look[1]],[0,r_look[2]],c='r')
    ax6.plot([0,1],[0,0],[0,0],c='b')
    ax6.plot([0,0],[0,1],[0,0],c='b')
    ax6.plot([0,0],[0,0],[0,1],c='b')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_zlabel('z')
    ax6.scatter(points[:,0],points[:,1],points[:,2],c='b',s=2)
    ax6.set_aspect('equal')
    plt.show(block=False)

    radius= 1.
    origin = np.asarray([0.,0.,0.])
    threshold = 1e-10
    vor = SphericalVoronoi(points, radius, origin, threshold) #r_loc/np.linalg.norm(r_loc))
    ax6.scatter(vor.vertices[:,0],vor.vertices[:,1],vor.vertices[:,2],c='g',s=2)



    fig10 = plt.figure(num=10)
    ax10 = fig10.add_subplot(111, projection='3d')


    #MOVE BACK TO TOP
    fig5, ax5 = plt.subplots(subplot_kw={'projection': 'polar'},num=5)
    # plot generator points
    out0 = xyz_rloc_rlook_to_phitheta(np.asarray([0.,0.,0.]),r_look,points)
    phis = out0[0]
    thetas = out0[1]
    ax5.scatter(thetas, phis*180/np.pi,s=5,c='blue',zorder=10)

    # plot Voronoi vertices, done here so it plots on top of the black lines
    out1 = xyz_rloc_rlook_to_phitheta(np.asarray([0.,0.,0.]),r_look,vor.vertices)
    ax5.scatter(out1[1], out1[0]*180/np.pi,c='g',s=5,zorder=10)

    # indicate Voronoi regions (as Euclidean polygons)
    for region in vor.regions:
        #region = vor.regions[0]
        n = len(region)
        for i in range(n):
            start = vor.vertices[region][i]
            end = vor.vertices[region][(i + 1) % n]
            result = geometric_slerp(start, end, t_vals)
            result_radial = np.zeros(result.shape[0])
            result_azimuth = np.zeros(result.shape[0])
            #for k in np.arange(result.shape[0]):
                #result_radial[k] = np.arccos(np.dot(r_look,result[k]))
                #tmp = result[k]-np.dot(r_look,result[k])*r_look
                #result_azimuth[k] = np.arctan2(np.dot(tmp,y_image),np.dot(tmp,x_image))#np.arctan2(np.dot(y_image,result[k]),np.dot(x_image,result[k]))
                ##result_radial[k] = np.arccos(np.dot(r_look,result[k]))
                #tmp = result[k]-np.dot(r_look,result[k])*r_look
                ##result_azimuth[k] = np.arctan2(np.dot(result[k],y_image),np.dot(result[k],x_image))#np.arctan2(np.dot(y_image,result[k]),np.dot(x_image,result[k]))
            out2 = xyz_rloc_rlook_to_phitheta(np.asarray([0.,0.,0.]),r_look,result)
            result_radial = out2[0]
            result_azimuth = out2[1]
                #result_radial[k] = result_radial[k]
            #ax.plot(result[..., 0],result[..., 1],result[..., 2],c='k')
            ax5.plot(result_azimuth,result_radial*180/np.pi,c='k',linewidth=1)
            #ax5.scatter(result_azimuth*180./np.pi,result_radial*180./np.pi,c='cyan',s=1)
            ax10.plot(result[..., 0],result[..., 1],result[..., 2],c='k',zorder=5)

    

    ax5.set_rmax(maxAngle)
    ax5.set_rticks([5,10,15])  # Less radial ticks
    ax5.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax5.grid(True)
    ax5.set_title("SV in SKY", va='bottom')
    plt.show(block=False)
    
    
    #fig.clf()
    #fig.canvas.draw()
    #fig.canvas.flush_events()
    #plt.show(block=False)
    plt.pause(0.05)
    #ax5.cla()
    #time.sleep(0.1)
#plt.ioff()
#plt.show(block=False)


